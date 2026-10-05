"""Gemini client with token accounting, retries and an on-disk cache.

Token counts come from the API's own ``usage_metadata`` rather than an
estimate, because the benchmark's central claim is about cost. Where the API
omits usage, the fallback is flagged in ``Reply.estimated`` so the dashboard
never silently mixes measured and guessed numbers.

The cache is keyed on (model, prompt, schema) and makes benchmark re-runs free,
which matters a great deal when iterating against a free-tier quota.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .trace import TokenUsage

_CACHE_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "llm_cache"


@dataclass
class Reply:
    text: str
    tokens: TokenUsage
    estimated: bool = False
    cached: bool = False

    def json(self) -> Any:
        """Parse the reply as JSON, tolerating ```json fences."""
        t = self.text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t
            t = t.rsplit("```", 1)[0]
        t = t.strip()
        if t.startswith("json"):
            t = t[4:].strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            start, end = t.find("{"), t.rfind("}")
            if start >= 0 and end > start:
                return json.loads(t[start : end + 1])
            raise


class RateLimiter:
    """Simple requests-per-minute gate, shared across threads."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self.interval:
                time.sleep(self.interval - delta)
            self._last = time.monotonic()


class GeminiClient:
    """Thin wrapper over google-genai with usage tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        rpm: int = 90,
        use_cache: bool = True,
    ):
        from google import genai  # imported lazily so the module loads without the dep

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add a key "
                "from https://aistudio.google.com/apikey"
            )
        self.model = model or os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
        self.embed_model = embed_model or os.environ.get(
            "GEMINI_EMBED_MODEL", "text-embedding-004"
        )
        self._client = genai.Client(api_key=self.api_key)
        self._limiter = RateLimiter(rpm)
        self.use_cache = use_cache
        if use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Session totals, for a cost line at the end of a benchmark run.
        self.session = TokenUsage()
        self.n_calls = 0
        self.n_cache_hits = 0

    # -- caching -----------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        return _CACHE_DIR / f"{key}.json"

    def _cache_key(self, prompt: str, system: str, temperature: float) -> str:
        blob = json.dumps(
            [self.model, system, prompt, temperature], ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:32]

    # -- generation --------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_retries: int = 5,
    ) -> Reply:
        key = self._cache_key(prompt, system, temperature)
        if self.use_cache:
            p = self._cache_path(key)
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                self.n_cache_hits += 1
                return Reply(
                    d["text"],
                    TokenUsage(d["prompt_tokens"], d["completion_tokens"]),
                    estimated=d.get("estimated", False),
                    cached=True,
                )

        from google.genai import types

        cfg = types.GenerateContentConfig(temperature=temperature)
        if system:
            cfg.system_instruction = system

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                self._limiter.wait()
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                )
                text = (resp.text or "").strip()
                um = getattr(resp, "usage_metadata", None)
                if um and getattr(um, "prompt_token_count", None) is not None:
                    tokens = TokenUsage(
                        int(um.prompt_token_count or 0),
                        int(getattr(um, "candidates_token_count", 0) or 0),
                    )
                    estimated = False
                else:
                    tokens = TokenUsage(_approx_tokens(system + prompt), _approx_tokens(text))
                    estimated = True

                self.session += tokens
                self.n_calls += 1
                if self.use_cache:
                    self._cache_path(key).write_text(
                        json.dumps(
                            {
                                "text": text,
                                "prompt_tokens": tokens.prompt,
                                "completion_tokens": tokens.completion,
                                "estimated": estimated,
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                return Reply(text, tokens, estimated=estimated)
            except Exception as exc:  # noqa: BLE001 - surface after retries
                last_err = exc
                if attempt < max_retries - 1:
                    if _is_rate_limit(exc):
                        time.sleep(min(2**attempt * 2, 60))
                    elif _is_transient(exc):
                        time.sleep(min(2**attempt * 3, 45))
                    else:
                        time.sleep(2**attempt)
                    continue
                break
        raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")

    # -- embeddings --------------------------------------------------------
    def embed(self, texts: list[str], batch_size: int = 100, task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """Embed a list of texts, batching to stay inside quota."""
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for attempt in range(5):
                try:
                    self._limiter.wait()
                    resp = self._client.models.embed_content(
                        model=self.embed_model,
                        contents=chunk,
                        config=types.EmbedContentConfig(task_type=task),
                    )
                    out.extend([list(e.values) for e in resp.embeddings])
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 4:
                        raise RuntimeError(f"embedding failed: {exc}") from exc
                    time.sleep(min(2**attempt * 2, 60))
        return out


def _approx_tokens(text: str) -> int:
    """~4 characters per token. Only used when the API omits usage metadata."""
    return max(1, len(text) // 4)


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or "quota" in s or "rate" in s


def _is_transient(exc: Exception) -> bool:
    """Connection resets and 5xx are worth a slower retry, not a failure.

    A dropped connection during synthesis otherwise discards an answer the
    retrieval layer already got exactly right.
    """
    s = str(exc).lower()
    return any(
        m in s
        for m in (
            "10054", "connection", "reset", "timeout", "timed out", "eof",
            "500", "502", "503", "504", "unavailable", "internal",
        )
    )


def load_env(path: str | Path | None = None) -> None:
    """Load .env without requiring python-dotenv."""
    p = Path(path) if path else Path(__file__).resolve().parents[2] / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
