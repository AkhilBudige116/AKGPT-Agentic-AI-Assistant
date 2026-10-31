"""Security helpers: filename/path safety, secret redaction, sensitive-data
detection, identifier validation and a small in-memory rate limiter."""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from collections import defaultdict, deque
from pathlib import Path

from exceptions import RateLimitedError, ValidationFailed

# ---------------------------------------------------------------- identifiers
THREAD_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SCOPE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")  # secrets.token_urlsafe(32)


def validate_thread_id(value: str) -> str:
    if not isinstance(value, str) or not THREAD_ID_RE.match(value):
        raise ValidationFailed("Invalid conversation identifier.")
    return value


def is_valid_scope_id(value: str | None) -> bool:
    return bool(value) and bool(SCOPE_ID_RE.match(value))


# ------------------------------------------------------------------ filenames
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_BAD_CHARS = re.compile(r'[<>:"/\\|?*]')
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".py"}


def sanitize_filename(name: str | None, default: str = "document") -> str:
    """Return a display-safe filename (never used to build filesystem paths)."""
    if not name:
        return default
    name = unicodedata.normalize("NFKC", name)
    name = name.replace("\\", "/").split("/")[-1]  # drop any directory component
    name = _CONTROL.sub("", name)
    name = _BAD_CHARS.sub("_", name).strip(" .")
    if not name:
        return default
    stem, dot, ext = name.rpartition(".")
    if len(name) > 150:
        keep = max(1, 150 - len(ext) - 1)
        name = (stem[:keep] + "." + ext) if dot else name[:150]
    return name


def safe_extension(filename: str | None) -> str:
    """Lower-case extension of a user-supplied filename (may not be allowed)."""
    return Path(sanitize_filename(filename)).suffix.lower()


def resolve_inside(base: Path, *parts: str) -> Path:
    """Join ``parts`` to ``base`` and ensure the result stays inside ``base``."""
    base_resolved = base.resolve()
    target = base_resolved.joinpath(*parts).resolve()
    if base_resolved != target and base_resolved not in target.parents:
        raise ValidationFailed("Invalid file path.")
    return target


# ------------------------------------------------------------------ redaction
_REDACTIONS = [
    (re.compile(r"(?i)(appid|api[_-]?key|apikey|access[_-]?token|token|secret|password)=([^&\s\"']+)"),
     r"\1=[REDACTED]"),
    (re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?)(bearer\s+)?[A-Za-z0-9._\-]{8,}"),
     r"\1[REDACTED]"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{16,}\b"), "[REDACTED]"),
    (re.compile(r"\btvly-[A-Za-z0-9_\-]{8,}\b"), "[REDACTED]"),
    (re.compile(r"\blsv2_[A-Za-z0-9_]{8,}\b"), "[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), "[REDACTED]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}"), "Bearer [REDACTED]"),
]


def redact(text: str) -> str:
    """Remove credential-looking substrings from text destined for logs/errors."""
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


# ------------------------------------------------- sensitive-data detection
_SENSITIVE_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgsk_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\btvly-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\b(password|passwd|pwd|api[_ -]?key|secret|token)\s*(is|=|:)\s*\S{4,}"),
]
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def looks_sensitive(text: str) -> bool:
    """Heuristic check used to refuse storing secrets/payment data in memory."""
    if any(p.search(text) for p in _SENSITIVE_PATTERNS):
        return True
    for match in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return False


# ---------------------------------------------------------------- rate limit
class RateLimiter:
    """Sliding-window limiter keyed by an arbitrary string (thread-safe)."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry = max(1, int(self.window - (now - hits[0])) + 1)
                raise RateLimitedError(headers={"Retry-After": str(retry)})
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory
                for k in [k for k, v in self._hits.items() if not v][:1000]:
                    self._hits.pop(k, None)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
