from __future__ import annotations

import os
import re
import secrets
import threading
import time
import unicodedata
import ipaddress
from collections import defaultdict, deque
from typing import Optional

from fastapi import HTTPException, Request, status

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_WHITESPACE_RE = re.compile(r"\s+")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:@/-]+$")

MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
MAX_IDENTIFIER_LENGTH = int(os.getenv("MAX_IDENTIFIER_LENGTH", "255"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
APP_ENV = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development")).lower()
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY") or ("dev-internal-api-key" if APP_ENV not in {"prod", "production"} else None)
INTERNAL_API_KEY_HEADER = "x-internal-api-key"

if APP_ENV in {"prod", "production"} and not INTERNAL_API_KEY:
    raise RuntimeError("INTERNAL_API_KEY must be set in production environments")


def normalize_text(value: str, *, field_name: str, max_length: int = MAX_TEXT_LENGTH, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field_name} must be a string",
        )

    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHARS_RE.sub("", normalized).strip()

    if not allow_empty and not normalized:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field_name} cannot be empty",
        )

    if len(normalized) > max_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field_name} exceeds max length of {max_length}",
        )

    return normalized


def normalize_optional_text(value: Optional[str], *, field_name: str, max_length: int = MAX_TEXT_LENGTH) -> Optional[str]:
    if value is None:
        return None
    normalized = normalize_text(value, field_name=field_name, max_length=max_length, allow_empty=True)
    return normalized or None


def normalize_identifier(value: str, *, field_name: str, max_length: int = MAX_IDENTIFIER_LENGTH) -> str:
    normalized = normalize_text(value, field_name=field_name, max_length=max_length)
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field_name} cannot be blank",
        )

    if _WHITESPACE_RE.search(normalized):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field_name} cannot contain whitespace",
        )

    if not _SAFE_IDENTIFIER_RE.fullmatch(normalized):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field_name} contains unsupported characters",
        )

    return normalized


def normalize_optional_identifier(value: Optional[str], *, field_name: str, max_length: int = MAX_IDENTIFIER_LENGTH) -> Optional[str]:
    if value is None:
        return None
    return normalize_identifier(value, field_name=field_name, max_length=max_length)


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def enforce(self, key: str, limit: int, window_seconds: int) -> None:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                )
            bucket.append(now)


rate_limiter = InMemoryRateLimiter()


def get_client_ip(request: Request) -> str:
    client_host = request.client.host if request.client and request.client.host else None
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for and client_host:
        try:
            proxy_ip = ipaddress.ip_address(client_host)
            candidate_ip = ipaddress.ip_address(forwarded_for.split(",")[0].strip())
            if proxy_ip.is_loopback or proxy_ip.is_private:
                return candidate_ip.compressed
        except ValueError:
            pass
    if client_host:
        return client_host
    return "unknown"


def enforce_rate_limit(request: Request, *, scope: str, limit: int, window_seconds: int) -> None:
    client_ip = get_client_ip(request)
    route_key = f"{scope}:{client_ip}"
    rate_limiter.enforce(route_key, limit=limit, window_seconds=window_seconds)


def require_internal_api_key(request: Request) -> None:
    provided = request.headers.get(INTERNAL_API_KEY_HEADER)
    if not provided or not INTERNAL_API_KEY or not secrets.compare_digest(provided, INTERNAL_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API key",
        )
