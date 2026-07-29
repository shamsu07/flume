from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_text(value: str) -> str:
    """Normalize user-authored text without changing meaningful line boundaries."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def canonical_identifier(value: str) -> str:
    """Normalize an identifier while keeping internal punctuation significant."""
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        raise ValueError("value cannot be empty")
    if any(unicodedata.category(character) == "Cc" for character in normalized):
        raise ValueError("value cannot contain control characters")
    return normalized


def canonical_json_value(value: Any) -> Any:
    """Return a JSON-safe, Unicode-normalized value suitable for stable hashing."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, str):
        return canonical_text(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            normalized_key = canonical_identifier(key)
            if normalized_key in normalized:
                raise ValueError(
                    f"JSON object contains colliding normalized key {normalized_key!r}"
                )
            normalized[normalized_key] = canonical_json_value(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [canonical_json_value(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_token_ids(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token ids must be integers")
        if not 0 <= token_id < 2**64:
            raise ValueError("token ids must be unsigned 64-bit integers")
        digest.update(token_id.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()
