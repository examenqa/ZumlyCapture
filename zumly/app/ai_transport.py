"""Gemini endpoint validation and payload translation primitives."""

import math
from typing import Any, List
from urllib.parse import urlparse

DEFAULT_AI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
STANDARD_GOOGLE_AI_HOSTS = frozenset({
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
})


class GeminiAPIError(RuntimeError):
    """HTTP-aware Gemini API failure used by the fallback controller."""

    def __init__(self, message: str, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def validate_ai_endpoint(endpoint: str) -> str:
    """Validate an HTTPS endpoint before credentials can be sent to it."""
    value = (endpoint or DEFAULT_AI_ENDPOINT).strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme.lower() != "https":
        raise ValueError("AI endpoints must use HTTPS; HTTP endpoints are not allowed.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("AI endpoint must be a valid HTTPS URL without embedded credentials.")
    return value


def is_standard_google_ai_endpoint(endpoint: str) -> bool:
    """Return whether an endpoint belongs to Google AI Studio or Vertex AI."""
    try:
        hostname = (urlparse(validate_ai_endpoint(endpoint)).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return (
        hostname in STANDARD_GOOGLE_AI_HOSTS
        or hostname.endswith(".aiplatform.googleapis.com")
        or hostname.endswith("-aiplatform.googleapis.com")
    )


def normalize_gemini_endpoint(endpoint: str) -> str:
    """Return an HTTPS Interactions API endpoint from user settings."""
    value = validate_ai_endpoint(endpoint)
    if not value:
        return DEFAULT_AI_ENDPOINT
    if value.endswith("/interactions"):
        return value
    if value.endswith("/v1beta"):
        return f"{value}/interactions"
    hostname = (urlparse(value).hostname or "").lower().rstrip(".")
    if hostname == "generativelanguage.googleapis.com":
        return f"{value}/v1beta/interactions"
    return value


def parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, 30.0)


def data_url_to_gemini_image(url: str) -> dict[str, str]:
    header, _, payload = url.partition(",")
    mime_type = "image/png"
    if header.startswith("data:"):
        mime_type = header[5:].split(";", 1)[0] or mime_type
    return {"type": "image", "data": payload, "mime_type": mime_type}


def legacy_content_to_gemini_input(
    content: str | List[dict[str, Any]],
) -> str | list[dict[str, Any]]:
    """Translate Zumly's existing prompt parts to Gemini Interactions input."""
    if isinstance(content, str):
        return content
    converted: list[dict[str, Any]] = []
    for item in content or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            converted.append({"type": "text", "text": str(item.get("text", ""))})
            continue
        if kind == "image":
            image = {"type": "image"}
            if item.get("data"):
                image["data"] = str(item.get("data"))
            if item.get("uri"):
                image["uri"] = str(item.get("uri"))
            image["mime_type"] = str(item.get("mime_type", "image/png"))
            converted.append(image)
            continue
        if kind == "image_url":
            image_url = item.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
            if isinstance(url, str) and url.startswith("data:"):
                converted.append(data_url_to_gemini_image(url))
            elif url:
                converted.append({
                    "type": "image",
                    "uri": str(url),
                    "mime_type": str(image_url.get("mime_type", "image/png")),
                })
    return converted


# Compatibility names for callers that used the private service helpers.
_normalize_gemini_endpoint = normalize_gemini_endpoint
_parse_retry_after_seconds = parse_retry_after_seconds
_data_url_to_gemini_image = data_url_to_gemini_image
_legacy_content_to_gemini_input = legacy_content_to_gemini_input
