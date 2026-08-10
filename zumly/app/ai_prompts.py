"""Pure prompt-normalization helpers shared by AI feature builders."""

NARRATION_SECTION_ORDER = (
    "Context",
    "Background",
    "Prompt / Action",
    "Walkthrough",
    "Result",
)


def format_time_label(timestamp_ms: float) -> str:
    """Render a millisecond timestamp as a compact mm:ss / hh:mm:ss label."""
    total_seconds = max(0, int(round(timestamp_ms / 1000.0)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def clean_markdown_response(text: str) -> str:
    """Normalize an LLM markdown response before saving it."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def clean_json_response(text: str) -> str:
    """Normalize an LLM JSON response before parsing."""
    cleaned = clean_markdown_response(text)
    if cleaned.lower().startswith("json\n"):
        cleaned = cleaned[5:]
    return cleaned.strip()


_NARRATION_SECTION_ORDER = NARRATION_SECTION_ORDER
_format_time_label = format_time_label
_clean_markdown_response = clean_markdown_response
_clean_json_response = clean_json_response
