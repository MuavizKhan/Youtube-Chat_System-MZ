"""Bounded transcript selection and validation helpers for video summaries."""
from __future__ import annotations
import json
import math
import re
from typing import Any, Sequence

MAX_SOURCE_CHARS = 48_000
MAX_DOCUMENTS = 60
MAX_SECTION_CHARS = 4_500
MAX_SECTIONS = 12


class OverviewError(RuntimeError):
    """Raised when a source-linked overview cannot be produced safely."""


def metadata(document: Any) -> dict:
    value = getattr(document, "metadata", {})
    return value if isinstance(value, dict) else {}


def sort_key(document: Any) -> tuple[float, float, str]:
    data = metadata(document)
    try:
        start, end = float(data.get("start", 0)), float(data.get("end", data.get("start", 0)))
    except (TypeError, ValueError, OverflowError):
        start = end = 0.0
    if not math.isfinite(start):
        start = 0.0
    if not math.isfinite(end):
        end = start
    return start, end, str(data.get("chunk_id", ""))


def even_sample(items: Sequence[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return list(items)
    if count <= 1:
        return [items[(len(items) - 1) // 2]]
    indexes = sorted({round(i * (len(items) - 1) / (count - 1)) for i in range(count)})
    return [items[index] for index in indexes]


def requested_count(question: str) -> int:
    match = re.search(
        r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:useful\s+)?(?:points?|pointers?|bullets?|takeaways|items?)\b",
        question.casefold(),
    )
    if not match:
        return 7
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    try:
        value = int(match.group(1))
    except ValueError:
        value = words[match.group(1)]
    return max(1, min(value, 10))


def parse_points(
    raw: Any,
    allowed_ids: set[str],
    limit: int,
    *,
    allow_empty: bool = False,
) -> list[dict]:
    if not isinstance(raw, str) or not raw.strip():
        raise OverviewError("Overview model returned an empty response.")
    content = raw.strip()
    fence = chr(96) * 3
    if content.startswith(fence):
        content = re.sub(r"^" + fence + r"(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*" + fence + r"$", "", content)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise OverviewError("Overview model returned invalid JSON.") from error
    if not isinstance(payload, dict) or set(payload) != {"points"} or not isinstance(payload["points"], list):
        raise OverviewError("Overview model returned an invalid response schema.")
    result = []
    for item in payload["points"]:
        if not isinstance(item, dict) or set(item) != {"text", "evidence_ids"}:
            continue
        text, ids = item["text"], item["evidence_ids"]
        if not isinstance(text, str) or not isinstance(ids, list) or not ids:
            continue
        text = " ".join(text.split())
        ids = list(dict.fromkeys(ids))
        if not text or len(text) > 600 or any(not isinstance(v, str) for v in ids):
            continue
        if not set(ids).issubset(allowed_ids):
            continue
        result.append({"text": text, "evidence_ids": ids})
        if len(result) >= limit:
            break
    if not result and not allow_empty:
        raise OverviewError("Overview contains no valid source-linked points.")
    return result


def build_sections(documents: Sequence[Any]) -> tuple[list[list[dict]], dict, dict]:
    valid, seen = [], set()
    for document in documents:
        text = getattr(document, "page_content", None)
        data = metadata(document)
        chunk_id = data.get("chunk_id")
        if not isinstance(text, str) or not text.strip() or chunk_id is None:
            continue
        key = (str(data.get("video_id", "")), str(chunk_id))
        if key not in seen:
            seen.add(key)
            valid.append(document)
    valid.sort(key=sort_key)
    if not valid:
        raise OverviewError("No indexed transcript chunks are available.")
    video_ids = {
        str(metadata(document).get("video_id"))
        for document in valid
        if metadata(document).get("video_id") is not None
    }
    if len(video_ids) > 1:
        raise OverviewError("Overview documents must belong to exactly one video.")

    total_chars = sum(len(item.page_content) for item in valid)
    selected, mode = valid, "full_transcript"
    if len(valid) > MAX_DOCUMENTS or total_chars > MAX_SOURCE_CHARS:
        mode = "timeline_stratified"
        average = max(1, math.ceil(total_chars / len(valid)))
        count = min(MAX_DOCUMENTS, max(1, MAX_SOURCE_CHARS // average))
        selected = even_sample(valid, count)
        while len(selected) > 1 and sum(len(item.page_content) for item in selected) > MAX_SOURCE_CHARS:
            selected = even_sample(selected, len(selected) - 1)

    def timestamp(document: Any) -> str:
        start, end, _ = sort_key(document)
        def fmt(seconds: float) -> str:
            value = max(0, int(seconds))
            hours, rem = divmod(value, 3600)
            mins, secs = divmod(rem, 60)
            return f"{hours:02d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"
        return f"{fmt(start)}–{fmt(end)}"

    sections, current, size, lookup = [], [], 0, {}
    for index, document in enumerate(selected, start=1):
        passage_id = f"P{index:05d}"
        chunk_id = metadata(document)["chunk_id"]
        line = f"[{passage_id}] [{timestamp(document)}] (chunk {chunk_id}) {document.page_content[:MAX_SECTION_CHARS]}"
        if current and size + len(line) > MAX_SECTION_CHARS:
            sections.append(current)
            current, size = [], 0
        current.append({"id": passage_id, "document": document, "line": line})
        lookup[passage_id] = document
        size += len(line)
    if current:
        sections.append(current)
    if len(sections) > MAX_SECTIONS:
        raise OverviewError("Transcript exceeds the bounded overview section limit.")
    stats = {
        "coverage_mode": mode,
        "indexed_document_count": len(valid),
        "selected_document_count": len(selected),
        "indexed_source_chars": total_chars,
        "selected_source_chars": sum(len(item.page_content) for item in selected),
        "selected_fraction": round(len(selected) / len(valid), 4),
        "section_count": len(sections),
    }
    return sections, lookup, stats
