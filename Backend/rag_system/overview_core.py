"""Bounded transcript selection and validation helpers for video summaries."""
from __future__ import annotations
import json
import math
import re
from typing import Any, Sequence

# Keep ordinary long videos fully represented while retaining hard upper bounds.
MAX_SOURCE_CHARS = 80_000
MAX_DOCUMENTS = 100
MAX_SECTION_CHARS = 9_000
MAX_SECTIONS = 10


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
    # Some chat-model completions append a short explanation or other
    # trailing text after a valid JSON object despite the JSON-only prompt.
    # Decode candidate objects independently and accept exactly one object
    # with the required top-level schema. The unparsed prefix/suffix is never
    # treated as evidence; every point still passes the strict ID/schema checks
    # below. Multiple matching objects remain ambiguous and fail closed.
    decoder = json.JSONDecoder()
    payload_candidates = []
    for match in re.finditer(r"\{", content):
        try:
            candidate, _end = decoder.raw_decode(content, match.start())
        except json.JSONDecodeError:
            continue
        if (
            isinstance(candidate, dict)
            and set(candidate) == {"points"}
            and isinstance(candidate.get("points"), list)
        ):
            payload_candidates.append(candidate)

    if not payload_candidates:
        raise OverviewError("Overview model returned invalid JSON.")
    if len(payload_candidates) != 1:
        raise OverviewError("Overview model returned multiple ambiguous JSON payloads.")
    payload = payload_candidates[0]
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

    def pack_sections(items: Sequence[Any]) -> tuple[list[list[dict]], dict]:
        sections: list[list[dict]] = []
        current: list[dict] = []
        size = 0
        lookup: dict[str, Any] = {}
        for index, document in enumerate(items, start=1):
            passage_id = f"P{index:05d}"
            chunk_id = metadata(document)["chunk_id"]
            prefix = f"[{passage_id}] [{timestamp(document)}] (chunk {chunk_id}) "
            if len(prefix) > MAX_SECTION_CHARS:
                raise OverviewError("Transcript chunk metadata exceeds the section limit.")
            # The complete rendered row, including its timestamp and evidence
            # identifier, must fit the section budget. Previously the text slice
            # alone used MAX_SECTION_CHARS, allowing a single row to exceed it.
            content_limit = MAX_SECTION_CHARS - len(prefix)
            line = prefix + document.page_content[:content_limit]
            if current and size + len(line) > MAX_SECTION_CHARS:
                sections.append(current)
                current, size = [], 0
            current.append({"id": passage_id, "document": document, "line": line})
            lookup[passage_id] = document
            size += len(line)
        if current:
            sections.append(current)
        return sections, lookup

    sections, lookup = pack_sections(selected)
    section_limit_sampling = False
    # Header overhead and uneven chunk lengths can create more sections than
    # the source-character and document-count bounds imply. Keep the map/reduce
    # call count bounded by sampling fewer documents evenly across the entire
    # timeline until the selected transcript fits the fixed section budget.
    while len(sections) > MAX_SECTIONS and len(selected) > 1:
        target_count = max(
            1,
            math.floor(len(selected) * MAX_SECTIONS / len(sections)),
        )
        if target_count >= len(selected):
            target_count = len(selected) - 1
        selected = even_sample(valid, target_count)
        while (
            len(selected) > 1
            and sum(len(item.page_content) for item in selected) > MAX_SOURCE_CHARS
        ):
            selected = even_sample(valid, len(selected) - 1)
        section_limit_sampling = True
        sections, lookup = pack_sections(selected)

    if len(sections) > MAX_SECTIONS:
        raise OverviewError("Transcript exceeds the bounded overview section limit.")
    if section_limit_sampling:
        mode = "timeline_stratified"

    stats = {
        "coverage_mode": mode,
        "indexed_document_count": len(valid),
        "selected_document_count": len(selected),
        "indexed_source_chars": total_chars,
        "selected_source_chars": sum(len(item.page_content) for item in selected),
        "selected_fraction": round(len(selected) / len(valid), 4),
        "section_count": len(sections),
        "section_limit_sampling": section_limit_sampling,
    }
    return sections, lookup, stats
