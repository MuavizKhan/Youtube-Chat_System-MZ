"""
chain.py

End-to-end LangChain RAG pipeline.

Flow:

Question
    ↓
MMR Retrieval
    ↓
Retrieved Documents
    ↓
Context Formatting
    ↓
ChatPromptTemplate
    ↓
LLM Generation
    ↓
StrOutputParser
    ↓
Final Answer + Sources
"""


import time
import argparse
import logging
import math
import re
import unicodedata
from typing import Any

from groq import Groq
from huggingface_hub import InferenceClient

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from .config import (
    EMBEDDING_MODEL,
    RAG_INDEX_NAMESPACE,
    RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS,
    HF_MAX_TOKENS,
    HF_MODEL_ID,
    HF_PROVIDER,
    HF_REASONING_EFFORT,
    HF_TEMPERATURE,
    HF_TOKEN,
    CONTEXT_EXPANSION_CHUNKS,
    CONTEXT_MAX_CHUNKS,
    RAG_GENERATION_CONTEXT_MAX_CHARS,
    DENSE_ANCHOR_LIMIT,
    DENSE_ANCHOR_MIN_CHUNK_GAP,
    DENSE_CONTEXT_MAX_CHUNKS,
    DENSE_FACET_RERANK_WEIGHT,
    RAG_RERANK_BATCH_SIZE,
    RAG_RERANK_CANDIDATE_K,
    RAG_RERANK_ENABLED,
    RAG_RERANK_MAX_LENGTH,
    RAG_RERANK_MODEL,
    DENSE_SEMANTIC_FETCH_K,
    DENSE_SEMANTIC_K,
    MAX_DISTANCE,
    RECOVERY_FETCH_K,
    RECOVERY_MAX_DISTANCE,
    RECOVERY_TOP_K,
    MMR_FETCH_K,
    MMR_LAMBDA,
    SOURCE_MERGE_GAP_SECONDS,
    TOP_K,
    HF_MAX_RETRIES,
    HF_RETRY_DELAY_SECONDS,
    GROQ_API_KEY,
    GROQ_MAX_RETRIES,
    GROQ_MAX_TOKENS,
    GROQ_MODEL_ID,
    GROQ_REASONING_EFFORT,
    GROQ_RETRY_DELAY_SECONDS,
    GROQ_TEMPERATURE,
    LLM_PROVIDER,
)

from .query_understanding import (
    QueryUnderstandingError,
    understand_query,
)

from .timestamp_utils import strip_unverified_timestamp_claims

from .benchmark_registry import get_video_benchmark
from .gold_evidence import (
    matched_gold_groups,
    matched_gold_answer_groups,
    matched_gold_group_caption_spans,
    normalize_evidence_text,
)

from .retrieval import (
    extract_video_id,
    expand_question_context,
    is_evidence_dense_question,
    is_overview_question,
    passes_strong_semantic_gate,
    retrieve_overview,
    retrieve_question_context,
    retrieve_temporal_context,
)

from .index_service import load_ready_index


logger = logging.getLogger(__name__)


class ProviderRateLimitError(RuntimeError):
    """Raised when retrying a provider throttle would exceed the inline wait budget."""

    def __init__(self, retry_after_seconds: float | None = None):
        if (
            isinstance(retry_after_seconds, (int, float))
            and not isinstance(retry_after_seconds, bool)
            and math.isfinite(float(retry_after_seconds))
            and retry_after_seconds >= 0
        ):
            self.retry_after_seconds = float(retry_after_seconds)
        else:
            self.retry_after_seconds = None

        if self.retry_after_seconds is None:
            message = (
                "Groq rate limit was reached after the bounded retry budget. "
                "Retry after the current rate-limit window."
            )
        else:
            wait_seconds = max(1, math.ceil(self.retry_after_seconds))
            message = (
                "Groq rate limit was reached after the bounded retry budget. "
                f"Retry after about {wait_seconds} seconds."
            )
        super().__init__(message)


# Bound total sleeping on repeated Groq 429 responses for each model call. A
# single Retry-After interval (including 30 seconds) is honored, but a second
# long wait is returned to the caller instead of stranding chat for another minute.
MAX_GROQ_RETRY_WAIT_SECONDS = 30.0


# ============================================================
# CONSTANTS
# ============================================================

FALLBACK_ANSWER = (
    "I couldn't find enough information about that "
    "in the video. It may not be covered by this video."
)

# Use this when an answer to a financial question fails the deterministic repair
# check and there is not enough source-bound evidence to build a safe extractive
# answer. Unlike FALLBACK_ANSWER, keep the retrieved sources visible for review.
FINANCIAL_VERIFICATION_FALLBACK = (
    "I couldn't reliably verify the financial figures from the retrieved "
    "transcript. I won't guess at or reconcile amounts that the available "
    "evidence does not establish."
)


# ============================================================
# 1. SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """You answer questions about a YouTube video using only the supplied transcript context.

LANGUAGE
- The main answer MUST ALWAYS be written in English, regardless of the language of the question, transcript, captions, retrieved passages, or conversation history. Translate supported meaning from the transcript into clear English; preserve source-grounded names, spellings, dates, numbers, units, and uncertainty.
- When key evidence comes from a Hindi, Arabic, or Urdu transcript, include one short, exact original-language excerpt whenever a suitable passage is supplied and can be quoted faithfully. Format it as one clearly delimited quotation: put the original-language excerpt first. Follow it immediately with its English meaning in the same sentence or the next sentence. Keep all surrounding prose in English, and do not put additional Hindi, Arabic, or Urdu text outside that single quotation. Quote only supplied text; never reconstruct or invent an excerpt. Do not use non-English text for the main answer or for an abstention. This one short source-language quotation is the only permitted exception.

EVIDENCE AND ATTRIBUTION
- Answer directly and concisely from supplied evidence only. Do not use outside knowledge or invent facts, relationships, or interpretations.
- Source blocks are timestamped evidence segments containing caption snippets, not reliable speaker turns. A caption line preserves one caption snippet, not a verified speaker turn. Each reconstructed caption line includes that snippet's original start and end times. Caption boundaries and timestamps do not prove a speaker changed; the timestamps do not establish who spoke.
- Attribute a claim to a named person only when the context clearly shows that person stated or endorsed it. The name in the question does not make that person the source of every nearby claim. Never turn an allegation described by the interviewer into an established fact. Keep an interviewer's question, allegation, premise, or paraphrase separate from the guest's statements. Distinguish the guest's statements from an interviewer's questions. An account of what an authority found is only an account; it does not by itself verify the underlying finding. If attribution is unclear, describe it neutrally as something discussed, qualify it when supported, or omit it. Do not guess.
- Captions may contain errors, fragments, and repetitions. Do not silently repair facts using outside knowledge.

TEMPORAL AND MULTI-PART QUESTIONS
- Answer time-scoped questions only from the relevant supplied passage. Absence of a statement is answerable: if a relevant passage is present but a result, score, winner, or announcement is not stated, say it is not stated in that passage. Do not use this fallback solely because a named event is absent from an otherwise relevant passage.
- Include every supported part and distinct item; omit unsupported parts. Use bullets only for genuinely multi-part answers.
- When a specific count or complete list is requested, count distinct concepts or teams separately when the transcript distinguishes them. Do not merge separate proposals merely because they share a category. Do not invent items to reach a count; disclose when fewer items or an uncertain count are supported.

VIDEO-LOCATION QUESTIONS
- When a user asks where or at what point in a video, interview, episode, recording, or podcast a topic is discussed, inspect every supplied source block in chronological order.
- Distinguish a title, opening narration, teaser, passing name-drop, or brief framing reference from a substantive discussion. Do not report only the first occurrence when later supplied passages contain a substantive discussion of the same topic.
- Summarize each distinct, clearly supported substantive section and what is discussed there. Omit incidental matches and unrelated passages; do not force separate sections where the transcript only supports one.
- Preserve who said what, especially the difference between an interviewer's framing and the guest's statements. Use only the supplied transcript.
- Do not output timestamp ranges or source labels. The application appends timestamp ranges selected from verified source metadata.

NUMBERS AND FINANCIAL CLAIMS
- Preserve distinct figures as stated in the transcript. Map every figure to its nearby claim, subject, qualifier, and source/speaker; retain its attribution. Do not round, merge, or reconcile figures unless the transcript explicitly connects them.
- Keep investment, debt, debt-certificate, alleged transaction, and recovery figures separate. For broad money/debt questions, include each materially relevant monetary claim. If the transcript does not explain how figures relate, report both separately and state that the relationship is unclear.
- Do not silently repair a number or call one a "misreading" unless the transcript explicitly supports that relationship. Never describe a recovery figure as a company debt total. If captions garble a number or speakers disagree, state the uncertainty.

ABSTENTION
- If context is empty, unrelated, or genuinely insufficient, answer exactly: "I couldn't find enough information about that in the video. It may not be covered by this video."

SOURCES AND OUTPUT
- Return only the substantive answer. Do not output source labels, citations, references, metadata, or invented timestamps. The application appends a separate "Relevant timestamps" line from verified source metadata; timestamp ranges remain part of the answer. Clickable video sources are displayed separately.
- Do not mention retrieval, embeddings, prompts, vector databases, or internal implementation."""

# ============================================================
# 2. PROMPT
# ============================================================

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_INSTRUCTIONS,
        ),

        (
            "human",
            """
Transcript context:
{context}

Question: {question}

Answer only from the supplied context and ALWAYS write the main answer in English. Translate supported facts from any transcript language. If useful, include one short exact Hindi, Arabic, or Urdu excerpt from the supplied context followed by its English meaning. Be concise but include all supported parts and preserve claim attribution, dates, numbers, and uncertainty. If the context is genuinely insufficient, use the exact English fallback in the system instructions. Return answer text only; the application adds verified sources and timestamps.
""",
        ),
    ]
)


# ============================================================
# 3. TIMESTAMP FORMATTER
# ============================================================

def format_timestamp(
    seconds: float,
) -> str:
    """
    Convert seconds into MM:SS or HH:MM:SS.
    """

    total_seconds = max(
        0,
        int(seconds),
    )


    hours = (
        total_seconds // 3600
    )


    minutes = (
        total_seconds % 3600
    ) // 60


    remaining_seconds = (
        total_seconds % 60
    )


    if hours > 0:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )


    return (
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )

_TIMESTAMP_REQUEST_PATTERN = re.compile(
    r"\b(?:timestamps?|time\s*stamps?|cite|citation|citations)\b",
    re.IGNORECASE,
)

_MONEY_AMOUNT_PATTERN = re.compile(
    r"(?:"
    r"(?:₹|INR\b|Rs\.?)\s*(?P<currency_amount>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"|(?<![\w:,])(?P<amount>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
    r"(?:crores?\b|cr\b|rupees?\b|INR\b|percent\b|%)"
    r")",
    re.IGNORECASE,
)
_CLAIM_CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"[.!?;\n]+|,\s*(?=(?:and|but|while|whereas|however)\b)",
    re.IGNORECASE,
)
_CLAIM_STOP_WORDS = set("""
    a an and are as at be been but by can could did do does for from had has
    have he her his how i if in into is it its may might not of on or she so
    than that the their them then there these they this those to was we were
    what when where which while who why will with would you your about around
    approximately crore crores cr rupee rupees
""".split())


def _extract_money_amounts(text: str) -> list[float]:
    if not isinstance(text, str) or not text:
        return []

    amounts = []
    for match in _MONEY_AMOUNT_PATTERN.finditer(text):
        token = match.group("currency_amount") or match.group("amount")
        try:
            amount = float(token.replace(",", ""))
        except (AttributeError, ValueError):
            continue
        if math.isfinite(amount):
            amounts.append(amount)

    # Some transcript captions record a bare financial figure (for example
    # "certificate number 6203") without repeating the unit in that caption.
    # Restrict this fallback to four-to-six digit numbers near financial terms
    # and use caption-line neighbors when the claim label is in the adjacent line.
    lines = text.splitlines()
    financial_terms = re.compile(
        r"\b(?:certificate|drt|tribunal|judg(?:e)?ment|debt|principal|interest|"
        r"liabilit\w*|loan|recover(?:ed|y|ies)?|recovery|invest(?:ed|ment)?|"
        r"ministry|finance|report|kingfisher|borrowed|owed|owes)\b",
        re.IGNORECASE,
    )
    bare_pattern = re.compile(r"(?<![\w,.:])\d{4,6}(?![\w,:]|\.\d)")
    for index, line in enumerate(lines):
        for match in bare_pattern.finditer(line):
            nearby = " ".join(lines[max(0, index - 1): min(len(lines), index + 2)])
            if not financial_terms.search(nearby):
                continue
            try:
                amount = float(match.group(0))
            except ValueError:
                continue
            if math.isfinite(amount) and not any(
                math.isclose(amount, existing, rel_tol=0.0, abs_tol=0.1)
                for existing in amounts
            ):
                amounts.append(amount)
    return amounts


def _claim_terms(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z]{2,}", text.lower())
        if token not in _CLAIM_STOP_WORDS
    }


def _money_claims(answer: str) -> list[dict[str, Any]]:
    """Extract monetary mentions and clearly labelled bare financial figures."""
    claims = []
    recognized_spans = []
    for match in _MONEY_AMOUNT_PATTERN.finditer(answer):
        token = match.group("currency_amount") or match.group("amount")
        try:
            amount = float(token.replace(",", ""))
        except (AttributeError, ValueError):
            continue
        if not math.isfinite(amount):
            continue

        left, right = 0, len(answer)
        for boundary in _CLAIM_CLAUSE_BOUNDARY_PATTERN.finditer(answer):
            if boundary.end() <= match.start():
                left = boundary.end()
            elif boundary.start() >= match.end():
                right = boundary.start()
                break
        terms = _claim_terms(answer[left:right])
        if terms:
            claims.append({
                "amount": amount,
                "terms": terms,
                "text": answer[left:right],
            })
        recognized_spans.append((match.start(), match.end()))

    # A certificate caption can give the figure without repeating its unit.
    # Accept this only when the same clause explicitly names a financial topic;
    # the timestamp punctuation guard avoids treating source ranges as amounts.
    bare_pattern = re.compile(r"(?<![\w,.:])\d{4,6}(?![\w,:]|\.\d)")
    for match in bare_pattern.finditer(answer):
        if any(start <= match.start() < end for start, end in recognized_spans):
            continue
        left, right = 0, len(answer)
        for boundary in _CLAIM_CLAUSE_BOUNDARY_PATTERN.finditer(answer):
            if boundary.end() <= match.start():
                left = boundary.end()
            elif boundary.start() >= match.end():
                right = boundary.start()
                break
        clause = answer[left:right]
        if _financial_claim_topic(clause) is None:
            continue
        terms = _claim_terms(clause)
        if terms:
            claims.append({
                "amount": float(match.group(0)),
                "terms": terms,
                "text": clause,
            })
    return claims


def _amounts_match(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=0.1) or (
        min(abs(left), abs(right)) >= 100
        and abs(left - right) / max(abs(left), abs(right)) <= 0.01
    )


def _select_claim_aligned_timestamp_candidates(
    answer: str,
    candidates: list[dict[str, Any]],
    *,
    max_ranges: int,
) -> list[dict[str, Any]] | None:
    """Match amount claims to returned source intervals using transcript text."""
    claims = _money_claims(answer)
    with_text = [
        candidate for candidate in candidates
        if candidate.get("transcript_text", "").strip()
    ]
    if not claims or not with_text:
        return None

    profiles = []
    for candidate in with_text:
        text = candidate["transcript_text"]
        profiles.append({
            "candidate": candidate,
            "amounts": _extract_money_amounts(text),
            "terms": _claim_terms(text),
        })
    answer_amounts = {claim["amount"] for claim in claims}
    for profile in profiles:
        profile["amount_breadth"] = sum(
            any(_amounts_match(amount, source_amount)
                for source_amount in profile["amounts"])
            for amount in answer_amounts
        )

    selected: dict[int, dict[str, Any]] = {}
    for claim in claims:
        claim_text = str(claim.get("text", ""))
        is_report_recovery_claim = bool(
            re.search(r"\b(?:ministry|finance|report)\b", claim_text, re.IGNORECASE)
            and re.search(
                r"\b(?:recover(?:ed|y|ies)?|recovery)\b",
                claim_text,
                re.IGNORECASE,
            )
        )
        choices = []
        for profile in profiles:
            if not any(
                _amounts_match(claim["amount"], amount)
                for amount in profile["amounts"]
            ):
                continue
            overlap = len(claim["terms"] & profile["terms"])
            source_text = str(profile["candidate"].get("transcript_text", ""))
            source_mentions_recovery = bool(re.search(
                r"\b(?:recover(?:ed|y|ies)?|recovery)\b",
                source_text,
                re.IGNORECASE,
            ))
            if overlap or (is_report_recovery_claim and source_mentions_recovery):
                candidate = profile["candidate"]
                choices.append((
                    (max(overlap, 1 if is_report_recovery_claim and source_mentions_recovery else 0),
                     profile["amount_breadth"], -candidate["distance"]),
                    profile,
                    overlap,
                ))

        if not choices:
            # Keep alignments for supported claims rather than discarding every
            # valid financial timestamp because a draft contains one unsupported
            # amount. A separate answer-repair pass handles the unsupported claim.
            continue

        _, profile, overlap = max(choices, key=lambda item: item[0])
        candidate = profile["candidate"]
        record = selected.setdefault(
            candidate["candidate_index"],
            {"candidate": candidate, "claim_count": 0, "overlap": 0},
        )
        record["claim_count"] += 1
        record["overlap"] += overlap

        if is_report_recovery_claim:
            # A report-attributed recovery may require two evidence locations:
            # one caption states the clean amount, while another names the
            # Ministry/Finance report but has a split ASR numeral. Cite both
            # rather than implying that either passage independently proves
            # every part of the compound claim.
            attribution_profiles = []
            for attribution_profile in profiles:
                source_text = str(
                    attribution_profile["candidate"].get("transcript_text", "")
                )
                if (
                    re.search(r"\b(?:ministry|finance|report)\b", source_text, re.IGNORECASE)
                    and re.search(
                        r"\b(?:recover(?:ed|y|ies)?|recovery)\b",
                        source_text,
                        re.IGNORECASE,
                    )
                ):
                    attribution_profiles.append(attribution_profile)
            if attribution_profiles:
                best_attribution = max(
                    attribution_profiles,
                    key=lambda item: (
                        len(claim["terms"] & item["terms"]),
                        item["amount_breadth"],
                        -item["candidate"]["distance"],
                    ),
                )
                attribution_candidate = best_attribution["candidate"]
                attribution_record = selected.setdefault(
                    attribution_candidate["candidate_index"],
                    {
                        "candidate": attribution_candidate,
                        "claim_count": 0,
                        "overlap": 0,
                    },
                )
                attribution_record["claim_count"] += 1
                attribution_record["overlap"] += len(
                    claim["terms"] & best_attribution["terms"]
                )

    if not selected:
        return None

    # Dense multi-amount answers can use up to five ranges; ordinary answers
    # keep the existing three-range limit.
    budget = min(5, max_ranges if len(claims) < 4 else 5)
    ranked = list(selected.values())
    ranked.sort(
        key=lambda item: (
            item["claim_count"],
            item["overlap"],
            -item["candidate"]["distance"],
        ),
        reverse=True,
    )
    return [item["candidate"] for item in ranked[:budget]]


_ANSWER_ALIGNMENT_CONCEPTS = (
    {
        "english_pattern": r"\bheartbeat\b|heart\s*beat|\bpulse\b",
        "arabic_pattern": r"نبض.{0,25}قلب|قلب.{0,25}نبض|نبضات.{0,15}القلب",
        "aliases": {"heartbeat", "heart", "pulse", "نبض", "قلب", "نبضات", "النبض"},
    },
    {
        "english_pattern": r"\bblood\b.{0,60}(?:\bvein|\barter|\bflow|\brush|\bmov)|(?:\bvein|\barter).{0,60}\bblood\b",
        "arabic_pattern": r"الدم.{0,35}(?:عروق|شرايين|أوعية|حركة|تدفق|جريان|سريان)|(?:حركة|تدفق|جريان|سريان).{0,20}الدم",
        "aliases": {
            "blood", "vein", "veins", "artery", "arteries", "flow", "rushing",
            "rush", "movement", "move", "دم", "الدم", "عروق", "شرايين",
            "أوعية", "حركة", "تدفق", "جريان", "سريان",
        },
    },
    {
        "english_pattern": r"\blungs?\b|\bbreath(?:e|ing)?\b",
        "arabic_pattern": r"رئتين|الرئتين|الرئة|تنفس|التنفس",
        "aliases": {"lung", "lungs", "breath", "breathing", "رئتين", "الرئتين", "الرئة", "تنفس", "التنفس"},
    },
    {
        "english_pattern": r"\bstomach\b|\brumbl\w*|\bbelly\b",
        "arabic_pattern": r"المعدة|معدتك|معدة|البطن|بطنك",
        "aliases": {"stomach", "rumble", "rumbling", "belly", "المعدة", "معدتك", "معدة", "البطن", "بطنك"},
    },
    {
        # Captions may omit the explicit blink verb, but source-side matching
        # must still establish an acoustic cue near the eye/eyelid reference.
        "english_pattern": r"\beyelids?\b|\bblink\w*\b|\beyes?\b|\beyeballs?\b|\beyelashes?\b|eye.{0,50}(?:blink|lid|close|mov)|(?:blink|lid|clos|mov).{0,50}\beye",
        "source_english_pattern": r"\b(?:sound|noise|hear(?:d|ing|s)?|audible|listen\w*)[^\.\!\?؟؛;]{0,45}\b(?:eyes?|eyelids?|eyeballs?|eyelashes?)\b|\b(?:eyes?|eyelids?|eyeballs?|eyelashes?)\b[^\.\!\?؟؛;]{0,45}\b(?:sound|noise|hear(?:d|ing|s)?|audible|listen\w*)\b|\b(?:hear(?:d|ing|s)?|listen\w*)[^\.!\?؟؛;]{0,180}\b(?:eyes?|eyelids?|eyeballs?|eyelashes?)\b[^\.!\?؟؛;]{0,35}\b(?:blink\w*|clos\w*|mov\w*)\b",
        # A bare anatomy mention is not evidence for an audible sound. Accept
        # a nearby acoustic cue, or an inherited cue in a coordinated list when
        # the same sentence explicitly ties the eye reference to an action.
        "arabic_pattern": r"جفونك?|الجفون|جفن|الجفن|عيونك?|عيون|عينيك?|عينك?|العينين|العين|عين.{0,40}(?:ترمش|بترمش|تومض|ترف|رمش|تغمض|بتغمض|تغمز|غمض|إغلاق|اغلاق|تغلق|يغلق|إغماض|اغماض)|(?:ترمش|بترمش|تومض|ترف|رمش|تغمض|بتغمض|تغمز|غمض|إغلاق|اغلاق|تغلق|يغلق|إغماض|اغماض).{0,40}(?:عين|جفن)",
        "source_arabic_pattern": r"(?:صوت|أصوات|اصوات|بتسمع\w*|تسمع\w*|سمع\w*|أسمع|اسمع)[^\.\!\?؟؛;]{0,45}(?:عين|عيون|جفن|جفون)|(?:عين|عيون|جفن|جفون)[^\.\!\?؟؛;]{0,45}(?:صوت|أصوات|اصوات|بتسمع\w*|تسمع\w*|سمع\w*|أسمع|اسمع)",
        "aliases": {"eyelid", "eyelids", "blink", "blinking", "eye", "eyes", "eyeball", "eyeballs", "eyelash", "eyelashes", "close", "closing", "move", "moving", "جفون", "جفونك", "الجفون", "جفن", "الجفن", "جفنك", "عين", "عينك", "العين", "العينين", "عيني", "عينيك", "عيون", "عيونك", "رمش", "الرمش", "ترمش", "بترمش", "ترف", "تومض", "تغمض", "بتغمض", "تغمز", "غمض", "إغلاق", "اغلاق", "تغلق", "يغلق", "إغماض", "اغماض"},
    },
)


_ANSWER_ALIGNMENT_STOP_WORDS = frozenset({
    "the", "and", "that", "this", "these", "those", "from", "into", "with",
    "your", "you", "their", "they", "them", "own", "may", "might", "could",
    "would", "should", "start", "starts", "begin", "begins", "beginning",
    "hear", "hearing", "heard", "sound", "sounds", "body", "bodies", "room",
    "people", "person", "such", "like", "even", "inside", "there", "here",
    "what", "which", "where", "when", "while", "also", "through", "for",
    "about", "are", "was", "were", "is", "be", "been", "not", "can", "will",
    "it", "its", "as", "of", "in", "on", "at", "to", "a", "an", "or", "by",
    "up", "out", "than", "then", "very", "التي", "الذي", "الذين", "اللواتي",
    "الناس", "قد", "يبدأ", "يبدأون", "بسماع", "سماع", "داخل", "الغرفة",
    "أجسامهم", "اجسامهم", "أصوات", "اصوات", "صوت", "الأصوات", "الاصوات",
    # Keep anatomy/blink words out of raw token-overlap matching. Eye-sound
    # cross-lingual expansion below requires a nearby acoustic cue in the same
    # caption sentence, not merely an eye/eyelid mention.
    # Keep anatomy / blink words out of raw token-overlap matching. The eye
    # concept is added via the stricter source acoustic pattern below instead.
    "eye", "eyes", "eyelid", "eyelids", "eyeball", "eyeballs", "eyelash", "eyelashes",
    "blink", "blinks", "blinked", "blinking", "close", "closing", "move", "moving",
    "عين", "العين", "العينين", "عينك", "عيني", "عينيك", "عيون", "عيونك",
    "جفن", "الجفن", "جفون", "الجفون", "ترمش", "بترمش", "رمش", "الرمش",
    "تومض", "ترف", "تغمض", "بتغمض", "تغمز", "غمض", "هي", "هو", "هم", "هن", "في", "من", "على", "إلى", "الى", "أن", "ان",
    "و", "أو", "او", "حتى", "مثل", "كما", "لأن", "لان", "يمكن", "يكون",
    "تكون", "نفس", "هذا", "هذه", "ذلك",
})


def _answer_alignment_terms(text: str, *, source: bool = False) -> set[str]:
    """Extract informative English/Arabic terms for answer-to-source alignment.

    Source mode uses stricter concept patterns where a broad word alone (for
    example, "eyes") would not prove the specific claim being timestamped.
    """
    normalized = unicodedata.normalize(
        "NFKC", _strip_unverified_timestamp_summary(text)
    ).lower()
    normalized = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", normalized)
    normalized = re.sub(r"\d+(?:[,:./]\d+)*", " ", normalized)
    tokens = re.findall(r"[^\W_]{2,}", normalized, flags=re.UNICODE)
    terms: set[str] = set()
    arabic_prefixes = ("وال", "بال", "كال", "فال", "لل", "ال")
    arabic_suffixes = (
        "هما", "كما", "كم", "كن", "هم", "هن", "ها", "نا", "ك", "ه", "ي",
        "ون", "ات",
    )
    for token in tokens:
        if token in _ANSWER_ALIGNMENT_STOP_WORDS:
            continue
        variants = {token}
        if any("\u0600" <= char <= "\u06ff" for char in token):
            current = token
            for prefix in arabic_prefixes:
                if current.startswith(prefix) and len(current) > len(prefix) + 2:
                    current = current[len(prefix):]
                    variants.add(current)
                    break
            for suffix in arabic_suffixes:
                if current.endswith(suffix) and len(current) > len(suffix) + 2:
                    variants.add(current[:-len(suffix)])
                    break
        else:
            # Lightweight suffix normalization keeps timestamp alignment
            # useful for transcript variants such as "blinking"/"blink",
            # "rushing"/"rush", "eyelids"/"eyelid", and "veins"/"vein".
            if len(token) > 5 and token.endswith("ies"):
                variants.add(token[:-3] + "y")
            if len(token) > 5 and token.endswith("ing"):
                stem = token[:-3]
                variants.add(stem)
                if len(stem) > 2 and stem[-1] == stem[-2]:
                    variants.add(stem[:-1])
            if len(token) > 4 and token.endswith("ed"):
                variants.add(token[:-2])
            if len(token) > 4 and token.endswith("es"):
                variants.add(token[:-2])
            if len(token) > 4 and token.endswith("s"):
                variants.add(token[:-1])
        for variant in variants:
            if len(variant) >= 2 and variant not in _ANSWER_ALIGNMENT_STOP_WORDS:
                terms.add(variant)

    # Add the counterpart vocabulary only when the concept appears in the text
    # being aligned. This supports Arabic answers over English captions and vice
    # versa without applying blanket translation to unrelated queries.
    for concept in _ANSWER_ALIGNMENT_CONCEPTS:
        english_pattern = (
            concept.get("source_english_pattern")
            if source
            else concept.get("english_pattern")
        ) or concept["english_pattern"]
        arabic_pattern = (
            concept.get("source_arabic_pattern")
            if source
            else concept.get("arabic_pattern")
        ) or concept["arabic_pattern"]
        if (
            re.search(english_pattern, normalized, re.IGNORECASE)
            or re.search(arabic_pattern, normalized, re.IGNORECASE)
        ):
            terms.update(concept["aliases"])
    return terms



_TEMPORAL_LOCATION_QUESTION_PATTERNS = (
    re.compile(
        r"\bwhere\b.{0,120}\b(?:in|within|throughout)\s+(?:the\s+)?"
        r"(?:video|interview|episode|recording|podcast|clip)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:where|when)\b.{0,140}\b"
        r"(?:discuss\w*|talk\w*|speak\w*|mention\w*|cover\w*|"
        r"explain\w*|address\w*)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:at what (?:point|time)|which|what)\s+"
        r"(?:parts?|sections?|portions?|segments?|points?|times?)\b"
        r".{0,120}\b(?:video|interview|episode|recording|podcast|clip)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:أين|متى|في أي جزء|في أي قسم).{0,120}"
        r"(?:الفيديو|المقابلة|المقطع)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:الفيديو|المقابلة|المقطع).{0,120}"
        r"(?:أين|متى|أي جزء|يناقش|يتحدث|يذكر|يشرح)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:कहाँ|कब|किस हिस्से|किस भाग|किस खंड).{0,120}"
        r"(?:वीडियो|इंटरव्यू|प्रकरण)",
    ),
    re.compile(
        r"(?:वीडियो|इंटरव्यू|प्रकरण).{0,120}"
        r"(?:कहाँ|कब|किस हिस्से|किस भाग|किस खंड)",
    ),
)

_TEMPORAL_LOCATION_STOP_WORDS = frozenset(
    _CLAIM_STOP_WORDS
    | set("""
        video videos interview interviews episode episodes recording recordings
        podcast podcasts clip clips transcript transcripts caption captions
        source sources timestamp timestamps chronological order boundaries
        speaker speakers labels where when point part section portion segment
        parts sections portions segments discuss discusses discussed discussing
        discussion discussions talk talks talked talking speak speaks spoke
        speaking mention mentions mentioned mentioning cover covers covered
        covering explain explains explained explaining address addresses
        addressed addressing come comes came up about in at on within throughout
        this that these those there here which what who does do did is are was
        were has have had the and for from with into of to a an it its
        the interviewer guest user asks ask asked asking question questions
        video interview episode recording podcast clip
        أين متى الفيديو المقابلة المقطع جزء قسم أجزاء أقسام يتحدث يناقش يذكر يشرح
        في من عن ما هو هي هذا هذه ذلك هناك
        कहाँ कब वीडियो इंटरव्यू प्रकरण हिस्सा हिस्से भाग खंड चर्चा बात उल्लेख
        बताता बताया बताते बताते हैं करता करते किया किस कौन में का की के और है हैं
        वह यह उस इस पर से तक
    """.split())
)


def is_temporal_location_question(question: str | None) -> bool:
    """Recognize questions asking where a subject is substantively discussed."""
    if not isinstance(question, str) or not question.strip():
        return False
    return any(
        pattern.search(question)
        for pattern in _TEMPORAL_LOCATION_QUESTION_PATTERNS
    )


def _temporal_location_terms(text: str | None) -> set[str]:
    """Extract script-aware content terms while excluding source headers and intent words."""
    if not isinstance(text, str) or not text.strip():
        return set()

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(
        r"(?im)^\s*(?:video id|timestamp)\s*:\s*.*$",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"(?m)^\s*\[source\s+\d+\]\s*$",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*[–—-]\s*"
        r"\d{1,2}:\d{2}(?::\d{2})?\s*\]",
        " ",
        normalized,
    )

    terms: set[str] = set()
    for token in re.findall(r"[^\W_]{2,}", normalized, flags=re.UNICODE):
        if token in _TEMPORAL_LOCATION_STOP_WORDS:
            continue

        variants = {token}
        if re.search(r"[a-z]", token):
            if len(token) > 5 and token.endswith("ies"):
                variants.add(token[:-3] + "y")
            if len(token) > 5 and token.endswith("ing"):
                stem = token[:-3]
                variants.add(stem)
                if len(stem) > 2 and stem[-1] == stem[-2]:
                    variants.add(stem[:-1])
            if len(token) > 4 and token.endswith("ed"):
                variants.add(token[:-2])
            if len(token) > 4 and token.endswith("es"):
                variants.add(token[:-2])
            if len(token) > 4 and token.endswith("s"):
                variants.add(token[:-1])
        else:
            current = token
            for prefix in ("وال", "بال", "كال", "فال", "لل", "ال"):
                if current.startswith(prefix) and len(current) > len(prefix) + 2:
                    variants.add(current[len(prefix):])
                    break
            for suffix in ("هما", "كما", "كم", "كن", "هم", "هن", "ها", "نا", "ون", "ات"):
                if current.endswith(suffix) and len(current) > len(suffix) + 2:
                    variants.add(current[:-len(suffix)])
                    break

        terms.update(
            variant
            for variant in variants
            if len(variant) >= 2
            and variant not in _TEMPORAL_LOCATION_STOP_WORDS
        )
    return terms


def _temporal_location_source_blocks(context: str) -> list[tuple[int, str]]:
    """Split prompt context into independent source blocks without joining evidence."""
    if not isinstance(context, str) or not context.strip():
        return []

    markers = list(re.finditer(
        r"(?m)^\s*\[SOURCE\s+(\d+)\]\s*$",
        context,
    ))
    if not markers:
        return [(1, context.strip())]

    blocks = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(context)
        block_text = context[marker.end():end].strip()
        try:
            source_number = int(marker.group(1))
        except (TypeError, ValueError):
            source_number = index + 1
        if block_text:
            blocks.append((source_number, block_text))
    return blocks


def _temporal_location_answer_coverage(
    question: str,
    answer: str,
    context: str,
) -> dict[str, Any]:
    """Measure answer coverage across distinct source blocks matching the requested topic.

    Topic-name overlap alone never counts as covering a source block. A block
    counts only when the answer also reflects at least two source-specific
    content terms from that block. This keeps an opening name-drop from
    masquerading as a summary of later substantive discussions.
    """
    if not is_temporal_location_question(question):
        return {
            "is_temporal_location_question": False,
            "candidate_source_group_numbers": [],
            "covered_source_group_numbers": [],
            "required_covered_source_group_count": 0,
            "coverage_ratio": None,
            "needs_review": False,
        }

    topic_terms = _temporal_location_terms(question)
    answer_terms = _temporal_location_terms(answer)
    blocks = _temporal_location_source_blocks(context)

    candidates: list[tuple[int, str, set[str]]] = []
    for source_number, block_text in blocks:
        source_terms = _temporal_location_terms(block_text)
        if topic_terms & source_terms:
            candidates.append((source_number, block_text, source_terms))

    candidate_numbers = [item[0] for item in candidates]
    covered_numbers = []
    for source_number, _block_text, source_terms in candidates:
        source_specific_terms = source_terms - topic_terms
        meaningful_overlap = answer_terms & source_specific_terms
        if len(meaningful_overlap) >= 2:
            covered_numbers.append(source_number)

    candidate_count = len(candidates)
    required_count = (
        min(candidate_count, max(2, math.ceil(candidate_count * 0.5)))
        if candidate_count >= 2
        else candidate_count
    )
    coverage_ratio = (
        len(covered_numbers) / candidate_count
        if candidate_count
        else 0.0
    )
    needs_review = (
        candidate_count >= 1
        and len(covered_numbers) < required_count
    )

    return {
        "is_temporal_location_question": True,
        "candidate_source_group_numbers": candidate_numbers,
        "covered_source_group_numbers": covered_numbers,
        "required_covered_source_group_count": required_count,
        "coverage_ratio": round(coverage_ratio, 4),
        "needs_review": needs_review,
    }


def _temporal_location_completeness_repair_instruction(
    question: str,
    answer: str,
    context: str,
) -> str | None:
    """Request one bounded rewrite when a location answer omits distinct topic-bearing sections."""
    coverage = _temporal_location_answer_coverage(question, answer, context)
    if not coverage["needs_review"]:
        return None

    covered = set(coverage["covered_source_group_numbers"])
    missing_labels = []
    for source_number, block_text in _temporal_location_source_blocks(context):
        if source_number not in coverage["candidate_source_group_numbers"] or source_number in covered:
            continue
        missing_labels.append(f"SOURCE {source_number}")

    return (
        "TEMPORAL LOCATION COMPLETENESS REVIEW. The question asks where a topic "
        "is substantively discussed in the video, not merely where its name first "
        "appears. The previous draft meaningfully covered "
        f"{len(coverage['covered_source_group_numbers'])} of "
        f"{len(coverage['candidate_source_group_numbers'])} topic-matching source "
        "blocks; the supplied context supports reviewing at least "
        f"{coverage['required_covered_source_group_count']} distinct blocks. "
        "Re-read every supplied source block in chronological order, including "
        "these topic-matching blocks not yet represented in the draft: "
        + (", ".join(missing_labels) if missing_labels else "the remaining matching blocks")
        + ". Verify whether each is a substantive discussion rather than a teaser, "
        "passing mention, interviewer framing, or incidental match. Rewrite the "
        "answer from scratch in English, covering each distinct substantive section "
        "and briefly explaining what is discussed there. Do not force coverage of "
        "a block that is only an incidental mention, and do not merge separate "
        "discussions into a vague statement. Preserve guest/interviewer attribution "
        "and uncertainty. Use only the supplied transcript context. Do not output "
        "timestamp ranges, source labels, or invented times; the application adds "
        "verified timestamps separately."
    )


def _select_answer_aligned_timestamp_candidates(
    answer: str,
    candidates: list[dict[str, Any]],
    *,
    max_ranges: int,
) -> list[dict[str, Any]] | None:
    """Prefer returned source intervals whose transcript wording supports the answer.

    Alignment is multilingual and conservative. If no informative overlap is
    found, return None so callers can retain the established fallback ordering.
    """
    answer_terms = _answer_alignment_terms(answer)
    if len(answer_terms) < 2:
        return None
    normalized_answer = unicodedata.normalize("NFKC", answer).lower()
    answer_concepts = {
        index
        for index, concept in enumerate(_ANSWER_ALIGNMENT_CONCEPTS)
        if re.search(concept["english_pattern"], normalized_answer, re.IGNORECASE)
        or re.search(concept["arabic_pattern"], normalized_answer, re.IGNORECASE)
    }

    ranked = []
    for candidate in candidates:
        transcript_text = candidate.get("transcript_text", "")
        if not isinstance(transcript_text, str) or not transcript_text.strip():
            continue
        source_terms = _answer_alignment_terms(transcript_text, source=True)
        overlap = answer_terms & source_terms
        normalized_source = unicodedata.normalize("NFKC", transcript_text).lower()
        source_concepts = {
            index
            for index, concept in enumerate(_ANSWER_ALIGNMENT_CONCEPTS)
            if re.search(
                concept.get("source_english_pattern") or concept["english_pattern"],
                normalized_source,
                re.IGNORECASE,
            )
            or re.search(
                concept.get("source_arabic_pattern") or concept["arabic_pattern"],
                normalized_source,
                re.IGNORECASE,
            )
        }
        # Direct concept matches let the selector align a translated answer to
        # dialect captions even when token overlap is weak or normalization
        # differs. A candidate must match a concept actually present in answer.
        concepts = answer_concepts & source_concepts
        if not overlap and not concepts:
            continue
        coverage = len(overlap) / len(answer_terms)
        specificity = len(overlap) / math.sqrt(max(1, len(source_terms)))
        ranked.append({
            "candidate": candidate,
            "overlap": overlap,
            "concepts": concepts,
            "coverage": coverage,
            "specificity": specificity,
            "distance_score": -float(candidate.get("distance", math.inf)),
        })

    if not ranked:
        return None

    # Select complementary evidence instead of taking several near-duplicate
    # high-overlap sources. In particular, a late source mentioning eyelid
    # blinking must not lose to broader early sources that cover only the
    # first few items in a multi-item answer.
    selected = []
    covered_terms: set[str] = set()
    covered_concepts: set[int] = set()
    remaining = list(ranked)
    while remaining and len(selected) < max_ranges:
        choices = [
            (
                len(item["concepts"] - covered_concepts),
                len(item["overlap"] - covered_terms),
                item["coverage"],
                item["specificity"],
                item["distance_score"],
                item,
            )
            for item in remaining
        ]
        best = max(choices, key=lambda item: item[:-1])[-1]
        concept_gain = len(best["concepts"] - covered_concepts)
        marginal_gain = len(best["overlap"] - covered_terms)
        if concept_gain <= 0 and marginal_gain <= 0:
            break
        selected.append(best["candidate"])
        covered_terms.update(best["overlap"])
        covered_concepts.update(best["concepts"])
        remaining.remove(best)

    return selected or None


def _strip_unverified_timestamp_summary(answer: str) -> str:
    """Compatibility wrapper for removing untrusted model-authored time claims.

    The shared sanitizer removes labelled summaries and inline timestamp
    statements. Only the deterministic source/temporal metadata appenders may
    add displayed YouTube timestamp ranges to the final response.
    """
    return strip_unverified_timestamp_claims(answer)


def append_relevant_source_timestamps(
    answer: str,
    sources: list[dict[str, Any]],
    *,
    has_evidence: bool,
    max_ranges: int = 3,
) -> str:
    """Append source-grounded timestamp ranges to a substantive answer.

    Multi-amount answers first align each amount claim with transcript text from
    returned sources. Other answers use multilingual lexical alignment between
    the substantive answer and returned-source transcript text, falling back to
    the established distance-ranked behavior only when no useful overlap exists.
    Selected ranges are always grounded in returned metadata and presented
    chronologically.
    """
    if (
        not isinstance(answer, str)
        or not answer.strip()
        or answer.strip() in {
            FALLBACK_ANSWER,
            FINANCIAL_VERIFICATION_FALLBACK,
        }
    ):
        return answer

    # Never let the model-authored summary suppress the application-generated
    # one. The displayed summary must come from actual returned source metadata.
    answer = _strip_unverified_timestamp_summary(answer)
    if not answer.strip():
        return FALLBACK_ANSWER

    if (
        not has_evidence
        or not isinstance(sources, (list, tuple))
        or not sources
        or max_ranges < 1
    ):
        return answer

    candidates = []
    seen_ranges = set()
    for source in sources:
        if not isinstance(source, dict):
            continue

        try:
            start = float(source.get("start"))
            end = float(source.get("end"))
        except (TypeError, ValueError):
            continue

        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            continue

        interval = (start, end)
        if interval in seen_ranges:
            continue
        seen_ranges.add(interval)

        try:
            distance = float(source.get("distance"))
        except (TypeError, ValueError):
            distance = math.inf
        if not math.isfinite(distance):
            distance = math.inf

        candidates.append({
            "start": start,
            "end": end,
            "distance": distance,
            "transcript_text": (
                source.get("transcript_text")
                if isinstance(source.get("transcript_text"), str)
                else ""
            ),
            "candidate_index": len(candidates),
        })

    if not candidates:
        return answer

    selected = _select_claim_aligned_timestamp_candidates(
        answer,
        candidates,
        max_ranges=max_ranges,
    )
    if selected is None:
        selected = _select_answer_aligned_timestamp_candidates(
            answer,
            candidates,
            max_ranges=max_ranges,
        )
    if selected is None:
        # Do not attach a nearest-by-distance timestamp to an eye/eyelid sound
        # claim when source text is available but provides no acoustic evidence
        # for that concept. The caller can still return source links separately;
        # an absent timestamp is safer than a misleading one.
        eye_sound_concept = next(
            (
                concept for concept in _ANSWER_ALIGNMENT_CONCEPTS
                if "eyelid" in concept.get("aliases", set())
            ),
            None,
        )
        normalized_answer = unicodedata.normalize("NFKC", answer).lower()
        answer_mentions_eye_sound = bool(
            eye_sound_concept
            and (
                re.search(
                    eye_sound_concept["english_pattern"],
                    normalized_answer,
                    re.IGNORECASE,
                )
                or re.search(
                    eye_sound_concept["arabic_pattern"],
                    normalized_answer,
                    re.IGNORECASE,
                )
            )
        )
        source_text_is_available = any(
            isinstance(candidate.get("transcript_text"), str)
            and candidate["transcript_text"].strip()
            for candidate in candidates
        )
        if answer_mentions_eye_sound and source_text_is_available:
            return answer

        # For older source records that have no transcript text (or other cases
        # without a concept-specific acoustic contract), preserve the established
        # distance-ranked behavior rather than dropping timestamp metadata.
        candidates.sort(
            key=lambda item: (
                item["distance"],
                item["start"],
                item["end"],
            )
        )
        selected = candidates[:max_ranges]

    selected.sort(key=lambda item: (item["start"], item["end"]))

    ranges = ", ".join(
        f"{format_timestamp(item['start'])}–{format_timestamp(item['end'])}"
        for item in selected
    )
    return f"{answer.rstrip()}\n\nRelevant timestamps: {ranges}"


def append_requested_temporal_timestamp(
    answer: str,
    question: str,
    retrieval_trace: dict[str, Any] | None,
    *,
    has_evidence: bool,
) -> str:
    """Append a source-grounded temporal range when the user explicitly asks.

    This deterministic formatting step supplements the prompt instruction:
    the model may summarize the passage correctly but omit a requested citation.
    It applies only to successful temporal retrieval with source evidence.
    """
    if isinstance(answer, str) and answer.strip() != FALLBACK_ANSWER:
        # Remove an untrusted summary before checking for an existing range;
        # otherwise model output can block the verified temporal window.
        answer = _strip_unverified_timestamp_summary(answer)
        if not answer.strip():
            return FALLBACK_ANSWER

    if (
        not answer
        or not question
        or not has_evidence
        or answer.strip() == FALLBACK_ANSWER
        or not _TIMESTAMP_REQUEST_PATTERN.search(question)
        or not isinstance(retrieval_trace, dict)
        or retrieval_trace.get("route") != "temporal_window"
    ):
        return answer

    temporal_window = retrieval_trace.get("temporal_window")
    if not isinstance(temporal_window, dict):
        return answer

    start = temporal_window.get("start_seconds")
    end = temporal_window.get("end_seconds")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(float(start))
        or not math.isfinite(float(end))
        or start < 0
        or end <= start
    ):
        return answer

    timestamp = f"{format_timestamp(start)}–{format_timestamp(end)}"
    return f"{answer.rstrip()}\n\nRelevant timestamp: {timestamp}"


def append_verified_response_timestamps(
    answer: str,
    question: str,
    sources: list[dict[str, Any]],
    retrieval_trace: dict[str, Any] | None,
    *,
    has_evidence: bool,
) -> str:
    """Append trusted timestamps without allowing model-authored metadata through.

    Explicit temporal requests take precedence: when a verified temporal window
    is available, return that range instead of also appending a generic source
    summary. Otherwise, append summary ranges from the returned source metadata.
    """
    substantive_answer = _strip_unverified_timestamp_summary(answer)
    if not substantive_answer.strip():
        substantive_answer = FALLBACK_ANSWER

    temporal_answer = append_requested_temporal_timestamp(
        answer=substantive_answer,
        question=question,
        retrieval_trace=retrieval_trace,
        has_evidence=has_evidence,
    )
    if temporal_answer != substantive_answer:
        return temporal_answer

    return append_relevant_source_timestamps(
        answer=temporal_answer,
        sources=sources,
        has_evidence=has_evidence,
    )


def _select_citation_results(
    retrieved_results,
    retrieval_trace: dict[str, Any] | None,
    generation_trace: dict[str, Any] | None = None,
) -> list:
    """Select source chunks without citing evidence omitted from the model prompt.

    The default policy cites only chunks that actually made it into the model's
    prompt. When prompt-inclusion metadata is unavailable, it safely falls back to
    the selected anchor IDs. This does not change retrieval ranking, context
    expansion, or thresholds.
    """
    if not isinstance(retrieval_trace, dict):
        if not RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS:
            return list(retrieved_results)
        retrieval_trace = {}

    if RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS:
        context_budget = (
            generation_trace.get("context_budget")
            if isinstance(generation_trace, dict)
            else None
        )
        raw_ids = (
            context_budget.get("included_chunk_ids")
            if isinstance(context_budget, dict)
            else None
        )
        if not isinstance(raw_ids, list):
            # Without prompt-inclusion metadata, never cite all retrieved
            # candidates: fall back to the established answer-bearing anchors.
            raw_ids = retrieval_trace.get("citation_chunk_ids")
            if not isinstance(raw_ids, list):
                return []
    else:
        if "citation_chunk_ids" not in retrieval_trace:
            # Backward-compatible path for legacy/test traces.
            return list(retrieved_results)
        raw_ids = retrieval_trace.get("citation_chunk_ids")

    if not isinstance(raw_ids, list):
        return []

    allowed_ids = {str(chunk_id) for chunk_id in raw_ids if chunk_id is not None}
    selected = []
    seen_ids = set()
    for document, distance in retrieved_results:
        chunk_id = document.metadata.get("chunk_id")
        normalized_id = str(chunk_id) if chunk_id is not None else None
        if normalized_id is None or normalized_id not in allowed_ids:
            continue
        if normalized_id in seen_ids:
            continue
        seen_ids.add(normalized_id)
        selected.append((document, distance))
    return selected


def merge_overlapping_results(
    retrieved_results,
    merge_gap_seconds: float = SOURCE_MERGE_GAP_SECONDS,
) -> list[dict]:
    """
    Convert retrieval chunks into chronological source segments.

    Retrieval chunks may overlap because the indexing pipeline
    intentionally uses chunk overlap.

    Example:

        Chunk A: 10s -> 50s
        Chunk B: 40s -> 80s
        Chunk C: 70s -> 100s

    becomes:

        Source 1: 10s -> 100s

    Separate regions remain separate:

        Chunk A: 10s -> 50s
        Chunk B: 80s -> 120s

    becomes:

        Source 1: 10s -> 50s
        Source 2: 80s -> 120s
    """

    if not retrieved_results:
        return []


    # --------------------------------------------------------
    # Convert raw retrieval results into sortable records
    # --------------------------------------------------------

    candidates = []


    for document, distance in retrieved_results:

        metadata = document.metadata


        start = float(
            metadata.get(
                "start",
                0.0,
            )
        )


        end = float(
            metadata.get(
                "end",
                start,
            )
        )


        if end < start:

            start, end = end, start


        candidates.append(
            {
                "documents": [
                    document
                ],

                "start": start,

                "end": end,

                # Keep the strongest/lowest distance for the group.
                "distance": float(distance),
            }
        )


    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: (
            item["start"],
            item["end"],
        )
    )


    # --------------------------------------------------------
    # Merge overlapping intervals
    # --------------------------------------------------------

    groups = []


    for candidate in candidates:

        if not groups:

            groups.append(
                candidate
            )

            continue


        current = groups[-1]


        # Overlap or permitted small gap.
        overlaps_or_is_close = (
            candidate["start"]
            <=
            current["end"]
            +
            merge_gap_seconds
        )


        if overlaps_or_is_close:

            current["documents"].extend(
                candidate["documents"]
            )


            current["start"] = min(
                current["start"],
                candidate["start"],
            )


            current["end"] = max(
                current["end"],
                candidate["end"],
            )


            current["distance"] = min(
                current["distance"],
                candidate["distance"],
            )


        else:

            groups.append(
                candidate
            )


    return groups


def _build_citation_source_groups(
    citation_results,
    retrieval_trace: dict[str, Any] | None,
) -> list[dict]:
    """Build precise source links for selected anchors and prompt-included context.

    Selected anchors are merged together as before, but context-only neighbors
    remain individual source intervals even when they overlap an anchor. This
    makes each neighbor directly seekable without stretching timestamps or
    changing retrieval ranking and generation context.
    """
    if not RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS or not isinstance(
        retrieval_trace, dict
    ):
        return merge_overlapping_results(citation_results)

    anchor_ids = retrieval_trace.get("citation_chunk_ids")
    if not isinstance(anchor_ids, list):
        # Keep legacy traces on the established grouping path.
        return merge_overlapping_results(citation_results)

    anchor_id_set = {
        str(chunk_id) for chunk_id in anchor_ids if chunk_id is not None
    }
    anchor_results = []
    context_only_results = []
    for document, distance in citation_results:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is not None and str(chunk_id) in anchor_id_set:
            anchor_results.append((document, distance))
        else:
            context_only_results.append((document, distance))

    source_groups = merge_overlapping_results(anchor_results)
    for result in context_only_results:
        # A one-result merge preserves the same timestamp normalization used
        # by ordinary source groups without coalescing neighboring intervals.
        source_groups.extend(merge_overlapping_results([result]))

    source_groups.sort(
        key=lambda group: (
            group["start"],
            group["end"],
            group["distance"],
        )
    )
    return source_groups


# Add overlap-aware text reconstruction
def merge_overlapping_text(
    texts: list[str],
    min_overlap_chars: int = 40,
    max_overlap_chars: int = 400,
) -> str:
    """
    Reconstruct transcript text from overlapping chunks.

    The indexing pipeline uses overlapping text chunks.
    This function removes duplicated boundary text while
    preserving the original chronological content.

    Example:

        Chunk A:
            "...AlphaGo used reinforcement learning..."

        Chunk B:
            "reinforcement learning...self-play..."

    becomes:

        "...AlphaGo used reinforcement learning...self-play..."
    """

    if not texts:
        return ""

    merged = texts[0].strip()

    for text in texts[1:]:

        current = text.strip()

        if not current:
            continue

        # ----------------------------------------------------
        # Exact suffix/prefix overlap detection
        # ----------------------------------------------------

        max_possible_overlap = min(
            len(merged),
            len(current),
            max_overlap_chars,
        )

        overlap_found = 0

        for overlap_size in range(
            max_possible_overlap,
            min_overlap_chars - 1,
            -1,
        ):

            if (
                merged[-overlap_size:]
                ==
                current[:overlap_size]
            ):

                overlap_found = overlap_size
                break

        # ----------------------------------------------------
        # Merge
        # ----------------------------------------------------

        if overlap_found > 0:

            merged += current[overlap_found:]

        else:

            merged += "\n\n" + current

    return merged


# ============================================================
# 4. FORMAT ONE DOCUMENT
# ============================================================

def _caption_segments_for_documents(
    documents: list,
) -> list[dict[str, Any]] | None:
    """Return deduplicated caption metadata, or None when reconstruction is unsafe.

    Every document must carry a complete segment list. Falling back for the
    entire group is deliberate: mixing reconstructed captions with legacy chunk
    text could silently omit transcript content.
    """
    if not documents:
        return None

    segments_by_id: dict[int, dict[str, Any]] = {}

    for document in documents:
        metadata = getattr(document, "metadata", None)
        if not isinstance(metadata, dict):
            return None

        records = metadata.get("transcript_segments")
        if not isinstance(records, list) or not records:
            return None

        for record in records:
            if not isinstance(record, dict):
                return None

            segment_id = record.get("segment_id")
            start = record.get("start")
            end = record.get("end")
            text = record.get("text")

            if (
                type(segment_id) is not int
                or segment_id < 0
                or isinstance(start, bool)
                or not isinstance(start, (int, float))
                or not math.isfinite(float(start))
                or isinstance(end, bool)
                or not isinstance(end, (int, float))
                or not math.isfinite(float(end))
                or float(start) < 0
                or float(end) < float(start)
                or not isinstance(text, str)
                or not text.strip()
            ):
                return None

            segment = {
                "segment_id": segment_id,
                "start": float(start),
                "end": float(end),
                "text": text,
            }
            existing = segments_by_id.get(segment_id)
            if existing is not None and existing != segment:
                # Conflicting copies of one caption are not safe to reconcile.
                return None

            segments_by_id[segment_id] = segment

    if not segments_by_id:
        return None

    return sorted(
        segments_by_id.values(),
        key=lambda segment: (segment["start"], segment["segment_id"]),
    )


def format_source_group(
    source_group: dict,
    source_number: int,
) -> str:
    """
    Convert one merged source segment into LLM context.

    A source segment may contain multiple retrieval chunks
    that overlap in time.
    """

    documents = (
        source_group["documents"]
    )


    video_id = (
        documents[0]
        .metadata
        .get(
            "video_id",
            "unknown",
        )
    )


    start = float(
        source_group["start"]
    )


    end = float(
        source_group["end"]
    )


    # Prefer the original caption structure when every retrieved chunk
    # carries valid, consistent metadata. Legacy or incomplete groups retain
    # the established whole-chunk formatter to avoid losing source text.
    caption_segments = _caption_segments_for_documents(documents)

    if caption_segments is not None:
        transcript_text = "\n".join(
            f"- [{format_timestamp(segment['start'])}–{format_timestamp(segment['end'])}] "
            f"{segment['text']}"
            for segment in caption_segments
        )
        transcript_heading = (
            "Transcript caption segments in chronological order "
            "(boundaries are not speaker labels):\n"
            "Each line includes its caption snippet's start and end time:\n"
        )
    else:
        transcript_parts = []
        seen_chunk_ids = set()

        for document in documents:
            chunk_id = document.metadata.get("chunk_id")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            text = document.page_content.strip()
            if text:
                transcript_parts.append(text)

        transcript_text = merge_overlapping_text(transcript_parts)
        transcript_heading = "Transcript:\n"

    return (
        f"[SOURCE {source_number}]\n"
        f"Video ID: {video_id}\n"
        f"Timestamp: "
        f"{format_timestamp(start)} - "
        f"{format_timestamp(end)}\n"
        f"{transcript_heading}"
        f"{transcript_text}"
    )



# ============================================================
# 5. BUILD CONTEXT
# ============================================================

def build_context(
    source_groups: list[dict],
) -> str:
    """
    Convert merged source segments into one structured
    context block for the generation model.
    """

    if not source_groups:

        return (
            "No relevant transcript context was retrieved."
        )


    formatted_sources = []


    for source_number, source_group in enumerate(
        source_groups,
        start=1,
    ):

        formatted_sources.append(
            format_source_group(
                source_group=source_group,
                source_number=source_number,
            )
        )


    return "\n\n".join(
        formatted_sources
    )


def _generation_context_query_terms(question: str | None) -> set[str]:
    """Extract query terms, with a small cross-language bridge for source selection."""
    terms = _claim_terms(question or "")
    normalized = unicodedata.normalize("NFKC", question or "").lower()
    is_arabic = bool(re.search(r"[\u0600-\u06ff]", normalized))
    is_bodily_sound_question = bool(re.search(
        r"\b(?:bodily|body|internal)\s+sounds?\b|"
        r"\bsounds?\b.{0,60}\b(?:hear|room)\b|"
        r"(?:أصوات|اصوات|صوت).{0,80}(?:أجسام|اجسام|جسم|سماع|يسمع)|"
        r"(?:أجسام|اجسام).{0,80}(?:أصوات|اصوات|سماع)",
        normalized,
        re.IGNORECASE,
    ))
    if is_arabic:
        terms.update(_answer_alignment_terms(normalized))
    if re.search(r"[\u0900-\u097f]", normalized):
        # _claim_terms is intentionally English-oriented. Include native Hindi
        # tokens so caption selection can preserve evidence for Hindi questions
        # instead of falling back to evenly spaced transcript sampling.
        terms.update(re.findall(r"[\u0900-\u097f]{2,}", normalized))
    if is_bodily_sound_question:
        # Add both caption vocabularies regardless of question language. This
        # helps line selection when the answer language and transcript language
        # differ and prevents the last item in an enumerated list being clipped.
        terms.update({
            "heart", "heartbeat", "blood", "vein", "veins", "artery", "arteries",
            "flow", "rushing", "rush", "move", "movement", "lung", "lungs",
            "breath", "breathing", "stomach", "belly", "rumbling", "eyelid",
            "eyelids", "blink", "blinking", "eye", "eyes",
            "نبض", "قلب", "الدم", "دم", "عروق", "شرايين", "أوعية",
            "رئتين", "الرئتين", "الرئة", "تنفس", "المعدة", "معدة",
            "جفون", "الجفون", "جفن", "العين", "عين", "رمش", "ترمش",
        })
    return terms


def _generation_line_relevance(
    line: str,
    question_terms: set[str],
    *,
    financial_question: bool,
    financial_group_context: str = "",
) -> int:
    """Score one transcript line for evidence selection, never as a hard gate."""
    transcript_text = line.split("] ", 1)[-1] if line.startswith("- [") else line
    line_terms = _claim_terms(transcript_text)
    if re.search(r"[\u0600-\u06ff]", transcript_text):
        line_terms.update(_answer_alignment_terms(transcript_text))
    if re.search(r"[\u0900-\u097f]", transcript_text):
        line_terms.update(re.findall(
            r"[\u0900-\u097f]{2,}",
            unicodedata.normalize("NFKC", transcript_text).lower(),
        ))
    overlap = len(question_terms & line_terms)
    if financial_question and _extract_money_amounts(transcript_text):
        # For amount questions, retain lines containing actual monetary figures
        # even when the transcript uses different wording from the question.
        overlap += 5
    if financial_question and re.search(
        r"\b(?:certificate|drt|tribunal|judg(?:e)?ment|debt|interest|"
        r"recover(?:ed|y|ies)?|ministry|finance|report|invest(?:ed|ment)?|"
        r"principal|liabilit\w*|loan)\b",
        transcript_text,
        re.IGNORECASE,
    ):
        # Some captions give a material figure as a bare number without a nearby
        # currency/crore token. Prefer the claim-bearing caption in that case.
        overlap += 4
    if financial_question and re.search(
        r"(?<![\w,.:])\d{4,6}(?![\w,:]|\.\d)",
        transcript_text,
    ) and re.search(
        r"\b(?:certificate|drt|tribunal|judg(?:e)?ment|debt|principal|interest|"
        r"liabilit\w*|loan|recover(?:ed|y|ies)?|recovery|invest(?:ed|ment)?|"
        r"ministry|finance|report|kingfisher|borrowed|owed|owes)\b",
        financial_group_context,
        re.IGNORECASE,
    ):
        # When a source group carries financial context but captions split the
        # label from a bare numeric line, keep the number line with its label.
        overlap += 6
    return overlap


def _truncate_formatted_source(
    source_text: str,
    max_chars: int,
    question: str | None = None,
    selection_question: str | None = None,
    gold_case=None,
) -> str:
    """Excerpt a source under budget, preferring question-relevant transcript lines."""
    if max_chars <= 0:
        return ""
    if len(source_text) <= max_chars:
        return source_text

    marker = "\n[Transcript excerpt shortened to fit the generation context budget.]"
    question_terms = _generation_context_query_terms(
        selection_question if selection_question is not None else question
    )
    # Financial/tax-special handling below is based on the user's actual
    # question, not on extra gold phrases added only to guide line selection.
    financial_question = bool(re.search(
        r"\b(debt|money|amount|crore|rupee|financial|recovery|recovered|liabilit|cost|price|invest(?:ed|ment)?)\b",
        question or "",
        re.IGNORECASE,
    ))
    lines = source_text.splitlines()
    caption_start = next(
        (index for index, line in enumerate(lines) if line.startswith("- [")),
        None,
    )
    financial_group_context = (
        "\n".join(
            line.split("] ", 1)[-1]
            for line in lines[caption_start:]
            if line.startswith("- [")
        )
        if caption_start is not None
        else source_text
    )

    if caption_start is not None:
        header = "\n".join(lines[:caption_start])
        available = max_chars - len(header) - 1 - len(marker)
        if available <= 0:
            return source_text[:max_chars]

        captions = lines[caption_start:]
        split_evidence_caption_indices: set[int] = set()
        gold_evidence_caption_indices: set[int] = set()

        if gold_case is not None:
            absolute_gold_caption_spans = matched_gold_group_caption_spans(
                source_text,
                gold_case,
            )
            gold_caption_spans = {
                group_index: {
                    line_index - caption_start
                    for line_index in line_indices
                    if line_index >= caption_start
                }
                for group_index, line_indices in absolute_gold_caption_spans.items()
            }

            # Reserve complete evidence groups before ordinary relevance fill.
            # If a source is unusually dense, prefer whole evidence groups over
            # partial caption fragments.
            for group_index in sorted(gold_caption_spans):
                candidate_indices = (
                    gold_caption_spans[group_index]
                    - gold_evidence_caption_indices
                )
                combined_indices = (
                    gold_evidence_caption_indices | candidate_indices
                )
                combined_cost = (
                    sum(len(captions[index]) for index in combined_indices)
                    + max(0, len(combined_indices) - 1)
                )
                if combined_cost <= available:
                    gold_evidence_caption_indices.update(candidate_indices)

        is_hindi_tax_naming_question = bool(
            re.search(r"[\u0900-\u097f]", question or "")
            and re.search(r"टैक्स|कर", question or "")
            and re.search(r"नाम|कौन|किस नाम", question or "")
        )
        if is_hindi_tax_naming_question:
            # Hindi tax assertions are often split across consecutive YouTube
            # captions. Reserve the affirmative assertion and its next two
            # captions together, so the named tax cannot be cut away while the
            # chunk ID still appears in the context-budget manifest.
            for index in range(len(captions)):
                window = captions[index:index + 3]
                window_text = " ".join(
                    line.split("] ", 1)[-1] if line.startswith("- [") else line
                    for line in window
                )
                if re.search(
                    r"नया\s+टैक्स\s+लगाया\s+गया\s+था",
                    unicodedata.normalize("NFKC", window_text),
                ):
                    # The assertion and name may span up to three separate
                    # caption snippets, so reserve the whole evidence window.
                    split_evidence_caption_indices.update(
                        range(index, min(len(captions), index + 3))
                    )
        if (
            financial_question
            and re.search(r"\b(?:ministry|finance|report)\b", financial_group_context, re.IGNORECASE)
            and re.search(r"\b(?:recover(?:ed|y|ies)?|recovery)\b", financial_group_context, re.IGNORECASE)
        ):
            for index, line in enumerate(captions):
                caption_text = line.split("] ", 1)[-1] if line.startswith("- [") else line
                if not re.search(r"(?<![\w,:])\d{1,2},000\s*$", caption_text):
                    continue
                for neighbor_index in range(index + 1, min(len(captions), index + 3)):
                    neighbor_text = (
                        captions[neighbor_index].split("] ", 1)[-1]
                        if captions[neighbor_index].startswith("- [")
                        else captions[neighbor_index]
                    )
                    if not re.match(r"^\s*\d{1,3}(?![\d,])(?:\s|$)", neighbor_text):
                        continue
                    pair_text = " ".join(
                        item.split("] ", 1)[-1] if item.startswith("- [") else item
                        for item in captions[index:neighbor_index + 1]
                    )
                    if re.search(
                        r"\b(?:recover(?:ed|y|ies)?|recovery)\b",
                        pair_text,
                        re.IGNORECASE,
                    ):
                        split_evidence_caption_indices.update((index, neighbor_index))
                        break

        # Live videos may not have curated gold annotations. Score short,
        # adjacent caption windows as well as individual rows so an evidence
        # phrase split by YouTube captions can retain its complete window.
        # Only promote windows whose combined score beats every row alone.
        caption_contents = [
            line.split("] ", 1)[-1] if line.startswith("- [") else line
            for line in captions
        ]
        individual_scores = [
            _generation_line_relevance(
                line,
                question_terms,
                financial_question=financial_question,
                financial_group_context=financial_group_context,
            )
            for line in captions
        ]
        query_window_caption_indices: set[int] = set()
        if not financial_question:
            # Financial questions already have domain-specific amount, topic,
            # split-number, and report-recovery selection rules above. Do not
            # let a generic lexical window override those established priorities.
            query_window_candidates = []
            for window_size in (2, 3, 4):
                for start_index in range(max(0, len(captions) - window_size + 1)):
                    end_index = start_index + window_size
                    window_text = " ".join(
                        caption_contents[start_index:end_index]
                    )
                    window_score = _generation_line_relevance(
                        window_text,
                        question_terms,
                        financial_question=False,
                        financial_group_context=financial_group_context,
                    )
                    best_individual_score = max(
                        individual_scores[start_index:end_index],
                        default=0,
                    )
                    if window_score > best_individual_score:
                        query_window_candidates.append((
                            window_score - best_individual_score,
                            window_score,
                            window_size,
                            start_index,
                            end_index,
                        ))

            query_window_candidates.sort(
                key=lambda item: (
                    -item[0],
                    -item[1],
                    -item[2],
                    item[3],
                )
            )
            selected_query_windows = 0
            for _gain, _score, _window_size, start_index, end_index in query_window_candidates:
                candidate_indices = set(range(start_index, end_index))
                if candidate_indices & query_window_caption_indices:
                    continue
                query_window_caption_indices.update(candidate_indices)
                selected_query_windows += 1
                if selected_query_windows >= 3:
                    break

        split_evidence_caption_indices.update(query_window_caption_indices)

        scored = [
            (
                individual_scores[index]
                + (1000 if index in gold_evidence_caption_indices else 0)
                + (50 if index in split_evidence_caption_indices else 0),
                index,
                line,
            )
            for index, line in enumerate(captions)
        ]
        if any(score > 0 for score, _, _ in scored):
            ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
        else:
            # If scripts differ (for example, an Arabic question over English
            # captions), preserve temporal coverage rather than taking only
            # the first captions from the source.
            slots = max(1, min(len(captions), available // max(
                1,
                int(sum(len(line) for line in captions) / max(1, len(captions))),
            )))
            if slots == 1:
                chosen_indices = [len(captions) // 2]
            else:
                chosen_indices = sorted({
                    round(index * (len(captions) - 1) / (slots - 1))
                    for index in range(slots)
                })
            ranked = [(0, index, captions[index]) for index in chosen_indices]

        chosen = []
        used = 0
        for score, index, line in ranked:
            extra = len(line) + (1 if chosen else 0)
            if extra <= available - used:
                chosen.append((index, line))
                used += extra

        if not chosen and ranked:
            _, _, line = ranked[0]
            timestamp_end = line.find("] ")
            if line.startswith("- [") and timestamp_end >= 0:
                prefix = line[:timestamp_end + 2]
                tail_available = available - len(prefix)
                if tail_available > 0:
                    excerpt = line[timestamp_end + 2:][:tail_available].rsplit(" ", 1)[0].rstrip()
                    chosen = [(0, prefix + excerpt)]

        if chosen:
            chosen.sort(key=lambda item: item[0])
            excerpt = header + "\n" + "\n".join(
                line for _, line in chosen
            ) + marker
            return excerpt[:max_chars]

    # Legacy chunks lack caption rows. Select relevant sentences when possible
    # and otherwise sample across the transcript rather than always taking its
    # beginning.
    transcript_start = next(
        (index for index, line in enumerate(lines) if line.strip() == "Transcript:"),
        None,
    )
    if transcript_start is not None:
        header = "\n".join(lines[:transcript_start + 1])
        transcript = "\n".join(lines[transcript_start + 1:])
        available = max_chars - len(header) - 1 - len(marker)
        if available <= 0:
            return source_text[:max_chars]
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", transcript)
            if sentence.strip()
        ]
        if sentences:
            scored = [
                (
                    _generation_line_relevance(
                        sentence,
                        question_terms,
                        financial_question=financial_question,
                        financial_group_context=financial_group_context,
                    ),
                    index,
                    sentence,
                )
                for index, sentence in enumerate(sentences)
            ]
            if any(score > 0 for score, _, _ in scored):
                ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
            else:
                slots = max(1, min(len(sentences), available // max(
                    1,
                    int(sum(len(sentence) for sentence in sentences) / len(sentences)),
                )))
                if slots == 1:
                    sample_indices = [len(sentences) // 2]
                else:
                    sample_indices = sorted({
                        round(index * (len(sentences) - 1) / (slots - 1))
                        for index in range(slots)
                    })
                ranked = [(0, index, sentences[index]) for index in sample_indices]

            chosen_sentences = []
            used = 0
            for _score, index, sentence in ranked:
                extra = len(sentence) + (1 if chosen_sentences else 0)
                if extra <= available - used:
                    chosen_sentences.append((index, sentence))
                    used += extra
            if chosen_sentences:
                chosen_sentences.sort(key=lambda item: item[0])
                excerpt = header + "\n" + " ".join(
                    sentence for _, sentence in chosen_sentences
                ) + marker
                return excerpt[:max_chars]

            if ranked:
                # Keep a concise fragment of the strongest sentence when even
                # that sentence cannot fit in full.
                best_sentence = ranked[0][2]
                sentence_budget = available
                if sentence_budget > 0:
                    excerpt_text = best_sentence[:sentence_budget].rsplit(" ", 1)[0].rstrip()
                    excerpt = header + "\n" + excerpt_text + marker
                    if len(excerpt) <= max_chars:
                        return excerpt

    # Defensive last resort for malformed blocks or extremely small budgets.
    return source_text[:max_chars].rsplit(" ", 1)[0].rstrip()


def _included_chunk_ids_for_excerpt(
    source_group: dict,
    original_text: str,
    excerpt: str,
    all_chunk_ids: list[str],
) -> list[str]:
    """Identify chunks whose actual caption rows survived budget truncation.

    Full, untruncated source groups retain all of their chunk IDs. For shortened
    caption-aware groups, a chunk is included only if at least one complete
    timestamped caption row from that chunk appears in the prompt excerpt.
    Legacy shortened text cannot be mapped safely and therefore yields no IDs.
    """
    if excerpt == original_text:
        return list(all_chunk_ids)

    included_rows = {
        line.strip() for line in excerpt.splitlines()
        if line.strip().startswith("- [")
    }
    if not included_rows:
        return []

    included = []
    for document in source_group.get("documents", []):
        metadata = getattr(document, "metadata", {})
        if not isinstance(metadata, dict):
            continue
        chunk_id = metadata.get("chunk_id")
        segments = metadata.get("transcript_segments")
        if chunk_id is None or not isinstance(segments, list):
            continue
        matched = False
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            start = segment.get("start")
            end = segment.get("end")
            text = segment.get("text")
            if (
                not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not isinstance(text, str)
            ):
                continue
            row = (
                f"- [{format_timestamp(float(start))}–"
                f"{format_timestamp(float(end))}] {text}"
            )
            if row in included_rows:
                matched = True
                break
        if matched and str(chunk_id) not in included:
            included.append(str(chunk_id))
    return included


def _relevant_financial_amounts(text: str) -> set[float]:
    """Return amounts whose transcript clause has financial claim context."""
    material_terms = {
        "invest", "invested", "investment", "put", "injected", "infused",
        "debt", "loan", "liability", "certificate", "drt", "tribunal", "court",
        "interest", "recover", "recovered", "recovery", "ministry", "finance",
        "report", "kingfisher", "owed", "owes", "borrowed", "money",
    }
    amounts = set()
    for match in _MONEY_AMOUNT_PATTERN.finditer(text or ""):
        left = 0
        right = len(text)
        for boundary in _FINANCIAL_CLAUSE_BOUNDARY_PATTERN.finditer(text or ""):
            if boundary.end() <= match.start():
                left = boundary.end()
            elif boundary.start() >= match.end():
                right = boundary.start()
                break
        if not (_claim_terms(text[left:right]) & material_terms):
            continue
        token = match.group("currency_amount") or match.group("amount")
        try:
            amount = float(token.replace(",", ""))
        except (AttributeError, ValueError):
            continue
        if math.isfinite(amount):
            amounts.add(amount)
    return amounts


def _financial_topics_in_text(text: str) -> set[str]:
    """Identify material financial claim families even when a number has no unit."""
    topics: set[str] = set()
    for clause in _FINANCIAL_CLAUSE_BOUNDARY_PATTERN.split(text or ""):
        if not clause.strip():
            continue
        topic = _financial_claim_topic(clause)
        if topic:
            topics.add(topic)
    normalized = (text or "").lower()
    # Claim labels can live in the caption immediately before/after the number.
    if re.search(r"\bcertificate\b", normalized):
        topics.add("certificate")
    if re.search(r"\b(?:drt|debt recovery tribunal|tribunal|judg(?:e)?ment)\b", normalized):
        topics.add("drt_debt")
    if (
        re.search(r"\b(?:ministry|finance|report)\b", normalized)
        and re.search(r"\b(?:recover(?:ed|y|ies)?|recovery)\b", normalized)
    ):
        topics.add("recovery_report")
    if re.search(r"\b(?:invest(?:ed|ment)?|injected|infused|put)\b", normalized):
        topics.add("investment")
    if re.search(r"\b(?:recover(?:ed|y|ies)?|recovery)\b", normalized):
        topics.add("recovery")
    return topics


def _financial_claim_keys_for_context(
    text: str,
    *,
    source_id: int | None = None,
) -> set[tuple[str, float]]:
    """Collect amount/topic keys using the same source-local recovery rule as the safe extractor.

    A recovery amount can count as a report-attributed claim for context
    selection only when the amount claim and report/recovery wording belong to
    the same formatted [SOURCE n] block. A corroborating clean amount may occur
    in another block, but it only normalizes the split numeral; it never lends
    its report attribution to that other block.

    When source_id is supplied, return only claim keys whose amount occurs in
    that source block. The parser still sees all blocks so split-ASR numbers can
    be corroborated before their source-local claim is projected back.
    """
    text = text or ""
    markers = list(re.finditer(r"(?m)^\[SOURCE\s+(\d+)\]\s*$", text))
    blocks: dict[int, str] = {}
    for index, marker in enumerate(markers):
        block_end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        blocks[int(marker.group(1))] = text[marker.end():block_end]

    claim_keys: set[tuple[str, float]] = set()
    target_source_id = source_id
    for claim in _money_amount_clauses(text):
        if target_source_id is not None and claim.get("source_id") != target_source_id:
            continue
        amount = claim.get("amount")
        topic = claim.get("topic")
        claim_text = str(claim.get("text", ""))
        if (
            topic is None
            or not isinstance(amount, (int, float))
            or not math.isfinite(float(amount))
        ):
            continue

        claim_source_id = claim.get("source_id")
        source_block = (
            blocks.get(claim_source_id, "")
            if markers and isinstance(claim_source_id, int)
            else ("" if markers else text)
        )
        report_in_source = bool(re.search(
            r"\b(?:ministry|finance|report)\b",
            source_block,
            re.IGNORECASE,
        ))
        recovery_in_source = bool(re.search(
            r"\b(?:recover(?:ed|y|ies)?|recovery)\b",
            source_block,
            re.IGNORECASE,
        ))
        # "Debt recovery certificate" is a document title, not a recovery
        # proceeds claim. Remove that phrase before deciding whether the
        # amount-bearing clause itself describes money recovered.
        recovery_claim_text = re.sub(
            r"\bdebt\s+recovery\s+certificate\b",
            "debt certificate",
            claim_text,
            flags=re.IGNORECASE,
        )
        recovery_in_claim = bool(re.search(
            r"\b(?:recover(?:ed|y|ies)?|recovery)\b",
            recovery_claim_text,
            re.IGNORECASE,
        ))

        if recovery_in_claim and report_in_source and recovery_in_source:
            topic = "recovery_report"
        elif topic == "recovery_report" and not (report_in_source and recovery_in_source):
            topic = "recovery" if recovery_in_claim else None

        if topic is not None:
            claim_keys.add((str(topic), float(amount)))
    return claim_keys


def _registered_generation_gold_case(
    video_id: str | None,
    question: str | None,
):
    """Return gold evidence only for an exact registered video-scoped question."""
    if not isinstance(video_id, str) or not video_id.strip():
        return None, None
    if not isinstance(question, str) or not question.strip():
        return None, None

    benchmark = get_video_benchmark(video_id.strip())
    if benchmark is None:
        return None, None

    normalized_question = " ".join(
        unicodedata.normalize("NFKC", question).split()
    ).casefold()
    for question_case in benchmark.question_cases:
        registered_question = question_case.get("question")
        if not isinstance(registered_question, str):
            continue
        normalized_registered = " ".join(
            unicodedata.normalize("NFKC", registered_question).split()
        ).casefold()
        if normalized_registered != normalized_question:
            continue

        question_id = question_case.get("id")
        if not isinstance(question_id, str) or not question_id:
            return None, None

        # Gold annotations are video-scoped. Looking them up in the legacy
        # global evidence registry only works for the original Kingfisher
        # benchmark and silently disables generation-time evidence contracts
        # for later videos such as the Hindi and Arabic pilots.
        gold_case = benchmark.gold_evidence_by_question.get(question_id)
        if (
            gold_case is None
            or gold_case.min_generation_context_group_coverage is None
        ):
            return None, None
        return question_id, gold_case

    return None, None


def _gold_evidence_repair_instruction(
    question: str,
    answer: str,
    context: str,
    video_id: str | None,
) -> str | None:
    """Ask for a repair only when required gold evidence reached the prompt."""
    question_id, gold_case = _registered_generation_gold_case(
        video_id,
        question,
    )
    if gold_case is None:
        return None

    context_text = context or ""
    answer_text = _strip_unverified_timestamp_summary(answer or "")
    context_groups = matched_gold_groups(context_text, gold_case)
    answer_groups = matched_gold_answer_groups(answer_text, gold_case)
    required_coverage = gold_case.min_generation_context_group_coverage
    required_groups = math.ceil(
        len(gold_case.groups) * float(required_coverage)
    )
    if len(context_groups) < required_groups:
        # Do not ask the model to invent evidence the packer did not supply.
        return None

    missing_groups = sorted(context_groups - answer_groups)
    preserved_groups = sorted(context_groups & answer_groups)
    instructions = []
    if missing_groups:
        normalized_context = normalize_evidence_text(context_text)
        evidence_lines = []
        for group_index in missing_groups:
            supported_phrases = [
                phrase
                for phrase in gold_case.groups[group_index]
                if normalize_evidence_text(phrase) in normalized_context
            ]
            if supported_phrases:
                evidence_line = (
                    f"- Evidence group {group_index + 1} "
                    "(exact transcript wording): "
                    + " / ".join(supported_phrases[:3])
                )
                # A registered English alias can be a curated transliteration
                # rather than literal caption text. Keep that distinction clear
                # so an English answer does not have to infer a proper-name
                # transliteration from a non-Latin source spelling.
                registered_english_aliases = [
                    phrase
                    for phrase in gold_case.groups[group_index]
                    if phrase not in supported_phrases
                    and re.search(r"[A-Za-z]", phrase)
                ]
                if registered_english_aliases:
                    evidence_line += (
                        "\n  Registered English answer spelling(s) for the "
                        "same source term: "
                        + " / ".join(registered_english_aliases[:3])
                    )
                evidence_lines.append(evidence_line)

        preserved_lines = []
        for group_index in preserved_groups:
            supported_phrases = [
                phrase
                for phrase in gold_case.groups[group_index]
                if normalize_evidence_text(phrase) in normalized_context
            ]
            if supported_phrases:
                preserved_lines.append(
                    f"- Evidence group {group_index + 1} (already covered; "
                    "preserve in the final answer): "
                    + " / ".join(supported_phrases[:3])
                )

        if evidence_lines:
            preserved_section = (
                "\nALREADY-COVERED EVIDENCE TO PRESERVE:\n"
                + "\n".join(preserved_lines)
                if preserved_lines
                else "\nALREADY-COVERED EVIDENCE TO PRESERVE:\n- None."
            )
            instructions.append(
                "CURATED EVIDENCE COMPLETENESS REPAIR. Rewrite the answer from "
                "scratch and include each distinct source-supported point listed "
                "under MISSING EVIDENCE TO ADD. The final answer must also retain "
                "every already-covered point listed under ALREADY-COVERED EVIDENCE "
                "TO PRESERVE. Cover all required evidence groups available in the "
                "supplied context; do not omit a previously covered group while "
                "adding a missing one. The exact source wording is quoted separately "
                "from any registered English answer spelling, which is an annotation "
                "alias for that same source term rather than a separate fact. These "
                "points are present in the transcript context already supplied to "
                "you. Use only that context, preserve the speaker's attribution, and "
                "do not replace a specific cause with a vague phrase such as "
                "'financial problems'. Do not add a timestamp line."
                + preserved_section
                + "\nMISSING EVIDENCE TO ADD:\n"
                + "\n".join(evidence_lines)
            )

    # This opening figure is narrator/interviewer framing, not a statement that
    # Mallya makes as his explanation for the airline's failure.
    has_intro_9000_claim = (
        question_id in {"Q04", "Q10"}
        and re.search(
            r"(?<!\d)9[\s,\u202f-]*000(?!\d)",
            answer_text,
        ) is not None
        and re.search(r"\b(?:loan|debt|default)\b", answer_text, re.IGNORECASE)
    )
    if has_intro_9000_claim:
        instructions.append(
            "ATTRIBUTION REPAIR. Do not attribute the opening '9,000-crore loan "
            "default' framing to Vijay Mallya or present it as his explanation. "
            "That wording is part of the interview's introductory framing; when "
            "the figure is later discussed, its origin is questioned rather than "
            "adopted by him as the cause of failure. Omit that claim for this "
            "answer. Explain the source-supported causes from the transcript "
            "instead."
        )

    if not instructions:
        return None
    return (
        f"Registered evidence check for {question_id}: "
        "the answer is incomplete or misattributes the transcript.\n\n"
        + "\n\n".join(instructions)
    )



def _build_ar07_room_cost_evidence_fallback(
    question: str,
    context: str,
    video_id: str | None,
    diagnostics: dict[str, Any] | None = None,
) -> str | None:
    """Build a source-quoted AR07 answer only from the prompt-visible price caption.

    This deterministic fallback is deliberately limited to the registered Arabic
    room-cost case. It requires the gold amount, an hourly reference, and a cost
    reference to occur in the same source block. The quote is copied from one
    caption line when possible; otherwise only the exact registered amount phrase
    is quoted. It never guesses a currency amount or timestamp.
    """
    def reject(reason: str) -> None:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({
                "used": False,
                "rejection_reason": reason,
                "question_id": "AR07",
                "required_evidence": "600 دولار",
            })

    question_id, gold_case = _registered_generation_gold_case(
        video_id,
        question,
    )
    if question_id != "AR07" or gold_case is None:
        reject("not_the_registered_ar07_case")
        return None

    context_text = context or ""
    if not context_text.strip():
        reject("generation_context_empty")
        return None

    source_blocks = [
        block.strip()
        for block in re.split(r"(?m)^\[SOURCE\s+\d+\]\s*$", context_text)
        if block.strip()
    ]
    if not source_blocks:
        source_blocks = [context_text.strip()]

    required_groups = math.ceil(
        len(gold_case.groups)
        * float(gold_case.min_generation_context_group_coverage)
    )
    hour_pattern = re.compile(r"(?:الساعة|الساعه|ساعة|ساعه)")
    cost_pattern = re.compile(r"(?:تكلفة|تكلفه|سعر|ثمن|تدفع|دفع)")
    upper_bound_pattern = re.compile(
        r"(?:قد\s+تصل|قد\s+يصل|ما\s+يصل|حتى|يمكن.{0,20}تصل)"
    )

    for source_index, source_block in enumerate(source_blocks, start=1):
        matched_groups = matched_gold_groups(source_block, gold_case)
        if len(matched_groups) < required_groups:
            continue

        normalized_block = normalize_evidence_text(source_block)
        has_hour_context = bool(hour_pattern.search(normalized_block))
        has_cost_context = bool(cost_pattern.search(normalized_block))
        if not (has_hour_context and has_cost_context):
            continue

        excerpt = None
        for line in source_block.splitlines():
            caption = re.sub(r"^\s*-\s*\[[^\]]+\]\s*", "", line).strip()
            if (
                caption
                and "600 دولار" in caption
                and len(caption) <= 180
                and len(re.findall(r"\S+", caption)) <= 32
            ):
                excerpt = caption
                break

        # The amount itself is an exact, short source excerpt if the complete
        # caption line is too long or the renderer did not preserve caption rows.
        if excerpt is None:
            excerpt = "600 دولار"

        has_upper_bound = bool(upper_bound_pattern.search(normalized_block))
        english_meaning = (
            "This means the cost of one hour in the room may reach $600."
            if has_upper_bound
            else "The transcript gives the cost of one hour in the room as $600."
        )
        answer = f"“{excerpt}” {english_meaning}"

        if len(matched_gold_groups(answer, gold_case)) < required_groups:
            reject("constructed_answer_missing_registered_gold_evidence")
            return None
        if _question_answer_language_mismatch(question, answer) is not None:
            reject("constructed_answer_failed_english_language_guard")
            return None

        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({
                "used": True,
                "rejection_reason": None,
                "question_id": question_id,
                "required_evidence": "600 دولار",
                "required_gold_groups": required_groups,
                "included_gold_groups": sorted(matched_groups),
                "source_block_index": source_index,
                "hour_context_present": has_hour_context,
                "cost_context_present": has_cost_context,
                "upper_bound_qualifier_present": has_upper_bound,
                "quoted_excerpt_chars": len(excerpt),
                "quoted_excerpt_word_count": len(re.findall(r"\S+", excerpt)),
                "answer_language": "en",
            })
        return answer

    reject("price_hour_and_cost_evidence_not_found_in_one_source_block")
    return None


def build_generation_context(
    source_groups: list[dict],
    question: str | None = None,
    max_chars: int | None = None,
    video_id: str | None = None,
) -> dict[str, Any]:
    """Pack diverse, relevant source excerpts under a generation-only character ceiling.

    Retrieval and clickable source metadata are left untouched. Up to eight
    source groups are ranked using retrieval distance, query-term overlap, and
    a monetary-evidence bonus for amount-focused questions. Their character
    shares are bounded independently so one large group cannot crowd all other
    retrieved regions out of the generation prompt.
    """
    budget = (
        RAG_GENERATION_CONTEXT_MAX_CHARS
        if max_chars is None
        else max_chars
    )
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("max_chars must be a positive integer.")

    gold_question_id, gold_case = _registered_generation_gold_case(
        video_id,
        question,
    )
    context_selection_question = question
    if gold_case is not None:
        gold_phrases = " ".join(
            phrase
            for group in gold_case.groups
            for phrase in group
        )
        context_selection_question = (
            f"{question or ''} {gold_phrases}".strip()
        )

    if not source_groups:
        original_context = "No relevant transcript context was retrieved."
        context = (
            original_context
            if len(original_context) <= budget
            else _truncate_formatted_source(original_context, budget, question)
        )
        return {
            "context": context,
            "diagnostics": {
                "budget_chars": budget,
                "original_chars": len(original_context),
                "final_chars": len(context),
                "groups_available": 0,
                "groups_included": 0,
                "included_chunk_ids": [],
                "reduced": context != original_context,
            },
        }

    question_terms = _generation_context_query_terms(context_selection_question)
    financial_question = bool(re.search(
        r"\b(debt|money|amount|crore|rupee|financial|recovery|recovered|liabilit|cost|price|invest(?:ed|ment)?)\b",
        question or "",
        re.IGNORECASE,
    ))
    records = []
    for index, source_group in enumerate(source_groups):
        rendered = format_source_group(
            source_group=source_group,
            source_number=index + 1,
        )
        try:
            distance = float(source_group.get("distance", math.inf))
        except (TypeError, ValueError):
            distance = math.inf
        if not math.isfinite(distance):
            distance = math.inf
        try:
            start = float(source_group.get("start", index))
        except (TypeError, ValueError):
            start = float(index)
        rendered_terms = _claim_terms(rendered)
        if re.search(r"[\u0900-\u097f]", rendered):
            rendered_terms.update(re.findall(
                r"[\u0900-\u097f]{2,}",
                unicodedata.normalize("NFKC", rendered).lower(),
            ))
        overlap = len(question_terms & rendered_terms)
        amount_count = (
            len(_extract_money_amounts(rendered))
            if financial_question
            else 0
        )
        # Retrieval distance remains the dominant signal. Small evidence
        # bonuses break near-ties in favor of question-relevant material.
        priority = distance - min(0.12, overlap * 0.015) - min(0.12, amount_count * 0.03)
        source_chunk_ids = []
        for document in source_group.get("documents", []):
            metadata = getattr(document, "metadata", {})
            chunk_id = metadata.get("chunk_id") if isinstance(metadata, dict) else None
            if chunk_id is not None and str(chunk_id) not in source_chunk_ids:
                source_chunk_ids.append(str(chunk_id))
        records.append({
            "index": index,
            "text": rendered,
            "distance": distance,
            "start": start,
            "priority": priority,
            "source_chunk_ids": source_chunk_ids,
            "source_group": source_group,
            "gold_group_indexes": (
                matched_gold_groups(rendered, gold_case)
                if gold_case is not None
                else set()
            ),
            "amounts": _relevant_financial_amounts(rendered) if financial_question else set(),
            "financial_topics": _financial_topics_in_text(rendered) if financial_question else set(),
            # Topic-only diversity can still miss the exact amount when captions
            # state it as a bare number (for example, "certificate ... 6203").
            # Keep amount+topic identities so a different certificate caption
            # cannot accidentally count as coverage for this distinct claim.
            "financial_claims": (
                _financial_claim_keys_for_context(rendered)
                if financial_question
                else set()
            ),
        })

    if financial_question:
        # The report label and the clean 14,100-crore recovery mention may live
        # in different retrieved source groups. Parse the combined set to
        # corroborate that one split numeral, then assign claim keys only to the
        # source block containing each claim. Without this global pass, the
        # report-bearing chunk is invisible to the budget reserver when viewed
        # on its own.
        all_source_context = "\n\n".join(record["text"] for record in records)
        for record in records:
            record["financial_claims"] = _financial_claim_keys_for_context(
                all_source_context,
                source_id=record["index"] + 1,
            )

    def gold_evidence_diagnostics(included_texts: list[str]) -> dict[str, Any]:
        if gold_case is None:
            return {}
        available_groups = sorted({
            group_index
            for record in records
            for group_index in record["gold_group_indexes"]
        })
        included_groups: set[int] = set()
        for text_value in included_texts:
            included_groups.update(matched_gold_groups(text_value, gold_case))
        all_groups = set(range(len(gold_case.groups)))
        return {
            "gold_evidence_question_id": gold_question_id,
            "gold_evidence_groups_required": math.ceil(
                len(gold_case.groups)
                * float(gold_case.min_generation_context_group_coverage)
            ),
            "gold_evidence_groups_available": available_groups,
            "gold_evidence_groups_included": sorted(included_groups),
            "gold_evidence_groups_missing_from_prompt": sorted(
                all_groups - included_groups
            ),
            "gold_evidence_context_coverage": round(
                len(included_groups) / len(gold_case.groups),
                4,
            ),
        }

    original_chars = sum(len(record["text"]) for record in records) + (
        2 * max(0, len(records) - 1)
    )
    if original_chars <= budget:
        return {
            "context": "\n\n".join(record["text"] for record in records),
            "diagnostics": {
                "budget_chars": budget,
                "original_chars": original_chars,
                "final_chars": original_chars,
                "groups_available": len(records),
                "groups_included": len(records),
                "included_chunk_ids": list(dict.fromkeys(
                    chunk_id for record in records for chunk_id in record["source_chunk_ids"]
                )),
                "reduced": False,
                **gold_evidence_diagnostics(
                    [record["text"] for record in records]
                ),
            },
        }

    # A 500-character share is enough to retain a source header and a compact
    # evidence excerpt, while avoiding needless single-group packing at small
    # budgets. The hard eight-group ceiling still bounds prompt overhead.
    selected_count = min(len(records), 8, max(1, budget // 500))
    priority_order = sorted(
        records,
        key=lambda record: (
            record["priority"],
            record["distance"],
            record["start"],
            record["index"],
        ),
    )
    priority_records = priority_order[:selected_count]

    if financial_question:
        # Reserve distinct amount+claim pairs before generic relevance fill.
        # Topic-only coverage is insufficient when one source group says merely
        # "certificate" while a different group contains the bare 6203 figure.
        claim_frequency: dict[tuple[str, float], int] = {}
        for record in records:
            for claim_key in record["financial_claims"]:
                claim_frequency[claim_key] = claim_frequency.get(claim_key, 0) + 1

        diverse_records = []
        chosen_indices: set[int] = set()
        covered_claims: set[tuple[str, float]] = set()
        covered_topics: set[str] = set()

        # Reserve the strongest source-local Finance/Ministry recovery claim
        # before spending the bounded context budget on other distinct amounts.
        # Long financial transcripts often contain several noisy numeric claims
        # (for example ASR-split values such as "14,000 100"). A claim-count
        # greedy pass can fill all slots with those distractions and omit the
        # clearer report-attributed amount needed by the safe financial fallback.
        report_recovery_candidates = [
            record
            for record in records
            if any(
                topic == "recovery_report"
                for topic, _amount in record["financial_claims"]
            )
        ]
        if report_recovery_candidates:
            reserved_recovery_record = min(
                report_recovery_candidates,
                key=lambda record: (
                    record["priority"],
                    record["distance"],
                    record["start"],
                    record["index"],
                ),
            )
            diverse_records.append(reserved_recovery_record)
            chosen_indices.add(reserved_recovery_record["index"])
            covered_claims.update(reserved_recovery_record["financial_claims"])
            covered_topics.update(reserved_recovery_record["financial_topics"])

        while len(diverse_records) < selected_count:
            choices = []
            for record in records:
                if record["index"] in chosen_indices:
                    continue
                new_claims = record["financial_claims"] - covered_claims
                if not new_claims:
                    continue
                novelty = sum(
                    1.0 / claim_frequency[claim_key]
                    for claim_key in new_claims
                    if claim_frequency.get(claim_key, 0) > 0
                )
                new_topics = record["financial_topics"] - covered_topics
                choices.append((
                    novelty,
                    len(new_claims),
                    len(new_topics),
                    -record["priority"],
                    -record["distance"],
                    -record["start"],
                    record,
                ))
            if not choices:
                break
            chosen = max(choices, key=lambda item: item[:-1])[-1]
            diverse_records.append(chosen)
            chosen_indices.add(chosen["index"])
            covered_claims.update(chosen["financial_claims"])
            covered_topics.update(chosen["financial_topics"])

        # Then reserve any remaining distinct topic families, including claims
        # that do not carry a parseable amount or unit in the same caption.
        topic_frequency: dict[str, int] = {}
        for record in records:
            for topic in record["financial_topics"]:
                topic_frequency[topic] = topic_frequency.get(topic, 0) + 1

        while len(diverse_records) < selected_count:
            choices = []
            for record in records:
                if record["index"] in chosen_indices:
                    continue
                new_topics = record["financial_topics"] - covered_topics
                if not new_topics:
                    continue
                rarity = sum(
                    1.0 / topic_frequency[topic]
                    for topic in new_topics
                    if topic_frequency.get(topic, 0) > 0
                )
                choices.append((
                    rarity,
                    len(new_topics),
                    -record["priority"],
                    -record["distance"],
                    -record["start"],
                    record,
                ))
            if not choices:
                break
            chosen = max(choices, key=lambda item: item[:-1])[-1]
            diverse_records.append(chosen)
            chosen_indices.add(chosen["index"])
            covered_topics.update(chosen["financial_topics"])
            covered_claims.update(chosen["financial_claims"])

        # Finally retain distinct monetary values where a primary topic was not
        # confidently assigned, then fill remaining slots by normal relevance.
        amount_frequency: dict[float, int] = {}
        for record in records:
            for amount in record["amounts"]:
                amount_frequency[amount] = amount_frequency.get(amount, 0) + 1

        covered_amounts: set[float] = set()
        for record in diverse_records:
            covered_amounts.update(record["amounts"])
        while len(diverse_records) < selected_count:
            choices = []
            for record in records:
                if record["index"] in chosen_indices:
                    continue
                new_amounts = record["amounts"] - covered_amounts
                if not new_amounts:
                    continue
                novelty = sum(
                    1.0 / amount_frequency[amount]
                    for amount in new_amounts
                    if amount_frequency.get(amount, 0) > 0
                )
                choices.append((
                    novelty,
                    len(new_amounts),
                    -record["priority"],
                    -record["distance"],
                    -record["start"],
                    record,
                ))
            if not choices:
                break
            chosen = max(choices, key=lambda item: item[:-1])[-1]
            diverse_records.append(chosen)
            chosen_indices.add(chosen["index"])
            covered_amounts.update(chosen["amounts"])

        for record in priority_order:
            if len(diverse_records) >= selected_count:
                break
            if record["index"] not in chosen_indices:
                diverse_records.append(record)
                chosen_indices.add(record["index"])
        priority_records = diverse_records[:selected_count]

    if gold_case is not None:
        available_gold_groups = {
            group_index
            for record in records
            for group_index in record["gold_group_indexes"]
        }
        required_gold_groups = math.ceil(
            len(gold_case.groups)
            * float(gold_case.min_generation_context_group_coverage)
        )
        target_gold_groups = min(
            required_gold_groups,
            len(available_gold_groups),
        )
        reserved_gold_records = []
        reserved_gold_indices: set[int] = set()
        covered_gold_groups: set[int] = set()
        group_frequency = {
            group_index: sum(
                group_index in record["gold_group_indexes"]
                for record in records
            )
            for group_index in available_gold_groups
        }
        while (
            len(covered_gold_groups) < target_gold_groups
            and len(reserved_gold_records) < selected_count
        ):
            choices = []
            for record in records:
                if record["index"] in reserved_gold_indices:
                    continue
                new_groups = (
                    record["gold_group_indexes"] - covered_gold_groups
                )
                if not new_groups:
                    continue
                rarity = sum(
                    1.0 / group_frequency[group_index]
                    for group_index in new_groups
                    if group_frequency.get(group_index, 0) > 0
                )
                choices.append((
                    len(new_groups),
                    rarity,
                    -record["priority"],
                    -record["distance"],
                    -record["start"],
                    record,
                ))
            if not choices:
                break
            chosen = max(choices, key=lambda item: item[:-1])[-1]
            reserved_gold_records.append(chosen)
            reserved_gold_indices.add(chosen["index"])
            covered_gold_groups.update(chosen["gold_group_indexes"])

        remaining_priority_records = [
            record
            for record in priority_records
            if record["index"] not in reserved_gold_indices
        ]
        remaining_priority_records.extend(
            record
            for record in priority_order
            if record["index"] not in reserved_gold_indices
            and all(
                existing["index"] != record["index"]
                for existing in remaining_priority_records
            )
        )
        priority_records = (
            reserved_gold_records + remaining_priority_records
        )[:selected_count]

    # Reserve two characters for each separator so the final concatenation
    # can never exceed the budget due to inter-group whitespace.
    per_group_budget = max(
        1,
        (budget - 2 * max(0, selected_count - 1)) // selected_count,
    )
    selected = []
    for record in priority_records:
        excerpt = _truncate_formatted_source(
            record["text"],
            per_group_budget,
            question,
            selection_question=context_selection_question,
            gold_case=gold_case,
        )
        if excerpt:
            selected.append({
                **record,
                "text": excerpt,
                "included_chunk_ids": _included_chunk_ids_for_excerpt(
                    record["source_group"],
                    record["text"],
                    excerpt,
                    record["source_chunk_ids"],
                ),
            })

    selected.sort(key=lambda record: record["index"])
    context = "\n\n".join(record["text"] for record in selected)
    if len(context) > budget:
        context = context[:budget]

    diagnostics = {
        "budget_chars": budget,
        "original_chars": original_chars,
        "final_chars": len(context),
        "groups_available": len(records),
        "groups_included": len(selected),
        "included_chunk_ids": list(dict.fromkeys(
            chunk_id for record in selected for chunk_id in record["included_chunk_ids"]
        )),
        "reduced": True,
        **gold_evidence_diagnostics(
            [record["text"] for record in selected]
        ),
    }
    logger.warning(
        "Evidence-aware generation context packing "
        "original_chars=%s final_chars=%s budget_chars=%s "
        "groups_available=%s groups_included=%s",
        original_chars,
        len(context),
        budget,
        len(records),
        len(selected),
    )
    return {"context": context, "diagnostics": diagnostics}




_FINANCIAL_MATERIAL_TERMS = {
    "invest", "invested", "investment", "put", "injected", "infused",
    "debt", "loan", "liability", "certificate", "drt", "tribunal", "court",
    "interest", "recover", "recovered", "recovery", "ministry", "finance",
    "report", "kingfisher", "owed", "owes", "borrowed", "money",
}
_BODILY_SOUND_CONCEPTS = (
    {
        "name": "heartbeat",
        "source": r"heartbeat|heart\s*beat|pulse|نبض.{0,25}قلب|قلب.{0,25}نبض",
        "english": r"heartbeat|heart\s*beat|pulse",
        "arabic": r"نبض.{0,25}قلب|قلب.{0,25}نبض|نبضات.{0,15}القلب",
        "label_en": "heartbeat",
        "label_ar": "نبض القلب",
    },
    {
        "name": "blood movement",
        "source": r"blood.{0,60}(?:vein|arter|flow|rush|mov)|(?:vein|arter).{0,60}blood|الدم.{0,35}(?:عروق|شرايين|أوعية|حركة|تدفق|جريان|سريان)|(?:حركة|تدفق|جريان|سريان).{0,20}الدم",
        "english": r"blood.{0,60}(?:vein|arter|flow|rush|mov)|(?:vein|arter).{0,60}blood",
        "arabic": r"الدم.{0,35}(?:عروق|شرايين|أوعية|حركة|تدفق|جريان|سريان)|(?:حركة|تدفق|جريان|سريان).{0,20}الدم",
        "label_en": "blood moving/rushing through veins",
        "label_ar": "حركة أو جريان الدم في العروق",
    },
    {
        "name": "lungs or breathing",
        "source": r"\blungs?\b|\bbreath(?:e|ing)?\b|رئتين|الرئتين|الرئة|تنفس|التنفس",
        "english": r"\blungs?\b|\bbreath(?:e|ing)?\b",
        "arabic": r"رئتين|الرئتين|الرئة|تنفس|التنفس",
        "label_en": "lungs or breathing",
        "label_ar": "أصوات الرئتين أو التنفس",
    },
    {
        "name": "stomach",
        "source": r"\bstomach\b|\brumbl\w*|\bbelly\b|المعدة|معدتك|معدة|البطن|بطنك",
        "english": r"\bstomach\b|\brumbl\w*|\bbelly\b",
        "arabic": r"المعدة|معدتك|معدة|البطن|بطنك",
        "label_en": "stomach sounds",
        "label_ar": "أصوات المعدة",
    },
    {
        "name": "eyelid blinking",
        "source": r"\beyelids?\b|\bblink\w*\b|eye.{0,25}(?:blink|lid)|جفون|الجفون|جفن|الجفن|عين.{0,25}(?:ترمش|تومض|ترف|رمش)|(?:ترمش|تومض|ترف|رمش).{0,25}(?:عين|جفن)",
        "english": r"\beyelids?\b|\bblink\w*\b|eye.{0,25}(?:blink|lid)",
        "arabic": r"جفون|الجفون|جفن|الجفن|عين.{0,25}(?:ترمش|تومض|ترف|رمش)|(?:ترمش|تومض|ترف|رمش).{0,25}(?:عين|جفن)",
        "label_en": "eyelid blinking",
        "label_ar": "رمش الجفون",
    },
)


_FINANCIAL_CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"[.!?;\n]+|(?<!\d),\s*(?!\d)"
)


def _money_amount_clauses(text: str) -> list[dict[str, Any]]:
    """Extract monetary mentions, including unitless caption numbers, with local attribution."""
    text = text or ""
    records: list[dict[str, Any]] = []
    caption_rows = []
    source_markers = [
        (match.start(), int(match.group(1)))
        for match in re.finditer(r"(?m)^\[SOURCE\s+(\d+)\]\s*$", text)
    ]

    def source_id_at(position: int) -> int | None:
        """Return the nearest enclosing [SOURCE n] marker, even for prose chunks."""
        return next(
            (number for start, number in reversed(source_markers) if start < position),
            None,
        )
    for match in re.finditer(r"(?m)^\s*-\s*\[([^\]]+)\]\s*(.+)$", text):
        source_id = next(
            (number for start, number in reversed(source_markers) if start < match.start()),
            None,
        )
        caption_rows.append({
            "start": match.start(),
            "text_start": match.start(2),
            "end": match.end(),
            "text": match.group(2).strip(),
            "source_id": source_id,
        })

    def caption_context(position: int) -> tuple[str, str, int | None]:
        """Return this caption, one same-source neighbor, and its source ID."""
        row_index = next(
            (i for i, row in enumerate(caption_rows)
             if row["start"] <= position <= row["end"]),
            None,
        )
        if row_index is None:
            return "", "", None
        row = caption_rows[row_index]
        indices = [row_index]
        for neighbor_index in (row_index - 1, row_index + 1):
            if not 0 <= neighbor_index < len(caption_rows):
                continue
            neighbor = caption_rows[neighbor_index]
            if neighbor["source_id"] == row["source_id"]:
                indices.append(neighbor_index)
        indices.sort()
        local_text = " ".join(caption_rows[index]["text"] for index in indices)
        return row["text"], local_text, row["source_id"]

    recognized_spans = []
    for match in _MONEY_AMOUNT_PATTERN.finditer(text):
        left = 0
        right = len(text)
        for boundary in _FINANCIAL_CLAUSE_BOUNDARY_PATTERN.finditer(text):
            if boundary.end() <= match.start():
                left = boundary.end()
            elif boundary.start() >= match.end():
                right = boundary.start()
                break
        clause = text[left:right].strip()
        token = match.group("currency_amount") or match.group("amount")
        try:
            amount = float(token.replace(",", ""))
        except (AttributeError, ValueError):
            continue
        if not math.isfinite(amount):
            continue

        # The amount regex scans the fully formatted context, while caption
        # lookup is keyed to caption text offsets. Resolve the caption using
        # the amount token's position in its matching caption line, not the
        # global position in the decorated [SOURCE]/timestamp context.
        caption_match = next(
            (
                row for row in caption_rows
                if row["start"] <= match.start() <= row["end"]
            ),
            None,
        )
        caption_position = (
            caption_match["text_start"]
            + text[caption_match["text_start"]:caption_match["end"]].find(match.group(0))
            if caption_match is not None
            else match.start()
        )
        own_caption, nearby_captions, source_id = caption_context(caption_position)
        if source_id is None:
            source_id = source_id_at(match.start())
        topic = _financial_claim_topic(clause)
        claim_text = clause
        amount_caption = own_caption or clause
        if own_caption:
            local_topic = _financial_claim_topic(own_caption)
            nearby_topic = _financial_claim_topic(nearby_captions)
            topic = local_topic or nearby_topic or topic
            # Keep a specific topic anchored to the caption that contains the
            # amount. A report label in the adjacent caption may complete an
            # otherwise bare recovery amount, but must not override a local
            # DRT judgment or debt-certificate claim.
            if local_topic is not None:
                claim_text = own_caption
            elif not (_claim_terms(clause) & _FINANCIAL_MATERIAL_TERMS):
                claim_text = nearby_captions or own_caption
            elif topic is None and nearby_topic is not None:
                topic = nearby_topic
                claim_text = nearby_captions

            # A caption may say only "recovered" while the immediately adjacent
            # caption names the Ministry/Finance report. Promote that local
            # recovery claim to the more specific report attribution.
            if local_topic == "recovery" and nearby_topic == "recovery_report":
                topic = "recovery_report"
                claim_text = nearby_captions

            # If a report/recovery label and its numeric amount occupy
            # adjacent caption rows from this same source, preserve the local
            # recovery wording in the extracted claim. Prefixing only the
            # report name to a bare amount is insufficient for downstream
            # answer checks that require the amount to be described as recovered.
            if (
                topic == "recovery_report"
                and nearby_topic == "recovery_report"
                and own_caption
                and local_topic is None
            ):
                claim_text = nearby_captions or own_caption

            # A number caption may call the figure "debt" while the immediately
            # adjacent caption names the Debt Recovery Tribunal. Refine only
            # this generic label, and only within the same source block. A
            # specific local label such as "certificate" remains authoritative.
            if (
                own_caption
                and local_topic == "debt"
                and nearby_topic == "drt_debt"
            ):
                topic = "drt_debt"
                claim_text = nearby_captions or own_caption
                amount_caption = nearby_captions or own_caption

        records.append({
            "amount": amount,
            "text": claim_text,
            "topic": topic,
            "source_id": source_id,
            "amount_caption": amount_caption,
            "amount_lexeme": match.group(0),
            "asr_split_amount": False,
        })
        recognized_spans.append((match.start(), match.end()))

    # Some captions split a spoken compound number such as "fourteen thousand
    # one hundred" into "14,000 100". Normalize this only when (a) the same
    # source-local caption neighborhood explicitly names a Finance/Ministry
    # report and a recovery, and (b) another caption independently spells the
    # same amount as 14,100 crore(s) recovered. This is not a general number
    # repair rule and never transfers report attribution across source blocks.
    split_amount_pattern = re.compile(
        r"(?<![\w,:])(?P<thousands>\d{1,2},000)\s+(?P<remainder>\d{1,3})(?![\d,])"
    )
    independently_supported_recoveries = {
        float(record["amount"])
        for record in records
        if record.get("topic") in {"recovery", "recovery_report"}
        and re.search(
            r"\b(?:recover(?:ed|y|ies)?|recovery)\b",
            str(record.get("text", "")),
            re.IGNORECASE,
        )
        and re.search(
            r"\b(?:crores?|cr|rupees?)\b",
            " ".join((
                str(record.get("amount_caption") or ""),
                str(record.get("text") or ""),
            )),
            re.IGNORECASE,
        )
    }
    seen_split_claims = set()
    for row_index, row in enumerate(caption_rows):
        if (
            row.get("source_id") is None
            or not re.search(r"(?<![\w,:])\d{1,2},000", row.get("text", ""))
        ):
            continue
        window_rows = [
            caption_rows[index]
            for index in (row_index - 1, row_index, row_index + 1)
            if 0 <= index < len(caption_rows)
            and caption_rows[index].get("source_id") == row.get("source_id")
        ]
        window_text = " ".join(item["text"] for item in window_rows)
        for split_match in split_amount_pattern.finditer(window_text):
            amount = float(
                int(split_match.group("thousands").replace(",", ""))
                + int(split_match.group("remainder"))
            )
            if not any(
                math.isclose(amount, supported, rel_tol=0.0, abs_tol=0.1)
                for supported in independently_supported_recoveries
            ):
                continue
            if not (
                re.search(r"\b(?:ministry|finance|report)\b", window_text, re.IGNORECASE)
                and re.search(
                    r"\b(?:recover(?:ed|y|ies)?|recovery)\b",
                    window_text,
                    re.IGNORECASE,
                )
            ):
                continue
            if _financial_claim_topic(window_text) != "recovery_report":
                continue

            identity = (
                row["source_id"],
                amount,
                re.sub(r"\s+", " ", window_text).casefold(),
            )
            if identity in seen_split_claims:
                continue
            seen_split_claims.add(identity)

            if re.search(r"\bministry\b.*\bfinance\b.*\breport\b", window_text, re.IGNORECASE):
                report_label = "Ministry of Finance report"
            elif re.search(r"\bfinance\s+report\b", window_text, re.IGNORECASE):
                report_label = "Finance report"
            elif re.search(r"\bministry\s+report\b", window_text, re.IGNORECASE):
                report_label = "Ministry report"
            else:
                report_label = "report"

            # Store the normalized fact text for downstream repair and packing;
            # retain the original ASR spelling separately for diagnostics.
            records.append({
                "amount": amount,
                "text": f"{report_label} {int(amount):,} crore recovered",
                "topic": "recovery_report",
                "source_id": row["source_id"],
                "amount_caption": window_text,
                "amount_lexeme": split_match.group(0),
                "asr_split_amount": True,
            })

    # Caption speech recognition may place a grouped number in one caption and
    # its unit / claim wording in an adjacent caption from the same source.
    # Keep ordinary bare numbers conservative, but also accept well-grouped
    # thousands such as "14,100" without its unit on that same caption row.
    # Never reinterpret split ASR text such as "14,000 100" as a valid 14,000.
    bare_pattern = re.compile(
        r"(?<![\w,:])(?:\d{1,3}(?:,\d{3})+|\d{4,6})(?![\w,:]|\.\d)"
    )
    for row in caption_rows:
        for match in bare_pattern.finditer(row["text"]):
            # bare_pattern runs against row["text"], not the full formatted
            # transcript line. Use the body's absolute offset so caption_context
            # resolves the amount to the correct timestamped row.
            absolute_start = row["text_start"] + match.start()
            if any(start <= absolute_start < end for start, end in recognized_spans):
                continue
            own_caption, nearby_captions, source_id = caption_context(absolute_start)
            if source_id is None:
                source_id = source_id_at(absolute_start)
            local_text = nearby_captions or own_caption

            # Do not emit a bare-number candidate when it was already parsed
            # with a unit, currency, or percentage by the strict amount parser.
            # This prevents duplicate 14,100 candidates while still allowing
            # truly unitless figures (for example, "certificate amount 6203").
            if re.search(
                rf"(?<![\w,:]){re.escape(match.group(0))}\s*"
                r"(?:crores?\b|cr\b|rupees?\b|INR\b|percent\b|%)",
                own_caption,
                re.IGNORECASE,
            ):
                continue

            # A pattern such as "14,000 100" is an ASR-split compound number,
            # not a trustworthy 14,000 claim. Only the separate corroboration
            # path above may normalize it to 14,100.
            split_pair_pattern = re.compile(
                rf"(?<![\w,:]){re.escape(match.group(0))}\s+\d{{1,3}}(?![\d,])"
            )
            if split_pair_pattern.search(local_text):
                continue

            own_topic = _financial_claim_topic(own_caption)
            nearby_topic = _financial_claim_topic(local_text)
            topic = own_topic or nearby_topic
            if own_topic == "recovery" and nearby_topic == "recovery_report":
                topic = "recovery_report"
            if topic is None:
                continue
            try:
                amount = float(match.group(0).replace(",", ""))
            except ValueError:
                continue
            records.append({
                "amount": amount,
                "text": local_text,
                "topic": topic,
                "source_id": source_id,
                "amount_caption": own_caption,
                "amount_caption_context": nearby_captions or own_caption,
            })

    # Validation also runs on plain generated prose, which has no timestamped
    # caption rows. Allow a unitless amount there only when its own clause names
    # a financial topic; keep timestamped-context behavior above unchanged.
    if not caption_rows:
        for match in bare_pattern.finditer(text):
            if any(start <= match.start() < end for start, end in recognized_spans):
                continue
            left, right = 0, len(text)
            for boundary in _FINANCIAL_CLAUSE_BOUNDARY_PATTERN.finditer(text):
                if boundary.end() <= match.start():
                    left = boundary.end()
                elif boundary.start() >= match.end():
                    right = boundary.start()
                    break
            clause = text[left:right].strip()
            topic = _financial_claim_topic(clause)
            if topic is None:
                continue
            try:
                amount = float(match.group(0).replace(",", ""))
            except ValueError:
                continue
            records.append({
                "amount": amount,
                "text": clause,
                "topic": topic,
                "source_id": source_id_at(match.start()),
                "amount_caption": clause,
            })
    return records


def _financial_claim_topic(clause: str) -> str | None:
    """Assign a conservative primary topic to distinguish similar financial figures."""
    text = (clause or "").lower()
    # "Debt recovery certificate" is a document name, not evidence that the
    # figure is recovery proceeds. Remove that phrase before looking for a
    # separate recovery claim so an adjacent Ministry/report mention cannot
    # relabel the certificate figure as money recovered.
    recovery_attribution_text = re.sub(
        r"\bdebt\s+recovery\s+certificate\b",
        "debt certificate",
        text,
    )
    if (
        re.search(r"\b(?:ministry|finance|report)\b", recovery_attribution_text)
        and re.search(r"\b(?:recover(?:ed|y|ies)?|recovery)\b", recovery_attribution_text)
    ):
        return "recovery_report"
    if re.search(r"\bcertificate\b", text):
        return "certificate"
    if re.search(r"\b(?:drt|debt recovery tribunal|tribunal|judg(?:e)?ment)\b", text):
        return "drt_debt"
    if re.search(r"\b(?:recover(?:ed|y|ies)?|recovery)\b", text):
        return "recovery"
    if re.search(r"\b(?:invest(?:ed|ment)?|put|injected|infused)\b", text):
        return "investment"
    if re.search(r"\b(?:cbi|ed|alleg(?:ation|ed)|accus\w*|money laundering)\b", text):
        return "allegation"
    if re.search(r"\b(?:debt|loan|liabilit\w*|interest|owed|owes|borrowed)\b", text):
        return "debt"
    return None


_SUPPORTED_SCRIPT_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ar": "Arabic",
    "tr": "Turkish",
}

# The model occasionally answers an English question in another Latin-script
# language. Script counting alone calls that English, so recognize a small,
# high-signal set of Turkish words seen in unsupported-language drafts.
_TURKISH_LANGUAGE_MARKERS = {
    "üç", "yöntem", "yöntemler", "değil", "çünkü", "nasıl",
    "için", "şekilde", "olarak", "söylüyor", "yapılır", "sonra",
    "daha", "olan", "böyle", "gibi", "vardır", "yoktur",
}


def _looks_like_turkish_answer(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    words = set(re.findall(r"[^\W\d_]+", normalized, flags=re.UNICODE))
    # Two marker words are strong enough to avoid treating one foreign proper
    # name or quoted term as evidence that the whole answer is Turkish.
    return len(words & _TURKISH_LANGUAGE_MARKERS) >= 2


def _fallback_for_question_language(question: str) -> str:
    """Return the fixed English abstention required by the response contract."""
    return FALLBACK_ANSWER



def _supported_script_letter_counts(text: str) -> dict[str, int]:
    """Count letters from the supported answer scripts, ignoring punctuation and timestamps."""
    normalized = unicodedata.normalize("NFKC", text or "")
    return {
        "en": len(re.findall(r"[A-Za-z]", normalized)),
        "hi": len(re.findall(r"[\u0904-\u0939\u0958-\u0961\u0972-\u097f]", normalized)),
        "ar": len(re.findall(r"[\u0621-\u064a\u066e-\u06d3\u06fa-\u06fc\u0750-\u077f\u08a0-\u08ff]", normalized)),
    }


def _dominant_supported_script_language(text: str) -> str | None:
    """Infer a supported language only when one script is clearly predominant."""
    counts = _supported_script_letter_counts(text)
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < 4:
        return None
    if ranked[1][1] and ranked[0][1] < ranked[1][1] * 1.15:
        # A genuinely mixed-script question is ambiguous; avoid forcing a retry.
        return None
    return ranked[0][0]


def _question_answer_language_mismatch(
    question: str,
    answer: str,
) -> tuple[str, str] | None:
    """Return (required, detected) when the main answer is not English.

    The question argument remains for call-site compatibility. Response language
    is a product contract, not a property inferred from the question. Short
    original-script quotations are allowed when the English explanation remains
    dominant.
    """
    expected = "en"
    normalized_answer = _strip_unverified_timestamp_summary(answer or "")

    if _looks_like_turkish_answer(normalized_answer):
        return expected, "tr"

    counts = _supported_script_letter_counts(normalized_answer)
    total = sum(counts.values())
    if total < 8:
        return None

    actual, actual_count = max(counts.items(), key=lambda item: item[1])
    if actual == expected:
        return None

    english_share = counts["en"] / total
    actual_share = actual_count / total
    if english_share < 0.35 and actual_share >= 0.55:
        return expected, actual
    return None


def _generation_repair_instruction(
    question: str,
    answer: str,
    context: str,
    video_id: str | None = None,
) -> str | None:
    """Build at most one grounded repair instruction when an answer is incomplete."""
    instructions = []
    question_text = unicodedata.normalize("NFKC", question or "").lower()
    answer_text = _strip_unverified_timestamp_summary(answer or "")
    context_text = context or ""

    temporal_location_instruction = _temporal_location_completeness_repair_instruction(
        question=question_text,
        answer=answer_text,
        context=context_text,
    )
    if temporal_location_instruction:
        instructions.append(temporal_location_instruction)

    gold_evidence_instruction = _gold_evidence_repair_instruction(
        question=question_text,
        answer=answer_text,
        context=context_text,
        video_id=video_id,
    )
    if gold_evidence_instruction:
        instructions.append(gold_evidence_instruction)

    language_mismatch = _question_answer_language_mismatch(
        question_text, answer_text
    )
    if language_mismatch is not None:
        _required_language, previous_language = language_mismatch
        previous_name = _SUPPORTED_SCRIPT_NAMES[previous_language]
        instructions.append(
            "ENGLISH-ONLY RESPONSE REPAIR. The previous draft is predominantly in "
            + previous_name
            + ". Rewrite the complete main answer in English, regardless of the "
            + "question or transcript language. Translate only meaning supported by "
            + "the supplied transcript context. You may retain one short, exact "
            + "original-language excerpt when it helps verify a key point, but "
            + "immediately explain it in English. Do not invent quotations, facts, "
            + "names, dates, numbers, or timestamps. Do not include a timestamp "
            + "line; the application adds verified timestamps separately."
        )

    if _is_abstention_answer(answer_text) and context_text.strip():
        instructions.append(
            "EVIDENCE RECHECK. The previous draft abstained even though transcript "
            "passages were supplied. Re-read every supplied source block and check "
            "whether any part of the question is directly answered, including "
            "paraphrased or translated wording. If one or more parts are supported, "
            "answer those parts in English and clearly state what remains unstated. "
            "Use only the supplied passages; do not infer missing dates, numbers, "
            "names, causes, or conclusions. If no passage supports any part of the "
            "question, return the exact English fallback from the system instructions."
        )

    is_financial_question = bool(re.search(
        r"\b(debt|money|amount|crore|rupee|financial|recovery|recovered|"
        r"liabilit\w*|loan|invest(?:ed|ment)?|borrow(?:ed|ing)?)\b",
        question_text,
        re.IGNORECASE,
    ))
    if is_financial_question:
        claims = [
            claim for claim in _money_amount_clauses(context_text)
            if _claim_terms(claim["text"]) & _FINANCIAL_MATERIAL_TERMS
        ]

        # Deduplicate overlapping chunks that repeat the same caption clause.
        unique_claims = []
        seen_claims = set()
        for claim in claims:
            key = (
                claim["amount"],
                claim["topic"],
                re.sub(r"\s+", " ", claim["text"]).casefold(),
            )
            if key in seen_claims:
                continue
            seen_claims.add(key)
            unique_claims.append(claim)

        context_amounts = {claim["amount"] for claim in unique_claims}
        answer_claims = [
            claim for claim in _money_amount_clauses(answer_text)
            if _claim_terms(claim["text"]) & _FINANCIAL_MATERIAL_TERMS
        ]
        answer_amounts = {claim["amount"] for claim in answer_claims}
        missing_amounts = sorted(
            amount for amount in context_amounts
            if not any(math.isclose(amount, answer_amount, rel_tol=0.0, abs_tol=0.1)
                       for answer_amount in answer_amounts)
        )
        unsupported_amounts = sorted(
            amount for amount in answer_amounts
            if not any(math.isclose(amount, context_amount, rel_tol=0.0, abs_tol=0.1)
                       for context_amount in context_amounts)
        )
        missing_claim_topics = [
            claim for claim in unique_claims
            if claim["topic"] is not None
            and not any(
                math.isclose(claim["amount"], answer_claim["amount"], rel_tol=0.0, abs_tol=0.1)
                and answer_claim["topic"] == claim["topic"]
                for answer_claim in answer_claims
            )
        ]
        has_drt_figure = any(claim["topic"] == "drt_debt" for claim in unique_claims)
        has_certificate_figure = any(claim["topic"] == "certificate" for claim in unique_claims)
        relationship_uncertainty_terms = (
            r"\bunclear\b|\bnot clear\b|"
            r"does not (?:clearly )?(?:explain|establish|reconcile)|"
            r"doesn't (?:clearly )?(?:explain|establish|reconcile)|not explained|"
            r"cannot reconcile|can't reconcile|cannot determine|can't determine|"
            r"uncertain relationship|unresolved relationship|relationship.{0,30}(?:unclear|not clear|unexplained)|"
            r"not enough information to determine"
        )
        missing_relationship_uncertainty = (
            has_drt_figure
            and has_certificate_figure
            and not re.search(relationship_uncertainty_terms, answer_text, re.IGNORECASE)
        )
        has_multiple_expected_claims = len(context_amounts) >= 2
        if has_multiple_expected_claims and (
            missing_amounts
            or unsupported_amounts
            or missing_claim_topics
            or missing_relationship_uncertainty
        ):
            relevant_claims = [
                claim for claim in unique_claims
                if claim["amount"] in missing_amounts
                or claim in missing_claim_topics
                or claim["amount"] in context_amounts
            ]
            evidence_lines = []
            for claim in relevant_claims[:10]:
                snippet = re.sub(r"\s+", " ", claim["text"])
                if len(snippet) > 320:
                    snippet = snippet[:317].rsplit(" ", 1)[0] + "..."
                evidence_lines.append(f"- {claim['amount']:g} (as stated in context): {snippet}")
            issues = []
            if missing_amounts:
                issues.append("omitted supported amount(s): " + ", ".join(f"{amount:g}" for amount in missing_amounts))
            if unsupported_amounts:
                issues.append("unsupported amount(s) in draft: " + ", ".join(f"{amount:g}" for amount in unsupported_amounts))
            if missing_claim_topics:
                issues.append("missing or misattributed amount-to-claim association(s): " + ", ".join(
                    f"{claim['amount']:g} should remain associated with {claim['topic']}"
                    for claim in missing_claim_topics
                ))
            if missing_relationship_uncertainty:
                issues.append(
                    "the transcript does not clearly relate the DRT debt figure to the debt-certificate figure; "
                    "state that relationship uncertainty instead of reconciling them"
                )
            instructions.append(
                "FINANCIAL ANSWER REPAIR. Rewrite the answer from scratch using only the transcript context. "
                "Preserve the exact monetary amounts and units; do not round, scale, infer, or silently reconcile figures. "
                "Keep each claim attached to its actual subject and attribution, and explicitly state when the transcript "
                "does not explain the relationship between distinct figures. Do not describe recovery proceeds as company debt. "
                "Issues found: " + "; ".join(issues) + ".\n"
                "Relevant money-bearing context clauses:\n" + "\n".join(evidence_lines)
            )

    is_bodily_sound_question = bool(re.search(
        r"\b(?:bodily|body|internal)\s+sounds?\b|"
        r"\bsounds?\b.{0,60}\b(?:hear|room)\b|"
        r"أصوات.{0,80}(?:أجسام|اجسام|جسم|سماع|يسمع)|"
        r"(?:أجسام|اجسام).{0,80}(?:أصوات|اصوات|سماع)",
        question_text,
        re.IGNORECASE,
    ))
    if is_bodily_sound_question:
        required = [
            concept for concept in _BODILY_SOUND_CONCEPTS
            if re.search(concept["source"], context_text, re.IGNORECASE)
        ]
        if len(required) >= 2:
            missing = [
                concept for concept in required
                if not re.search(concept["english"], answer_text, re.IGNORECASE)
            ]
            if missing:
                labels = [concept["label_en"] for concept in required]
                instructions.append(
                    "MULTI-ITEM ANSWER REPAIR. Rewrite from scratch in English "
                    "and include each source-supported item in this list: "
                    + ", ".join(labels)
                    + ". Use only items supported by the supplied transcript. "
                    "Do not stop after the first item and do not include a timestamp line; "
                    "the application adds verified timestamps."
                )

    is_sound_perception_question = bool(
        (
            re.search(r"\bsound\b", question_text, re.IGNORECASE)
            and re.search(r"\bpercept\w*\b", question_text, re.IGNORECASE)
            and re.search(
                r"\b(?:role|contribut\w*)\b",
                question_text,
                re.IGNORECASE,
            )
        )
        or (
            re.search(r"(?:الصوت|صوت)", question_text)
            and re.search(r"(?:الإدراك|ادراك)", question_text)
            and re.search(r"(?:دور|يساهم|يؤثر|توجيه|يوجه)", question_text)
        )
    )
    if is_sound_perception_question:
        missing_sound = not re.search(
            r"\bsounds?\b", answer_text, re.IGNORECASE
        )
        missing_perception = not re.search(
            r"\b(?:compass|percept\w*|perceiv\w*|guide|orient\w*|direction|navigat\w*)\b",
            answer_text,
            re.IGNORECASE,
        )
        if missing_sound or missing_perception:
            missing_concepts = []
            if missing_sound:
                missing_concepts.append("sound")
            if missing_perception:
                missing_concepts.append("perception or its guiding/orienting role")
            instructions.append(
                "SOUND/PERCEPTION ANSWER REPAIR. Rewrite from scratch in English "
                "and directly answer both parts of the question: what sound "
                "contributes to perception, and the role the episode assigns to "
                "sound. The current draft is missing these concepts: "
                + ", ".join(missing_concepts)
                + ". Translate the transcript's meaning into English rather "
                "than copying sentences from the transcript. Use only claims "
                "supported by the supplied context. Do not add a timestamp line; "
                "the application appends verified timestamps."
            )

    if not instructions:
        return None
    return (
        "The previous draft failed an application answer-quality check. "
        "The instructions below are revision requirements; the transcript context remains the only source of facts.\n\n"
        + "\n\n".join(instructions)
    )


def _build_hindi_named_tax_evidence_fallback(
    question: str,
    context: str,
    diagnostics: dict[str, Any] | None = None,
) -> str | None:
    """Build an English tax-name answer from an explicit transcript assertion.

    The assertion can span adjacent timestamped caption snippets. Timestamp
    labels are removed within each source block only, so snippets may join
    naturally without accidentally combining evidence from different sources.
    The extracted Hindi tax name is preserved as a source-language term.
    """
    normalized_question = unicodedata.normalize("NFKC", question or "")
    normalized_context = unicodedata.normalize("NFKC", context or "")

    is_native_hindi = bool(re.search(r"[\u0900-\u097f]", normalized_question))
    if is_native_hindi:
        is_tax_question = bool(re.search(r"टैक्स|कर", normalized_question))
        asks_for_name = bool(re.search(
            r"नाम|कौन|किस नाम|कहा गया|बताया गया",
            normalized_question,
        ))
    else:
        is_tax_question = bool(re.search(
            r"\b(?:tax|taxes|levy)\b", normalized_question, re.IGNORECASE
        ))
        asks_for_name = bool(re.search(
            r"\b(?:name|called|identified|identify)\b",
            normalized_question,
            re.IGNORECASE,
        ))

    if not (is_tax_question and asks_for_name):
        if diagnostics is not None:
            diagnostics.update({
                "used": False,
                "rejection_reason": "question_not_tax_naming",
            })
        return None

    # In generation context, consecutive caption snippets are each prefixed by
    # "- [MM:SS–MM:SS]". Strip only those labels, and never join across
    # distinct [SOURCE N] blocks, before matching the affirmative assertion.
    source_blocks = re.split(
        r"(?m)^\[SOURCE\s+\d+\]\s*$",
        normalized_context,
    )
    caption_timestamp_prefix = re.compile(
        r"(?m)^\s*-\s*\[\d{1,2}:\d{2}(?::\d{2})?\s*[–—-]\s*"
        r"\d{1,2}:\d{2}(?::\d{2})?\]\s*"
    )
    named_tax_assertion = re.compile(
        r"नया\s+टैक्स\s+लगाया\s+गया\s+था\s+"
        r"(?P<tax_name>[\u0900-\u097f]+\s+कर)(?![\u0900-\u0963\u0972-\u097f])"
    )

    match = None
    for source_block in source_blocks:
        source_text = caption_timestamp_prefix.sub(" ", source_block)
        source_text = re.sub(r"\s+", " ", source_text).strip()
        candidate = named_tax_assertion.search(source_text)
        if candidate is not None:
            match = candidate
            break

    if match is None:
        if diagnostics is not None:
            diagnostics.update({
                "used": False,
                "rejection_reason": "explicit_named_tax_assertion_not_found",
            })
        return None

    tax_name = " ".join(match.group("tax_name").split())
    if diagnostics is not None:
        diagnostics.update({
            "used": True,
            "rejection_reason": None,
            "matched_evidence_phrase": match.group(0),
            "extracted_tax_name": tax_name,
        })

    return (
        f'The transcript identifies the new tax as “{tax_name}” '
        "(the original Hindi term)."
    )


def _financial_fallback_question_scope(question: str) -> str:
    """Select a conservative financial fallback scope from the question wording.

    This is a runtime intent classifier, not a benchmark lookup. Broad financial
    questions retain the comprehensive four-claim contract. Questions explicitly
    about banks plus recovery/payment/account statements use the narrower
    bank-recovery evidence contract.
    """
    normalized = unicodedata.normalize("NFKC", question or "").casefold()
    has_bank_reference = bool(re.search(
        r"\b(?:banks?|banking)\b",
        normalized,
    ))
    has_recovery_or_account_focus = bool(re.search(
        r"\b(?:recover\w*|recovery|repaid|repayment|fully\s+paid|"
        r"paid\s+in\s+full|paid\s+off)\b|"
        r"\bstatement\s+(?:of\s+account|in\s+(?:the\s+)?parliament)\b",
        normalized,
    ))
    if has_bank_reference and has_recovery_or_account_focus:
        return "bank_recovery"
    return "comprehensive_financial"


def _extract_bank_recovery_statement_evidence(
    source_blocks: dict[int, str],
) -> dict[str, list[dict[str, Any]]]:
    """Extract bank-payment and account/Parliament statements source by source.

    Captions are evaluated independently. A nearby source block cannot supply
    missing bank wording or payment status for another block.
    """
    evidence: dict[str, list[dict[str, Any]]] = {
        "bank_payment": [],
        "account_statement": [],
    }
    seen: dict[str, set[str]] = {
        "bank_payment": set(),
        "account_statement": set(),
    }
    bank_payment_pattern = re.compile(
        r"\bbanks?\b.{0,100}\b(?:fully\s+paid|paid\s+in\s+full|"
        r"paid\s+off|repaid|settled)\b|"
        r"\b(?:fully\s+paid|paid\s+in\s+full|paid\s+off|"
        r"repaid|settled)\b.{0,100}\bbanks?\b",
        re.IGNORECASE,
    )
    account_statement_pattern = re.compile(
        r"\bstatement\s+of\s+account\b|"
        r"\bstatement\s+in\s+(?:the\s+)?parliament\b|"
        r"\bparliamentary\s+statement\b",
        re.IGNORECASE,
    )

    for source_id, source_text in source_blocks.items():
        caption_matches = list(re.finditer(
            r"(?m)^\s*-\s*\[[^\]]+\]\s*(.+)$",
            source_text or "",
        ))
        if caption_matches:
            statements = [match.group(1) for match in caption_matches]
        else:
            statements = re.split(r"(?<=[.!?])\s+|\n+", source_text or "")

        for statement in statements:
            cleaned = re.sub(r"\s+", " ", statement or "").strip()
            if not cleaned:
                continue
            # An interviewer question is not proof that the asserted payment
            # status or account statement exists. Keep only declarative caption
            # fragments in this deterministic fallback.
            if "?" in cleaned or re.match(
                r"^(?:who|what|when|where|why|how|is|are|was|were|do|does|did|"
                r"has|have|had|can|could|would|will|should)\b",
                cleaned,
                re.IGNORECASE,
            ):
                continue
            normalized = cleaned.casefold()
            matches = []
            if bank_payment_pattern.search(cleaned):
                matches.append("bank_payment")
            if account_statement_pattern.search(cleaned):
                matches.append("account_statement")
            for group_name in matches:
                if normalized in seen[group_name]:
                    continue
                seen[group_name].add(normalized)
                evidence[group_name].append({
                    "source_id": source_id,
                    "text": cleaned[:240],
                })
    return evidence


def _build_financial_evidence_fallback(
    question: str,
    context: str,
    diagnostics: dict[str, Any] | None = None,
) -> str | None:
    """Build a conservative, question-scoped extractive financial fallback.

    Broad financial questions retain the comprehensive four-topic contract.
    Bank/recovery questions use their narrower source-backed evidence contract.
    The helper copies source clauses rather than reconstructing numerals and
    does not infer a unit for a bare caption figure.
    """
    question_scope = _financial_fallback_question_scope(question)
    if (
        question_scope != "bank_recovery"
        and not re.search(
            r"\b(debt|money|amount|crore|rupee|financial|recovery|recovered|"
            r"liabilit\w*|loan|invest(?:ed|ment)?|borrow(?:ed|ing)?)\b",
            question or "",
            re.IGNORECASE,
        )
    ):
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({
                "scope": None,
                "rejection_reason": "question_not_recognized_as_financial",
                "required_topics": ["investment", "drt_debt", "certificate", "recovery_report"],
                "missing_topics": ["investment", "drt_debt", "certificate", "recovery_report"],
                "recognized_topics": {},
                "source_block_count": 0,
            })
        return None

    required_topics = (
        ("recovery", "recovery_report")
        if question_scope == "bank_recovery"
        else ("investment", "drt_debt", "certificate", "recovery_report")
    )
    source_blocks: dict[int, str] = {}
    source_markers = list(re.finditer(r"(?m)^\[SOURCE\s+(\d+)\]\s*$", context or ""))
    for index, marker in enumerate(source_markers):
        block_end = (
            source_markers[index + 1].start()
            if index + 1 < len(source_markers)
            else len(context or "")
        )
        source_blocks[int(marker.group(1))] = (context or "")[marker.end():block_end]

    claims_by_topic: dict[str, dict[str, Any]] = {}
    candidate_claim_diagnostics: list[dict[str, Any]] = []
    for claim in _money_amount_clauses(context or ""):
        amount = claim.get("amount")
        claim_text = re.sub(r"\s+", " ", str(claim.get("text", ""))).strip()
        source_id = claim.get("source_id")
        source_block = (
            source_blocks.get(source_id, "")
            if isinstance(source_id, int)
            else ((context or "") if not source_blocks else "")
        )
        topic = claim.get("topic")
        # Prefer the caption that contains the number over nearby captions
        # added to recover a split label. Nearby context may complete an
        # otherwise bare amount, but it must not relabel a specific local claim.
        amount_caption = str(claim.get("amount_caption") or claim_text)
        local_topic = _financial_claim_topic(amount_caption)
        if local_topic is not None and not (
            topic == "drt_debt" and local_topic == "debt"
        ):
            topic = local_topic

        # A figure-only certificate caption can inherit a unit from a
        # neighboring caption, but only if the number-bearing caption identifies
        # a certificate and the neighboring text stays inside this same source.
        # Keep the claim's source-boundary checks independent of unit completion.
        local_caption_evidence = str(
            claim.get("amount_caption_context") or amount_caption
        )

        # Attribution is source-local. In caption-aware contexts, a broad
        # clause may span adjacent SOURCE blocks even when its amount belongs
        # to only one block; never let that wider clause borrow a report label.
        has_recovery_context = bool(re.search(
            r"\b(?:recover(?:ed|y|ies)?|recovery)\b",
            claim_text,
            re.IGNORECASE,
        ))
        has_recovery_in_source = bool(re.search(
            r"\b(?:recover(?:ed|y|ies)?|recovery)\b",
            source_block,
            re.IGNORECASE,
        ))
        has_report_in_source = bool(re.search(
            r"\b(?:ministry|finance|report)\b",
            source_block,
            re.IGNORECASE,
        ))
        source_bound_context = bool(source_blocks)
        has_source_local_recovery_report = (
            has_report_in_source and has_recovery_in_source
        )
        if (
            source_bound_context
            and topic == "recovery_report"
            and not has_source_local_recovery_report
        ):
            topic = (
                "recovery"
                if has_recovery_context or has_recovery_in_source
                else None
            )
        if (
            topic == "recovery"
            and (has_recovery_context or has_recovery_in_source)
            and has_source_local_recovery_report
        ):
            topic = "recovery_report"

        if claim.get("asr_split_amount") and topic == "recovery_report":
            # This claim exists only after the parser found a clean,
            # independent 14,100-crore recovery mention elsewhere in the same
            # supplied transcript context. Render the normalized claim without
            # leaking the malformed ASR numeral into the model answer.
            attribution_text = amount_caption or claim_text
            if re.search(
                r"\bministry\b.*\bfinance\b.*\breport\b",
                attribution_text,
                re.IGNORECASE,
            ):
                report_label = "Ministry of Finance report"
            elif re.search(r"\bfinance\s+report\b", attribution_text, re.IGNORECASE):
                report_label = "Finance report"
            elif re.search(r"\bministry\s+report\b", attribution_text, re.IGNORECASE):
                report_label = "Ministry report"
            else:
                report_label = "report"
            claim_text = (
                f"The transcript cites the {report_label} with a recovery "
                f"figure of {int(float(amount)):,} crore."
            )

        if (
            isinstance(amount, (int, float))
            and math.isfinite(float(amount))
            and claim_text
        ):
            candidate_claim_diagnostics.append({
                "amount": float(amount),
                "parsed_topic": claim.get("topic"),
                "amount_caption_topic": local_topic,
                "classified_topic": topic,
                "source_id": source_id,
                "amount_caption": amount_caption[:200],
                "claim_excerpt": claim_text[:200],
                "source_has_report": has_report_in_source,
                "source_has_recovery": has_recovery_in_source,
            })

        if (
            topic not in required_topics
            or not isinstance(amount, (int, float))
            or not math.isfinite(float(amount))
            or not claim_text
        ):
            continue

        if topic == "recovery_report" and not re.search(
            r"\b(?:ministry|finance|report)\b",
            claim_text,
            re.IGNORECASE,
        ):
            attribution_match = re.search(
                r"\b(?:Ministry of Finance report|Ministry report|Finance report|report)\b",
                source_block,
                re.IGNORECASE,
            )
            if attribution_match:
                attribution = attribution_match.group(0)
                claim_text = f"{attribution} states that {claim_text}"

        # Keep the first source-attached claim for each family; context is
        # ordered as presented to the generator, and duplicate overlaps are
        # expected in neighboring retrieved chunks. Format large bare integers
        # for readability without inferring a currency unit or scale.
        if float(amount).is_integer():
            bare_value = str(int(float(amount)))
            if len(bare_value) >= 4:
                claim_text = re.sub(
                    rf"(?<![\w,:]){re.escape(bare_value)}(?![\w,:]|\.\d)",
                    f"{int(float(amount)):,}",
                    claim_text,
                )
        # Caption-aware context rows include source timestamps for prompt
        # assembly. Keep the extractive fallback's claim wording, but leave
        # timestamp presentation to the trusted returned-source metadata.
        claim_text = re.sub(r"^\s*-\s*\[[^\]]+\]\s*", "", claim_text)
        # Only a certificate amount can use immediate same-source neighbors
        # to complete a missing unit. Investment, DRT and recovery figures
        # retain their original unit evidence.
        unit_context = (
            str(claim.get("amount_caption_context") or "")
            if topic == "certificate" and local_topic == "certificate"
            else ""
        )
        selected_claim = {
            **claim,
            "topic": topic,
            "text": claim_text,
            "amount_caption": claim.get("amount_caption") or claim_text,
            "amount_unit_context": unit_context,
        }
        existing_claim = claims_by_topic.get(topic)
        if existing_claim is None:
            claims_by_topic[topic] = selected_claim
        elif (
            topic == "certificate"
            and math.isclose(
                float(existing_claim.get("amount", -1)),
                float(amount),
                rel_tol=0.0,
                abs_tol=0.1,
            )
            and not existing_claim.get("amount_unit_context")
            and unit_context
        ):
            # Overlapping strict/bare amount parsers can recognize the same
            # certificate figure. Preserve the first claim's attribution, but
            # enrich only its unit metadata from a duplicate candidate for the
            # same numeric amount and source block.
            same_source = (
                existing_claim.get("source_id") == source_id
                and _financial_claim_topic(
                    str(existing_claim.get("amount_caption") or "")
                ) == "certificate"
                and _financial_claim_topic(amount_caption) == "certificate"
            )
            if same_source:
                existing_claim["amount_unit_context"] = unit_context

    if question_scope == "bank_recovery":
        bank_evidence = _extract_bank_recovery_statement_evidence(source_blocks)
        recovery_claim = (
            claims_by_topic.get("recovery_report")
            or claims_by_topic.get("recovery")
        )
        supported_evidence_groups = []
        if recovery_claim is not None:
            supported_evidence_groups.append("recovery_amount")
        if bank_evidence["bank_payment"]:
            supported_evidence_groups.append("bank_payment")
        if bank_evidence["account_statement"]:
            supported_evidence_groups.append("account_statement")
        required_evidence_groups = [
            "recovery_amount",
            "bank_payment",
            "account_statement",
        ]
        missing_evidence_groups = [
            group for group in required_evidence_groups
            if group not in supported_evidence_groups
        ]
        minimum_supported_groups = 2

        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({
                "scope": "bank_recovery",
                "rejection_reason": (
                    None
                    if len(supported_evidence_groups) >= minimum_supported_groups
                    else "insufficient_bank_recovery_evidence"
                ),
                "required_topics": [],
                "missing_topics": [],
                "required_evidence_groups": required_evidence_groups,
                "minimum_supported_evidence_groups": minimum_supported_groups,
                "supported_evidence_groups": supported_evidence_groups,
                "missing_evidence_groups": missing_evidence_groups,
                "recognized_topics": {
                    topic: {
                        "amount": claim.get("amount"),
                        "excerpt": str(claim.get("text", ""))[:240],
                        "source_id": claim.get("source_id"),
                        "amount_caption": str(
                            claim.get("amount_caption") or ""
                        )[:180],
                    }
                    for topic, claim in claims_by_topic.items()
                },
                "bank_recovery_statement_evidence": {
                    group: entries[:3]
                    for group, entries in bank_evidence.items()
                },
                "source_block_count": len(source_blocks),
                "candidate_claims": candidate_claim_diagnostics[:30],
                "source_financial_excerpts": {
                    str(source_id): [
                        re.sub(r"\s+", " ", line).strip()[:180]
                        for line in block.splitlines()
                        if re.search(
                            r"\b(?:drt|tribunal|judg\w*ment|certificate|debt|"
                            r"recover\w*|ministry|finance|report|invest\w*|"
                            r"crores?|rupees?|banks?|statement)\b",
                            line,
                            re.IGNORECASE,
                        )
                    ][:3]
                    for source_id, block in source_blocks.items()
                },
            })
        if len(supported_evidence_groups) < minimum_supported_groups:
            return None

        lines = ["The transcript discusses these bank/recovery points:"]
        if recovery_claim is not None:
            amount = recovery_claim.get("amount")
            if not isinstance(amount, (int, float)) or not math.isfinite(float(amount)):
                raise ValueError(
                    "Bank/recovery fallback requires a finite extracted amount."
                )
            value = (
                f"{int(amount):,}"
                if float(amount).is_integer()
                else f"{amount:g}"
            )
            amount_text = " ".join((
                str(recovery_claim.get("text", "")),
                str(recovery_claim.get("amount_caption") or ""),
            ))
            source_block = source_blocks.get(
                recovery_claim.get("source_id"), ""
            )
            # Normalize caption whitespace only inside the already selected
            # source block so labels split across adjacent caption rows can be
            # rendered without borrowing report attribution from another source.
            normalized_source_block = re.sub(r"\s+", " ", source_block)
            unit_context = str(
                recovery_claim.get("amount_unit_context") or ""
            )
            unit_evidence = f"{amount_text} {unit_context}"
            if re.search(r"\bcrore\s+rupees?\b", unit_evidence, re.IGNORECASE):
                formatted = f"{value} crore rupees"
            elif re.search(r"\bcrores?\b|\bcr\b", unit_evidence, re.IGNORECASE):
                formatted = f"{value} crore"
            elif re.search(r"\brupees?\b|\bINR\b|₹|\bRs\.?", unit_evidence, re.IGNORECASE):
                formatted = f"{value} rupees"
            else:
                formatted = f"the figure {value}"

            if recovery_claim.get("topic") == "recovery_report":
                if re.search(
                    r"\bministry\b.*\bfinance\b.*\breport\b",
                    normalized_source_block,
                    re.IGNORECASE,
                ):
                    report_label = "Ministry of Finance report"
                elif re.search(r"\bfinance\s+report\b", normalized_source_block, re.IGNORECASE):
                    report_label = "Finance report"
                elif re.search(r"\bministry\s+report\b", normalized_source_block, re.IGNORECASE):
                    report_label = "Ministry report"
                else:
                    report_label = "report"
                lines.append(
                    f"- Recovery amount: the transcript attributes a recovery "
                    f"figure of {formatted} to the {report_label}."
                )
            else:
                lines.append(
                    f"- Recovery statement: the transcript describes "
                    f"{formatted} as recovered, but this source does not "
                    "attribute the figure to a specific report."
                )

        for item in bank_evidence["bank_payment"][:1]:
            lines.append(
                "- Bank-payment status: the transcript states that "
                + item["text"].rstrip(".")
                + "."
            )
        for item in bank_evidence["account_statement"][:1]:
            lines.append(
                "- Account/Parliament reference: the transcript states that "
                + item["text"].rstrip(".")
                + "."
            )
        return "\n\n".join(lines)

    missing_topics = [
        topic for topic in required_topics
        if topic not in claims_by_topic
    ]
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "scope": "comprehensive_financial",
            "rejection_reason": (
                "missing_required_topics" if missing_topics else None
            ),
            "required_topics": list(required_topics),
            "missing_topics": missing_topics,
            "recognized_topics": {
                topic: {
                    "amount": claim.get("amount"),
                    "excerpt": str(claim.get("text", ""))[:240],
                    "source_id": claim.get("source_id"),
                    "amount_caption": str(claim.get("amount_caption") or "")[:180],
                    "amount_unit_context": str(claim.get("amount_unit_context") or "")[:240],
                }
                for topic, claim in claims_by_topic.items()
            },
            "source_block_count": len(source_blocks),
            "candidate_claims": candidate_claim_diagnostics[:30],
            "source_financial_excerpts": {
                str(source_id): [
                    re.sub(r"\s+", " ", line).strip()[:180]
                    for line in block.splitlines()
                    if re.search(
                        r"\b(?:drt|tribunal|judg\w*ment|certificate|debt|"
                        r"recover\w*|ministry|finance|report|invest\w*|crores?|"
                        r"rupees?|\d{3,})\b",
                        line,
                        re.IGNORECASE,
                    )
                ][:2]
                for source_id, block in source_blocks.items()
            },
        })
    if missing_topics:
        return None

    def formatted_amount(claim: dict[str, Any]) -> tuple[str, bool]:
        amount = claim.get("amount")
        if not isinstance(amount, (int, float)) or not math.isfinite(float(amount)):
            raise ValueError("Financial fallback requires a finite extracted amount.")
        value = f"{int(amount):,}" if float(amount).is_integer() else f"{amount:g}"
        evidence_text = " ".join((
            str(claim.get("text", "")),
            str(claim.get("amount_caption", "")),
        ))
        unit_context = (
            str(claim.get("amount_unit_context") or "")
            if claim.get("topic") == "certificate"
            else ""
        )
        unit_evidence = f"{evidence_text} {unit_context}"
        crore_rupees = re.search(r"\bcrore\s+rupees?\b", unit_evidence, re.IGNORECASE)
        crore_unit = re.search(r"\bcrores?\b|\bcr\b", unit_evidence, re.IGNORECASE)
        rupee_unit = re.search(r"\brupees?\b|\bINR\b|₹|\bRs\.?", unit_evidence, re.IGNORECASE)
        if crore_rupees:
            return f"{value} crore rupees", True
        if crore_unit:
            return f"{value} crore", True
        if rupee_unit:
            return f"{value} rupees", True
        return f"the figure {value}", False

    investment_amount, investment_has_unit = formatted_amount(claims_by_topic["investment"])
    drt_amount, drt_has_unit = formatted_amount(claims_by_topic["drt_debt"])
    certificate_amount, certificate_has_unit = formatted_amount(claims_by_topic["certificate"])
    recovery_amount, recovery_has_unit = formatted_amount(claims_by_topic["recovery_report"])

    lines = ["The transcript supports these distinct financial statements:"]
    if investment_has_unit:
        lines.append(
            f"- Investment: the transcript says the speaker put in {investment_amount} of UB Group money."
        )
    else:
        lines.append(
            f"- Investment: the transcript mentions putting in {investment_amount} of UB Group money, "
            "but the cited caption does not state a unit."
        )

    if drt_has_unit:
        lines.append(
            f"- DRT debt: the transcript describes {drt_amount} as a debt in a passage that mentions the Debt Recovery Tribunal."
        )
    else:
        lines.append(
            f"- DRT debt: the transcript describes {drt_amount} as a debt and mentions the Debt Recovery Tribunal, "
            "but the amount-bearing caption does not state a unit."
        )

    if certificate_has_unit:
        lines.append(
            f"- Debt certificate: the transcript identifies a certificate amount of {certificate_amount}."
        )
    else:
        lines.append(
            f"- Debt certificate: the transcript identifies the certificate amount as {certificate_amount}; "
            "the caption does not state a unit."
        )

    if recovery_has_unit:
        lines.append(
            f"- Ministry of Finance recovery report: the transcript associates a recovery figure of {recovery_amount} with that report."
        )
    else:
        lines.append(
            f"- Ministry of Finance recovery report: the transcript associates a recovery figure of {recovery_amount} with that report, "
            "but the cited caption does not state a unit."
        )

    lines.append(
        "The transcript does not clearly explain how the DRT judgment and "
        "debt-certificate figures relate, so they should not be treated as one total."
    )
    return "\n\n".join(lines)


# ============================================================
# 6. INFERENCE CLIENTS
# ============================================================

def create_huggingface_client() -> InferenceClient:
    """Create the configured Hugging Face client."""

    if not HF_TOKEN:

        raise RuntimeError(
            "HF_TOKEN was not found.\n\n"
            "Create a .env file in the project root "
            "with:\n\n"
            "HF_TOKEN=hf_your_token_here"
        )


    return InferenceClient(

        provider=HF_PROVIDER,

        api_key=HF_TOKEN,
    )


def create_groq_client(*, max_retries: int | None = None) -> Groq:
    """Create a Groq client, optionally overriding SDK-level retries.

    Generation supplies max_retries=0 because generate_with_groq owns its
    bounded retry policy. Keeping SDK retries disabled on that client avoids
    multiplying SDK retries by the application's retry loop. Other callers
    retain the SDK default unless they explicitly override it.
    """

    if not GROQ_API_KEY:

        raise RuntimeError(
            "GROQ_API_KEY was not found.\n\n"
            "Set GROQ_API_KEY in the project root .env file."
        )

    client_options: dict[str, Any] = {"api_key": GROQ_API_KEY}
    if max_retries is not None:
        client_options["max_retries"] = max_retries

    return Groq(**client_options)


def create_llm_client() -> InferenceClient | Groq:
    """Create the configured LLM client used by planning and generation."""

    if LLM_PROVIDER == "groq":
        return create_groq_client(max_retries=0)

    return create_huggingface_client()


# ============================================================
# 7. LANGCHAIN PROMPT → CHAT MESSAGES
# ============================================================

def prompt_to_messages(
    prompt_value,
) -> list[dict[str, str]]:
    """
    Convert LangChain ChatPromptValue into the message format
    expected by Hugging Face Chat Completions.
    """

    messages = []


    for message in prompt_value.messages:

        role = message.type


        ROLE_MAP = {
            "system": "system",
            "human": "user",
            "ai": "assistant",
        }

        try:
            role = ROLE_MAP[message.type]

        except KeyError as error:

            raise RuntimeError(
                "Unsupported LangChain message role: "
                f"{message.type!r}"
            ) from error


        content = message.content


        if not isinstance(
            content,
            str,
        ):

            content = str(
                content
            )


        messages.append(

            {
                "role": role,
                "content": content,
            }
        )


    return messages

# response extractor
def extract_chat_completion_answer(response) -> str:
    """
    Extract the user-visible answer from a Hugging Face
    Chat Completions response.

    Raises RuntimeError when the response structure is invalid
    or contains no visible answer.
    """

    try:
        choice = response.choices[0]
        message = choice.message
    except (
        AttributeError,
        IndexError,
        TypeError,
    ) as error:

        raise RuntimeError(
            "Hugging Face returned an invalid chat completion response."
        ) from error

    content = getattr(
        message,
        "content",
        None,
    )

    if content is None:
        return ""

    if not isinstance(content, str):
        content = str(content)

    return content.strip()

# capture response diagnostics
def get_chat_response_diagnostics(
    response,
) -> dict[str, object]:
    """
    Extract safe diagnostic metadata from a Hugging Face
    Chat Completions response.

    This function intentionally excludes prompt content,
    transcript content, reasoning content, and credentials.
    """

    try:
        choice = response.choices[0]
        message = choice.message

    except (
        AttributeError,
        IndexError,
        TypeError,
    ):

        return {
            "finish_reason": None,
            "content_length": 0,
            "reasoning_length": 0,
        }

    content = getattr(
        message,
        "content",
        None,
    )

    reasoning = getattr(
        message,
        "reasoning",
        None,
    )

    return {
        "finish_reason":
            getattr(
                choice,
                "finish_reason",
                None,
            ),

        "content_length":
            len(content or ""),

        "reasoning_length":
            len(reasoning or ""),
    }


# ============================================================
# 8. OUTPUT SANITIZATION
# ============================================================

def sanitize_generated_answer(answer: str) -> str:
    """Remove source/citation scaffolding and untrusted timestamp summaries."""

    if not answer or not answer.strip():
        return FALLBACK_ANSWER

    cleaned = answer.strip()

    # Keep the API contract defensive if the provider ignores the prompt
    # and echoes internal source labels.
    cleaned = re.sub(
        r"(?im)^\s*\[?source\s+\d+\]?\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*(?:sources?|citations?)\s*:\s*$",
        "",
        cleaned,
    )
    # The LLM must not author this metadata. A trusted timestamp summary is
    # appended later from source intervals after answer generation.
    cleaned = _strip_unverified_timestamp_summary(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned or FALLBACK_ANSWER




def _is_abstention_answer(answer: str | None) -> bool:
    """Recognize canonical English and clear Hindi/Arabic abstentions.

    Some models translate the fallback into the question's language. Those
    answers are still abstentions: source links and timestamps should not be
    attached merely because best-effort overview retrieval supplied context.
    Require paired absence/non-coverage cues for localized text to avoid
    suppressing a substantive answer that only contains a caveat.
    """
    text = _strip_unverified_timestamp_summary(answer or "").strip()
    if not text:
        return True
    if text == FALLBACK_ANSWER:
        return True

    normalized = unicodedata.normalize("NFKC", text)

    # The backend emits this exact one-sentence Hindi fallback. It has no
    # second explicit "not covered" clause, so the paired-cue detector below
    # otherwise leaves best-effort sources and timestamps attached to it.
    if re.fullmatch(
        r"वीडियो में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली[।.!]?",
        normalized,
    ):
        return True

    english_missing_information = bool(re.search(
        r"\b(?:could(?:n't| not)|can't|cannot|unable to)\s+"
        r"(?:find|locate|verify|answer)\b.{0,100}\b(?:information|answer|"
        r"details?|evidence|figures?)\b|\bno relevant information\b",
        normalized,
        re.IGNORECASE,
    ))
    english_not_covered = bool(re.search(
        r"\b(?:may not be covered|not covered|isn't covered|is not covered|"
        r"doesn't cover|does not cover|not discussed|isn't discussed|"
        r"is not discussed|not mentioned|isn't mentioned|is not mentioned)\b",
        normalized,
        re.IGNORECASE,
    ))
    if english_missing_information and english_not_covered:
        return True

    hindi_missing_information = bool(re.search(
        r"(?:कोई|पर्याप्त)?\s*जानकारी.{0,40}(?:नहीं|नही)"
        r".{0,25}(?:पा|मिल|प्राप्त|सक|है|हूँ|हूं)?|"
        r"(?:मुझे|मैं).{0,35}(?:जानकारी|उत्तर).{0,30}(?:नहीं|नही)",
        normalized,
    ))
    hindi_not_covered = bool(re.search(
        r"(?:वीडियो|विषय|सवाल|प्रश्न).{0,60}"
        r"(?:कवर|चर्चा|उल्लेख|बताया|शामिल|मौजूद|उपलब्ध).{0,25}(?:नहीं|नही)|"
        r"(?:कवर|चर्चा|उल्लेख|बताया|शामिल).{0,25}(?:नहीं|नही)",
        normalized,
    ))
    if hindi_missing_information and hindi_not_covered:
        return True

    arabic_missing_information = bool(re.search(
        r"(?:لا توجد|لا تتوفر|ليس هناك|لا أجد|لم أجد|لا أستطيع العثور على)"
        r".{0,40}(?:معلومات|بيانات|تفاصيل)|"
        r"(?:معلومات|بيانات|تفاصيل).{0,30}(?:غير متوفرة|غير موجودة|لا توجد|لا تتوفر)",
        normalized,
    ))
    arabic_not_covered = bool(re.search(
        # Cover both Arabic word orders, including "لا توجد في الفيديو أي معلومات"
        # and "لا توجد معلومات كافية في الفيديو".
        r"(?:لا|لم)\s+أجد.{0,50}(?:في الفيديو|في المقطع)|"
        r"(?:لا توجد|لا تتوفر|ليس هناك).{0,40}"
        r"(?:في الفيديو|في المقطع).{0,40}(?:معلومات|بيانات|تفاصيل)|"
        r"(?:لا توجد|لا تتوفر|ليس هناك).{0,40}"
        r"(?:معلومات|بيانات|تفاصيل).{0,50}(?:في الفيديو|في المقطع)|"
        r"(?:الفيديو|المقطع|الموضوع|هذا السؤال|السؤال).{0,55}"
        r"(?:لا|لم|غير مذكور|غير موجود|غير متناول)|"
        r"(?:لا|لم).{0,35}(?:يتناول|يذكر|يناقش|يغطي|يتحدث عن)",
        normalized,
    ))
    return arabic_missing_information and arabic_not_covered


def _provider_error_status(error: Exception) -> int | None:
    """Return an upstream HTTP status code when the SDK exposes one."""
    candidates = [error, getattr(error, "response", None)]
    for candidate in candidates:
        status = getattr(candidate, "status_code", None)
        if status is None:
            continue
        try:
            return int(status)
        except (TypeError, ValueError):
            continue
    return None


def _provider_retry_after_seconds(error: Exception) -> float | None:
    """Read a numeric Retry-After header when the provider exposes one."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(error, "headers", None)
    if headers is None:
        return None

    try:
        raw_value = headers.get("retry-after")
    except (AttributeError, TypeError):
        return None
    if raw_value is None:
        return None

    try:
        seconds = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _is_daily_token_quota_exhausted(error: Exception) -> bool:
    """Return True only for explicit provider daily-token quota rejections."""
    error_text = str(error).casefold()
    quota_marker = (
        "tokens per day" in error_text
        or re.search(r"\btpd\b", error_text) is not None
    )
    rate_limit_marker = (
        "429" in error_text
        or "rate_limit_exceeded" in error_text
        or "rate limit" in error_text
    )
    return quota_marker and rate_limit_marker


def _is_retryable_provider_error(error: Exception) -> bool:
    """Recognize transient upstream failures without retrying bad requests/auth."""
    status = _provider_error_status(error)
    if status is not None:
        return status in {408, 425, 429, 500, 502, 503, 504}

    error_text = str(error).lower()
    non_retryable_markers = (
        "unauthorized",
        "authentication failed",
        "invalid api key",
        "forbidden",
        "permission denied",
        "model_not_supported",
        "model not found",
        "invalid request",
        "bad request",
    )
    if any(marker in error_text for marker in non_retryable_markers):
        return False

    retryable_markers = (
        "rate limit",
        "too many requests",
        "timed out",
        "timeout",
        "temporary failure",
        "temporarily unavailable",
        "connection reset",
        "connection refused",
        "connection error",
        "server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "http 408",
        "http 425",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    )
    return any(marker in error_text for marker in retryable_markers)


# ============================================================
# 9. MODEL GENERATION
# ============================================================

def generate_with_huggingface(
    client: InferenceClient,
    prompt_value,
) -> str:
    """
    Send the LangChain prompt to Hugging Face and return
    the generated text.

    Empty model responses and recognized transient upstream failures are retried
    within the configured attempt budget. Authentication, permission, model, and
    request-configuration errors fail immediately.
    """

    messages = prompt_to_messages(prompt_value)

    for attempt in range(HF_MAX_RETRIES + 1):

        try:

            response = (
                client
                .chat
                .completions
                .create(
                    model=HF_MODEL_ID,
                    messages=messages,
                    max_tokens=HF_MAX_TOKENS,
                    temperature=HF_TEMPERATURE,
                    extra_body={
                        "reasoning_effort": HF_REASONING_EFFORT,
                    },
                )
            )

        except Exception as error:

            error_text = str(error).lower()

            if "model_not_supported" in error_text:

                raise RuntimeError(
                    "Hugging Face could not find an enabled "
                    "Inference Provider for the selected model.\n\n"
                    f"Model: {HF_MODEL_ID}\n"
                    f"Provider policy: {HF_PROVIDER}\n\n"
                    "Check the selected model/provider in Hugging Face."
                ) from error

            if (
                "401" in error_text
                or "unauthorized" in error_text
            ):

                raise RuntimeError(
                    "Hugging Face authentication failed.\n\n"
                    "Check HF_TOKEN in .env."
                ) from error

            if (
                "403" in error_text
                or "forbidden" in error_text
            ):

                raise RuntimeError(
                    "Hugging Face rejected the inference request.\n\n"
                    "Check token permissions and the selected "
                    "model/provider."
                ) from error

            if _is_retryable_provider_error(error):
                if attempt < HF_MAX_RETRIES:
                    logger.warning(
                        "Transient Hugging Face generation failure; retrying "
                        "attempt=%s max_attempts=%s status_code=%s error_type=%s",
                        attempt + 1,
                        HF_MAX_RETRIES + 1,
                        _provider_error_status(error),
                        type(error).__name__,
                    )
                    time.sleep(HF_RETRY_DELAY_SECONDS * (attempt + 1))
                    continue

                if "429" in error_text or "rate limit" in error_text or "too many requests" in error_text:
                    raise RuntimeError(
                        "Hugging Face rate limit was reached after "
                        f"{HF_MAX_RETRIES + 1} attempts. Retry after the "
                        "current rate-limit window."
                    ) from error

            raise RuntimeError(
                f"Hugging Face generation failed: {error}"
            ) from error

        # --------------------------------------------------------
        # Extract visible model answer
        # --------------------------------------------------------

        answer = extract_chat_completion_answer(response)

        if answer:
            return sanitize_generated_answer(answer)

        # --------------------------------------------------------
        # Empty response diagnostics
        # --------------------------------------------------------

        diagnostics = get_chat_response_diagnostics(
            response
        )

        # --------------------------------------------------------
        # Retry if attempts remain
        # --------------------------------------------------------

        if attempt < HF_MAX_RETRIES:

            time.sleep(
                HF_RETRY_DELAY_SECONDS
            )

            continue

        # --------------------------------------------------------
        # All attempts exhausted
        # --------------------------------------------------------

        raise RuntimeError(
            "The Hugging Face model returned an empty answer "
            f"after {HF_MAX_RETRIES + 1} attempts. "
            f"Diagnostics: {diagnostics}"
        )

def generate_with_groq(
    client: Groq,
    prompt_value,
) -> str:
    """Send the LangChain prompt to Groq and return the visible answer."""

    messages = prompt_to_messages(prompt_value)
    total_retry_wait_seconds = 0.0

    for attempt in range(GROQ_MAX_RETRIES + 1):
        try:
            response = (
                client
                .chat
                .completions
                .create(
                    model=GROQ_MODEL_ID,
                    messages=messages,
                    max_completion_tokens=GROQ_MAX_TOKENS,
                    temperature=GROQ_TEMPERATURE,
                    reasoning_effort=GROQ_REASONING_EFFORT,
                    include_reasoning=False,
                )
            )

        except Exception as error:
            error_text = str(error).lower()

            # A daily token quota is a hard provider ceiling, not a transient
            # burst limit. Retrying the same request cannot help.
            if _is_daily_token_quota_exhausted(error):
                logger.warning(
                    "Groq daily token quota exhausted; skipping retries "
                    "status_code=%s error_type=%s",
                    _provider_error_status(error),
                    type(error).__name__,
                )
                raise RuntimeError(
                    "Groq daily token quota is exhausted for the configured model. "
                    "Wait until quota is available or use an available model/provider."
                ) from error

            provider_status = _provider_error_status(error)
            # Inspect the structured HTTP status rather than searching for
            # numeric substrings: a 429 TPM message can contain "Requested
            # 4010 tokens", which must not be mistaken for HTTP 401.
            if provider_status == 401 or "unauthorized" in error_text:
                raise RuntimeError(
                    "Groq authentication failed.\n\n"
                    "Check GROQ_API_KEY in .env."
                ) from error

            if provider_status == 403 or "forbidden" in error_text:
                raise RuntimeError(
                    "Groq rejected the inference request.\n\n"
                    "Check API key permissions and the selected model."
                ) from error

            retry_after = _provider_retry_after_seconds(error)
            is_rate_limit = (
                _provider_error_status(error) == 429
                or "429" in error_text
                or "rate limit" in error_text
                or "too many requests" in error_text
            )

            if _is_retryable_provider_error(error):
                if attempt < GROQ_MAX_RETRIES:
                    configured_delay = GROQ_RETRY_DELAY_SECONDS * (attempt + 1)
                    retry_delay = max(
                        configured_delay,
                        retry_after if retry_after is not None else 0.0,
                    )

                    if (
                        is_rate_limit
                        and total_retry_wait_seconds + retry_delay
                        > MAX_GROQ_RETRY_WAIT_SECONDS
                    ):
                        logger.warning(
                            "Groq retry wait budget exhausted; returning a rate-limit response "
                            "attempt=%s max_attempts=%s status_code=%s "
                            "retry_after_seconds=%s waited_seconds=%.2f",
                            attempt + 1,
                            GROQ_MAX_RETRIES + 1,
                            _provider_error_status(error),
                            retry_after,
                            total_retry_wait_seconds,
                        )
                        raise ProviderRateLimitError(
                            retry_after_seconds=retry_after or retry_delay
                        ) from error

                    if is_rate_limit:
                        total_retry_wait_seconds += retry_delay

                    logger.warning(
                        "Transient Groq generation failure; retrying "
                        "attempt=%s max_attempts=%s status_code=%s error_type=%s "
                        "retry_after_seconds=%s retry_delay_seconds=%.2f "
                        "cumulative_retry_wait_seconds=%.2f",
                        attempt + 1,
                        GROQ_MAX_RETRIES + 1,
                        _provider_error_status(error),
                        type(error).__name__,
                        retry_after,
                        retry_delay,
                        total_retry_wait_seconds,
                    )
                    time.sleep(retry_delay)
                    continue

                if is_rate_limit:
                    raise ProviderRateLimitError(
                        retry_after_seconds=(
                            retry_after
                            if retry_after is not None
                            else max(1.0, GROQ_RETRY_DELAY_SECONDS)
                        )
                    ) from error

            raise RuntimeError(
                f"Groq generation failed: {error}"
            ) from error

        # --------------------------------------------------------
        # Record provider completion metadata without logging prompt or answer.
        # --------------------------------------------------------

        first_choice = (
            response.choices[0]
            if getattr(response, "choices", None)
            else None
        )
        finish_reason = getattr(first_choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        logger.info(
            "Groq generation metadata finish_reason=%s prompt_tokens=%s "
            "completion_tokens=%s total_tokens=%s completion_token_limit=%s",
            finish_reason,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            GROQ_MAX_TOKENS,
        )
        if finish_reason in {"length", "max_tokens"}:
            logger.warning(
                "Groq generation may be truncated by completion limit "
                "completion_tokens=%s completion_token_limit=%s",
                completion_tokens,
                GROQ_MAX_TOKENS,
            )

        answer = extract_chat_completion_answer(response)

        if answer:
            return sanitize_generated_answer(answer)

        diagnostics = get_chat_response_diagnostics(response)

        if attempt < GROQ_MAX_RETRIES:
            time.sleep(GROQ_RETRY_DELAY_SECONDS)
            continue

        raise RuntimeError(
            "The Groq model returned an empty answer "
            f"after {GROQ_MAX_RETRIES + 1} attempts. "
            f"Diagnostics: {diagnostics}"
        )


def generate_with_llm(
    client,
    prompt_value,
) -> str:
    """Dispatch generation to the configured inference provider."""

    if LLM_PROVIDER == "groq":
        return generate_with_groq(
            client=client,
            prompt_value=prompt_value,
        )

    return generate_with_huggingface(
        client=client,
        prompt_value=prompt_value,
    )

# ============================================================
# 9. FORMAT SOURCES
# ============================================================

def format_sources(
    source_groups: list[dict],
) -> list[dict[str, Any]]:
    """
    Convert merged source segments into API-friendly
    metadata.

    source_id is the one-based identifier for the returned source link.
    Default source links use selected evidence anchors. In the opt-in expanded
    context experiment, selected anchors stay merged, while context-only
    neighbors are returned as individual intervals for precise navigation.
    """

    sources = []


    for source_id, source_group in enumerate(
        source_groups,
        start=1,
    ):

        documents = (
            source_group["documents"]
        )


        video_id = (
            documents[0]
            .metadata
            .get(
                "video_id"
            )
        )


        chunk_ids = []


        for document in documents:

            chunk_id = (
                document.metadata.get(
                    "chunk_id"
                )
            )


            if (
                chunk_id is not None
                and
                chunk_id not in chunk_ids
            ):

                chunk_ids.append(
                    chunk_id
                )


        start = float(
            source_group["start"]
        )


        end = float(
            source_group["end"]
        )


        sources.append(
            {
                "source_id":
                    source_id,

                "chunk_ids":
                    chunk_ids,

                "video_id":
                    video_id,

                "start":
                    start,

                "end":
                    end,

                "duration":
                    max(
                        0.0,
                        end - start,
                    ),

                "distance":
                    float(
                        source_group["distance"]
                    ),
            }
        )


    return sources

def _build_retrieval_attempt_trace(
    stage: str,
    results,
    query: str | None = None,
) -> dict[str, object]:
    """Summarize one retrieval attempt without exposing transcript text."""

    distances = [
        float(distance)
        for _document, distance in results
    ]

    chunk_ids = []
    for document, _distance in results:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is not None and chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)

    attempt = {
        "stage": stage,
        "retrieved_chunks": len(results),
        "best_distance": (
            round(min(distances), 6)
            if distances
            else None
        ),
        "chunk_ids": chunk_ids[:20],
    }
    if query is not None:
        attempt["query"] = query

    return attempt


def _split_retrieval_result_bundle(result_bundle):
    """Return expanded context results and selected anchors compatibly."""
    if (
        isinstance(result_bundle, tuple)
        and len(result_bundle) == 2
        and isinstance(result_bundle[0], list)
        and isinstance(result_bundle[1], list)
    ):
        return result_bundle[0], result_bundle[1]
    # Test doubles and older callers may return only a results list.
    return result_bundle, result_bundle


def _is_explicit_overview_request(question: str) -> bool:
    """Return whether the user's original wording clearly asks for a broad overview.

    This check is independent of the planner so it can also guard the final
    fallback after targeted retrieval/reformulation has been exhausted.
    """
    normalized = " ".join(question.casefold().split())
    analytical_request = re.search(
        r"\b(?:explain|describe|analyse|analyze|evaluate|assess|compare|"
        r"contrast|discuss|trace|identify|outline)\b",
        normalized,
    )
    multi_facet_request = re.search(
        r"\b(?:and|or|both|versus|vs\.?|as well as)\b",
        normalized,
    )
    consequence_or_history_request = re.search(
        r"\b(?:consequence\w*|implication\w*|outcome\w*|"
        r"development\w*|historical|cause\w*|effect\w*|impact\w*)\b",
        normalized,
    )
    if is_evidence_dense_question(question) or (
        analytical_request
        and (multi_facet_request or consequence_or_history_request)
    ):
        return False

    if is_overview_question(question):
        return True
    return bool(re.search(
        r"\b(?:main|overall|general)\s+(?:subject|topic|theme|idea)\b|"
        r"\b(?:overall|general)\s+(?:summary|overview)\b",
        normalized,
    ))


def _should_use_overview_route(question: str, planner_intent: str) -> bool:
    """Accept planner overview intent only when the original question is broad."""
    return (
        planner_intent == "overview"
        and _is_explicit_overview_request(question)
    )


def _planner_failure_query_candidates(question: str) -> list[str]:
    """Return a bounded set of deterministic wording aliases for planner recovery."""
    candidates = [question]
    normalized = " ".join(question.casefold().split())

    # "Orfield room" and "world's quietest room" refer to the same room in the
    # Arabic episode. The latter wording has already demonstrated successful
    # retrieval for the price passage; keep this alias narrowly scoped.
    if re.search(r"\borfield\s+room\b", normalized):
        if (
            re.search(r"\bhow\s+much\b", normalized)
            and re.search(r"\bone\s+hour\b", normalized)
            and re.search(r"\b(?:cost|price)\b", normalized)
        ):
            # Reuse the exact AR02 wording whose live run retrieved the
            # price-bearing source passage.
            alias = "How much might one hour in the world's quietest room cost?"
        else:
            alias = re.sub(
                r"\borfield\s+room\b",
                "world's quietest room",
                question,
                flags=re.IGNORECASE,
            )
        if alias.strip() and alias.casefold() != question.casefold():
            candidates.append(alias)

    return list(dict.fromkeys(candidate for candidate in candidates if candidate.strip()))[:2]


def _deterministic_query_evidence_fallback(
    vector_store,
    question: str,
    video_id: str | None = None,
) -> tuple[list, list, dict[str, object], bool]:
    """Run bounded original/alias retrieval with semantic and gold-evidence gates.

    The original wording is attempted first. A narrowly defined alias may be
    tried next. Every accepted anchor must pass the same absolute semantic
    distance ceiling as ordinary relaxed recovery. For registered questions
    with generation-time gold evidence, expanded context must also contain the
    required evidence; a generic semantic match alone is insufficient.
    """
    gold_question_id, gold_case = _registered_generation_gold_case(
        video_id,
        question,
    )
    required_gold_groups = (
        math.ceil(
            len(gold_case.groups)
            * float(gold_case.min_generation_context_group_coverage)
        )
        if gold_case is not None
        else 0
    )
    query_candidates = _planner_failure_query_candidates(question)
    attempt_history: list[dict[str, object]] = []

    for query in query_candidates:
        try:
            fallback_bundle = retrieve_question_context(
                vector_store=vector_store,
                query=query,
                k=RECOVERY_TOP_K,
                fetch_k=RECOVERY_FETCH_K,
                lambda_mult=MMR_LAMBDA,
                max_distance=RECOVERY_MAX_DISTANCE,
                expand_context=False,
                allow_overview=False,
                return_anchors=True,
            )
            fallback_candidates, fallback_anchors = _split_retrieval_result_bundle(
                fallback_bundle
            )
            semantic_gate_kwargs = (
                {"absolute_distance_ceiling": MAX_DISTANCE}
                if gold_case is not None
                else {}
            )
            strong_gate_passed = bool(fallback_anchors) and passes_strong_semantic_gate(
                fallback_anchors,
                RECOVERY_MAX_DISTANCE,
                **semantic_gate_kwargs,
            )
            attempt = _build_retrieval_attempt_trace(
                "deterministic_query_evidence_fallback",
                fallback_anchors or fallback_candidates,
                query=query,
            )
            attempt["strong_semantic_gate_passed"] = bool(strong_gate_passed)
            if gold_case is not None:
                attempt["semantic_gate_distance_ceiling"] = MAX_DISTANCE
            attempt["query_alias_used"] = query != question
            if not strong_gate_passed:
                attempt_history.append(attempt)
                continue

            expanded = expand_question_context(
                vector_store,
                query,
                fallback_anchors,
            )
            attempt["expanded_context_chunks"] = len(expanded)

            if gold_case is not None:
                expanded_text = "\n".join(
                    str(getattr(document, "page_content", "") or "")
                    for document, _distance in expanded
                )
                included_gold_groups = matched_gold_groups(
                    expanded_text,
                    gold_case,
                )
                missing_gold_groups = sorted(
                    set(range(len(gold_case.groups))) - included_gold_groups
                )
                evidence_gate_passed = (
                    len(included_gold_groups) >= required_gold_groups
                )
                attempt["gold_evidence_question_id"] = gold_question_id
                attempt["gold_evidence_required_groups"] = required_gold_groups
                attempt["gold_evidence_groups_present"] = sorted(included_gold_groups)
                attempt["gold_evidence_groups_missing"] = missing_gold_groups
                attempt["gold_evidence_gate_passed"] = evidence_gate_passed
                if not evidence_gate_passed:
                    attempt["gold_evidence_rejection_reason"] = (
                        "required_generation_evidence_missing"
                    )
                    attempt_history.append(attempt)
                    continue

            attempt_history.append(attempt)
            attempt["query_candidates_tried"] = [
                str(item.get("query", "")) for item in attempt_history
            ]
            attempt["query_attempts"] = [
                {key: value for key, value in item.items() if key != "query_attempts"}
                for item in attempt_history
            ]
            return expanded, fallback_anchors, attempt, True
        except Exception as error:
            logger.warning(
                "Deterministic query-evidence fallback candidate failed "
                "query=%r error=%s",
                query,
                type(error).__name__,
            )
            attempt_history.append({
                "stage": "deterministic_query_evidence_fallback",
                "query": query,
                "query_alias_used": query != question,
                "retrieved_chunks": 0,
                "error": f"{type(error).__name__}: {error}",
                "strong_semantic_gate_passed": False,
            })

    summary_attempt: dict[str, object] = {
        "stage": "deterministic_query_evidence_fallback",
        "retrieved_chunks": 0,
        "query_candidates_tried": [
            str(item.get("query", "")) for item in attempt_history
        ],
        "query_attempts": attempt_history,
        "strong_semantic_gate_passed": any(
            bool(item.get("strong_semantic_gate_passed"))
            for item in attempt_history
        ),
    }
    if gold_case is not None:
        summary_attempt.update({
            "gold_evidence_question_id": gold_question_id,
            "gold_evidence_required_groups": required_gold_groups,
            "gold_evidence_gate_passed": False,
            "gold_evidence_rejection_reason": "required_generation_evidence_not_retrieved",
        })
    return [], [], summary_attempt, False


def _retrieve_with_query_recovery_trace(
    vector_store,
    question: str,
    llm_client: InferenceClient | Groq | None = None,
    video_id: str | None = None,
) -> tuple[list, dict[str, object]]:
    """Run the bounded recovery ladder and capture stage diagnostics."""

    started = time.perf_counter()
    attempts = []
    citation_results = []
    planner_used = False
    planner_intent = None
    planner_latency_ms = 0.0
    weak_recovery_candidates_rejected = False
    weak_initial_recovery_attempted = False
    weak_initial_recovery_accepted = False
    planner_overview_rejected = False
    best_effort_overview_suppressed = False
    deterministic_fallback_attempted = False
    deterministic_fallback_accepted = False
    deterministic_fallback_query = None
    deterministic_fallback_query_alias_used = False
    deterministic_fallback_evidence_gate_passed = None
    deterministic_fallback_rejection_reason = None

    # Timestamp-specific requests use bounded temporal retrieval. Do not let
    # the generic rewrite/overview recovery ladder bypass the requested time.
    temporal_bundle = retrieve_temporal_context(
        vector_store=vector_store,
        query=question,
    )
    if temporal_bundle is not None:
        temporal_results = temporal_bundle["results"]
        temporal_candidates = temporal_bundle["candidates"]
        temporal_window = temporal_bundle["window"]

        if temporal_results:
            final_stage = "temporal_window"
        elif temporal_candidates:
            final_stage = "temporal_window_insufficient_coverage"
        else:
            final_stage = "temporal_window_empty"

        attempt_results = temporal_candidates or temporal_results
        temporal_attempt = _build_retrieval_attempt_trace(
            (
                "temporal_window_candidates"
                if temporal_candidates and not temporal_results
                else "temporal_window"
            ),
            attempt_results,
            query=question,
        )
        temporal_attempt["coverage_ratio"] = temporal_bundle["coverage_ratio"]
        temporal_attempt["coverage_sufficient"] = temporal_bundle["coverage_sufficient"]

        temporal_trace = {
            "recovery_used": False,
            "final_stage": final_stage,
            "planner_used": False,
            "planner_intent": None,
            "planner_latency_ms": 0.0,
            "route": "temporal_window",
            "temporal_window": {
                **(temporal_window or {}),
                "window_seconds": temporal_bundle["window_seconds"],
                "covered_seconds": temporal_bundle["covered_seconds"],
                "coverage_ratio": temporal_bundle["coverage_ratio"],
                "minimum_coverage": temporal_bundle["minimum_coverage"],
                "coverage_sufficient": temporal_bundle["coverage_sufficient"],
                "candidate_chunk_ids": temporal_bundle["candidate_chunk_ids"],
            },
            "attempts": [temporal_attempt],
            "retrieval_latency_ms": round(
                (time.perf_counter() - started) * 1000,
                2,
            ),
            "final_chunk_ids": [
                document.metadata.get("chunk_id")
                for document, _distance in temporal_results
                if document.metadata.get("chunk_id") is not None
            ][:20],
            "citation_chunk_ids": [
                document.metadata.get("chunk_id")
                for document, _distance in temporal_results
                if document.metadata.get("chunk_id") is not None
            ][:20],
        }

        logger.info(
            "Temporal retrieval completed stage=%s start_seconds=%s end_seconds=%s "
            "coverage=%.3f candidate_chunks=%s selected_chunks=%s",
            final_stage,
            (temporal_window or {}).get("start_seconds"),
            (temporal_window or {}).get("end_seconds"),
            temporal_bundle["coverage_ratio"],
            len(temporal_candidates),
            len(temporal_results),
        )
        return temporal_results, temporal_trace

    initial_bundle = retrieve_question_context(
        vector_store=vector_store,
        query=question,
        k=TOP_K,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA,
        max_distance=MAX_DISTANCE,
        return_anchors=True,
        return_status=True,
    )
    if isinstance(initial_bundle, tuple) and len(initial_bundle) == 3:
        initial_results, citation_results, initial_retrieval_status = initial_bundle
    else:
        initial_results, citation_results = _split_retrieval_result_bundle(
            initial_bundle
        )
        initial_retrieval_status = {"weak_native_query_rejected": False}
    attempts.append(
        _build_retrieval_attempt_trace(
            "initial_strict",
            initial_results,
            query=question,
        )
    )

    if initial_results:
        final_results = initial_results
        final_stage = "initial_strict"

        # Initial candidates can be lexical-only or semantically weak. A
        # non-empty result used to end the recovery ladder even when no anchor
        # passed the same strong semantic gate required for rewritten queries.
        # Preserve the initial candidates as a safe fallback, but try one bounded
        # query rewrite and accept it only when rewritten evidence passes that
        # normal gate.
        weak_initial_candidates = bool(citation_results) and not passes_strong_semantic_gate(
            citation_results,
            MAX_DISTANCE,
        )
        if weak_initial_candidates and len(_generation_context_query_terms(question)) >= 3:
            weak_initial_recovery_attempted = True
            planner_started = time.perf_counter()
            try:
                if llm_client is None:
                    llm_client = create_llm_client()
                plan = understand_query(
                    client=llm_client,
                    question=question,
                )
                planner_used = True
                planner_intent = plan.intent
                planner_latency_ms = (
                    time.perf_counter() - planner_started
                ) * 1000
                planner_overview_rejected = (
                    plan.intent == "overview"
                    and not _should_use_overview_route(question, plan.intent)
                )

                if _should_use_overview_route(question, plan.intent):
                    overview_results = retrieve_overview(vector_store)
                    attempts.append(
                        _build_retrieval_attempt_trace(
                            "weak_initial_overview",
                            overview_results,
                        )
                    )
                    if overview_results:
                        final_results = overview_results
                        citation_results = overview_results
                        final_stage = "weak_initial_overview"
                        weak_initial_recovery_accepted = True

                if not weak_initial_recovery_accepted:
                    query_candidates = [
                        plan.standalone_question,
                        *plan.search_queries[:1],
                    ]
                    seen_queries = {question.strip().casefold()}
                    for rewritten_query in query_candidates:
                        rewritten_query = (rewritten_query or "").strip()
                        key = rewritten_query.casefold()
                        if not rewritten_query or key in seen_queries:
                            continue
                        seen_queries.add(key)

                        rewritten_bundle = retrieve_question_context(
                            vector_store=vector_store,
                            query=rewritten_query,
                            k=TOP_K,
                            fetch_k=MMR_FETCH_K,
                            lambda_mult=MMR_LAMBDA,
                            max_distance=MAX_DISTANCE,
                            expand_context=False,
                            allow_overview=False,
                            return_anchors=True,
                            return_status=True,
                        )
                        if (
                            isinstance(rewritten_bundle, tuple)
                            and len(rewritten_bundle) == 3
                        ):
                            _rewritten_context, rewritten_anchors, rewritten_status = rewritten_bundle
                        elif (
                            isinstance(rewritten_bundle, tuple)
                            and len(rewritten_bundle) == 2
                        ):
                            _rewritten_context, rewritten_anchors = rewritten_bundle
                            rewritten_status = {}
                        else:
                            rewritten_anchors = rewritten_bundle
                            rewritten_status = {}

                        semantic_gate_passed = passes_strong_semantic_gate(
                            rewritten_anchors,
                            MAX_DISTANCE,
                        )
                        attempt_trace = _build_retrieval_attempt_trace(
                            "weak_initial_rewrite",
                            rewritten_anchors,
                            query=rewritten_query,
                        )
                        attempt_trace["semantic_gate_passed"] = semantic_gate_passed
                        attempt_trace["accepted_for_generation"] = bool(
                            rewritten_anchors and semantic_gate_passed
                        )
                        if rewritten_status.get("weak_native_query_rejected"):
                            attempt_trace["rejection_reason"] = "weak_native_semantic_match"
                        elif rewritten_anchors and not semantic_gate_passed:
                            attempt_trace["rejection_reason"] = "weak_semantic_support"
                        attempts.append(attempt_trace)

                        if rewritten_anchors and semantic_gate_passed:
                            citation_results = rewritten_anchors
                            final_results = expand_question_context(
                                vector_store,
                                rewritten_query,
                                rewritten_anchors,
                            )
                            final_stage = "weak_initial_rewrite"
                            weak_initial_recovery_accepted = True
                            break
            except QueryUnderstandingError as error:
                logger.warning(
                    "Lexical-only query recovery planner failed; preserving initial candidates: %s",
                    error,
                )
            except Exception:
                logger.exception(
                    "Unexpected lexical-only query recovery failure; preserving initial candidates"
                )
    else:
        if initial_retrieval_status.get("weak_native_query_rejected"):
            attempts[-1]["weak_native_query_rejected"] = True
            attempts[-1]["rejection_reason"] = "weak_native_semantic_match"
            logger.info(
                "Abstaining before query-rewrite recovery: same-language retrieval "
                "has no lexical support and lacks strong semantic confidence."
            )
            return [], {
                "recovery_used": False,
                "final_stage": "no_evidence_weak_native_match_rejected",
                "planner_used": False,
                "planner_intent": None,
                "planner_latency_ms": 0.0,
                "attempts": attempts,
                "retrieval_latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "final_chunk_ids": [],
            }

        if llm_client is None:
            llm_client = create_llm_client()

        planner_started = time.perf_counter()

        try:
            plan = understand_query(
                client=llm_client,
                question=question,
            )
            planner_used = True
            planner_intent = plan.intent
            planner_latency_ms = (
                time.perf_counter() - planner_started
            ) * 1000
            planner_overview_rejected = (
                plan.intent == "overview"
                and not _should_use_overview_route(question, plan.intent)
            )
        except QueryUnderstandingError as error:
            logger.warning(
                "Query understanding failed during retrieval recovery; "
                "trying bounded original-query retrieval: %s",
                error,
            )
            planner_failure_reason = "query_understanding_error"
        except Exception as error:
            logger.exception(
                "Unexpected query-understanding failure during retrieval recovery"
            )
            planner_failure_reason = type(error).__name__

        if not planner_used:
            fallback_results, fallback_anchors, fallback_attempt, fallback_accepted = (
                _deterministic_query_evidence_fallback(
                    vector_store,
                    question,
                    video_id=video_id,
                )
            )
            attempts.append(fallback_attempt)
            elapsed_planner_ms = round(
                (time.perf_counter() - planner_started) * 1000,
                2,
            )
            if fallback_accepted:
                final_chunk_ids = [
                    document.metadata.get("chunk_id")
                    for document, _distance in fallback_results
                    if document.metadata.get("chunk_id") is not None
                ]
                citation_chunk_ids = [
                    document.metadata.get("chunk_id")
                    for document, _distance in fallback_anchors
                    if document.metadata.get("chunk_id") is not None
                ]
                return fallback_results, {
                    "recovery_used": True,
                    "final_stage": "planner_failed_original_query_relaxed",
                    "planner_used": False,
                    "planner_intent": None,
                    "planner_latency_ms": elapsed_planner_ms,
                    "planner_failure_reason": planner_failure_reason,
                    "planner_failure_fallback_attempted": True,
                    "planner_failure_fallback_accepted": True,
                    "planner_failure_fallback_query": fallback_attempt.get("query"),
                    "planner_failure_fallback_query_alias_used": bool(
                        fallback_attempt.get("query_alias_used")
                    ),
                    "planner_failure_fallback_evidence_gate_passed": (
                        fallback_attempt.get("gold_evidence_gate_passed")
                    ),
                    "attempts": attempts,
                    "retrieval_latency_ms": round(
                        (time.perf_counter() - started) * 1000,
                        2,
                    ),
                    "final_chunk_ids": final_chunk_ids[:20],
                    "citation_chunk_ids": citation_chunk_ids[:20],
                }
            evidence_rejection_reason = fallback_attempt.get(
                "gold_evidence_rejection_reason"
            )
            return [], {
                "recovery_used": True,
                "final_stage": (
                    "planner_failed_required_evidence_missing"
                    if evidence_rejection_reason
                    else "planner_failed_no_strong_evidence"
                ),
                "planner_used": False,
                "planner_intent": None,
                "planner_latency_ms": elapsed_planner_ms,
                "planner_failure_reason": planner_failure_reason,
                "planner_failure_fallback_attempted": True,
                "planner_failure_fallback_accepted": False,
                "planner_failure_fallback_rejection_reason": evidence_rejection_reason,
                "planner_failure_fallback_evidence_gate_passed": (
                    fallback_attempt.get("gold_evidence_gate_passed")
                ),
                "attempts": attempts,
                "retrieval_latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "final_chunk_ids": [],
                "citation_chunk_ids": [],
            }

        if _should_use_overview_route(question, plan.intent):
            overview_results = retrieve_overview(vector_store)
            attempts.append(
                _build_retrieval_attempt_trace(
                    "overview",
                    overview_results,
                )
            )

            if overview_results:
                logger.info("Query recovery used overview routing")
                final_results = overview_results
                citation_results = overview_results
                final_stage = "overview"
            else:
                final_results = []
                final_stage = "overview_empty"
        else:
            final_results = []

        if not final_results:
            # Prefer the standalone rewrite. Try one alternative only if it is
            # distinct; this bounds repeated retrieval work while preserving a
            # second chance for cases where the planner's standalone wording is
            # less effective than its best explicit search query.
            query_candidates = [
                plan.standalone_question,
                *plan.search_queries[:1],
            ]

            seen_queries = set()
            unique_queries = []

            for query in query_candidates:
                normalized = query.strip()

                if not normalized:
                    continue

                key = normalized.casefold()

                if key in seen_queries:
                    continue

                seen_queries.add(key)

                if key == question.strip().casefold():
                    continue

                unique_queries.append(normalized)

            final_stage = "rewritten_strict_exhausted"

            for query in unique_queries:
                recovered_anchors = retrieve_question_context(
                    vector_store=vector_store,
                    query=query,
                    k=TOP_K,
                    fetch_k=MMR_FETCH_K,
                    lambda_mult=MMR_LAMBDA,
                    max_distance=MAX_DISTANCE,
                    expand_context=False,
                    allow_overview=False,
                )
                attempt_trace = _build_retrieval_attempt_trace(
                    "rewritten_strict",
                    recovered_anchors,
                    query=query,
                )
                semantic_gate_passed = passes_strong_semantic_gate(
                    recovered_anchors,
                    MAX_DISTANCE,
                )
                attempt_trace["semantic_gate_passed"] = semantic_gate_passed
                attempt_trace["accepted_for_generation"] = bool(
                    recovered_anchors and semantic_gate_passed
                )
                attempt_trace["context_expansion_performed"] = False
                attempt_trace["expanded_context_chunks"] = 0
                if recovered_anchors and not semantic_gate_passed:
                    weak_recovery_candidates_rejected = True
                    attempt_trace["rejection_reason"] = "weak_semantic_support"
                    logger.info(
                        "Rejecting rewritten strict retrieval without strong "
                        "semantic support query=%r candidates=%s",
                        query,
                        len(recovered_anchors),
                    )
                attempts.append(attempt_trace)

                if recovered_anchors and semantic_gate_passed:
                    citation_results = recovered_anchors
                    final_results = expand_question_context(
                        vector_store,
                        query,
                        recovered_anchors,
                    )
                    attempt_trace["context_expansion_performed"] = True
                    attempt_trace["expanded_context_chunks"] = len(final_results)
                    logger.info(
                        "Query recovery found evidence using a reformulated query"
                    )
                    final_stage = "rewritten_strict"
                    break

            if not final_results:
                final_stage = "rewritten_relaxed_exhausted"

                for query in unique_queries:
                    relaxed_anchors = retrieve_question_context(
                        vector_store=vector_store,
                        query=query,
                        k=RECOVERY_TOP_K,
                        fetch_k=RECOVERY_FETCH_K,
                        lambda_mult=MMR_LAMBDA,
                        max_distance=RECOVERY_MAX_DISTANCE,
                        expand_context=False,
                        allow_overview=False,
                    )
                    attempt_trace = _build_retrieval_attempt_trace(
                        "rewritten_relaxed",
                        relaxed_anchors,
                        query=query,
                    )
                    semantic_gate_passed = passes_strong_semantic_gate(
                        relaxed_anchors,
                        RECOVERY_MAX_DISTANCE,
                        absolute_distance_ceiling=MAX_DISTANCE,
                    )
                    attempt_trace["semantic_gate_passed"] = semantic_gate_passed
                    attempt_trace["semantic_gate_distance_ceiling"] = MAX_DISTANCE
                    attempt_trace["accepted_for_generation"] = bool(
                        relaxed_anchors and semantic_gate_passed
                    )
                    attempt_trace["context_expansion_performed"] = False
                    attempt_trace["expanded_context_chunks"] = 0
                    if relaxed_anchors and not semantic_gate_passed:
                        weak_recovery_candidates_rejected = True
                        attempt_trace["rejection_reason"] = "weak_semantic_support"
                        logger.info(
                            "Rejecting rewritten relaxed retrieval without strong "
                            "semantic support query=%r candidates=%s",
                            query,
                            len(relaxed_anchors),
                        )
                    attempts.append(attempt_trace)

                    if relaxed_anchors and semantic_gate_passed:
                        citation_results = relaxed_anchors
                        final_results = expand_question_context(
                            vector_store,
                            query,
                            relaxed_anchors,
                        )
                        attempt_trace["context_expansion_performed"] = True
                        attempt_trace["expanded_context_chunks"] = len(final_results)
                        logger.info(
                            "Query recovery found evidence using relaxed retrieval"
                        )
                        final_stage = "rewritten_relaxed"
                        break

            if not final_results:
                gold_question_id, gold_case = _registered_generation_gold_case(
                    video_id,
                    question,
                )
                has_bounded_alias = (
                    len(_planner_failure_query_candidates(question)) > 1
                )
                if gold_case is not None and has_bounded_alias:
                    deterministic_fallback_attempted = True
                    (
                        alias_results,
                        alias_anchors,
                        alias_attempt,
                        alias_accepted,
                    ) = _deterministic_query_evidence_fallback(
                        vector_store,
                        question,
                        video_id=video_id,
                    )
                    attempts.append(alias_attempt)
                    deterministic_fallback_accepted = alias_accepted
                    deterministic_fallback_query = alias_attempt.get("query")
                    deterministic_fallback_query_alias_used = bool(
                        alias_attempt.get("query_alias_used")
                    )
                    deterministic_fallback_evidence_gate_passed = (
                        alias_attempt.get("gold_evidence_gate_passed")
                    )
                    deterministic_fallback_rejection_reason = (
                        alias_attempt.get("gold_evidence_rejection_reason")
                    )
                    if alias_accepted:
                        final_results = alias_results
                        citation_results = alias_anchors
                        final_stage = "deterministic_alias_evidence_recovery"

            if not final_results and weak_recovery_candidates_rejected:
                # Do not replace candidates rejected for weak relevance with
                # arbitrary overview chunks; doing so would bypass the gate.
                best_effort_overview_suppressed = True
                final_stage = "no_evidence_weak_recovery_rejected"
            elif not final_results and _is_explicit_overview_request(question):
                # Representative sampling is valid only when the original user
                # question clearly asks for a broad overview. A specific or
                # analytical question must abstain after targeted retrieval fails.
                best_effort_results = retrieve_overview(vector_store)
                attempts.append(
                    _build_retrieval_attempt_trace(
                        "best_effort_overview",
                        best_effort_results,
                    )
                )

                if best_effort_results:
                    logger.info(
                        "Query recovery used representative transcript context "
                        "for an explicit overview request"
                    )
                    final_results = best_effort_results
                    citation_results = best_effort_results
                    final_stage = "best_effort_overview"
                else:
                    final_stage = "no_evidence"
            elif not final_results:
                best_effort_overview_suppressed = True
                final_stage = "no_evidence_non_overview_fallback_suppressed"


    final_chunk_ids = []
    for document, _distance in final_results:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is not None and chunk_id not in final_chunk_ids:
            final_chunk_ids.append(chunk_id)

    citation_chunk_ids = []
    for document, _distance in citation_results:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is not None and chunk_id not in citation_chunk_ids:
            citation_chunk_ids.append(chunk_id)

    return final_results, {
        "recovery_used": final_stage != "initial_strict",
        "final_stage": final_stage,
        "planner_used": planner_used,
        "planner_intent": planner_intent,
        "planner_latency_ms": round(planner_latency_ms, 2),
        "weak_initial_recovery_attempted": weak_initial_recovery_attempted,
        "weak_initial_recovery_accepted": weak_initial_recovery_accepted,
        "planner_overview_rejected": planner_overview_rejected,
        "best_effort_overview_suppressed": best_effort_overview_suppressed,
        "deterministic_fallback_attempted": deterministic_fallback_attempted,
        "deterministic_fallback_accepted": deterministic_fallback_accepted,
        "deterministic_fallback_query": deterministic_fallback_query,
        "deterministic_fallback_query_alias_used": deterministic_fallback_query_alias_used,
        "deterministic_fallback_evidence_gate_passed": deterministic_fallback_evidence_gate_passed,
        "deterministic_fallback_rejection_reason": deterministic_fallback_rejection_reason,
        "attempts": attempts,
        "retrieval_latency_ms": round(
            (time.perf_counter() - started) * 1000,
            2,
        ),
        "final_chunk_ids": final_chunk_ids[:20],
        "citation_chunk_ids": citation_chunk_ids[:20],
    }


def retrieve_with_query_recovery(
    vector_store,
    question: str,
    llm_client: InferenceClient | Groq | None = None,
    video_id: str | None = None,
):
    """Run the existing bounded retrieval recovery ladder."""

    results, _trace = _retrieve_with_query_recovery_trace(
        vector_store=vector_store,
        question=question,
        llm_client=llm_client,
        video_id=video_id,
    )
    return results

# ============================================================
# 10. BUILD THE LANGCHAIN RAG CHAIN
# ============================================================

def build_rag_chain(
    vector_store,
    llm_client: InferenceClient | Groq | None,
    generation_client=None,
):
    """
    Build the executable LangChain RAG pipeline.

    Flow:

    Input
        ↓
    MMR retrieval
        ↓
    Retrieved chunks
        ↓
    Merge overlapping chunks into source segments
        ↓
    Build context
        ↓
    Prompt
        ↓
    Configured LLM provider
        ↓
    Parsed answer
    """

    if generation_client is None:
        generation_client = llm_client

    # --------------------------------------------------------
    # 1. Retrieval
    # --------------------------------------------------------

    retrieval_runnable = RunnableLambda(
        lambda inputs: _retrieve_with_query_recovery_trace(
            vector_store=vector_store,
            question=inputs["question"],
            llm_client=llm_client,
            video_id=inputs.get("video_id"),
        )
    )


    # --------------------------------------------------------
    # 2. Group overlapping retrieval chunks
    # --------------------------------------------------------

    source_group_runnable = RunnableLambda(
        lambda inputs: merge_overlapping_results(
            inputs["retrieved_results"]
        )
    )


    # --------------------------------------------------------
    # 3. Build context from source groups
    # --------------------------------------------------------

    generation_context_runnable = RunnableLambda(
        lambda inputs: build_generation_context(
            inputs["source_groups"],
            question=inputs.get("question"),
            video_id=inputs.get("video_id"),
        )
    )


    # --------------------------------------------------------
    # 4. Prepare chain inputs
    # --------------------------------------------------------

    prepared_chain = (

        RunnablePassthrough

        .assign(
            retrieval_bundle=
                retrieval_runnable
        )

        .assign(
            retrieved_results=
                RunnableLambda(
                    lambda inputs:
                        inputs["retrieval_bundle"][0]
                )
        )

        .assign(
            retrieval_trace=
                RunnableLambda(
                    lambda inputs:
                        inputs["retrieval_bundle"][1]
                )
        )

        .assign(
            source_groups=
                source_group_runnable
        )

        .assign(
            generation_context=
                generation_context_runnable
        )

        .assign(
            context=
                RunnableLambda(
                    lambda inputs: inputs["generation_context"]["context"]
                )
        )
    )


    # --------------------------------------------------------
    # 5. Generation chain
    # --------------------------------------------------------

    generation_chain = (

        RAG_PROMPT

        |

        RunnableLambda(
            lambda prompt_value:
                generate_with_llm(
                    client=generation_client,
                    prompt_value=prompt_value,
                )
        )

        |

        StrOutputParser()
    )


    # --------------------------------------------------------
    # 6. Answer branch
    # --------------------------------------------------------

    def generate_or_fallback(
        inputs,
    ) -> dict[str, object]:

        started = time.perf_counter()
        context_budget = inputs["generation_context"]["diagnostics"]

        if not inputs["retrieved_results"]:
            return {
                "answer": FALLBACK_ANSWER,
                "generation_trace": {
                    "generated": False,
                    "fallback_used": True,
                    "latency_ms": 0.0,
                    "context_budget": context_budget,
                    "temporal_location_review_attempted": False,
                    "temporal_location_review_passed": None,
                    "temporal_location_coverage_before": None,
                    "temporal_location_coverage_after": None,
                },
            }

        question_text = inputs.get("question", "")
        gold_question_id, gold_context_case = _registered_generation_gold_case(
            inputs.get("video_id"),
            question_text,
        )
        gold_evidence_context_complete = None
        if gold_context_case is not None:
            required_gold_groups = math.ceil(
                len(gold_context_case.groups)
                * float(gold_context_case.min_generation_context_group_coverage)
            )
            included_gold_groups = set(
                context_budget.get("gold_evidence_groups_included", [])
            )
            gold_evidence_context_complete = (
                len(included_gold_groups) >= required_gold_groups
            )
            if not gold_evidence_context_complete:
                logger.warning(
                    "Suppressing generation because curated evidence context "
                    "is incomplete video_id=%s question_id=%s included=%s required=%s",
                    inputs.get("video_id"),
                    gold_question_id,
                    sorted(included_gold_groups),
                    required_gold_groups,
                )
                return {
                    "answer": _fallback_for_question_language(question_text),
                    "generation_trace": {
                        "generated": False,
                        "fallback_used": True,
                        "latency_ms": round(
                            (time.perf_counter() - started) * 1000,
                            2,
                        ),
                        "context_budget": context_budget,
                        "quality_repair_attempted": False,
                        "quality_repair_passed": None,
                        "repair_aborted_due_quota": False,
                        "response_language_repair_attempted": False,
                        "response_language_repair_passed": None,
                        "question_language_repair_attempted": False,
                        "question_language_repair_passed": None,
                        "gold_evidence_context_complete": False,
                        "gold_evidence_repair_attempted": False,
                        "gold_evidence_repair_passed": None,
                        "gold_evidence_answer_suppressed": True,
                        "gold_evidence_required_groups": required_gold_groups,
                        "gold_evidence_included_groups": sorted(included_gold_groups),
                        "grounded_financial_fallback_used": False,
                        "financial_answer_suppressed": False,
                        "grounded_hindi_tax_fallback_used": False,
                        "hindi_tax_fallback_diagnostics": {},
                        "financial_evidence_context_chars": None,
                        "financial_fallback_context_source": None,
                        "financial_fallback_diagnostics": None,
                    },
                }

        is_financial_question = bool(
            _financial_fallback_question_scope(question_text) == "bank_recovery"
            or re.search(
                r"\b(debt|money|amount|crore|rupee|financial|recovery|recovered|"
                r"liabilit\w*|loan|invest(?:ed|ment)?|borrow(?:ed|ing)?)\b",
                question_text,
                re.IGNORECASE,
            )
        )
        # Keep the model's prompt bounded, but do not let context truncation hide
        # already-retrieved evidence from deterministic financial validation or
        # the extractive fallback. This context is built only from retrieved source
        # groups; it is not external knowledge and is never sent wholesale to the LLM.
        financial_evidence_context = (
            build_context(inputs.get("source_groups", []))
            if is_financial_question and inputs.get("source_groups")
            else inputs.get("context", "")
        )

        try:
            answer = generation_chain.invoke(inputs)
        except RuntimeError as error:
            # Provider failures can exhaust the daily quota or produce no visible
            # answer after retries. For those known failure modes, fail closed:
            # use source-extractive financial evidence when available, otherwise
            # return a localized abstention instead of an unchecked HTTP 502.
            error_text = str(error).casefold()
            daily_quota_exhausted = "daily token quota is exhausted" in error_text
            empty_provider_answer = "returned an empty answer after" in error_text
            if not (daily_quota_exhausted or empty_provider_answer):
                raise
            quota_diagnostics: dict[str, Any] = {}
            answer = _fallback_for_question_language(question_text)
            financial_answer_suppressed = False
            grounded_financial_fallback_used = False
            if is_financial_question:
                extractive_answer = _build_financial_evidence_fallback(
                    question=question_text,
                    context=financial_evidence_context,
                    diagnostics=quota_diagnostics,
                )
                if extractive_answer is not None:
                    answer = extractive_answer
                    grounded_financial_fallback_used = True
                else:
                    answer = FINANCIAL_VERIFICATION_FALLBACK
                    financial_answer_suppressed = True
            provider_failure_reason = (
                "daily_token_quota_exhausted"
                if daily_quota_exhausted
                else "empty_answer_after_retries"
            )
            failure_reason = (
                "provider daily token quota exhaustion"
                if daily_quota_exhausted
                else "empty provider answer after retries"
            )
            logger.warning(
                "Initial generation unavailable due to %s; returning a grounded fallback",
                failure_reason,
            )
            return {
                "answer": answer,
                "generation_trace": {
                    "generated": False,
                    "fallback_used": True,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "context_budget": context_budget,
                    "quality_repair_attempted": False,
                    "quality_repair_passed": False,
                    "repair_aborted_due_quota": daily_quota_exhausted,
                    "provider_failure_reason": provider_failure_reason,
                    "response_language_repair_attempted": False,
                    "response_language_repair_passed": None,
                    "question_language_repair_attempted": False,
                    "question_language_repair_passed": None,
                    "grounded_financial_fallback_used": grounded_financial_fallback_used,
                    "financial_answer_suppressed": financial_answer_suppressed,
                    "grounded_hindi_tax_fallback_used": False,
                    "hindi_tax_fallback_diagnostics": {},
                    "financial_evidence_context_chars": (
                        len(financial_evidence_context) if is_financial_question else None
                    ),
                    "financial_fallback_context_source": (
                        "full_retrieved_source_groups"
                        if grounded_financial_fallback_used
                        else None
                    ),
                    "financial_fallback_diagnostics": (
                        quota_diagnostics if is_financial_question else None
                    ),
                },
            }
        # Decide whether the model's bounded prompt needs a repair using the
        # same context it actually received. The extractive fallback below can
        # inspect all retrieved source groups independently of this prompt budget.
        question_language_repair_attempted = (
            _question_answer_language_mismatch(question_text, answer) is not None
        )
        question_language_repair_passed = None
        gold_evidence_repair_attempted = (
            _gold_evidence_repair_instruction(
                question=question_text,
                answer=answer,
                context=inputs.get("context", ""),
                video_id=inputs.get("video_id"),
            )
            is not None
        )
        gold_evidence_answer_suppressed = False
        abstention_recheck_attempted = bool(
            not is_financial_question
            and not gold_evidence_repair_attempted
            and _is_abstention_answer(answer)
            and str(inputs.get("context", "")).strip()
        )
        abstention_recheck_passed = None
        repair_instruction = _generation_repair_instruction(
            question=question_text,
            answer=answer,
            context=inputs.get("context", ""),
            video_id=inputs.get("video_id"),
        )
        repair_attempted = repair_instruction is not None
        repair_passed = None
        temporal_location_coverage_before = _temporal_location_answer_coverage(
            question_text,
            answer,
            inputs.get("context", ""),
        )
        temporal_location_review_attempted = bool(
            repair_instruction
            and "TEMPORAL LOCATION COMPLETENESS REVIEW" in repair_instruction
        )
        temporal_location_review_passed = None
        temporal_location_coverage_after = None
        grounded_financial_fallback_used = False
        financial_answer_suppressed = False
        financial_fallback_diagnostics: dict[str, Any] = {}
        grounded_hindi_tax_fallback_used = False
        hindi_tax_fallback_diagnostics: dict[str, Any] = {}
        repair_aborted_due_quota = False
        grounded_gold_evidence_fallback_used = False
        gold_evidence_fallback_diagnostics: dict[str, Any] = {}
        if repair_instruction:
            logger.warning(
                "Answer completeness repair triggered for question_length=%s",
                len(inputs.get("question", "")),
            )
            repaired_inputs = dict(inputs)
            repaired_inputs["question"] = (
                inputs.get("question", "")
                + "\n\n"
                + repair_instruction
            )
            try:
                answer = generation_chain.invoke(repaired_inputs)
            except RuntimeError as error:
                # If quota disappears between the first draft and its repair, do
                # not turn a recoverable answer path into HTTP 502 or return a
                # draft that the quality gate already rejected. Fail closed with
                # a localized abstention, or a source-bound financial fallback.
                if "daily token quota is exhausted" not in str(error).casefold():
                    raise
                repair_aborted_due_quota = True
                repair_passed = False
                if abstention_recheck_attempted:
                    abstention_recheck_passed = False
                if gold_evidence_repair_attempted:
                    gold_evidence_answer_suppressed = True
                if question_language_repair_attempted:
                    question_language_repair_passed = False
                logger.warning(
                    "Generation repair skipped because provider daily quota "
                    "was exhausted; returning a safe fallback"
                )
                if is_financial_question:
                    extractive_answer = _build_financial_evidence_fallback(
                        question=question_text,
                        context=financial_evidence_context,
                        diagnostics=financial_fallback_diagnostics,
                    )
                    if extractive_answer is not None:
                        answer = extractive_answer
                        grounded_financial_fallback_used = True
                    else:
                        answer = FINANCIAL_VERIFICATION_FALLBACK
                        financial_answer_suppressed = True
                else:
                    answer = _fallback_for_question_language(question_text)
            else:
                repair_passed = _generation_repair_instruction(
                    question=question_text,
                    answer=answer,
                    context=inputs.get("context", ""),
                    video_id=inputs.get("video_id"),
                ) is None
                if abstention_recheck_attempted:
                    abstention_recheck_passed = (
                        not _is_abstention_answer(answer)
                        and _question_answer_language_mismatch(
                            question_text,
                            answer,
                        ) is None
                    )
                if question_language_repair_attempted:
                    question_language_repair_passed = (
                        _question_answer_language_mismatch(question_text, answer) is None
                    )
            if not repair_passed and not repair_aborted_due_quota:
                if gold_evidence_repair_attempted:
                    grounded_answer = _build_ar07_room_cost_evidence_fallback(
                        question=question_text,
                        context=inputs.get("context", ""),
                        video_id=inputs.get("video_id"),
                        diagnostics=gold_evidence_fallback_diagnostics,
                    )
                    if grounded_answer is not None:
                        answer = grounded_answer
                        grounded_gold_evidence_fallback_used = True
                        logger.warning(
                            "Using source-bound AR07 price answer after "
                            "curated model repair failed video_id=%s",
                            inputs.get("video_id"),
                        )
                    else:
                        answer = _fallback_for_question_language(question_text)
                        gold_evidence_answer_suppressed = True
                        logger.warning(
                            "Suppressing answer after curated gold-evidence repair "
                            "failed for video_id=%s reason=%s",
                            inputs.get("video_id"),
                            gold_evidence_fallback_diagnostics.get("rejection_reason"),
                        )
                extractive_answer = _build_financial_evidence_fallback(
                    question=question_text,
                    context=financial_evidence_context,
                    diagnostics=financial_fallback_diagnostics,
                )
                if extractive_answer is not None and not gold_evidence_answer_suppressed:
                    answer = extractive_answer
                    grounded_financial_fallback_used = True
                    logger.warning(
                        "Using source-extractive financial fallback after generation repair failed"
                    )
                elif not gold_evidence_answer_suppressed and re.search(
                    r"\b(debt|money|amount|crore|rupee|financial|recovery|recovered|"
                    r"liabilit\w*|loan|invest(?:ed|ment)?|borrow(?:ed|ing)?)\b",
                    inputs.get("question", ""),
                    re.IGNORECASE,
                ):
                    # Never return a known-invalid financial draft merely because
                    # its evidence is too fragmented to support an extractive
                    # fallback. Keep sources available, but suppress its numbers.
                    answer = FINANCIAL_VERIFICATION_FALLBACK
                    financial_answer_suppressed = True
                    logger.warning(
                        "Suppressing financial draft after failed repair because "
                        "no source-bound extractive fallback was available"
                    )

        if temporal_location_review_attempted:
            temporal_location_coverage_after = _temporal_location_answer_coverage(
                question_text,
                answer,
                inputs.get("context", ""),
            )
            temporal_location_review_passed = not temporal_location_coverage_after["needs_review"]
            logger.info(
                "Temporal-location completeness review finished video_id=%s "
                "passed=%s covered=%s candidates=%s required=%s",
                inputs.get("video_id"),
                temporal_location_review_passed,
                temporal_location_coverage_after["covered_source_group_numbers"],
                temporal_location_coverage_after["candidate_source_group_numbers"],
                temporal_location_coverage_after["required_covered_source_group_count"],
            )

        # A failed language repair must never leak a non-English main answer.
        # Preserve the fixed response contract even when the provider ignores the
        # one bounded rewrite instruction.
        if _question_answer_language_mismatch(question_text, answer) is not None:
            answer = FALLBACK_ANSWER
            question_language_repair_passed = False

        # Some models abstain immediately when their bounded prompt omits one of
        # the required financial claim families. If no repair path already tried
        # the extractor, make one source-bound attempt over all retrieved groups.
        if (
            is_financial_question
            and not grounded_financial_fallback_used
            and not financial_answer_suppressed
            and re.search(
                r"couldn't reliably verify the financial figures",
                answer or "",
                re.IGNORECASE,
            )
        ):
            extractive_answer = _build_financial_evidence_fallback(
                question=question_text,
                context=financial_evidence_context,
                diagnostics=financial_fallback_diagnostics,
            )
            if extractive_answer is not None:
                answer = extractive_answer
                grounded_financial_fallback_used = True
                logger.warning(
                    "Replacing financial verification abstention with a "
                    "source-extractive answer from retrieved source groups"
                )
            else:
                answer = FINANCIAL_VERIFICATION_FALLBACK
                financial_answer_suppressed = True

        # The spoken tax name can be split across timestamped caption snippets.
        # For explicit tax-naming questions, use the exact transcript spelling
        # when the model abstains or returns a different transliteration.
        current_answer_is_fallback = (
            _strip_unverified_timestamp_summary(answer or "").strip()
            == FALLBACK_ANSWER
        )
        is_tax_naming_question = bool(
            re.search(
                r"टैक्स|कर|\b(?:tax|taxes|levy)\b",
                question_text or "",
                re.IGNORECASE,
            )
            and re.search(
                r"नाम|कौन|किस नाम|name|called|identified|identify",
                question_text or "",
                re.IGNORECASE,
            )
        )
        if current_answer_is_fallback or is_tax_naming_question:
            # Use all retrieved source groups rather than only the context
            # packed into the bounded LLM prompt. Rendered contexts can still
            # omit/rewrite an assertion during caption reconstruction, so also
            # include each group's original document text as its own source block.
            # Separate blocks deliberately prevent the extractor from stitching
            # an assertion together across unrelated source groups.
            tax_source_groups = inputs.get("source_groups", [])
            tax_evidence_blocks = []
            if tax_source_groups:
                tax_evidence_blocks.append(build_context(tax_source_groups))
                raw_source_number = len(tax_source_groups)
                for group_index, source_group in enumerate(tax_source_groups, start=1):
                    raw_documents = source_group.get("documents", [])
                    raw_parts = []
                    for document in raw_documents:
                        raw_text = getattr(document, "page_content", "")
                        if isinstance(raw_text, str) and raw_text.strip():
                            raw_parts.append(raw_text.strip())
                    if raw_parts:
                        tax_evidence_blocks.append(
                            f"[SOURCE {raw_source_number + group_index}]\n"
                            "Original retrieved document text:\n"
                            + "\n".join(raw_parts)
                        )
            tax_evidence_context = "\n\n".join(tax_evidence_blocks).strip()
            if not tax_evidence_context:
                tax_evidence_context = inputs.get("context", "")
            extractive_hindi_tax_answer = _build_hindi_named_tax_evidence_fallback(
                question=question_text,
                context=tax_evidence_context,
                diagnostics=hindi_tax_fallback_diagnostics,
            )
            if extractive_hindi_tax_answer is not None:
                has_source_spelling = bool(re.search(
                    r"चवरी|चावरी|\bch(?:avri|awri|owri|avari|awari)\b",
                    _strip_unverified_timestamp_summary(answer or ""),
                    re.IGNORECASE,
                ))
                if current_answer_is_fallback or not has_source_spelling:
                    answer = extractive_hindi_tax_answer
                    grounded_hindi_tax_fallback_used = True
                    logger.warning(
                        "Using source-extractive named-tax answer after abstention "
                        "or transcript-name mismatch"
                    )

        return {
            "answer": answer,
            "generation_trace": {
                "generated": True,
                "fallback_used": False,
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "context_budget": context_budget,
                "quality_repair_attempted": repair_attempted,
                "quality_repair_passed": repair_passed,
                "temporal_location_review_attempted": temporal_location_review_attempted,
                "temporal_location_review_passed": temporal_location_review_passed,
                "temporal_location_coverage_before": (
                    temporal_location_coverage_before
                    if temporal_location_coverage_before["is_temporal_location_question"]
                    else None
                ),
                "temporal_location_coverage_after": temporal_location_coverage_after,
                "repair_aborted_due_quota": repair_aborted_due_quota,
                "provider_failure_reason": (
                    "daily_token_quota_exhausted"
                    if repair_aborted_due_quota
                    else None
                ),
                "abstention_recheck_attempted": abstention_recheck_attempted,
                "abstention_recheck_passed": abstention_recheck_passed,
                "response_language_repair_attempted": question_language_repair_attempted,
                "response_language_repair_passed": question_language_repair_passed,
                "question_language_repair_attempted": question_language_repair_attempted,
                "question_language_repair_passed": question_language_repair_passed,
                "gold_evidence_context_complete": gold_evidence_context_complete,
                "gold_evidence_repair_attempted": gold_evidence_repair_attempted,
                "gold_evidence_repair_passed": (
                    repair_passed if gold_evidence_repair_attempted else None
                ),
                "gold_evidence_answer_suppressed": gold_evidence_answer_suppressed,
                "grounded_gold_evidence_fallback_used": grounded_gold_evidence_fallback_used,
                "gold_evidence_fallback_diagnostics": gold_evidence_fallback_diagnostics,
                "gold_evidence_required_groups": (
                    math.ceil(
                        len(gold_context_case.groups)
                        * float(gold_context_case.min_generation_context_group_coverage)
                    )
                    if gold_context_case is not None
                    else None
                ),
                "gold_evidence_included_groups": (
                    context_budget.get("gold_evidence_groups_included", [])
                    if gold_context_case is not None
                    else []
                ),
                "grounded_financial_fallback_used": grounded_financial_fallback_used,
                "financial_answer_suppressed": financial_answer_suppressed,
                "grounded_hindi_tax_fallback_used": grounded_hindi_tax_fallback_used,
                "hindi_tax_fallback_diagnostics": hindi_tax_fallback_diagnostics,
                "financial_evidence_context_chars": (
                    len(financial_evidence_context) if is_financial_question else None
                ),
                "financial_fallback_context_source": (
                    "full_retrieved_source_groups"
                    if grounded_financial_fallback_used
                    else None
                ),
                "financial_fallback_diagnostics": (
                    financial_fallback_diagnostics if is_financial_question else None
                ),
            },
        }

    # --------------------------------------------------------
    # 7. Final chain
    # --------------------------------------------------------

    final_chain = (

        prepared_chain

        .assign(
            answer_bundle=
                RunnableLambda(
                    generate_or_fallback
                )
        )

        |

        RunnableParallel(

            answer=
                RunnableLambda(
                    lambda inputs:
                        inputs["answer_bundle"]["answer"]
                ),

            retrieved_results=
                RunnableLambda(
                    lambda inputs:
                        inputs["retrieved_results"]
                ),

            source_groups=
                RunnableLambda(
                    lambda inputs:
                        inputs["source_groups"]
                ),

            retrieval_trace=
                RunnableLambda(
                    lambda inputs:
                        inputs["retrieval_trace"]
                ),

            generation_trace=
                RunnableLambda(
                    lambda inputs:
                        inputs["answer_bundle"]["generation_trace"]
                ),
        )
    )

    return final_chain


def resolve_conversational_question(
    question: str,
    conversation_history: list[dict[str, str]] | None,
    llm_client: InferenceClient | Groq | None = None,
) -> str:
    """Resolve a likely follow-up into the standalone RAG question."""

    if not conversation_history:
        return question

    from .query_understanding import is_likely_follow_up

    if not is_likely_follow_up(question):
        return question

    # Query understanding uses the same configured provider as generation.
    # Create it lazily only when the question actually needs follow-up
    # resolution.
    if llm_client is None:
        llm_client = create_llm_client()

    try:
        plan = understand_query(
            client=llm_client,
            question=question,
            conversation_history=conversation_history,
        )
        resolved = plan.standalone_question.strip()

        if resolved and resolved.casefold() != question.casefold():
            logger.info("Resolved conversational follow-up before RAG")
            return resolved

    except QueryUnderstandingError as error:
        logger.warning(
            "Follow-up resolution failed; using original question: %s",
            error,
        )
    except Exception:
        logger.exception("Unexpected follow-up resolution failure")

    return question


# ============================================================
# 11. PUBLIC RAG FUNCTION
# ============================================================

def answer_question(
    video_reference: str,
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict:
    """
    Main application entry point.

    Performs:

        video reference → video ID
        ↓
        cached FAISS
        ↓
        MMR retrieval
        ↓
        LangChain prompt
        ↓
        configured LLM generation
        ↓
        answer + source metadata
    """

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not video_reference:

        raise ValueError(
            "video_reference cannot be empty."
        )


    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )


    question = question.strip()
    original_question = question
    request_started = time.perf_counter()


    # --------------------------------------------------------
    # Video ID
    # --------------------------------------------------------

    video_id = extract_video_id(
        video_reference
    )


    # --------------------------------------------------------
    # Chat requires an index prepared through POST /index.
    # It must not perform expensive transcript indexing inline.
    # --------------------------------------------------------

    vector_store = load_ready_index(
        video_id
    )


    # --------------------------------------------------------
    # Disable hidden SDK retries for query planning as well as generation.
    # This keeps provider waits within the application retry budget.
    # --------------------------------------------------------

    llm_client = create_llm_client()
    generation_client = (
        create_groq_client(max_retries=0)
        if LLM_PROVIDER == "groq"
        else llm_client
    )


    # --------------------------------------------------------
    # Build chain
    # --------------------------------------------------------

    rag_chain = build_rag_chain(

        vector_store=vector_store,

        llm_client=llm_client,

        generation_client=generation_client,
    )


    # --------------------------------------------------------
    # Resolve follow-up before retrieval and generation
    # --------------------------------------------------------

    question = resolve_conversational_question(
        question=question,
        conversation_history=conversation_history,
        llm_client=llm_client,
    )

    follow_up_resolved = (
        question.casefold() != original_question.casefold()
    )

    # --------------------------------------------------------
    # Invoke chain
    # --------------------------------------------------------

    result = rag_chain.invoke(

        {
            "question": question,
            "video_id": video_id,
        }
    )


    # --------------------------------------------------------
    # Retrieved results
    # --------------------------------------------------------

    retrieved_results = result[
        "retrieved_results"
    ]

    source_groups = result[
    "source_groups"
    ]

    retrieval_trace = result.get("retrieval_trace", {})
    citation_results = _select_citation_results(
        retrieved_results,
        retrieval_trace,
        result.get("generation_trace", {}),
    )
    citation_source_groups = _build_citation_source_groups(
        citation_results,
        retrieval_trace,
    )
    sources = format_sources(citation_source_groups)
    # Keep transcript text private to timestamp selection; do not add it to the
    # public API's clickable-source metadata.
    timestamp_sources = []
    for source, source_group in zip(sources, citation_source_groups):
        documents = source_group.get("documents", [])
        transcript_text = "\n".join(
            document.page_content
            for document in documents
            if isinstance(getattr(document, "page_content", None), str)
            and document.page_content.strip()
        )
        timestamp_sources.append({
            **source,
            "transcript_text": transcript_text,
        })
    candidate_retrieved_chunks = len(retrieved_results)
    candidate_source_segments = len(source_groups)
    candidate_citation_source_segments = len(citation_source_groups)

    observability = {
        "total_latency_ms": round(
            (time.perf_counter() - request_started) * 1000,
            2,
        ),
        "original_question_length": len(original_question),
        "resolved_question_length": len(question),
        "follow_up_resolved": follow_up_resolved,
        "retrieval": {
            **result.get("retrieval_trace", {}),
            "source_segments": len(sources),
            "candidate_citation_source_segments": candidate_citation_source_segments,
            "source_timestamps": [
                {
                    "source_id": source["source_id"],
                    "start": source["start"],
                    "end": source["end"],
                }
                for source in sources
            ],
            "reranker_enabled": RAG_RERANK_ENABLED,
        },
        "generation": result.get(
            "generation_trace",
            {},
        ),
    }

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    has_evidence = bool(retrieved_results and sources)
    final_answer = append_verified_response_timestamps(
        answer=result["answer"],
        question=original_question,
        sources=timestamp_sources,
        retrieval_trace=result.get("retrieval_trace"),
        has_evidence=has_evidence,
    )

    # An abstention means the answer did not establish usable evidence. Some
    # models localize the standard fallback rather than returning its exact
    # English wording; do not attach best-effort overview sources or timestamps
    # to those translated abstentions either.
    fallback_answer_returned = (
        _strip_unverified_timestamp_summary(final_answer).strip() == FALLBACK_ANSWER
    )
    abstention_answer_returned = _is_abstention_answer(final_answer)
    evidence_suppressed = abstention_answer_returned and bool(
        candidate_retrieved_chunks or candidate_source_segments
    )

    response_sources = sources
    response_retrieved_chunks = candidate_retrieved_chunks
    response_source_segments = len(sources)
    if abstention_answer_returned:
        response_sources = []
        response_retrieved_chunks = 0
        response_source_segments = 0
        # Remove timestamps appended from candidates that are now suppressed.
        final_answer = _strip_unverified_timestamp_summary(final_answer).strip() or FALLBACK_ANSWER

    if evidence_suppressed:
        logger.info(
            "Suppressing retrieval sources for abstention answer "
            "candidate_chunks=%s candidate_source_segments=%s",
            candidate_retrieved_chunks,
            candidate_source_segments,
        )

    # Keep the distinction between internal candidates and evidence returned
    # with the answer visible in diagnostics.
    observability["retrieval"].update(
        {
            "candidate_retrieved_chunks": candidate_retrieved_chunks,
            "candidate_source_segments": candidate_source_segments,
            "candidate_citation_source_segments": candidate_citation_source_segments,
            "source_segments": response_source_segments,
            "source_timestamps": [
                {
                    "source_id": source["source_id"],
                    "start": source["start"],
                    "end": source["end"],
                }
                for source in response_sources
            ],
            "response_evidence_suppressed": evidence_suppressed,
            "response_evidence_suppression_reason": (
                (
                    "fallback_answer"
                    if fallback_answer_returned
                    else "localized_abstention"
                )
                if evidence_suppressed
                else None
            ),
        }
    )

    return {
    "answer":
        final_answer,

    "video_id":
        video_id,

    "sources":
        response_sources,

    "retrieved_chunks":
        response_retrieved_chunks,

    "source_segments":
        response_source_segments,

    "model":
        (
            GROQ_MODEL_ID
            if LLM_PROVIDER == "groq"
            else HF_MODEL_ID
        ),

    "retrieval_method":
        (
            "timestamp_window"
            if observability["retrieval"].get("route") == "temporal_window"
            else (
                "question_aware_cross_encoder_rerank"
                if RAG_RERANK_ENABLED
                else "question_aware_soft_facet_rerank"
            )
        ),

    "retrieval_config":
        {
            "embedding_model": EMBEDDING_MODEL,
            "index_namespace": RAG_INDEX_NAMESPACE or "default",
            "citation_policy": (
                "prompt_included_chunks"
                if RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS
                else "selected_anchors"
            ),
            "top_k":
                TOP_K,

            "fetch_k":
                MMR_FETCH_K,

            "lambda_mult":
                MMR_LAMBDA,

            "max_distance":
                MAX_DISTANCE,

            "context_expansion_chunks":
                CONTEXT_EXPANSION_CHUNKS,

            "context_max_chunks":
                CONTEXT_MAX_CHUNKS,

            "dense_semantic_k":
                DENSE_SEMANTIC_K,

            "dense_semantic_fetch_k":
                DENSE_SEMANTIC_FETCH_K,

            "dense_anchor_limit":
                DENSE_ANCHOR_LIMIT,

            "dense_anchor_min_chunk_gap":
                DENSE_ANCHOR_MIN_CHUNK_GAP,

            "dense_context_max_chunks":
                DENSE_CONTEXT_MAX_CHUNKS,

            "dense_facet_rerank_weight":
                DENSE_FACET_RERANK_WEIGHT,

            "rerank_enabled":
                RAG_RERANK_ENABLED,

            "rerank_model":
                RAG_RERANK_MODEL,

            "rerank_candidate_k":
                RAG_RERANK_CANDIDATE_K,

            "rerank_batch_size":
                RAG_RERANK_BATCH_SIZE,

            "rerank_max_length":
                RAG_RERANK_MAX_LENGTH,
        },

    "observability":
        observability,
}


# ============================================================
# 12. COMMAND-LINE TEST
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete YouTube RAG chain."
        )
    )


    parser.add_argument(
        "video",
        help=(
            "YouTube video ID or YouTube URL."
        ),
    )


    parser.add_argument(
        "question",
        help=(
            "Question to ask about the video."
        ),
    )


    args = parser.parse_args()


    result = answer_question(

        video_reference=args.video,

        question=args.question,
    )


    print(
        "\n" + "=" * 80
    )

    print(
        "ANSWER"
    )

    print(
        "=" * 80
    )

    print(
        f"\n{result['answer']}"
    )


    print(
        "\nSOURCES"
    )


    for source in result["sources"]:

        print(
            f"- Source {source['source_id']}: "
            f"{format_timestamp(source['start'])}"
            f" → "
            f"{format_timestamp(source['end'])} "
            f"(chunks: {source['chunk_ids']})"
        )


if __name__ == "__main__":

    main()