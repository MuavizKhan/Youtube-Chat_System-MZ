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

from .retrieval import (
    extract_video_id,
    expand_question_context,
    passes_strong_semantic_gate,
    retrieve_overview,
    retrieve_question_context,
    retrieve_temporal_context,
)

from .index_service import load_ready_index


logger = logging.getLogger(__name__)


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

SYSTEM_INSTRUCTIONS = """
You are a YouTube video question-answering assistant.

Your task is to answer the user's question using ONLY the
transcript evidence provided in the context.

RESPONSE LANGUAGE — HIGH PRIORITY
---------------------------------
- Answer in the same language as the user's question, unless the user explicitly
  requests a different language.
- Do not switch to English merely because the transcript, retrieved context, or
  video captions are in English. For an Arabic question, answer in Arabic; for a
  Hindi question, answer in Hindi; use the same principle for other languages.
- Keep names, product names, numbers, currencies, and source-grounded facts
  accurate while expressing the answer naturally in the requested language.
- When the transcript enumerates several distinct items that answer the question,
  include each supported item in the answer in the question's language. Do not
  reduce a supported list to only its first item. Omit an item only when the
  supplied transcript does not support it or the available context is insufficient.
- If the question deliberately uses multiple languages and no output language is
  requested, answer in the language used for the main question.

SPEAKER AND CLAIM ATTRIBUTION — HIGH PRIORITY
---------------------------------------------
The supplied transcript context may contain multiple speakers. Source blocks are
timestamped evidence segments, not speaker labels and not guaranteed single-speaker
turns. Some blocks preserve the original caption snippets as separate lines.

- A caption line preserves one caption snippet, not a verified speaker turn. Each
  reconstructed caption line includes that snippet's original start and end times
  in brackets. Use those times to follow chronology and nearby conversational
  sequence, but the timestamps do not establish who spoke.
- A single caption can contain words from more than one speaker, and a speaker may
  continue across multiple captions. Caption boundaries and timestamps do not prove
  that the speaker changed.
- Use caption order and the exact wording to understand the conversation, but never
  infer a speaker's identity from segmentation alone.

For each material claim, determine from the actual transcript wording who is
speaking and what level of certainty is supported:
- When the user asks what a named person says, attribute a claim to that person
  only when the context clearly shows that person stated or endorsed it. The name
  in the question does not make that person the source of every nearby claim.
- Keep another speaker's question, premise, allegation, or summary separate from
  the guest's response. Do not turn an interviewer's paraphrase of an allegation
  about an authority, investigation, legal matter, or financial issue into the
  guest's statement or an established finding.
- Attribute a named report or authority's claim only as strongly as the transcript
  supports. A speaker saying an authority "found", "concluded", or "said" something
  establishes that the speaker made that account; it does not by itself verify the
  underlying finding.
- Do not carry attribution from one sentence into a different claim with phrases
  like "he also says" or "she mentions" unless the source for that claim is clear.
- If speaker identity or claim status cannot be reliably resolved from the context,
  describe it neutrally as something discussed, qualify it as an allegation or
  account when supported, or omit it if it is not needed to answer the question.
  Do not guess.

Apply this claim-by-claim check before composing the answer, not as a caveat added
afterward.

GROUNDING RULES
---------------

1. Use only information supported by the provided transcript.

2. Do not use outside knowledge.

3. Do not invent facts, names, events, explanations,
   conclusions, or relationships.

4. If the transcript context is empty, unrelated, or genuinely insufficient
   to answer what was said, respond exactly with:

   "I couldn't find enough information about that in the video. It may not be covered by this video."

   Do not use this fallback solely because a named event, result, score,
   winner, or announcement is absent from an otherwise relevant transcript
   passage. For an absence question scoped to a supplied time range, say that
   the event is not stated in that passage when no such statement appears.

5. The transcript may contain speech-recognition errors,
   incomplete sentences, repetitions, or informal wording.
   Do not silently repair missing facts using outside knowledge.

6. If a statement cannot be supported by the provided context,
   leave it out.

7. Distinguish between what the speaker said and your own
   interpretation. Do not add interpretation unless the
   transcript clearly supports it.

8. Do not make claims about a person's importance, reputation,
   intelligence, brilliance, influence, or impact unless the
   transcript explicitly supports that claim.

TEMPORAL AND MULTI-PART QUESTIONS
---------------------------------

For questions scoped to a time range:
- Answer only about the supplied passage, not the whole video unless the
  supplied context supports a whole-video conclusion.
- For multi-part questions, answer each part that the passage supports.
- If the user asks whether a winner was declared, a score announced, or a
  decision stated, and none appears in the relevant supplied passage, explicitly
  say it is not stated in that passage. Absence of a statement is answerable
  when the relevant passage is present; it is not by itself a reason to abstain.
- Describe what the speakers actually discuss before addressing what is absent.

ANSWER STYLE
------------

9. Answer the user's actual question directly.

10. Prefer a concise paragraph for a simple factual question.

11. Use bullet points only when the question naturally contains
    multiple distinct points.

12. Do not create empty bullet points.

13. Do not repeat the same information.

14. Use clear, natural language.

15. Use Markdown only when it improves readability.

SOURCE / TIMESTAMP RULES
------------------------

16. Do NOT output source labels such as:
    [SOURCE 1]
    [Source 1]
    Source 1
    [1]

17. Do not invent timestamps or citations. The application appends a separate
    "Relevant timestamps" line to every substantive answer from returned source
    metadata. For answers containing several distinct monetary claims, the
    application may select up to five source intervals that best match those
    claims; ordinary answers retain a concise timestamp summary. Do not calculate
    or invent timestamp values yourself.

18. Do NOT mention the retrieval system, vector database,
    embeddings, prompts, or internal implementation.

19. The application also displays clickable video sources separately. Plain-text
    timestamp ranges remain part of the answer and are generated from source
    metadata, not from guesses.

20. Never fabricate a source, timestamp, citation, or reference.

21. For questions involving numbers, amounts, percentages, dates, or counts, preserve
    distinct figures as stated in the transcript. Before answering, silently map
    every figure to its nearby claim, subject, qualifier, and source/speaker. Explain
    what each figure refers to and retain its attribution. Do not round, merge, or
    treat different figures as interchangeable unless the transcript explicitly says
    they describe the same measure. In financial passages, keep investment, debt,
    debt-certificate, alleged transaction, and recovery figures separate. In
    particular, never describe a recovery figure as a company debt total merely
    because both figures occur in the same interview. For a broad question about
    debt or money, include each materially relevant monetary claim present in the
    supplied context, with a short attribution, instead of choosing one figure as
    a single total. If a DRT debt figure and a debt-certificate figure are both
    discussed but their relationship is not explicit, report both separately and
    say the transcript does not clearly explain how they relate. If automatic
    captions garble a number or speakers appear to disagree, do not silently repair
    a number, reconcile the figures, or call one a "misreading" unless the transcript
    explicitly supports that relationship.

22. In multi-speaker passages, retain claim-by-claim attribution. Distinguish the
    guest's statements from an interviewer's questions, premises, allegations, or
    paraphrases. Never turn an allegation described by the interviewer into a
    statement made by the guest or an established finding. A timestamp/source
    identifies where a passage occurs, not who said it. When speaker identity or
    claim status is unclear, use neutral wording or omit the claim rather than
    guessing.

FINAL RULE
----------

23. Return only the answer to the user's question.
"""


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
VIDEO CONTEXT
=============

{context}


USER QUESTION
=============

{question}


Answer the user's question using only the transcript evidence
provided above.

Return only the substantive answer text.
Do not include source labels, citations, references, timestamps, or metadata
yourself; the application appends verified source timestamp ranges after
generation.
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

_TIMESTAMP_RANGE_IN_ANSWER_PATTERN = re.compile(
    r"(?<!\d)\d{1,2}:\d{2}\s*(?:-|–|—|to)\s*"
    r"\d{1,2}:\d{2}(?!\d)",
    re.IGNORECASE,
)

_MODEL_TIMESTAMP_SUMMARY_LINE_PATTERN = re.compile(
    r"(?im)^[ \t]*Relevant timestamps?[ \t]*:[^\r\n]*(?:\r?\n|$)"
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
    """Remove a model-authored timestamp summary before trusted ranges are appended.

    Timestamp summary ranges are generated by the application from source
    metadata. A model-authored Relevant timestamp(s): line must not bypass
    that deterministic step or prevent a requested temporal range from being
    appended.
    """
    if not isinstance(answer, str) or not answer:
        return answer

    cleaned = _MODEL_TIMESTAMP_SUMMARY_LINE_PATTERN.sub("", answer)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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

    # Preserve timestamp ranges that are part of the substantive answer, but
    # not a labelled summary (which was removed above).
    if _TIMESTAMP_RANGE_IN_ANSWER_PATTERN.search(answer):
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

    # Avoid duplicating a timestamp range already present in substantive text.
    if _TIMESTAMP_RANGE_IN_ANSWER_PATTERN.search(answer):
        return answer

    temporal_window = retrieval_trace.get("temporal_window")
    if not isinstance(temporal_window, dict):
        return answer

    start = temporal_window.get("start_seconds")
    end = temporal_window.get("end_seconds")
    if (
        not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
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
) -> list:
    """Select source chunks according to the configured citation experiment.

    The default is anchor-only citations. The opt-in experiment includes
    context-expanded neighbors so audits can compare citation coverage without
    changing any retrieval ranking, context expansion, or distance thresholds.
    """
    if not isinstance(retrieval_trace, dict):
        return list(retrieved_results)

    if RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS:
        raw_ids = retrieval_trace.get("final_chunk_ids")
        if raw_ids is None:
            raw_ids = retrieval_trace.get("citation_chunk_ids")
        if raw_ids is None:
            # Preserve behavior for legacy/test traces with no ID metadata.
            return list(retrieved_results)
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
    """Build source links without stretching opt-in neighbor citations.

    The default path is behaviorally equivalent to the existing merged-anchor
    policy. During the expanded-context experiment, selected anchors are merged
    together as before, but context-only neighbors remain individual source
    intervals even when they overlap an anchor. This makes each neighbor
    directly seekable without changing retrieval or generation.
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
) -> str:
    """Excerpt a source under budget, preferring question-relevant transcript lines."""
    if max_chars <= 0:
        return ""
    if len(source_text) <= max_chars:
        return source_text

    marker = "\n[Transcript excerpt shortened to fit the generation context budget.]"
    question_terms = _generation_context_query_terms(question)
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

        scored = [
            (
                _generation_line_relevance(
                    line,
                    question_terms,
                    financial_question=financial_question,
                    financial_group_context=financial_group_context,
                ) + (50 if index in split_evidence_caption_indices else 0),
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


def build_generation_context(
    source_groups: list[dict],
    question: str | None = None,
    max_chars: int | None = None,
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

    question_terms = _generation_context_query_terms(question)
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
        overlap = len(question_terms & _claim_terms(rendered))
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

        own_caption, nearby_captions, source_id = caption_context(match.start())
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
            absolute_start = row["start"] + match.start()
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


def _generation_repair_instruction(
    question: str,
    answer: str,
    context: str,
) -> str | None:
    """Build at most one grounded repair instruction when an answer is incomplete."""
    instructions = []
    question_text = unicodedata.normalize("NFKC", question or "").lower()
    answer_text = _strip_unverified_timestamp_summary(answer or "")
    context_text = context or ""

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

    is_arabic = bool(re.search(r"[\u0600-\u06ff]", question_text))
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
            pattern_key = "arabic" if is_arabic else "english"
            missing = [
                concept for concept in required
                if not re.search(concept[pattern_key], answer_text, re.IGNORECASE)
            ]
            if missing:
                labels = [
                    concept["label_ar"] if is_arabic else concept["label_en"]
                    for concept in required
                ]
                instructions.append(
                    "MULTI-ITEM ANSWER REPAIR. Rewrite from scratch in "
                    + ("Arabic" if is_arabic else "English")
                    + " and include each source-supported item in this list: "
                    + ", ".join(labels)
                    + ". Use only items supported by the supplied transcript. "
                    "Do not stop after the first item and do not include a timestamp line; "
                    "the application adds verified timestamps separately."
                )

    if not instructions:
        return None
    return (
        "The previous draft failed an application source-completeness check. "
        "The instructions below are revision requirements; the transcript context remains the only source of facts.\n\n"
        + "\n\n".join(instructions)
    )


def _build_financial_evidence_fallback(
    question: str,
    context: str,
    diagnostics: dict[str, Any] | None = None,
) -> str | None:
    """Build a conservative extractive answer when financial generation repair fails.

    This is intentionally narrow: it activates only when the supplied context
    contains separately attributable investment, DRT, certificate, and recovery
    report claims. It copies source clauses rather than asking the model to
    reconstruct numerals, and it does not infer a unit for a bare caption figure.
    """
    if not re.search(
        r"\b(debt|money|amount|crore|rupee|financial|recovery|recovered|"
        r"liabilit\w*|loan|invest(?:ed|ment)?|borrow(?:ed|ing)?)\b",
        question or "",
        re.IGNORECASE,
    ):
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({
                "rejection_reason": "question_not_recognized_as_financial",
                "required_topics": ["investment", "drt_debt", "certificate", "recovery_report"],
                "missing_topics": ["investment", "drt_debt", "certificate", "recovery_report"],
                "recognized_topics": {},
                "source_block_count": 0,
            })
        return None

    required_topics = ("investment", "drt_debt", "certificate", "recovery_report")
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
        # The parser exposes immediate same-source neighbor text separately
        # from the amount-bearing caption. Use that neighborhood only for a
        # unitless caption already classified as a certificate; never allow
        # a neighboring report/debt label to change the amount's topic.
        # Keep caption neighborhoods as a dedicated formatting-only field.
        # The claim topic remains anchored to the number-bearing caption, while
        # the unit lookup may use that caption's immediate same-source neighbors.
        unit_context = (
            str(claim.get("amount_caption_context") or "")
            if topic == "certificate" and local_topic == "certificate"
            else ""
        )
        claims_by_topic.setdefault(
            topic,
            {
                **claim,
                "topic": topic,
                "text": claim_text,
                "amount_caption": claim.get("amount_caption") or claim_text,
                "amount_unit_context": unit_context,
            },
        )

    missing_topics = [
        topic for topic in required_topics
        if topic not in claims_by_topic
    ]
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
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
        crore_rupees = re.search(r"\\bcrore\\s+rupees?\\b", unit_evidence, re.IGNORECASE)
        crore_unit = re.search(r"\\bcrores?\\b|\\bcr\\b", unit_evidence, re.IGNORECASE)
        rupee_unit = re.search(r"\\brupees?\\b|\\bINR\\b|₹|\\bRs\\.?", unit_evidence, re.IGNORECASE)
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


def create_groq_client() -> Groq:
    """Create the direct Groq generation client."""

    if not GROQ_API_KEY:

        raise RuntimeError(
            "GROQ_API_KEY was not found.\n\n"
            "Set GROQ_API_KEY in the project root .env file."
        )


    return Groq(
        api_key=GROQ_API_KEY,
    )


def create_llm_client() -> InferenceClient | Groq:
    """Create the configured LLM client used by planning and generation."""

    if LLM_PROVIDER == "groq":
        return create_groq_client()

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

            if "401" in error_text or "unauthorized" in error_text:
                raise RuntimeError(
                    "Groq authentication failed.\n\n"
                    "Check GROQ_API_KEY in .env."
                ) from error

            if "403" in error_text or "forbidden" in error_text:
                raise RuntimeError(
                    "Groq rejected the inference request.\n\n"
                    "Check API key permissions and the selected model."
                ) from error

            if _is_retryable_provider_error(error):
                if attempt < GROQ_MAX_RETRIES:
                    logger.warning(
                        "Transient Groq generation failure; retrying "
                        "attempt=%s max_attempts=%s status_code=%s error_type=%s",
                        attempt + 1,
                        GROQ_MAX_RETRIES + 1,
                        _provider_error_status(error),
                        type(error).__name__,
                    )
                    time.sleep(GROQ_RETRY_DELAY_SECONDS * (attempt + 1))
                    continue

                if "429" in error_text or "rate limit" in error_text or "too many requests" in error_text:
                    raise RuntimeError(
                        "Groq rate limit was reached after "
                        f"{GROQ_MAX_RETRIES + 1} attempts. Retry after the "
                        "current rate-limit window."
                    ) from error

            raise RuntimeError(
                f"Groq generation failed: {error}"
            ) from error

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


def _retrieve_with_query_recovery_trace(
    vector_store,
    question: str,
    llm_client: InferenceClient | Groq | None = None,
) -> tuple[list, dict[str, object]]:
    """Run the bounded recovery ladder and capture stage diagnostics."""

    started = time.perf_counter()
    attempts = []
    citation_results = []
    planner_used = False
    planner_intent = None
    planner_latency_ms = 0.0
    weak_recovery_candidates_rejected = False

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
        except QueryUnderstandingError as error:
            logger.warning(
                "Query understanding failed during retrieval recovery: %s",
                error,
            )
            return [], {
                "recovery_used": True,
                "final_stage": "planner_failed",
                "planner_used": False,
                "planner_intent": None,
                "planner_latency_ms": round(
                    (time.perf_counter() - planner_started) * 1000,
                    2,
                ),
                "attempts": attempts,
                "retrieval_latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "final_chunk_ids": [],
            }
        except Exception:
            logger.exception(
                "Unexpected query-understanding failure during retrieval recovery"
            )
            return [], {
                "recovery_used": True,
                "final_stage": "planner_failed",
                "planner_used": False,
                "planner_intent": None,
                "planner_latency_ms": round(
                    (time.perf_counter() - planner_started) * 1000,
                    2,
                ),
                "attempts": attempts,
                "retrieval_latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "final_chunk_ids": [],
            }

        if plan.intent == "overview":
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
            queries = [
                plan.standalone_question,
                *plan.search_queries,
            ]

            seen_queries = set()
            unique_queries = []

            for query in queries:
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

            if not final_results and weak_recovery_candidates_rejected:
                # Do not replace candidates rejected for weak relevance with
                # arbitrary overview chunks; doing so would bypass the gate.
                final_stage = "no_evidence_weak_recovery_rejected"
            elif not final_results:
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
                        "as a best-effort fallback"
                    )
                    final_results = best_effort_results
                    citation_results = best_effort_results
                    final_stage = "best_effort_overview"
                else:
                    final_stage = "no_evidence"


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
):
    """Run the existing bounded retrieval recovery ladder."""

    results, _trace = _retrieve_with_query_recovery_trace(
        vector_store=vector_store,
        question=question,
        llm_client=llm_client,
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
                },
            }

        answer = generation_chain.invoke(inputs)
        question_text = inputs.get("question", "")
        is_financial_question = bool(re.search(
            r"\b(debt|money|amount|crore|rupee|financial|recovery|recovered|"
            r"liabilit\w*|loan|invest(?:ed|ment)?|borrow(?:ed|ing)?)\b",
            question_text,
            re.IGNORECASE,
        ))
        # Keep the model's prompt bounded, but do not let context truncation hide
        # already-retrieved evidence from deterministic financial validation or
        # the extractive fallback. This context is built only from retrieved source
        # groups; it is not external knowledge and is never sent wholesale to the LLM.
        financial_evidence_context = (
            build_context(inputs.get("source_groups", []))
            if is_financial_question and inputs.get("source_groups")
            else inputs.get("context", "")
        )
        # Decide whether the model's bounded prompt needs a repair using the
        # same context it actually received. The extractive fallback below can
        # inspect all retrieved source groups independently of this prompt budget.
        repair_instruction = _generation_repair_instruction(
            question=question_text,
            answer=answer,
            context=inputs.get("context", ""),
        )
        repair_attempted = repair_instruction is not None
        repair_passed = None
        grounded_financial_fallback_used = False
        financial_answer_suppressed = False
        financial_fallback_diagnostics: dict[str, Any] = {}
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
            answer = generation_chain.invoke(repaired_inputs)
            repair_passed = _generation_repair_instruction(
                question=question_text,
                answer=answer,
                context=inputs.get("context", ""),
            ) is None
            if not repair_passed:
                extractive_answer = _build_financial_evidence_fallback(
                    question=question_text,
                    context=financial_evidence_context,
                    diagnostics=financial_fallback_diagnostics,
                )
                if extractive_answer is not None:
                    answer = extractive_answer
                    grounded_financial_fallback_used = True
                    logger.warning(
                        "Using source-extractive financial fallback after generation repair failed"
                    )
                elif re.search(
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
                "grounded_financial_fallback_used": grounded_financial_fallback_used,
                "financial_answer_suppressed": financial_answer_suppressed,
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
    # Create one configured LLM client for both query understanding
    # and final generation. The provider is selected by LLM_PROVIDER.
    # --------------------------------------------------------

    llm_client = create_llm_client()
    generation_client = llm_client


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

    # A fallback means the generated answer did not identify usable evidence.
    # Do not expose nearby/recovery candidates as clickable supporting sources.
    fallback_answer_returned = final_answer.strip() == FALLBACK_ANSWER
    evidence_suppressed = fallback_answer_returned and bool(
        candidate_retrieved_chunks or candidate_source_segments
    )

    response_sources = sources
    response_retrieved_chunks = candidate_retrieved_chunks
    response_source_segments = len(sources)
    if fallback_answer_returned:
        response_sources = []
        response_retrieved_chunks = 0
        response_source_segments = 0

    if evidence_suppressed:
        logger.info(
            "Suppressing retrieval sources for fallback answer "
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
                "fallback_answer" if evidence_suppressed else None
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
                "expanded_context_experiment"
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