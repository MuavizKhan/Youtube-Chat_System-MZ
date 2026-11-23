# General Video Answer Quality (Phase 13)

## Purpose

This phase improves transcript-grounded answers across unfamiliar videos and
subjects. It does not contain rules for particular creators, video IDs, names,
studies, companies, games, or financial examples.

It builds on Phase 12 browser-assisted transcript acquisition and preserves the
accepted Phase 11 retrieval, embedding, chunking, and evidence thresholds.

## 1. Conversation-aware follow-up retrieval

A bounded query-understanding plan may use recent conversation turns to resolve
references and retain up to two alternate search formulations. The prior
conversation is used only to understand the user's wording; it is never treated
as transcript evidence.

Each alternate formulation is independently searched against the current
video. Results are admitted only through the existing strict semantic evidence
gate. Accepted anchors are merged with the original strict anchors and expanded
using the established context-expansion code. Weak or rejected alternatives
cannot override the evidence gate.

The follow-up detector covers pronouns and deictic references, as well as common
elliptical follow-ups such as requests for a source, study, result, method,
person, organization, product, event, or place.

## 2. Video-wide overview and full-story requests

Broad overview requests use the indexed transcript documents in chronological
order rather than relying on the targeted-question top-k result set.

Processing is bounded:

- Up to 60 transcript chunks and 48,000 source characters are passed through
  the full-transcript route.
- Larger transcripts are sampled evenly over the complete indexed timeline;
  the response and diagnostics disclose when this happens.
- Selected source passages are grouped into chronological sections, with a
  maximum of 4,500 characters per section and 12 sections.
- Multi-section videos use map/reduce: each section produces a small set of
  source-ID-linked findings, followed by a chronological synthesis.
- Every accepted summary point must cite evidence IDs from supplied passages.
  Reducer output cannot invent a new reference or cite a passage that did not
  support a section finding.

If a section contains no useful information, it may return no findings.
Invalid output or a summary without source-linked points is rejected and the
existing evidence-gated RAG path remains the fallback.

This is a bounded transcript-wide summary, not a promise that every detail of
an arbitrarily long transcript can fit into one model call. Sampling status and
selected fraction are kept in diagnostics so coverage limitations are visible.

## 3. Claim-level answer verification

Compound and fact-sensitive answers are audited after generation using the
existing evidence-assessment contracts. The audit is bounded to at most eight
claims, eight evidence passages, and 8,000 input characters. Trivial short
answers avoid an additional model call.

The audit is restricted to:

1. chunk IDs recorded as included in the generation context; and
2. exact normalized text excerpts that can be found inside the prompt-visible
   transcript context.

A chunk selected by retrieval but omitted from the prompt cannot verify an
answer. Text outside a truncated prompt excerpt cannot verify a claim either.
Each passage retains its video ID, chunk ID, and timestamp interval.

Completed assessments keep supported claims, qualify partially supported
claims, and remove claims assessed as missing, contradicted, or conflicting.
If all claims are rejected, the existing video-evidence abstention is returned.
Supported chunk IDs determine which source links and timestamps are returned.
Claim-to-chunk support is retained in diagnostics.

A provider outage, quota error, or invalid assessment response is explicitly
recorded as an unavailable verification status. It is not labelled as verified
and does not trigger destructive answer rewriting.

This checks whether the transcript supports what the answer says. It does not
authenticate a paper, independently fact-check a speaker, or establish that a
medical, financial, advertising, or other claim is true in the real world.

## 4. Timestamps and sources

Video-wide summaries can retain more than the default three answer-aligned
timestamp ranges. Compound answers can expose up to the number of supported
claims (capped at eight). Every displayed range still comes from returned
transcript-source metadata; model-authored timestamps are not trusted.

Sources are filtered after a completed claim audit to the transcript chunks
supporting retained claims. The internal prompt text used for auditing is not
included in the public API response or observability payload.

## 5. Preserved contracts

This phase does not modify:

- the accepted Phase 11 embedding model or index namespace;
- chunk size, overlap, FAISS indexing, MMR parameters, similarity thresholds,
  reranking configuration, or the standard targeted-question retrieval ladder;
- Phase 12 browser caption acquisition, transcript staging, durable index jobs,
  browser/API fallback, or transcript provenance;
- the existing registered live-audit case requirements.

The new route is used only for explicit broad video-summary requests. Targeted
questions continue to use the established RAG path.

## 6. Generic acceptance coverage

The regression suite covers generic follow-up references, query-plan alternate
retention, strict semantic-gate behavior, full-video and complete-story wording,
chronological map/reduce, explicit timeline sampling, empty sections, mixed
video rejection, model-invented evidence IDs, prompt-visible excerpt boundaries,
claim filtering, exact chunk provenance, and verifier/provider failure modes.
