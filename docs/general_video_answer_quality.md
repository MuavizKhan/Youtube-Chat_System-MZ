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

- When the indexed transcript contains at most 100 chunks and 80,000 source
  characters, every valid chunk is processed by the overview map stage.
- Larger transcripts are sampled near evenly spaced timestamps over the
  complete indexed timeline; the response and diagnostics disclose this.
  Sampling is time-based rather than based only on chunk order.
- Selected source passages are grouped into chronological, equal-duration
  timeline sections (normally matching the requested point count), while each
  section stays within the 9,000-character input bound. At most 10 map sections
  are processed.
- Multi-section videos use map/reduce: each non-empty timeline section produces
  up to three source-ID-linked findings, followed by chronological synthesis.
  The map stage is prompted to retain distinct developments within each
  interval, including late conclusions and new examples of earlier themes.
- Every accepted summary point retains its own exact transcript chunk IDs.
  The verifier receives this point-to-source mapping rather than an
  unassociated global pool of candidate passages. Excerpts are selected using
  the vocabulary of the claims that cite them. Mapped alternatives are
  reserved across the timeline before primary points are audited. The reserve
  uses the remaining claim-audit budget, so primary points plus alternatives
  never exceed fourteen candidates. This lets fully supported alternatives
  replace partial or unsupported primary points even when the primary list
  already fills the requested count.
- The map candidate coverage, selected-point coverage, and final verified
  answer coverage are reported separately. Final coverage is calculated only
  from evidence IDs attached to claims that survive verification.
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
existing evidence-assessment contracts. Each assessment request is bounded to
eight aspects, 12 evidence passages, and 8,000 input characters. Video overview
answers support up to ten requested points plus as many source-linked reserve
candidates as fit within the fourteen-candidate verification ceiling, through
bounded assessment batches. Trivial
short non-overview answers avoid an additional model call.

The audit is restricted to:

1. chunk IDs recorded as included in the generation or overview map context; and
2. exact normalized text excerpts that can be found inside the prompt-visible
   transcript context.

Overview points provide their own evidence chunk IDs. Those sources are
prioritized under the passage ceiling, and each excerpt is selected against
the claims that cite it. Other answer types retain bounded claim-aware source
selection. Up to fourteen overview candidates can be audited in two requests,
with at most eight aspects per request; up to three source-linked reserve
candidates may also be verified to replace unsupported primary points. No requested point is silently dropped
merely because it exceeds one request's aspect limit.

A chunk selected for indexing but omitted from the relevant model prompt cannot
verify an answer. Text outside a prompt-visible excerpt cannot verify a claim.
Each passage retains its video ID, chunk ID, and timestamp interval.

For ordinary targeted answers, completed assessments keep supported
claims, qualify partially supported claims, and remove claims assessed as
missing, contradicted, or conflicting. For video-wide overviews, only fully
supported points are retained; partial or unsupported points are omitted and
replaced by fully supported reserved findings when available. The response
discloses when fewer than the requested number of fully supported points can
be produced. If all claims are rejected, the existing video-evidence abstention
is returned.
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
