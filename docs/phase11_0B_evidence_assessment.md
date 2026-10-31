# Phase 11.0B — bounded semantic evidence assessment

**Status:** implementation PR based on merged Phase 11.0A  
**Base commit:** `b1ff077f6700d6655752491617eb13c3d38c92f3`  
**Phase 10.11 accepted retrieval configuration:** unchanged  
**Production integration:** intentionally deferred

## 1. Purpose

Phase 11.0B implements the first real evidence-support assessor on top of the
Phase 11.0A contracts. It checks whether selected transcript passages substantively
support, partially support, contradict, conflict with, or fail to address each
requested question aspect.

It is a component-level implementation. It does not change the live RAG chain,
retrieval ranking, context packing, API behavior, or answer generation. A later
phase must evaluate its semantic accuracy on a labelled assessment set before
considering production integration.

## 2. Architecture

The component lives in `Backend/rag_system/evidence_assessment.py`.

### Inputs

- A `QuestionAnalysis` with one or more `EvidenceAspect` entries.
- A bounded sequence of `EvidencePassage` records. Each passage has a caller-
  assigned passage ID, transcript text, transcript language tag, and one or more
  exact `EvidenceReference` values.
- A configured Hugging Face or Groq chat-completion client.

An `EvidencePassage` may represent one transcript chunk or a merged source group.
Its references must identify one video. A single assessment request cannot mix
videos. When `QuestionAnalysis.video_id` is set, every passage must match it.
No benchmark registry or benchmark-specific gold phrase is consulted.

### Model output

The model must return one strict JSON assessment for every requested aspect:

- `supported`: clear substantive support, with supporting passage IDs and score
  at or above the configured support threshold;
- `partial`: relevant but incomplete support, with a score below the threshold;
- `missing`: none of the supplied passages addresses the aspect;
- `contradicted`: supplied evidence directly refutes the aspect;
- `conflicted`: the supplied passages contain material support and contradiction;
- `unknown`: the assessor cannot reliably determine support.

The returned passage IDs are checked against the exact input passage IDs and mapped
back to source chunk/timestamp references. Unknown or fabricated IDs are rejected.
The response is atomic: if any aspect is omitted, duplicated, malformed, or cites
an unknown passage, all aspects become `unknown`. This avoids accepting a partial
or internally inconsistent model response.

The module then calls the Phase 11.0A `evaluate_evidence_coverage` aggregator.
The aggregator itself remains deterministic; it does not perform semantic
entailment.

## 3. Defences and bounded execution

Each assessment uses at most one provider call and does not retry. Input passages
are explicitly treated as untrusted data; instructions embedded inside transcript
text must not change the assessor's task or schema. The model receives no tools.

Default limits:

| Limit | Default |
|---|---:|
| Transcript passages | 12 |
| Requested aspects | 8 |
| Total system + user input | 8,000 characters |
| Completion tokens | 900 |
| Minimum score for `supported` | 0.75 |
| Provider retries | 0 |
| Provider calls per assessment | 1 |

Inputs are not silently truncated. If the passage, aspect, or character budget is
exceeded, the assessor does not call the provider and returns `unknown` results
with a budget diagnostic. The completion ceiling is further capped by the active
provider's configured token limit.

The assessor supports the existing `groq` and `huggingface` provider configuration.
It reuses the repository's configured model IDs and reasoning-effort settings,
uses temperature zero, and requests JSON mode on Groq. No dependency, model,
embedding, index, or environment-variable change is introduced.

## 4. Failure behavior

The assessor fails closed:

| Situation | Result |
|---|---|
| No passages supplied | `missing` assessments, no provider call |
| No required aspects | `indeterminate` coverage, no provider call |
| Input/item budget exceeded | `unknown`, no provider call |
| Client unavailable | `unknown`, no provider call |
| Provider exception / 429 / timeout | `unknown`, one attempted call |
| Invalid JSON or schema | `unknown` for every aspect |
| Missing/duplicate aspect | `unknown` for every aspect |
| Invented passage ID | `unknown` for every aspect |
| Cross-video input | Rejected before provider call |

Diagnostics record the provider/model, video ID, elapsed time, passage/aspect
counts, input character count, stable error code, and token usage when the provider
returns it. Raw transcript text, full prompts, provider exception messages, and
secrets are not copied into diagnostics or the serialized result.

## 5. What the score does and does not mean

The model's `support_score` is an assessor-strength label used for conservative
gating. It is **not a calibrated probability**, and temperature zero does not make
a language model deterministic or guarantee factual correctness. A model can still
misread negation, causality, speaker attribution, fragmented captions, or a
cross-lingual statement.

A `supported` assessment means only that the configured model judged the supplied
passages to substantively support the specified aspect. It is not independent
verification that the speaker's statement is true in the real world.

This PR's mocked-provider tests verify request construction, strict schema handling,
reference mapping, failure modes, budgets, multilingual data transport, and safe
aggregation. They do **not** prove the live model's semantic accuracy. That must
be measured with a held-out labelled assessment set before production integration.

## 6. Tests

`tests/test_evidence_assessment.py` exercises:

- Single-aspect and multi-aspect support;
- Complementary passages for one aspect;
- Partial and missing evidence;
- Contradicted and conflicting statements;
- Hindi and Arabic transcript passages assessed against an English question;
- Prompt-injection-like transcript content;
- Invalid JSON, duplicate keys, omitted/duplicate aspects, invalid scores/statuses,
  unexpected fields, and fabricated passage IDs;
- Provider errors, empty evidence, absent client, and no-required-aspect behavior;
- Passage/aspect/character budgets;
- Cross-video rejection and duplicate passage IDs;
- Groq and Hugging Face request contracts;
- Token usage and transcript-text privacy in serialized diagnostics.

All tests use a fake chat-completion client. No live API key or external provider
request is used by the unit tests.

## 7. Explicit non-goals

- No integration into `chain.py`, `retrieval.py`, or the FastAPI route.
- No question-aspect generation; callers must provide `QuestionAnalysis`.
- No query rewriting, retrieval recovery, or vector-store access.
- No automatic answer generation or answer repair.
- No changes to top-k/fetch-k, MMR, dense retrieval, reranking, context expansion,
  the accepted embedding model, the 8,000-character generation-context ceiling,
  citation policy, index namespace, or Phase 10.11 reports.
- No use of registered `GoldEvidenceCase` phrases in the production assessor.

## 8. Gate before 11.0C / production integration

Before a later phase connects this assessor to live RAG behavior, it must:

1. Build a labelled, held-out aspect-assessment set that is separate from the
   production prompt and not used for tuning.
2. Include unrelated same-topic passages, wrong-direction relationships, negation,
   causal-vs-correlational claims, speaker/narrator attribution, conflicting sources,
   fragmented transcript evidence, unanswerable questions, and prompt-injection text.
3. Include independently reviewed English, Hindi, and Arabic cases, including
   English questions over Hindi/Arabic transcripts where caption tracks are verified.
4. Measure per-status precision/recall and confusion, especially false-positive
   `supported` decisions and false negatives that would trigger abstention.
5. Calibrate the threshold on a development split and report performance on a
   separate held-out split. Do not tune and report on the same examples.
6. Record provider latency, actual token usage, failure rate, and rate-limit behavior.
7. Preserve the frozen Phase 10.11 baseline and prove there is no material
   regression before any production rollout.

A passing unit suite or a plausible model explanation is not a substitute for
these semantic-quality gates.
