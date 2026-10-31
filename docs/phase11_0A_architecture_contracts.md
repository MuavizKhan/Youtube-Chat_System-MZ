# Phase 11.0A — Stabilization and evidence-contract foundation

**Status:** implementation branch based on the frozen main-branch baseline  
**Baseline commit:** `8a3b3a7592c1fe31bad165b4756a08f1c1534b8b` (merged PR #154)  
**Phase 10.11 acceptance configuration:** unchanged

## 1. Purpose

Phase 11.0A stabilizes the failed PR #155 experiment and establishes provider-neutral,
video-scoped interfaces for the next general evidence-retrieval work.

This phase is deliberately non-invasive. It defines and validates the data contracts
that later phases can connect to retrieval and answer generation. It does not change
the production RAG chain, retrieval ranking, context packing, prompt, provider calls,
index contents, or API response.

## 2. PR #155 postmortem

PR #155 was opened from the frozen main baseline to recover curated evidence for
Kingfisher questions Q04, Q09, and Q10. It remains unmerged and must not be treated
as production code.

Two CI runs failed on the same test:

| Run | Result |
|---|---|
| #985 | 718 passed, 1 failed |
| #986 | 718 passed, 1 failed |

The failing test was
`tests/test_chain.py::test_rag_chain_abstains_when_registered_required_gold_context_is_incomplete`.
The test intentionally supplies a dummy vector-store object and verifies that
generation is skipped when the registered evidence context is incomplete. PR #155
replaced the production retrieval runnable, so the new registered-evidence recovery
wrapper invoked the real `retrieve_question_context` function. The dummy store did
not have an `embedding_function`, causing an `AttributeError` before the safe
abstention assertion could complete.

Dependency consistency, Python compilation, and Chrome extension JavaScript syntax
checks passed in both runs. The full test step reported 718 passed and one failed.

### Architectural findings

The CI failure was a concrete test-isolation regression. The design review also
identified reasons not to repair PR #155 by merely mocking the failing call:

1. Recovery was enabled only for exact registered Kingfisher question/video pairs;
   it could not improve arbitrary questions on unseen videos.
2. Queries were generated from the same curated phrases later used for acceptance.
   Phrase overlap is not independent proof that a passage answers the user's intent.
3. Query fan-out grew with the number of missing groups and alternative phrases,
   without a single explicit total work/latency budget.
4. The production retrieval runnable was changed before a reusable question/evidence
   interface existed.
5. The new tests validated one phrase-recovery example and a non-registered bypass,
   but did not establish generalization across held-out videos, languages, question
   types, or unanswerable controls.

The original Q04/Q09/Q10 post-PR-154 diagnostic remains the evidence for the problem:
all three requests succeeded at HTTP level and the accepted retrieval configuration
matched, but answer quality failed. Q04 and Q10 lacked required evidence in retrieved
candidate context; Q09 had the evidence in its prompt but its answer/repair did not
cover the required details. These findings are retained as regression targets, not
as a reason to hard-code those phrases into production retrieval.

## 3. Scope of this phase

### Included

- Add provider-neutral immutable contracts for question analysis, evidence aspects,
  transcript provenance references, aspect assessments, aggregate coverage, retrieval
  attempts, and retrieval diagnostics.
- Add deterministic validation and aggregation rules.
- Require every supported aspect assessment to retain transcript provenance.
- Distinguish missing, partial, contradicted, conflicted, and unknown evidence.
- Make "no required aspects" indeterminate, never automatically sufficient.
- Keep retrieval telemetry video-scoped and free of raw transcript text.
- Add isolated unit tests for English, Hindi, and Arabic language tags; multi-aspect
  questions; temporal constraints; incomplete/contradictory evidence; and invalid
  provenance.
- Document how later phases may consume these contracts.

### Explicitly excluded

- No changes to `Backend/rag_system/chain.py` or `retrieval.py`.
- No changes to `Backend/rag_system/config.py`, retrieval thresholds, top-k/fetch-k,
  context expansion, generation-context ceiling, embedding model, or reranker.
- No changes to the Phase 10.11 suite, reports, accepted index namespace, or indexes.
- No production use of benchmark-specific gold phrases.
- No new LLM calls, query rewriting, semantic verifier, or recovery retries.
- No change to public API or Chrome extension response shapes.

## 4. Contract design

The contracts live in `Backend/rag_system/evidence_contracts.py`.

### QuestionAnalysis

Describes the user's intent, language, subject, constraints, optional time window,
and distinct requested information aspects. It does not state whether the transcript
can answer the question.

### EvidenceAspect

Represents one requested information need. `required` and `weight` control
aggregation; optional retrieval hints are query-planning suggestions only. They are
not gold phrases and do not prove support.

### EvidenceReference

Identifies a chunk by video ID and chunk ID, with an optional timestamp interval.
No transcript text is stored in the diagnostic contract.

### EvidenceAssessment

Records an assessor's status and score for an aspect, plus the exact supporting
and/or contradicting transcript references. Supported or partial claims without a
supporting source are rejected by validation. Conflicting evidence is explicit.

### EvidenceCoverageSummary

Aggregates assessments without doing semantic matching. The aggregate is
`sufficient` only when every required aspect has a `supported` assessment, meets
the configured minimum support score, and includes at least one transcript reference.
A high weighted average cannot hide a missing mandatory aspect. If there are no
required aspects, the result is `indeterminate`, not a pass.

The function `evaluate_evidence_coverage` is a deterministic contract aggregator.
It does not decide whether a transcript passage semantically entails a claim. That
assessment mechanism belongs to Phase 11.0B and must be tested independently.

### RetrievalAttempt and RetrievalDiagnostics

Describe retrieval method, stage, candidate and accepted provenance references,
elapsed time, recovery rounds, and whether a work budget was exhausted. Query text
and raw transcript passages are not copied into these telemetry structures.

## 5. Generalization and language boundaries

The contracts are independent of benchmark IDs, video registry entries, embedding
providers, vector-store implementations, and LLM vendors. The video ID remains part
of every evidence reference so future multi-video or shared-index implementations
can enforce provenance boundaries.

Language tags are metadata, not a claim that a caption track exists. The existing
live transcript support remains English, Hindi, and Arabic. Urdu remains blocked
pending verification of an accessible publisher-uploaded Urdu-script caption track.

The existing unseen-video indexing smoke validates transcript acquisition and
index lifecycle behavior; it does not, by itself, establish answer-quality
generalization. Phase 11.0F must separately validate answers against held-out videos.

## 6. Test and acceptance gates

This phase passes only when:

1. Main remains the exact PR #154 baseline until this foundation is reviewed.
2. No production retrieval, generation, API, or index behavior changes.
3. All contract validation and aggregation tests pass.
4. The existing incomplete-evidence abstention test passes unchanged with its dummy
   vector store; it must not require a real embedding model or provider.
5. Dependency consistency, Python compilation, extension syntax, and the full
   automated test suite pass.
6. The frozen Phase 10.11 configuration and report/index artifacts remain untouched.

After this phase, Phase 11.0B will implement actual evidence-aspect coverage assessment.
It must not be merged until the contract semantics and tests are accepted.
