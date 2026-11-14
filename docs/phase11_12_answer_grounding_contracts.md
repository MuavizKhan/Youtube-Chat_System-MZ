# Phase 11.12 — Answer-grounding contracts

## Baseline and acceptance interpretation

The consolidated Phase 11.11 report generated on 2026-11-14 recorded 43/43
successful cases, zero failed cases, and `quality_passed=true`. It was
assembled from nine November 12 batch reports and a November 14 AR09 retry.
The report used main commit
`d1e19473d1f5c2c556f29f094b0d82f993299456`.

That consolidated result is a valid transport/source-provenance/registered
phrase-gate result for its 43 returned answers. It is not proof that all claims
were factually correct. Several answer contracts only looked for one broad
phrase; several existing curated gold cases had no generation-time coverage
requirement and therefore did not participate in the generation repair/fallback
gate. This phase separates:
- transport and rate-limit outcomes;
- retrieval/prompt evidence coverage;
- citation and timestamp provenance;
- answer completeness, attribution, numeric direction, and factual semantics.

A provider quota failure is not an answer-quality failure. A successful HTTP
response is not automatically a semantically accepted answer.

## Defects reviewed

The review was based on the consolidated 43-answer report, earlier focused
runs, the registered transcript evidence, and the merged Phase 11.10 code.

| Cases | Observed answer defect | Contract added |
| --- | --- | --- |
| Q01 | Opening 9,000-crore narrator framing was repeated as a verified loan default; single-class/low-cost model and service differentiators were incomplete; launch/decision dates ran together. | Keep the old broad retrieval group; require prompt-visible claim groups for the operating model and differentiators; remove the unqualified narrator label in the registered answer normalizer; test date punctuation. |
| Q02 | Force India was incorrectly explained as Mallya's own brand; the £10m transfer direction was reversed or mislabelled as sponsorship. | Preserve the original single retrieval group and its threshold; add separate source-checked answer-claim groups for the Indian-flag rationale, brand promotion, and F1 team's return of £10m to Kingfisher Airlines. |
| Q03 | The answer's list was shallow/poorly formatted and classified the 17-bank lender consortium as a business. | Require the three distinct business families; normalize escaped/bold list items; remove the consortium from the operating-business list. |
| Q04/Q10 | The interviewer Raj Shamani's “money stopped / it got dry” paraphrase was attributed to Mallya. | Correct the Q04 deterministic fallback, add speaker-preserving prompt guidance and repair instructions, and require interviewer (not guest) attribution in the audit. |
| Q06 | The answer added visual identity/packaging details not established by the transcript and omitted brand personality, surrogate advertising, and the Kingfisher water business. | Require the transcript-supported brand-strategy claims and reject the observed unsupported black-and-gold visual claim. |
| Q08 | The answer omitted the separate court claim that the Recovery Officer/Official Liquidator had said the banks were fully paid. | Require separate recovery figure, court-officer statement, and missing statement-of-account claims. |
| Q11 | Opening 9,000-crore loan-default framing was repeated without attribution during a temporal-location answer. | Remove only the unverified introductory label while preserving Q11's earlier entity and temporal-location corrections. |
| AR03/AR08 | One answer substituted arteries where the transcript-supported term is veins. | Require veins and reject an arteries-only substitution. |
| AR06 | “Orfield” was rendered as “Orville.” | Add exact registered-case normalization and a corresponding audit check. |

EW03 was also flagged for manual semantic review: its returned answer largely restated that conversation with strangers is difficult and may not explain the specific difficulty requested. The third-party transcript page could not be opened during this review, so no new deterministic answer or acceptance assertion is added for EW03 in this phase. Verify the source passage before adding one.

## Implementation

- Existing retrieval gold groups and retrieval thresholds are deliberately retained. Additional answer-only claims use the existing source-scoped caption matcher.
- The generation repair requests a missing claim only when every required claim has been found in the prompt-visible source blocks. It does not ask the model to invent a claim that was not retrieved.
- The deterministic registered fallback uses its own source and answer groups and remains gated on all required prompt-visible claims.
- The claim-level live audit checks speaker attribution, financial transfer direction, business/lender classification, required multi-part answer coverage, and high-risk proper names.
- Regression tests deliberately feed prior wrong-answer patterns to the quality checks and require those patterns to fail.
- Existing Q05, Q07, Q08 financial-scope, Q11 temporal-location, Hindi, AR07, source-scoped matching, citation provenance, and quota-accounting safeguards are not removed or weakened.

## Retrieval configuration freeze

No retrieval/ranking settings are changed in this phase. Continue using the accepted configuration unless a separate diagnostic proves a retrieval defect:

- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Namespace: `phase10-10-acceptance-minilm-20261023`
- Citation policy: `prompt_included_chunks`
- `top_k=4`, `fetch_k=10`, `lambda_mult=0.7`, `max_distance=1.3`
- Context expansion: `1`; maximum context chunks: `12`; generation context ceiling: `8000`
- Dense semantic K: `24`; dense fetch K: `32`; anchor limit: `12`; minimum anchor gap: `2`; dense context maximum: `16`
- Facet rerank weight: `0.5`; reranker disabled

## Regression protections carried forward

Do not re-open resolved problems while addressing answer semantics:

- Caption phrase matching stays scoped to one source block; timestamp rows from different sources must never be stitched together.
- Q05's atomic financial/operational evidence and interviewer-versus-guest attribution remain intact.
- Q07's £4,999 crore judgment debt, ₹6,203 crore with interest, and ₹14,100 crore recovery statement remain separate claims.
- Q08 remains question-scoped and must not inherit generic financial fallback claims.
- Q11 temporal-location coverage and timestamp/source normalization remain intact.
- Hindi Bijolia/tax-name/tax-count corrections and AR07's source-bound $600 answer remain protected.
- Provider quota/rate-limit/transport errors remain separate from semantic-quality failures.
- Urdu remains blocked pending a verified publisher-uploaded Urdu caption track.

## Required validation before merge

1. Run the focused unit tests for `tests/test_chain.py` and
   `tests/test_chat_answer_source_audit.py`.
2. Run the full existing unit suite.
3. Run the live 43-case suite in the nine established batches with the frozen
   configuration. Retry only transport/quota failures; do not replace a semantic
   failure with a duplicate successful retry.
4. Review per-case semantic checks, transcript/source attribution, money
   direction and units, proper names, and timestamps before declaring acceptance.

This PR must not be merged solely because the historical consolidated
Phase 11.11 report says 43/43. The new tests and fresh live acceptance are
required gates.


## CI remediation note (2026-11-14)

- Q08 answer completeness does not block its existing question-scoped financial fallback. The fallback remains responsible for verifying all required source claims before producing an answer.
- Q04/Q10 interviewer attribution repair is independent of the broader financial gold-group threshold, but only runs when the disputed wording and interviewer speaker marker both appear in the supplied context.
- Prompt safeguard wording for English translation and financial claim attribution remains compatible with the earlier regression tests.

- Gold completeness repair now requires the case's configured minimum prompt-evidence coverage; independently checked speaker-attribution and fully source-matched claim repairs remain eligible on partial contexts. A regression test protects this separation.
- The system-prompt contract is tested below 4,900 characters to leave a reserved budget below the 5,000-character ceiling, while retaining the English-output, attribution, timestamp, and financial-figure safeguards.
