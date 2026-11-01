# Phase 11.2 — Retrieval Routing, Generation Integrity, and Provider Resilience

## Objective

Make the live YouTube RAG path robust when the query planner misclassifies a question or the inference provider is temporarily unavailable. Improve observability so incomplete generation can be distinguished from retrieval failure and genuine transcript abstention.

The product contract remains unchanged: users ask in English, the main answer is English, and short original-language excerpts may support answers from Hindi, Arabic, or Urdu captions when the exact excerpt is present in the supplied evidence.

## Production failure patterns motivating this phase

The 2026-11-01 manual run on Hindi and Arabic videos exposed three independent failure classes:

1. A multi-part historical question was classified by query understanding as `overview`. The recovery path sampled six evenly distributed transcript chunks, which is intended for broad video summaries and is not sufficient for an analytical question about developments and consequences.
2. Groq returned repeated HTTP 429 token-per-minute errors. When query understanding failed, the recovery path returned zero chunks immediately, even though deterministic retrieval using the original question could still be attempted.
3. Generation was configured for 800 completion tokens. The response path did not record the provider's `finish_reason` or usage counts, so a truncated completion could not be distinguished reliably from a complete answer.

These are independent failure modes. A provider quota failure is not evidence that the transcript lacks an answer.

## Scope and implementation

### 11.2A — Conservative overview routing

- Treat planner intent as advisory, not authoritative.
- Permit overview sampling only when the user's original question is recognised as an overview request or explicitly asks for the main/overall/general subject, topic, theme, or summary.
- Never allow a question identified by the existing evidence-dense detector to be downgraded to overview sampling.
- Apply the same guard to both weak-initial-candidate recovery and empty-initial-result recovery.
- Preserve the existing overview route for normal overview questions.

Real-world cases:
- "What is this video about?" remains an overview request.
- "What is the main subject discussed here?" remains an overview request.
- "Explain the major historical developments and their consequences" must use targeted retrieval even if the planner says `overview`.
- "Compare the risks and benefits discussed by the speaker" must not be replaced with representative samples.

### 11.2B — Planner-independent retrieval fallback

- If query understanding fails, retry retrieval with the original user question using the existing bounded recovery candidate count and distance configuration.
- Disable overview retrieval for this fallback.
- Apply the existing strong semantic evidence gate before accepting any candidate.
- Expand context only after an anchor passes that gate.
- If no anchor passes, return no evidence rather than fabricate support or substitute arbitrary overview passages.
- Record whether the fallback was attempted and accepted, and preserve the planner failure reason in the retrieval trace.

Real-world cases:
- Groq TPM 429 during planning.
- Temporary network/provider outage during query reformulation.
- Malformed planner JSON or unusable planner output.
- Original question still matches a relevant passage even when query rewriting is unavailable.

### 11.2C — Provider completion observability

- Record Groq `finish_reason`, prompt tokens, completion tokens, total tokens, and configured completion-token limit.
- Do not log prompts, transcript passages, generated answer text, credentials, or user content.
- Emit an explicit warning when the provider reports `length` / `max_tokens`.
- Keep the existing completion-token setting unchanged in this patch. Raising it without a TPM budget analysis can worsen quota failures.

Real-world cases:
- A long list answer stops at the completion ceiling.
- A detailed explanation is cut off mid-sentence.
- A response finishes normally but is mistakenly suspected of being truncated.
- TPM pressure increases because prompt and completion usage are not understood.

## Deliberately not changed

- Embedding model, index namespace, chunking, vector-store contents, MMR parameters, hybrid ranking, reranker settings, semantic thresholds, citation policy, and timestamp selection.
- No reindexing is required.
- No automatic increase to `GROQ_MAX_TOKENS`.
- No provider failover to a different model without an explicit configured provider and compatibility tests.
- No semantic answer-verifier integration. The Phase 11.0B assessor remains an offline/experimental component and is not treated as proof of live answer correctness.
- No claim of universal error-free answers. Generalisation must be established through held-out videos and independently checked evidence.

## Validation gates

### Automated regression tests

- Explicit overview questions still route to overview.
- Evidence-dense analytical questions reject planner-generated overview intent.
- Planner failure retries the original query without overview sampling.
- The planner-independent fallback accepts only candidates passing the existing strong semantic gate.
- Weak fallback candidates are rejected before context expansion.
- Groq usage/finish metadata is logged without answer or transcript content.
- Existing retrieval, citation, language, and API tests remain green.

### Manual real-world validation

Use English questions only. Cover:
- Hindi historical video: multi-event chronology, causes/consequences, names and dates.
- Arabic acoustics video: mechanism explanation, perception, and exact Arabic excerpt fidelity.
- Long English lecture: binary representation and multi-step explanation.
- English AI discussion: contrasting positions and multi-facet risk/application questions.
- Unrelated control questions to confirm abstention.
- Genuine overview requests to confirm overview behaviour.

For every answer, compare each claim to the retrieved transcript passages and timestamps. Do not use model output as gold truth.

### Acceptance criteria

1. No evidence-dense question is routed to overview solely because of planner intent.
2. Planner failure no longer short-circuits retrieval without one bounded original-query attempt.
3. The fallback cannot bypass the existing strong semantic gate.
4. Provider completion metadata distinguishes normal completion from token-limit termination.
5. All existing automated tests pass.
6. Live quality is reported separately from software test status; held-out evaluation is required before declaring generalisation.

## Follow-on work

The following should be separate, evidence-driven changes after Phase 11.2's first validation:
- Answer completeness and duplicate-section detection using observed failure cases and explicit tests.
- Token-budget optimisation based on provider usage telemetry rather than arbitrary prompt truncation.
- Broader provider resilience or failover only after provider-specific compatibility, privacy, cost, and quota policies are defined.
