# Phase 11.7 — question-scoped financial fallback and answer integrity

- **Status:** implementation in progress
- **Base commit:** `d1b60b5ca8592394683200fc657a67b66fd1bf96` (merged Phase 11.6)
- **Target:** repair the live Q08 bank/recovery answer without regressing Q07 or the existing financial-attribution safeguards
- **Production retrieval changes:** none
- **Index rebuild or mutation:** none
- **Frozen Phase 10.11 acceptance configuration:** unchanged

## 1. Problem statement

The November 5, 2026 Phase 11.0 live subset reports show:

- Batch 01 (Q01–Q05): 5/5 cases succeeded; all answer/source quality checks passed.
- Batch 02 (Q06–Q10): 5/5 HTTP/API requests succeeded; Q06, Q07, Q09 and Q10 passed; Q08 failed answer quality.
- Q08 returned HTTP 200 but its generation trace reports `generated=false`, `fallback_used=true`, `repair_aborted_due_quota=false`, and `financial_answer_suppressed=true`. The captured request had three attempts and two rate-limit retries. This is not a transport failure and is not the daily-token-quota branch; generation returned no usable answer after retries.
- Retrieval returned 12 chunks across 11 source segments. All 12 retrieved chunks were included in four source groups; the generation context was 7,977 / 8,000 characters. All three curated Q08 evidence groups were represented in the generation context. The failure is therefore downstream of retrieval and context selection.
- The financial fallback recognized a source-attributed 14,100-crore recovery claim, the 4,999-crore DRT debt claim, and the 6,203-crore debt-certificate figure. It rejected the answer because the separate 3,000-crore investment topic was absent from these retrieved source groups.

The current fallback requires four topics—investment, DRT debt, debt certificate, and Ministry/Finance recovery report—for every financial question. That is appropriate for broad financial-summary questions such as Q07, but over-scoped for Q08, which asks specifically about banks and recovery.

## 2. Correctness contract

### 2.1 Question-scoped fallback

Introduce a small, conservative question-intent classifier that is based on the question text, not benchmark case IDs:

- **Comprehensive financial scope:** broad questions asking about the amount of debt/money involved or multiple financial amounts continue using the existing four-topic contract.
- **Bank/recovery scope:** questions that combine bank/account-payment language with recovery language use a separate evidence contract relevant to that request.

No production helper may import or inspect Q07/Q08 benchmark IDs, gold phrases, or the benchmark registry. The registry remains an independent acceptance oracle.

### 2.2 Bank/recovery evidence contract

For a bank/recovery question, assess three source-backed evidence families:

1. **Recovery amount/claim:** a monetary amount is explicitly connected to recovery in its own source block. When the wording attributes it to a Ministry/Finance/report, that attribution must also be present within that same source block. A generic recovery claim must never be relabelled as an official report claim.
2. **Bank-payment status:** a transcript statement directly connects banks with a payment/repayment status. A question, speculation, or unrelated mention of a bank is not enough.
3. **Account/Parliament statement:** a transcript statement explicitly refers to a statement of account or a statement in Parliament.

The fallback requires at least two of these three evidence families before producing a substantive answer. It includes only the evidence families actually found and preserves each statement's source identity. This avoids requiring an unrelated investment figure while also avoiding an answer based on one isolated keyword.

### 2.3 Financial claim safeguards

Preserve the existing conservative financial parser and source boundaries:

- Do not combine claims across `[SOURCE n]` blocks to invent attribution.
- Keep 4,999-crore DRT debt, 6,203-crore debt-certificate amount, and 14,100-crore recovery distinct.
- Do not infer a currency/unit when the amount-bearing caption does not support it.
- Do not emit malformed ASR figures or unsupported distractor amounts (including 22,280, 14,000, or malformed variants) as if they were the verified recovery amount.
- Do not imply that a recovered amount is the company's debt total or that the DRT/certificate amounts have a relationship the transcript does not establish.
- If fewer than two bank/recovery evidence families are supported, fail closed with an explicit diagnostic rather than guessing.

### 2.4 Provider failure observability

The early generation fallback path must expose a stable, non-sensitive reason code distinguishing:

- `daily_token_quota_exhausted`
- `empty_answer_after_retries`

Keep the existing HTTP/API behavior and safe fallback. Do not serialize raw provider exceptions, prompt contents, transcript bodies, or credentials. The generation trace must continue to distinguish provider generation success from a successful HTTP response.

### 2.5 Timestamp and citation contract

Substantive fallback answers must pass through the existing trusted timestamp append path. Timestamp ranges must be selected from actual returned-source metadata, remain within returned source intervals, and align with the answer evidence. Do not manufacture timestamps from text or broaden the citation policy.

If the bank/recovery fallback abstains, preserve the existing intentional abstention/source-suppression behavior. Do not label unverified review material as answer-supporting citations.

## 3. Sub-phases and execution order

| Sub-phase | Work | Required evidence before advancing |
|---|---|---|
| **11.7A — failure contract** | Freeze the Batch 02 Q08 trace, classify the empty-provider-answer path, and record the source/evidence facts above. | Reproducible unit fixture represents the actual failure shape; no changes to retrieval or index. |
| **11.7B — question scope** | Add conservative intent selection for comprehensive financial questions versus bank/recovery questions. | Q08-like wording selects bank/recovery scope; Q07-like wording retains comprehensive scope; ambiguous wording defaults conservatively. |
| **11.7C — evidence extraction** | Add source-local bank-payment and account/Parliament evidence extraction; reuse the existing monetary parser for recovery claims. | Two-of-three evidence contract; source-local recovery attribution; no malformed or cross-source claim leakage. |
| **11.7D — fallback rendering** | Render a concise answer from only the verified evidence groups. | Q08-like fixture returns 14,100-crore report evidence without requiring 3,000-crore investment; no unsupported bank-payment claim is added. |
| **11.7E — observability** | Add a stable provider failure reason code and explicit fallback-scope diagnostics. | Empty answer and daily quota are distinguishable; sensitive raw exception/provider content remains excluded. |
| **11.7F — timestamps and regression suite** | Verify source-linked timestamp behavior and add focused regressions. | Answer timestamps are grounded in the returned source intervals; abstentions retain safe behavior. |
| **11.7G — validation and handoff** | Run targeted tests, the full test suite, GitHub Actions CI, and code review. Document exact local sync and live-audit commands. | CI green; no frozen-config/index changes; PR remains unmerged until local/live checks are complete. |

## 4. Required regression matrix

1. A bank/recovery question with two supported evidence families returns a substantive fallback without investment, DRT-debt, or certificate evidence.
2. A bank/recovery question with only one supported family abstains.
3. A source-local recovery report amount is attributed to that report; an attribution found only in another source block cannot be borrowed.
4. A generic recovery amount is never relabelled as a Ministry/Finance report amount.
5. Bank-payment status is included only when an actual source statement supports it.
6. Account/Parliament wording is included only when an actual source statement supports it.
7. A bank/recovery answer does not leak 3,000, 4,999, 6,203, 22,280, 14,000, malformed ASR amounts, or unsupported totals unless a claim is independently required and source-supported.
8. The comprehensive Q07-like question still requires all four existing financial claim families and retains the DRT/certificate distinction and uncertainty caveat.
9. Existing plain-prose and timestamped-caption financial parser tests remain green.
10. A successful bank/recovery fallback gets only relevant, returned-source timestamps.
11. An abstention is not decorated with misleading timestamps or citations.
12. Empty provider output after retries and daily quota exhaustion produce distinct trace reason codes while both retain safe fallback behavior.
13. The production helper does not import or consult the gold-evidence registry or question IDs.

## 5. Explicit non-goals

- No changes to embeddings, vector-store contents, index namespace, chunking, top-k/fetch-k, MMR, dense retrieval, reranking, recovery thresholds, or retrieval query planning.
- No changes to the Phase 10.11 frozen acceptance configuration or the 8,000-character generation-context ceiling.
- No loosening of source-local amount/report attribution.
- No benchmark-specific production condition such as `question_id == "Q08"`.
- No changes to Q05 answer-only paraphrase handling.
- No live semantic evidence-assessor integration; the independent human-labelled Phase 11.0C validation gate remains separate.
- No Chrome extension/UI changes and no Urdu support claim.

## 6. Frozen acceptance baseline

The Phase 10.11 baseline remains:

- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Namespace: `phase10-10-acceptance-minilm-20261023`
- Citation policy: `prompt_included_chunks`
- `top_k=4`, `fetch_k=10`, `lambda_mult=0.7`, `max_distance=1.3`
- Context expansion: 1; maximum context chunks: 12
- Dense semantic k/fetch k: 24/32
- Dense anchor limit/minimum gap: 12/2
- Dense context maximum chunks: 16
- Dense facet rerank weight: 0.5
- Reranker disabled; model metadata remains `cross-encoder/ms-marco-MiniLM-L6-v2`
- Generation context ceiling: 8,000 characters

No index rebuild is authorized by this phase.

## 7. Release and live-validation rule

The existing Q01–Q10 subset reports remain historical diagnosis artifacts on commit `d1b60b5ca8592394683200fc657a67b66fd1bf96`. Q08 is an HTTP-success attempt with a quality failure; Phase 11.6 correctly does not permit a second HTTP-success attempt for the same case to be cherry-picked into the same consolidation.

After the Phase 11.7 code is reviewed and merged, execute the complete nine-batch, 43-case live audit against one clean runner commit and the unchanged accepted namespace/configuration. Save unique report filenames and consolidate using `scripts.merge_chat_answer_source_audit_reports`. Do not mix reports from different runner commits. Do not call the full 43-case acceptance complete until the consolidated quality gate is green.

The PR must remain unmerged until repository CI and the local validation procedure documented in the PR pass. Live acceptance is a separate post-merge execution.
