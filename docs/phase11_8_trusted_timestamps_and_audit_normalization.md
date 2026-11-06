# Phase 11.8 — trusted timestamps and audit normalization

- **Status:** implementation branch; pending CI, review, merge, and fresh live acceptance
- **Base:** merged Phase 11.7 (0060541db7de34cc00c2f427a7fe3a13e2362ad7)
- **Scope:** shared response timestamp handling and answer/source audit correctness
- **Production video-specific branches:** none
- **Retrieval/index/configuration changes:** none

## 1. Why this phase exists

The November 6 Phase 11.7 Batch 03 report had five successful HTTP responses. Only Q11 and EW02 failed answer/source quality:

- **Q11 — timestamp integrity.** The model returned inline timestamps such as 00:00 to 01:08. The production timestamp helper treated a model-authored range as sufficient and returned before adding the deterministic source-metadata footer. The audit parser recognized dash-separated ranges only, so it did not see the model's "to" range. Separately, the canonical benchmark registry already contained coarse temporal evidence windows for Q11, but the live answer/source audit did not apply those windows.
- **EW02 — Unicode hyphen false negative.** The answer used a non-breaking Unicode hyphen in 30‑second. NFKC normalization changed the code point to another Unicode hyphen, while the registered pattern accepted only ASCII hyphen or whitespace.

These are downstream response/audit contract problems. The Batch 03 retrieval and frozen server configuration were consistent; no retrieval tuning or index change is justified by these failures.

## 2. Phase 11.8 design

### 2.1 One shared timestamp parser

Backend/rag_system/timestamp_utils.py is a small, provider-neutral utility used by both production response handling and the live audit runner.

It recognizes:

- MM:SS and HH:MM:SS endpoints;
- common ASCII and Unicode dash characters;
- natural-language range separators "to" and "and";
- only valid, forward-moving time intervals.

It rejects invalid clock fields, malformed values, and reversed intervals. The audit and production code no longer maintain separate timestamp-range grammars.

### 2.2 Model-authored timestamps are not source metadata

The model may explain the transcript, but it does not own clickable timestamp metadata.

Before final response formatting, the shared sanitizer removes:
- model-authored "Relevant timestamp(s):" footer lines;
- sentences or bullet fragments containing inline timestamp ranges;
- single timestamp claims explicitly framed as video locations (for example, "around 01:20").

Substantive sentences that do not make timestamp claims are retained. Ordinary durations such as 30-second and 90-minute are not interpreted as YouTube timestamp ranges.

The production finalizer then always gets the opportunity to append:
- a validated temporal-retrieval window when the question explicitly requests a timestamp and the retrieval trace supplies a valid temporal window; or
- ranges selected from actual returned source metadata using the existing answer/source alignment logic.

An inline model range can no longer suppress that deterministic step. If there is no valid source/temporal metadata, the application does not invent a replacement time.

### 2.3 Temporal evidence is an evaluation contract, not production routing

The live audit runner now loads optional temporal evidence from the canonical, video-scoped benchmark registry when a case has it.

For a case with registered temporal windows, the audit adds a separate check requiring at least one displayed timestamp to overlap a registered relevant section. The existing check still requires every displayed range to be fully covered by returned source intervals.

This is deliberately an evaluation-only check:
- production chain code does not import question IDs, video IDs, or gold windows;
- videos/questions without temporal annotations keep the ordinary source-coverage checks;
- the current Q11 annotation remains a coarse chapter-level reference, not a word-level timestamp claim;
- future videos can use the same contract when independently verified temporal annotations are added to their registry entry.

### 2.4 Unicode answer requirement normalization

After NFKC normalization, the audit maps common Unicode hyphen/dash characters to ASCII hyphen before applying answer requirements. This preserves the strict semantic requirement—EW02 still needs a 30-second structure—but avoids rejecting equivalent typography.

## 3. Scope guardrails

This phase does **not** change:

- embedding model, index namespace, persisted index contents, or reindexing;
- chunk size/overlap, retrieval candidates, hybrid fusion, MMR, top-k, distance thresholds, reranking, or query planner;
- context expansion, context packing, or the 8,000-character generation cap;
- financial fallback logic from Phase 11.7;
- the frozen Phase 10.11 expected server configuration;
- Urdu support status or the full-registry case list.

No Q11/Q08/EW02-specific behavior is added to production.

## 4. Regression coverage

The deterministic tests cover:

- MM:SS and HH:MM:SS timestamps;
- dash variants, "to", and "and" separators;
- invalid seconds, invalid clock fields, and reversed ranges;
- removal of inline model-authored ranges and labelled timestamp summaries;
- retention of ordinary explanatory prose and duration phrases;
- trusted source metadata replacing an inline model timestamp;
- timestamp removal when no trusted metadata is available;
- Q11 timestamps that are source-covered but outside registered temporal sections failing the temporal audit;
- Q11 timestamps that overlap a registered section and returned source intervals passing;
- EW02's U+2011 and U+2010 hyphen variants satisfying the unchanged answer requirement.

## 5. Validation and release sequence

1. Run the focused timestamp, chain, and live-audit tests.
2. Run the complete pytest suite, dependency checks, Python compilation, and Chrome extension syntax checks through GitHub Actions.
3. Review the branch diff and CI. Merge only after explicit approval.
4. After merge, restart the local backend so it loads the merged code.
5. Run the full 43-case Phase 11.0 live suite from Batch 01 through Batch 09 on one clean runner commit and the unchanged accepted index/configuration.
6. Consolidate only those new batch reports with scripts.merge_chat_answer_source_audit_reports.

The earlier Phase 11.7 Q08 smoke and Batch 01–03 reports remain useful diagnostics. They must not be combined with the Phase 11.8 acceptance run because they were generated from a different code revision. No selective retry of Q11 or EW02 from the old report set is valid.
