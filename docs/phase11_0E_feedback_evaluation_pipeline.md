# Phase 11.0E — automated feedback intake and evaluation queue

- **Implementation slice:** user-feedback intake and private review queue
- **Base:** Phase 11.0B semantic assessor, 11.0C evaluation harness, 11.0D blinded annotation workflow
- **Production RAG path:** unchanged
- **Data status:** feedback is unverified and is not ground truth

## 1. Goal and boundaries

This slice adds a user-facing helpful/not-helpful control below each completed answer, a bounded feedback API, local SQLite persistence, and a private JSONL export/summary command.

It does **not** change `chain.py`, retrieval, index creation, context packing, answer prompts, model/provider settings, embedding configuration, or the accepted 43-case baseline. The semantic assessor is never called when a user clicks a feedback button. It can only be run later through the local `--assess` batch command, and only for reports with separate AI-provider consent and a complete exact-source evidence snapshot.

## 2. Feedback interaction

Every successfully rendered answer gets two controls:

- **Yes:** records a positive signal.
- **No:** opens a short reason form before submission.

Negative feedback reasons are one of:

- `answer_incorrect_or_incomplete`
- `question_or_context_misunderstood`
- `transcript_missing_or_inaccurate`
- `source_or_timestamp_issue`
- `language_or_translation_issue`
- `other`

A free-text comment is optional and limited to 1,000 characters. The extension explains that a submitted comment is stored with the report and asks users not to include sensitive personal information.

The form includes an unchecked-by-default context-sharing control. Only when the user checks it does the extension submit the question, answer, up to six preceding conversation turns (capped at 4,000 characters), and source timing/chunk references. A second, separate checkbox—also unchecked by default and disabled until context sharing is enabled—asks for permission to send the shared answer and cited transcript excerpts to the configured AI provider for a bounded quality assessment. Without the first opt-in, the backend stores no question/answer/conversation-history/source content. Without the second opt-in, the pipeline must not call an AI provider for that record. The backend stores the rating, reason/comment (if given), video ID, answer/request identifiers and technical model/retrieval metadata. Conversation history, user identity, IP address, cookies, and transcript text are not stored at submission time.

## 3. Backend persistence and limits

The `POST /feedback` endpoint is separate from `POST /chat` and uses the existing API rate-limit decorator. Payloads are strictly validated, unknown fields are rejected, comments and answer context are bounded, source timestamps must be valid, and every source must belong to the same video as the rating.

SQLite is the development/local persistence adapter. The default file is `data/feedback.sqlite3`; set `RAG_FEEDBACK_DB_PATH` to choose another path. The database file is ignored by Git. A unique answer ID makes submissions idempotent: retrying a submission cannot create a second feedback record for the same displayed answer.

This local adapter is not a production storage guarantee. Phase 12 must select a persistent managed database or durable mounted volume, backup/retention policy, deletion policy, and access controls before public deployment.

## 4. Review queue commands

Run from the repository root in PowerShell:

~~~powershell
$python = ".\.venv\Scripts\python.exe"

# Aggregate counts only; does not print submitted question/answer/comment text.
& $python -m scripts.phase11_feedback_queue summary

# Export the review queue with the shared fields only.
# Keep this file private; evaluation_data/private/ is ignored by Git.
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$output = "evaluation_data/private/feedback_queue_$stamp.jsonl"
& $python -m scripts.phase11_feedback_queue export --output $output

# Optional: resolve the exact cited transcript chunks from existing local indexes.
# This step does not call a text-generation model.
$evidenceOutput = "evaluation_data/private/feedback_evidence_$stamp.jsonl"
& $python -m scripts.phase11_feedback_queue export --include-evidence --output $evidenceOutput

# Optional: run the bounded Phase 11.0B assessor only for reports with
# separate AI-provider consent. Default maximum is 10 reports per command.
$assessmentOutput = "evaluation_data/private/feedback_assessment_$stamp.jsonl"
& $python -m scripts.phase11_feedback_queue export --assess --output $assessmentOutput
~~~

To use a non-default database:

~~~powershell
& $python -m scripts.phase11_feedback_queue summary --db "D:\app-data\feedback.sqlite3"
& $python -m scripts.phase11_feedback_queue export --db "D:\app-data\feedback.sqlite3" --output "evaluation_data/private/feedback_queue_custom.jsonl"
~~~

The exporter refuses paths outside `evaluation_data/private/` and refuses to overwrite an existing export. Summary output contains counts only. Export output is sensitive and must not be committed or sent to public issue trackers.

With `--include-evidence`, the local command loads the existing index for each opted-in video and resolves only the chunk IDs returned with that answer. It includes the exact chunk text, caption-segment text, transcript language, and source timing; it reports missing chunks instead of silently substituting other text. It does not build or modify an index or change the accepted retrieval configuration. The evidence-enriched export contains YouTube transcript text and must remain in the private ignored directory.

With `--assess`, the command runs the existing Phase 11.0B evidence assessor only for negative reports where the user explicitly consented to AI-provider analysis and where all cited chunks were resolved. It uses the configured provider/model, one bounded call per report, no retries, at most eight sentence/bullet answer claims and twelve passages per report, and at most ten reports by default. The assessor output is marked `ai_proposed_assessment`, never ground truth; every result requires human verification. Reports without consent, complete evidence, or within-budget claims are skipped with a reason. This is an optional local command, not a per-click provider call.

Shared-context feedback is checked client-side against a 14 KB payload ceiling so it stays below the backend's existing request-body limit. If it is too large, the user can uncheck the sharing option and submit the rating/reason without the question, answer, or source references.

## 5. How feedback becomes evaluation data

A user report is a **candidate case**, not a verdict. A negative rating can reflect an answer defect, an ambiguous question, missing conversational context, a misleading/incomplete transcript, a source/timestamp issue, or a user misunderstanding.

Recommended review sequence:

1. Cluster new feedback by reason, video ID, and repeated answer/report IDs.
2. Inspect the actual YouTube video and the relevant caption track around the reported source timestamps.
3. Determine whether the issue is in the question/context, transcript, retrieval/evidence, answer wording, attribution, or citation/timestamp mapping.
4. For candidate semantic cases, a curator prepares exact transcript excerpts and a source-only manifest using the Phase 11.0D workflow.
5. Two independent reviewers label every aspect without model predictions; a distinct third person adjudicates disagreements.
6. Only the adjudicated development cases may be used for development evaluation/calibration. Keep held-out videos and labels isolated under the Phase 11.0C rules.
7. Add verified bug reproductions to regression tests where appropriate. Do not automatically change prompts, thresholds, models, or production behavior based on feedback.

The queue is designed to make these steps much faster, not to eliminate source verification. The UI does not ask users to decide the six semantic evidence labels.

## 6. Interpretation and next implementation slice

The aggregate helpful rate is a product signal, not an answer-accuracy metric. Feedback is self-selected and may be biased toward unusually good or bad experiences. A user clicking "No" does not prove the answer is false; a "Yes" does not establish that its claims are supported.

The extension response includes exact citation chunk IDs and timestamps. When the user opts into context sharing, the local evidence exporter resolves those IDs against the existing FAISS docstore and includes exact indexed transcript text and caption segments. It reports missing or duplicate chunks as incomplete rather than substituting nearby passages. This hydration step makes no generation-provider call and does not alter the accepted index or retrieval configuration.

When the user also opts into AI-provider assessment, the `--assess` batch command splits the generated answer into candidate claims and asks the existing bounded Phase 11.0B assessor whether the exact cited passages support each claim. This is deliberately heuristic and can mis-segment claims or miss conversational context; outputs are proposals only, not final labels or calibrated probabilities. All outputs require human review and must never be used as ground truth, used to tune against held-out data, or trigger automatic model/prompt changes.

This export is a review bundle, not a ready-to-assemble Phase 11.0D source manifest. A curator must still verify the actual video, caption track and timestamps, define source/independence groups, confirm each question aspect, and attest the verification fields before using the blinded human-annotation workflow. The feedback pipeline accelerates candidate discovery and evidence collection; it does not fabricate those human attestations. Phase 11.0F remains a separate calibrated development/held-out assessment.

## 7. Validation

The new unit tests cover validation, explicit context and AI-provider consent, bounded recent conversation history, source/video consistency, SQLite persistence and migrations, idempotent retries, aggregate-only summaries, and rejection of unexpected fields.

Run:

~~~powershell
& $python -m pytest tests/test_feedback_pipeline.py -q
& $python -m pytest -q
~~~

A subset test passing does not substitute for running the complete existing suite before merge.
