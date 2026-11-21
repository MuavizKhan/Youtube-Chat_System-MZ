# Phase 11.0E — automated feedback intake and evaluation queue

- **Implementation slice:** user-feedback intake and private review queue
- **Base:** Phase 11.0B semantic assessor, 11.0C evaluation harness, 11.0D blinded annotation workflow
- **Production RAG path:** unchanged
- **Data status:** feedback is unverified and is not ground truth

## 1. Goal and boundaries

This slice adds a user-facing helpful/not-helpful control below each completed answer, a bounded feedback API, local SQLite persistence, and a private JSONL export/summary command.

It does **not** change `chain.py`, retrieval, index creation, context packing, answer prompts, model/provider settings, embedding configuration, or the accepted 43-case baseline. The semantic assessor is not called when a user clicks a feedback button. That is intentional: the current chat response contains answer text and source IDs/timestamps, but not the actual transcript passage text needed for a grounded semantic assessment.

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

A free-text comment is optional and limited to 1,000 characters.

The form includes an unchecked-by-default consent control. Only when the user checks it does the extension submit the question, answer, and source timing references. Without that opt-in, the backend stores the rating, reason/comment (if given), video ID, answer/request identifiers, model/retrieval metadata, and no question/answer/source timing content. Conversation history, user identity, IP address, cookies, and transcript text are not stored by this feedback feature.

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

# Export the full review queue (including only context users explicitly shared).
# Keep this file private; evaluation_data/private/ is ignored by Git.
$output = "evaluation_data/private/feedback_queue_20261121.jsonl"
& $python -m scripts.phase11_feedback_queue export --output $output
~~~

To use a non-default database:

~~~powershell
& $python -m scripts.phase11_feedback_queue summary --db "D:\app-data\feedback.sqlite3"
& $python -m scripts.phase11_feedback_queue export --db "D:\app-data\feedback.sqlite3" --output "evaluation_data/private/feedback_queue_custom.jsonl"
~~~

The exporter refuses paths outside `evaluation_data/private/` and refuses to overwrite an existing export. Summary output contains counts only. Export output is sensitive and must not be committed or sent to public issue trackers.

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

The current extension response does not include transcript excerpts. Therefore fully automated semantic triage is deferred until a separate evidence-snapshot component can safely bind an opted-in report to exact indexed transcript passages and caption provenance. Only then should the bounded Phase 11.0B assessor produce a clearly marked **AI-proposed** label for human review. It must not create final labels, tune against held-out data, or trigger automatic model updates.

The next part of Phase 11.0E can add that evidence snapshot and AI-assisted triage workflow, then feed independently verified records into Phase 11.0C/11.0D. Phase 11.0F remains a separate calibrated development/held-out assessment.

## 7. Validation

The new unit tests cover validation, explicit context consent, source/video consistency, SQLite persistence, idempotent retries, aggregate-only summaries, and rejection of unexpected fields.

Run:

~~~powershell
& $python -m pytest tests/test_feedback_pipeline.py -q
& $python -m pytest -q
~~~

A subset test passing does not substitute for running the complete existing suite before merge.
