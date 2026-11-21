# Phase 11.16 — user feedback and evaluation pipeline

- Base: Phase 11.15 accepted baseline (dd330859).
- Scope: feedback capture, local triage, privacy-conscious candidate export, and connection to the existing 11.0C/11.0D evaluation process.
- Production RAG generation, retrieval, indexes, embedding model, prompts, and frozen 43-case acceptance configuration are not changed.
- This phase adds a feedback API and extension UI. It does not represent a public-production security approval.

## 1. Objective

Capture answer-specific feedback without assuming the user is always correct. A report is a signal to investigate, not a gold label and not an automatic model correction.

The pipeline distinguishes answer-quality issues (unsupported claim, missing detail, wrong attribution, incorrect timestamp/source), transcript-quality issues (missing, truncated, misheard, or language-mismatched captions), question/context issues (ambiguous wording, follow-up context, or a mismatch between what the user intended and what the assistant understood), and uncertain reports that need more context.

A user's report must never directly change prompts, retrieval configuration, evidence-assessor thresholds, indexes, or production behavior.

## 2. User experience

Each completed assistant answer has Yes and No feedback controls.

- Yes submits a helpful vote immediately.
- No opens an optional issue form.
- Issue categories: incorrect_or_unsupported, missing_information, wrong_attribution, wrong_timestamp_or_source, transcript_problem, question_context_misunderstood, uncertain_context, and other.
- An optional free-text note can explain the report.
- A separate, unchecked consent control permits the user to share that answer's question, answer, source references, model identifier, and retrieval configuration for quality review.
- The user is told that no transcript text or conversation history is shared by the extension feedback action, and is asked not to include private or sensitive information in notes.

If the user does not opt in to context sharing, the database stores only the answer request correlation ID, video ID, vote, selected category, and user-written note. It does not store the question, answer, or sources.

## 3. Backend contract and storage

POST /feedback accepts a bounded request containing:

- request_id: the X-Request-ID response header returned by the corresponding /chat response (the chat JSON contract remains unchanged);
- video_id: validated as a YouTube video reference;
- vote: helpful or not_helpful;
- optional reason and note;
- share_context: explicit boolean opt-in;
- optional bounded answer context (question, answer, source metadata, model and retrieval config).

The API validates that every included source belongs to the submitted video. It stores one feedback record per chat request, making retries idempotent and preventing multiple ratings of the same answer. There is no public read/list endpoint. Existing API rate limiting is applied to feedback submission.

The SQLite database defaults to Backend/rag_system/feedback.sqlite3 and may be relocated with FEEDBACK_DB_PATH. The existing .gitignore rule for *.sqlite3 keeps the runtime database out of source control. Back up and protect this database as user-submitted data in any deployment.

The database stores a review_status lifecycle:
- new: unreviewed user signal;
- triaged: grouped/initially inspected;
- verified_issue: a human checked the answer against the source and confirmed an issue;
- not_reproducible: the report could not be substantiated;
- transcript_issue: the underlying caption/source appears incomplete or inaccurate;
- needs_context: evidence is insufficient to classify the report;
- evaluation_candidate: human-approved for curation into a future evaluation dataset;
- closed: review completed without promoting it to evaluation.

These are operator workflow states, not model labels. In particular, verified_issue does not automatically decide one of the six Phase 11.0B evidence statuses.

## 4. Automated operator workflow

All queue-reading and review commands are local operator commands; the unauthenticated API never exposes stored reports to the public.

From the repository root in PowerShell:

~~~powershell
$python = ".\.venv\Scripts\python.exe"
& $python scripts/feedback_pipeline.py summary
& $python scripts/feedback_pipeline.py list --status new --limit 100
& $python scripts/feedback_pipeline.py review --feedback-id "<feedback-id>" --status needs_context --reviewer-id R01 --notes "The caption excerpt around the reported timestamp needs checking."
& $python scripts/feedback_pipeline.py review --feedback-id "<feedback-id>" --status evaluation_candidate --reviewer-id R01 --notes "Checked the YouTube caption and answer source; candidate for independent annotation."
& $python scripts/feedback_pipeline.py export-candidates --output evaluation_data/private/feedback_candidates.jsonl
~~~

The export is deliberately not a Phase 11.0C dataset. It contains the user report and answer context the user explicitly consented to share, but it does not invent source excerpts, transcript-track verification, aspect labels, independent reviews, or adjudication. A curator must revisit the original video, verify the caption track and exact timestamped excerpt, and prepare a source-only Phase 11.0D manifest. Only two blinded independent reviews plus a distinct adjudicator can produce an adjudicated label.

The existing Phase 11.0C runner remains the evaluation engine. Once the private development manifest is complete, run its development evaluation and select thresholds from development only. Keep held-out data separate and untouched; feedback-derived cases must not enter held-out data if they influenced prompt/model/threshold decisions.

## 5. What can be automated safely

Automated now:
- feedback capture and request correlation;
- storage minimisation based on explicit consent;
- basic vote/category/video aggregation;
- per-answer duplicate prevention;
- private handoff export for human-approved candidates;
- the existing six-class assessor evaluation, metric computation, language breakdowns, threshold-lock validation, and held-out leakage checks once real adjudicated manifests exist.

Still requires human verification:
- deciding whether a user's interpretation matches the video;
- checking caption completeness, transcript track and exact timestamps;
- confirming that a passage supports, partially supports, contradicts, conflicts with, or does not address a claim;
- independent annotation and adjudication;
- approval of any production behavior change.

A future AI pre-triage step may suggest likely issue categories, extract likely aspects, or point a curator to the relevant transcript interval. Its output must be labelled as a suggestion, hidden from blind reviewers, and excluded from gold labels and held-out threshold selection.

## 6. Privacy and public-beta boundaries

- No identity, email, account ID, IP address, or conversation history is stored by this feature.
- The API uses the existing IP-based rate limiter; that limiter is an abuse-control mechanism and not an identity system.
- The feedback endpoint accepts writes only; queue reads and review operations are local.
- User comments may contain personal information. The UI warns users not to include sensitive information; operators must apply an appropriate retention/deletion policy before inviting public users.
- Context sharing is opt-in and off by default.
- The initial SQLite implementation is appropriate for a single-instance controlled beta only. Multi-worker/shared deployment, authentication, admin access control, database backups, deletion requests, retention enforcement, and abuse controls belong to the Phase 12 release-infrastructure gate.
- The current release supports verified English, Hindi and Arabic transcript flows. Urdu must not be advertised as supported until the publisher-uploaded Urdu-script caption requirement is verified.

## 7. Acceptance checks

- A chat response's server request ID is carried through the extension and linked to one feedback record.
- Yes and No controls are attached to each successful AI answer, not to the global chat or the wrong video.
- Context is discarded unless the user explicitly opts in.
- Shared context contains only question, answer, source metadata, model and retrieval config; no raw transcript or conversation history.
- One answer can receive at most one rating.
- Public clients cannot read the feedback queue.
- Feedback categories and review lifecycle remain user signal / human workflow metadata, not ground-truth model labels.
- Only context-consented records manually marked evaluation_candidate can be exported, and exports must be under evaluation_data/private/.
- Unit/API/extension tests and the full existing suite pass.
- No production RAG generation/retrieval/index/config behavior changes.

## 8. Next work

After this feedback slice is reviewed and merged, continue with:
1. a curator-facing report review screen or richer local CLI;
2. optional AI-assisted triage that does not see or alter blind-review labels;
3. automated private source-candidate generation from verified transcript passages;
4. Phase 11.0E real development/held-out datasets with independent reviews;
5. Phase 11.0F development calibration and a once-only held-out evaluation;
6. Phase 12 single-instance deployment hardening, then multi-user authentication, persistent shared storage, abuse and cost controls before a public beta.
