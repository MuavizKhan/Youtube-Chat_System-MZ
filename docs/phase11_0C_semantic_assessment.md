# Phase 11.0C — independent semantic evidence evaluation

- **Status:** isolated evaluation harness and provisional synthetic fixtures
- **Base:** merged Phase 11.0B / PR #157
- **Production integration:** explicitly excluded
- **Frozen Phase 10.11 baseline:** unchanged

## 1. Objective

Phase 11.0B implemented a bounded semantic evidence assessor. Passing mocked-provider tests establishes that request construction, response validation, provenance mapping, and failure handling work as designed. It does not establish that a live model correctly distinguishes semantic support from same-topic relevance, partial evidence, contradiction, conflict, or ambiguity.

Phase 11.0C supplies the independent measurement process required before any production integration is considered.

The phase does not modify retrieval, answer generation, FastAPI, the extension, model configuration, index contents, citation policy, or Phase 10.11 reports.

## 2. Evaluation boundary

The runner is scripts/phase11_0c_semantic_assessment_eval.py.

For each case, it constructs a provider-neutral QuestionAnalysis and bounded EvidencePassage list, then calls the Phase 11.0B assess_evidence_support function. It never calls retrieval or the production RAG chain to obtain passages.

Only question analysis and transcript passages are sent to the assessor. Gold labels, split metadata, annotator identities, and expected passage IDs are not included in the model prompt.

Reports contain case IDs, expected/predicted labels, scores, source passage IDs, safe diagnostics, and aggregate metrics. They deliberately omit raw transcript text, prompts, raw provider responses, and assessor summaries.

## 3. Dataset contract and split isolation

Dataset schema version: **1.0.0**.

Every case contains:
- a unique case ID and split (development or held_out);
- an independence group, question-language group, and transcript-language group;
- a QuestionAnalysis with distinct required evidence aspects;
- one or more transcript passages with exact video/chunk/timestamp references;
- one reference label per aspect, including expected status and expected supporting/contradicting passage IDs;
- source metadata and annotation metadata.

The validator checks the contract types, status/reference consistency, duplicate passage IDs and source references, question-language and transcript-language coverage, and cross-split leakage. Every passage in a case must match that case's declared transcript-language group.

**Leakage rule:** the same video, source group, or independence group may not occur in both development and held-out. Split by source video and scenario family, not by randomly scattering paraphrases of one underlying example.

Supported question and transcript languages in this phase are English (en), Hindi (hi), and Arabic (ar). They are measured separately: an English question over a Hindi or Arabic transcript is a cross-lingual case, not an English-transcript case. The seed includes one provisional cross-lingual case in each split. Urdu is intentionally excluded until a verified publisher-uploaded Urdu-script caption source is available.

### Seed fixture warning

evaluation_data/phase11_0C_semantic_assessment_seed.json contains 26 assistant-authored synthetic cases: 13 development and 13 held-out. Each split contains all six labels and all three question languages and transcript languages.

These examples exist only to validate the runner, split checks, metrics, and reporting pipeline. They are not real YouTube transcripts, have not been independently reviewed, and must not be used to claim live-model accuracy or select a production threshold.

The CLI refuses provisional datasets by default. The --allow-provisional-labels flag is an explicit smoke-test override; reports generated from provisional labels remain ineligible for a quality gate.

## 4. Human annotation protocol for the real dataset

Replace or extend the seed fixture with transcript excerpts from verified source videos. Every case in a real dataset must be reviewed independently by two human annotators before adjudication.

### Blind independent review

1. Two annotators independently label every aspect without seeing assessor predictions, model scores, each other's labels, or the eventual adjudicated label.
2. Each annotator records the expected status and exact passage IDs that support and/or contradict the aspect.
3. Each reviewer uses a distinct reviewer ID. The adjudicator uses a third, distinct ID.
4. A separate adjudicator reviews disagreements, records the final adjudicated label in \`labels\`, and resolves the source-reference decision. Do not overwrite or discard the two original review records.
5. Preserve source video ID, caption language/track, source URL, chunk identity, and timestamp interval for each passage.
6. Use source videos and scenario families that do not cross the development/held-out boundary.
7. Keep the held-out labels inaccessible to anyone tuning the prompt, model, or threshold.

The case-level \`annotation\` object must contain:
- \`status\`: \`adjudicated\`;
- \`reviewer_ids\`: exactly the two independent reviewer IDs;
- \`adjudicator_id\`: the distinct final adjudicator ID;
- \`blinding_attested\`: \`true\`, confirming neither reviewer saw model predictions or the other reviewer's labels;
- \`independent_reviews\`: exactly two objects, each containing \`reviewer_id\` and a complete \`labels\` array.

Each independent review's \`labels\` array must have exactly one entry for every aspect and use the same reference-label schema as the final \`labels\` array:

\`\`\`json
{
  "status": "adjudicated",
  "reviewer_ids": ["annotator-A", "annotator-B"],
  "adjudicator_id": "annotator-C",
  "blinding_attested": true,
  "independent_reviews": [
    {
      "reviewer_id": "annotator-A",
      "labels": [
        {
          "aspect_id": "cause",
          "expected_status": "supported",
          "supporting_passage_ids": ["p1"],
          "contradicting_passage_ids": []
        }
      ]
    },
    {
      "reviewer_id": "annotator-B",
      "labels": [
        {
          "aspect_id": "cause",
          "expected_status": "partial",
          "supporting_passage_ids": ["p1"],
          "contradicting_passage_ids": []
        }
      ]
    }
  ]
}
\`\`\`

The example above intentionally shows a disagreement: the adjudicator must determine the final case label and retain both initial reviews. It is a schema illustration, not a real annotation.

Reference-label semantics must match the Phase 11.0A/11.0B contracts:
- \`supported\`: clear substantive support;
- \`partial\`: relevant but incomplete support;
- \`missing\`: supplied passages do not address the aspect;
- \`contradicted\`: supplied evidence directly refutes the aspect;
- \`conflicted\`: material evidence supports and contradicts the aspect;
- \`unknown\`: the excerpt is too ambiguous or incomplete to assess reliably.

Do not label a same-topic passage as supported merely because it shares keywords. Preserve negation, direction, quantities, dates, attribution, causality, and temporal context. For unknown, record why the excerpt is genuinely indeterminate rather than merely difficult.

The validator rejects incomplete reviewer labels, duplicated reviewer IDs, reviewer-ID mismatches, missing blinding attestation, and an adjudicator who is also one of the reviewers. A final dataset must set the manifest and every case annotation status to \`adjudicated\`, use real source metadata including \`source_url\` and \`transcript_track\`, and retain the individual review records. Synthetic fixtures cannot be marked adjudicated.

### Inter-annotator agreement

Before adjudication, the harness computes nominal two-rater Cohen's kappa from the independent status labels:
- overall across all reviewed aspects;
- separately by question language;
- separately by transcript language.

The report also includes observed agreement and expected agreement. Kappa is undefined when chance agreement is 1.0; in that case the value is recorded as unavailable rather than fabricated. Agreement statistics measure consistency, not truth. A high kappa does not prove that labels are correct; review disagreements and final adjudications still require human scrutiny. No automatic kappa cut-off is imposed by this phase.


## 5. Metrics and interpretation

The report computes:
- six-class confusion matrix, per-class precision/recall/F1, overall accuracy and macro F1;
- pre-adjudication inter-annotator observed agreement, expected agreement, and Cohen's kappa overall and by question/transcript language;
- binary supported precision/recall/F1, false-positive supported rate on all non-supported labels, and specificity;
- Wilson 95% intervals for supported precision and recall;
- per-question-language and per-transcript-language confusion matrices and classification metrics;
- alignment of predicted supporting/contradicting passage IDs with reference passage IDs;
- provider call count, result/failure categories, input/output token usage when available, and latency p50/p95/mean.

The most safety-critical error is a false-positive supported result: the assessor accepts a passage that does not actually establish the requested aspect. Review these examples individually, especially same-topic irrelevance, reversed relationships, negation, speaker attribution, and cross-lingual cases.

The support_score remains an uncalibrated assessor-strength label, not a probability. Do not compute or report Brier scores or interpret it as calibrated confidence. Token counts are observed from provider response usage when supplied; missing usage is recorded as unavailable, not zero. The 8,000-character request ceiling is not tokenizer preflight. The report must distinguish characters from provider-reported input tokens; it must not infer token counts from characters. If provider usage is absent, token usage remains unavailable.

A report never emits an automatic quality_passed decision. It reports whether the dataset and run are eligible for final human review. Eligibility requires a complete held-out run with human-adjudicated labels, every assessment call completed with valid output, every held-out case represented exactly once, at least 100 aspect observations, at least 10 examples per status, and at least 20 cases in each question-language and transcript-language cohort. These are minimum review-volume safeguards, not proof of statistical adequacy or model quality.

## 6. Threshold calibration protocol

The development set is the only place to choose minimum_support_score.

1. Run the complete development split at a documented candidate threshold.
2. Repeat development runs at other predeclared thresholds only when needed; retain every report.
3. Compare supported precision/recall, false-supported rate, per-status errors, language metrics, operational errors, and the underlying false-positive examples.
4. Record the selected threshold and rationale in a threshold-lock JSON file that references the exact dataset SHA-256 and the SHA-256 of a completed development report.
5. Run the held-out split using --threshold-lock. The runner verifies the lock and rejects a threshold, provider, model, reasoning-effort, output-token configuration, assessor-code, or evaluation-runner mismatch.

Do not run multiple candidate thresholds on held-out data and do not use held-out results to revise the threshold, prompt, examples, or model. Any such change requires a new untouched held-out cohort.

Example lock file (replace the placeholders with actual values):

evaluation_data/phase11_0C_threshold_lock.json

\`\`\`json
{
  "schema_version": "1.0.0",
  "dataset_id": "your-adjudicated-dataset-id",
  "dataset_sha256": "64-character-dataset-sha256",
  "annotation_status": "adjudicated",
  "assessor_protocol": {
    "provider": "groq",
    "model": "openai/gpt-oss-20b",
    "temperature": 0.0,
    "reasoning_effort": "low",
    "max_input_chars": 8000,
    "max_passages": 12,
    "max_aspects": 8,
    "configured_max_output_tokens": 1200,
    "effective_max_output_tokens": 900,
    "assessor_module_sha256": "64-character-assessor-module-sha256",
    "contracts_module_sha256": "64-character-contracts-module-sha256",
    "config_module_sha256": "64-character-config-module-sha256",
    "client_factory_module_sha256": "64-character-client-factory-module-sha256",
    "evaluation_runner_sha256": "64-character-evaluation-runner-sha256"
  },
  "minimum_support_score": 0.75,
  "development_report_path": "../evaluation_reports/phase11_0C_development.json",
  "development_report_sha256": "64-character-report-sha256",
  "selected_by": "reviewer-or-owner-id",
  "selection_rationale": "Describe development-only precision/recall and error review."
}
\`\`\`

The lock is bound to an exact human-adjudicated dataset revision, complete development case set, completed development report, provider/model settings, and hashes of the assessor, evidence contracts, configuration, provider-client factory, and evaluation runner. Provisional/synthetic development data cannot create a threshold lock. The development and held-out runs must use the same provider/model and non-secret inference configuration. It is not a substitute for preserving all development reports and annotator records.

## 7. Commands

From the repository root in PowerShell:

Validate the provisional seed without calling a model:

\`\`\`powershell
$python = ".\\.venv\\Scripts\\python.exe"
& $python scripts/phase11_0c_semantic_assessment_eval.py --dataset evaluation_data/phase11_0C_semantic_assessment_seed.json --split development --allow-provisional-labels --validate-only
& $python scripts/phase11_0c_semantic_assessment_eval.py --dataset evaluation_data/phase11_0C_semantic_assessment_seed.json --split held_out --allow-provisional-labels --validate-only
\`\`\`

A small provider smoke run (this makes up to two live provider calls and is not a quality evaluation):

\`\`\`powershell
& $python scripts/phase11_0c_semantic_assessment_eval.py --dataset evaluation_data/phase11_0C_semantic_assessment_seed.json --split development --allow-provisional-labels --max-cases 2 --minimum-support-score 0.75 --sleep-seconds 2
\`\`\`

The smoke run is partial by design. It cannot generate a threshold-lockable development report.

A full human-adjudicated development run:

\`\`\`powershell
& $python scripts/phase11_0c_semantic_assessment_eval.py --dataset evaluation_data/phase11_0C_semantic_assessment.json --split development --minimum-support-score 0.75 --sleep-seconds 2 --output evaluation_reports/phase11_0C_development_075.json
\`\`\`

The held-out run requires a completed development report and its threshold lock:

\`\`\`powershell
& $python scripts/phase11_0c_semantic_assessment_eval.py --dataset evaluation_data/phase11_0C_semantic_assessment.json --split held_out --threshold-lock evaluation_data/phase11_0C_threshold_lock.json --sleep-seconds 2 --output evaluation_reports/phase11_0C_held_out_final.json
\`\`\`

Each case receives at most one application-level assessment call. There are no runner retries. On a provider rate-limit response, the runner stops further calls and saves a partial report. Check provider quota before a full live run.

## 8. Acceptance and non-goals

Phase 11.0C is accepted as an evaluation-harness implementation when:
1. Dataset validation rejects malformed labels, invalid provenance, and cross-split leakage.
2. Metrics tests cover all six statuses, supported false positives, per-question-language and per-transcript-language aggregation, and confidence intervals.
3. Threshold locks reject changed datasets, changed development reports, incomplete development runs, and threshold mismatch.
4. Reports contain no transcript text or raw prompts.
5. Direct CLI help and validate-only invocation work without API keys.
6. Full local and CI tests pass.
7. No production RAG code, Phase 10.11 config/index/report, dependency, or API behavior changes.

The phase is **not** complete as a semantic-quality claim until real excerpts have been independently adjudicated and the development and held-out runs have been performed and reviewed. No production integration is authorized by this phase.
