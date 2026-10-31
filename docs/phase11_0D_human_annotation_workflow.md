# Phase 11.0D — real human annotation workflow

- Depends on merged Phase 11.0A, 11.0B and 11.0C.
- Implements local/private human-review packet operations only.
- Makes no provider/model calls.
- Does not integrate the assessor into production RAG or alter Phase 10.11.

## 1. Objective and boundary

Phase 11.0C provides the evaluator and final adjudicated-dataset contract. Phase 11.0D provides the operational process to produce those labels without exposing model predictions or one reviewer's labels to the other reviewer.

The workflow has four operations:

1. Validate a source-only manifest containing question analysis, transcript passages and source provenance.
2. Export two separate blind reviewer packets.
3. Combine the completed independent reviews into a packet for a third, distinct adjudicator.
4. Assemble and validate the final Phase 11.0C manifest.

The tool has no provider client, API calls, retrieval path, generated predictions, or access to model responses. All source manifests, reviewer packets, adjudication packets, and final adjudicated datasets contain transcript excerpts or human labels. They must remain under evaluation_data/private/, which is excluded by .gitignore. The tool resolves and enforces that path boundary for every input and output file and refuses to overwrite an existing packet or dataset.

## 2. Source-only manifest

Use one source manifest per split. Never put development and held-out cases in one file.

Suggested private layout:

- evaluation_data/private/phase11_0C_development_sources.json
- evaluation_data/private/phase11_0C_held_out_sources.json
- evaluation_data/private/annotation_packets/development/reviewer_R01.json
- evaluation_data/private/annotation_packets/development/reviewer_R02.json
- evaluation_data/private/annotation_packets/development/adjudicator_R03.json
- evaluation_data/private/phase11_0C_development.json
- equivalent separate files and packet folder for held-out

Use opaque case IDs matching CASE- followed by 3–16 uppercase letters/digits (for example, CASE-001 or CASE-00A91). Never encode language, expected status, challenge type, source cohort, or split in a case ID. Use pseudonymous reviewer IDs such as R01, R02, and R03. Do not store annotators' names, email addresses, employer details, or other personal information in dataset files. Reviewer IDs and the adjudicator ID must be unique within every case.

A source manifest has schema version 1.0.0:

~~~json
{
  "schema_version": "1.0.0",
  "dataset_id": "phase11_0C_development_v1",
  "split": "development",
  "cases": [
    {
      "case_id": "CASE-001",
      "split": "development",
      "independence_group": "video-and-scenario-family-001",
      "question_language_group": "en",
      "transcript_language_group": "en",
      "question_text": "What evidence does the speaker give for the stated cause?",
      "question_analysis": {
        "intent": "explanation",
        "language": "en",
        "video_id": "VIDEO_ID",
        "aspects": [
          {
            "aspect_id": "cause",
            "description": "The cause the speaker explicitly gives for the event.",
            "required": true,
            "weight": 1.0,
            "retrieval_hints": []
          }
        ]
      },
      "passages": [
        {
          "passage_id": "p1",
          "text": "Exact transcript excerpt, not a paraphrase.",
          "transcript_language": "en",
          "references": [
            {
              "video_id": "VIDEO_ID",
              "chunk_id": "stable-source-chunk-001",
              "start_seconds": 125.0,
              "end_seconds": 142.5
            }
          ]
        }
      ],
      "source": {
        "kind": "youtube_caption",
        "source_group": "video-and-source-family-001",
        "source_url": "https://www.youtube.com/watch?v=VIDEO_ID",
        "transcript_track": "en",
        "caption_track_kind": "auto_generated",
        "verification": {
          "video_id_confirmed": true,
          "caption_track_confirmed": true,
          "timestamps_confirmed": true,
          "verified_by": "CURATOR-01"
        }
      }
    }
  ]
}
~~~

Replace VIDEO_ID, timestamps and excerpt text with verified values. This is a schema illustration, not annotation evidence.

### Source-manifest requirements

- Every case must match the manifest's single split.
- Each case needs a stable unique case_id and independence_group.
- Question and transcript language groups must each be en, hi, or ar.
- question_analysis.video_id must match every passage reference and the video ID in the HTTPS YouTube source URL.
- Keep the actual user question in question_text, and describe every independently assessable evidence aspect in question_analysis.aspects.
- Use 1–8 aspects and 1–12 passages per case, matching the Phase 11.0B request limits.
- Each case must contain at least one required aspect. Each passage must be an exact transcript excerpt with at least one exact video/chunk/timestamp reference. Every reference needs both start_seconds and end_seconds, with end strictly after start.
- Keep the exact Phase 11.0B request within its 8,000-character hard limit. The validator uses the assessor's prompt builder and a maximum-length threshold representation, rather than estimating from transcript-text characters alone.
- source.kind must be youtube_caption. caption_track_kind must be auto_generated or publisher_uploaded, and transcript_track must match the declared transcript language group.
- Set source.verification.video_id_confirmed, caption_track_confirmed, and timestamps_confirmed to true only after manually checking the actual YouTube video, caption track, and every excerpt/timestamp. verified_by is a pseudonymous curator ID. These are audit attestations, not automatic YouTube API verification.
- Use stable source chunk IDs. A source chunk must not be assigned to two different passage IDs in the same case.
- The source URL must be an HTTPS YouTube watch, share, Shorts, live, or embed URL identifying the exact video.
- source_group and independence_group must identify source/scenario relationships that could create leakage. Reused or near-duplicate material belongs to the same group.
- The source-only manifest must not contain labels, assessor predictions, scores, confidence values, retrieval hints exposed to reviewers, or threshold-lock data.

The existing five-video Phase 11.0 registry can be considered as a development source shortlist, not as pre-adjudicated semantic-assessor data. Its registered video IDs include MdeQMVBuGgY, pOnl-Zndcls, A-JBaNvv3Tk, INpnuL7qnzY, and fAd2nKfrDB8. Reconfirm the actual YouTube caption track and exact timestamps for every excerpt. Existing retrieval gold phrases and answer-audit outcomes are not substitutes for two independent semantic evidence reviews. Do not reuse any development video in held-out data.

## 3. Sampling and split design

Question language (the language in which the question/aspects are expressed) and transcript language (the language of the source passage) are separate axes. Plan coverage across all nine pairs:

| Question language | Transcript language |
|---|---|
| en | en, hi, ar |
| hi | en, hi, ar |
| ar | en, hi, ar |

A cross-lingual pair is a real evaluation cohort, not an excuse to translate the source passage or relabel its language. Urdu remains excluded until a verified publisher-uploaded Urdu-script caption track is available.

### Development cohort

Development is the only cohort used for threshold selection and prompt/model/protocol decisions. Include all six evidence statuses and all three question and transcript languages. Prefer broad video and scenario diversity; do not fill a language pair with paraphrases of one source question.

A practical planning target is at least 100 cases and 150 aspect observations in development, balanced across statuses and language pairs. This is a planning recommendation, not a statistical guarantee or a validator-enforced minimum. Small pilot batches may test the annotation process but are not suitable for threshold claims.

### Held-out cohort

The Phase 11.0C eligibility floor is:

- at least 100 aspect observations;
- at least 10 aspect observations for each of the six statuses;
- at least 20 cases in each question-language cohort;
- at least 20 cases in each transcript-language cohort;
- at least 10 cases in each of the nine question→transcript language pairs;
- at least 20 distinct source videos, source groups, and independence groups.

The pair floor implies at least 90 held-out cases before exclusions. The Phase 11.0D source preflight enforces these cohort, aspect-volume, and source-diversity floors before exporting any reviewer packet. After adjudication, assembly also enforces at least 10 adjudicated aspects per status before allowing the held-out dataset to be evaluated. Plan above the minimum (for example, 120 or more cases, 30 or more distinct videos, and 150–200 aspects) to retain a buffer for invalid source tracks, ambiguous excerpts, or adjudication exclusions. These are safeguards against obviously underfilled or heavily repeated samples, not a statistical-power claim.

Do not split by random case rows. Assign videos, source groups and independence/scenario groups to one split only. Do not put alternate excerpts, paraphrases, or related segments of one source video on opposite sides. The Phase 11.0C threshold lock independently checks hashed source identities before a held-out provider call.

## 4. Blind independent review

1. Choose two distinct reviewer IDs and export two separate packet files.
2. Give each reviewer only their own packet. Do not share the other review file, model responses, support scores, previous gold labels, answer-audit outcomes, or threshold choices.
3. Each reviewer labels every aspect using the supplied passage IDs. Do not infer support from topic similarity or from the question's wording.
4. Each reviewer may record a concise rationale. It is a human audit note, not an instruction to the assessor.
5. Each reviewer sets blinding_attested to true only after confirming they did not see model predictions or the other reviewer's labels.
6. Return the completed packets without changing the source text or references. The assembler checks the source hash, exact source content, reviewer identity, all case IDs, and one label per aspect.

Use the six status meanings consistently:

- supported: the cited excerpt clearly establishes the requested aspect;
- partial: some substantive support exists, but it is incomplete;
- missing: the supplied passages do not address the aspect;
- contradicted: the cited evidence directly refutes the aspect;
- conflicted: material evidence both supports and contradicts the aspect;
- unknown: the supplied excerpt is too ambiguous or incomplete to decide reliably.

For each aspect, use supporting_passage_ids for passages that substantively support it, and contradicting_passage_ids for passages that materially contradict it. The Phase 11.0C contract checks that cited IDs exist and match the selected status. supported and partial require support; contradicted requires contradiction only; conflicted requires both; missing and unknown carry neither.

Preserve negation, direction, chronology, quantities, speaker attribution, causality, qualifications and uncertainty. Do not silently resolve a contradiction by choosing the passage that best matches the question. A same-topic excerpt that does not establish the requested relationship is not supported.

## 5. Third-person adjudication

The adjudicator must be distinct from both reviewers. The adjudication packet contains both independent labels side by side for each aspect, the original question and passages, and a blank final label.

The adjudicator must inspect disagreements and ambiguous/unknown cases, resolve the final status and exact passage IDs, retain both original reviews unchanged, and set adjudication_attested to true only after completing every final label.

Do not replace independent review records with the adjudicated labels. The final manifest retains both reviewer records and the distinct adjudicator ID so Phase 11.0C can calculate pre-adjudication Cohen's kappa.

## 6. Commands

Run from the repository root in PowerShell. All input and output paths below must be under the ignored private directory.

Validate a source-only manifest (no model calls):

~~~powershell
$python = ".\.venv\Scripts\python.exe"
& $python scripts/phase11_0d_annotation_workflow.py validate-source --source-manifest evaluation_data/private/phase11_0C_development_sources.json
~~~

Export two separate reviewer packets:

~~~powershell
& $python scripts/phase11_0d_annotation_workflow.py export-review-packets --source-manifest evaluation_data/private/phase11_0C_development_sources.json --reviewer-ids R01 R02 --output-dir evaluation_data/private/annotation_packets/development
~~~

Give reviewer_R01.json only to R01 and reviewer_R02.json only to R02. Each reviewer completes every label and changes blinding_attested to true. Do not overwrite or regenerate completed packets.

After both completed files are returned, build the adjudication packet:

~~~powershell
& $python scripts/phase11_0d_annotation_workflow.py build-adjudication-packet --source-manifest evaluation_data/private/phase11_0C_development_sources.json --reviewer-packets evaluation_data/private/annotation_packets/development/reviewer_R01.json evaluation_data/private/annotation_packets/development/reviewer_R02.json --adjudicator-id R03 --output evaluation_data/private/annotation_packets/development/adjudicator_R03.json
~~~

Give the adjudication packet only to R03. The adjudicator completes every adjudicated_labels entry and changes adjudication_attested to true.

Assemble and validate the final Phase 11.0C manifest:

~~~powershell
& $python scripts/phase11_0d_annotation_workflow.py assemble --source-manifest evaluation_data/private/phase11_0C_development_sources.json --reviewer-packets evaluation_data/private/annotation_packets/development/reviewer_R01.json evaluation_data/private/annotation_packets/development/reviewer_R02.json --adjudication-packet evaluation_data/private/annotation_packets/development/adjudicator_R03.json --output evaluation_data/private/phase11_0C_development.json
~~~

The final command runs the Phase 11.0C validate_manifest contract before writing. It rejects incomplete status/language coverage as well as malformed labels, unverified provenance, missing timestamps, incomplete reviews, and invalid annotator identities. The output is private and is not committed.

### Held-out commands

The held-out source preflight, reviewer export, adjudication-packet build, and final assembly must all be supplied the development source manifest. The workflow checks video IDs, source groups, and independence groups for overlap before any held-out reviewer packets can be issued.

~~~powershell
$devSource = "evaluation_data/private/phase11_0C_development_sources.json"
$testSource = "evaluation_data/private/phase11_0C_held_out_sources.json"

& $python scripts/phase11_0d_annotation_workflow.py validate-source --source-manifest $testSource --development-source-manifest $devSource

& $python scripts/phase11_0d_annotation_workflow.py export-review-packets --source-manifest $testSource --development-source-manifest $devSource --reviewer-ids R11 R12 --output-dir evaluation_data/private/annotation_packets/held_out

& $python scripts/phase11_0d_annotation_workflow.py build-adjudication-packet --source-manifest $testSource --development-source-manifest $devSource --reviewer-packets evaluation_data/private/annotation_packets/held_out/reviewer_R11.json evaluation_data/private/annotation_packets/held_out/reviewer_R12.json --adjudicator-id R13 --output evaluation_data/private/annotation_packets/held_out/adjudicator_R13.json

& $python scripts/phase11_0d_annotation_workflow.py assemble --source-manifest $testSource --development-source-manifest $devSource --reviewer-packets evaluation_data/private/annotation_packets/held_out/reviewer_R11.json evaluation_data/private/annotation_packets/held_out/reviewer_R12.json --adjudication-packet evaluation_data/private/annotation_packets/held_out/adjudicator_R13.json --output evaluation_data/private/phase11_0C_held_out.json
~~~

Use a distinct reviewer/adjudicator group for held-out where practical. Keep held-out labels inaccessible to anyone changing the assessor or selecting the development threshold.

The adjudication packet is bound to the canonical SHA-256 of each completed reviewer packet. Changing either review after the adjudicator packet was produced invalidates assembly; regenerate the adjudication packet only after the adjudicator has reviewed the correct two revisions.

Finally, validate both assembled real datasets using the Phase 11.0C evaluator with --validate-only (without --allow-provisional-labels). The held-out assembler also enforces at least 10 adjudicated aspect observations per status before the evaluator can make any provider calls. Only after the development cohort is adjudicated and validated should threshold calibration begin.

## 7. Acceptance gates and stop conditions

Phase 11.0D is accepted as annotation-workflow tooling when:

1. The source-manifest validator rejects fake/synthetic sources, malformed language/aspect structures, unsupported passage counts, invalid/mismatched YouTube provenance, missing timestamps and duplicate IDs.
2. Source manifests and every generated packet are restricted to evaluation_data/private/.
3. Exported reviewer packets contain no existing labels, predictions, support scores, thresholds, split identity, or independence-group metadata.
4. The two reviewer IDs are distinct, every reviewer completes every aspect, and each explicitly attests to blind independent review.
5. The adjudicator is a distinct third identity and completes every aspect.
6. Review and adjudication packets remain bound to the exact source-manifest hash and source passage contents; edits to source text, timestamps, reviewer labels or identities are rejected.
7. The final assembled manifest passes the Phase 11.0C adjudicated dataset validator.
8. Unit tests, full CI, and local Windows validation pass.
9. No production RAG behavior, Phase 10.11 artifacts, dependencies, indexes, or API behavior change.

This phase does not itself create real labels, assert inter-annotator agreement, calibrate a threshold, run the model, establish semantic accuracy, or authorize production integration. Those require completed human work and the separate Phase 11.0C development/held-out evaluation protocol.
