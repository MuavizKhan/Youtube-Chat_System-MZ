# Phase 11.6 — resumable full-registry live answer/source acceptance

- **Status:** implementation PR
- **Base:** merged PR #167, commit **10b248c09411166d8c44890c6d39a371ef4fd95b**
- **Release target:** all 43 registered answer/source cases across the five pilot videos
- **Production retrieval/configuration changes:** none
- **Index rebuild or mutation:** none

## 1. Why this phase exists

The Phase 11.0 runner already supports subset selection, but subset runs
deliberately disable the full-registry coverage gate. Each invocation writes
one report after its selected cases finish. This makes a 43-case live run
difficult to resume cleanly when provider quota interruptions leave a mixture
of successful and failed requests.

Phase 11.6 adds a separate consolidator. It accepts saved Phase 11.0 subset
reports, validates that they came from the same clean audit-runner commit and
backend base URL, checks the frozen acceptance configuration, and recomputes
the complete 43-case registry accounting and quality gate. It does not call
the backend or an inference provider.

## 2. New tooling

The new module is:

    scripts/merge_chat_answer_source_audit_reports.py

Example:

    .\.venv\Scripts\python.exe -m scripts.merge_chat_answer_source_audit_reports --reports evaluation_reports/batch-01.json evaluation_reports/batch-02.json --output evaluation_reports/phase11_6_full_registry_consolidated.json

The report includes:
- Cases attempted, succeeded, and failed, distinct from semantic quality failures.
- Full registry accounting and unevaluated case IDs.
- Effective configuration consistency and frozen-configuration comparison.
- Every selected per-case answer/source result, source intervals and quality checks.
- Source report names and SHA-256 hashes.
- Case attempt history and explicit handling of transport retries.
- Separate retry and manual-review case lists.

A valid but incomplete consolidation is still written for diagnosis, but returns
exit code 1 and cannot pass acceptance.

## 3. Provenance and duplicate handling

Every source report must:
1. Use schema 1.2.0 and the Phase 11.0 subset report ID.
2. Declare the exact frozen Phase 10.11 configuration below.
3. Come from a clean working tree and include a full 40-character commit SHA.
4. Use the same audit-runner commit and backend base URL as the other batches.
5. Contain unique case IDs that map to the canonical 43-case registry.
6. Preserve the canonical case-to-video identity.
7. Have internally consistent attempted, succeeded and failed counts.

A later successful HTTP attempt may replace an earlier transport failure for
the same case. A later transport failure cannot replace an earlier success.
Two successful attempts for the same case are rejected: the consolidator will
not select a more favorable stochastic answer. A quality-failed HTTP-success
case is listed for diagnosis and manual review, not blind retries against the
same code version.

The consolidator records the raw source report SHA-256 hashes so the combined
result can be traced back to the exact batch files.

## 4. Frozen acceptance configuration

The consolidator rejects a report that does not declare this exact configuration:

| Setting | Required value |
|---|---|
| Embedding model | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Namespace | phase10-10-acceptance-minilm-20261023 |
| Citation policy | prompt_included_chunks |
| top_k / fetch_k | 4 / 10 |
| lambda_mult / max_distance | 0.7 / 1.3 |
| Context expansion / max chunks | 1 / 12 |
| Dense semantic k / fetch k | 24 / 32 |
| Dense anchor limit / min gap | 12 / 2 |
| Dense context max chunks | 16 |
| Dense facet rerank weight | 0.5 |
| Reranker | disabled |
| Reranker model | cross-encoder/ms-marco-MiniLM-L6-v2 |
| Reranker candidate / batch / max length | 24 / 16 / 512 |

No accepted index, retrieval, prompt, citation, timestamp, or model setting is
changed by this phase.

## 5. Recommended live execution

After this PR is merged, sync the local repository to the resulting clean
main commit. Restart the backend on that commit with
RAG_INDEX_NAMESPACE=phase10-10-acceptance-minilm-20261023 set before importing
backend modules. Confirm the accepted Kingfisher index is ready. Keep the
backend running and run each batch from a second terminal. Use a new output
filename for every batch.

Suggested batch plan (all 43 IDs exactly once):

| Batch | Case IDs |
|---|---|
| 01 | Q01, Q02, Q03, Q04, Q05 |
| 02 | Q06, Q07, Q08, Q09, Q10 |
| 03 | Q11, Q12, Q13, EW01, EW02 |
| 04 | EW03, EW04, EW05, AI01, AI02 |
| 05 | AI03, AI04, AI05, HI01, HI02 |
| 06 | HI03, HI04, HI05, HI06, HI07 |
| 07 | HI08, HI09, HI10, AR01, AR02 |
| 08 | AR03, AR04, AR05, AR06, AR07 |
| 09 | AR08, AR09, AR10 |

Use this complete PowerShell block from the audit terminal after syncing to main
and restarting the backend on the same clean commit. It checks the working tree,
captures the runner commit, verifies the accepted index, and saves a separate
JSON report after each batch. It stops if any batch returns a non-zero result,
so inspect that batch report before deciding whether to retry or continue.

    $ErrorActionPreference = "Stop"
    Set-Location -LiteralPath "C:\Users\Muaviz Khan\OneDrive\Desktop\All Projects\Youtube_Video_ChatSystem"

    $python = ".\.venv\Scripts\python.exe"
    $env:RAG_INDEX_NAMESPACE = "phase10-10-acceptance-minilm-20261023"

    $gitStatus = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0 -or $gitStatus.Count -ne 0) {
        throw "The working tree must be clean before the live audit."
    }
    $runnerCommit = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $runnerCommit.Length -ne 40) {
        throw "Could not verify the current Git commit."
    }

    $status = Invoke-RestMethod "http://127.0.0.1:8000/index/MdeQMVBuGgY"
    if ($status.state -ne "ready" -or -not $status.ready) {
        $status | Format-List
        throw "The accepted Kingfisher index is not ready."
    }

    $batches = @(
        @{ Name = "01"; CaseIds = "Q01,Q02,Q03,Q04,Q05" }
        @{ Name = "02"; CaseIds = "Q06,Q07,Q08,Q09,Q10" }
        @{ Name = "03"; CaseIds = "Q11,Q12,Q13,EW01,EW02" }
        @{ Name = "04"; CaseIds = "EW03,EW04,EW05,AI01,AI02" }
        @{ Name = "05"; CaseIds = "AI03,AI04,AI05,HI01,HI02" }
        @{ Name = "06"; CaseIds = "HI03,HI04,HI05,HI06,HI07" }
        @{ Name = "07"; CaseIds = "HI08,HI09,HI10,AR01,AR02" }
        @{ Name = "08"; CaseIds = "AR03,AR04,AR05,AR06,AR07" }
        @{ Name = "09"; CaseIds = "AR08,AR09,AR10" }
    )

    foreach ($batch in $batches) {
        $output = "evaluation_reports/phase11_6_batch_$($batch.Name).json"
        Write-Host "Running batch $($batch.Name): $($batch.CaseIds)"
        & $python -m scripts.chat_answer_source_audit --suite phase11_0 --case-ids $batch.CaseIds --output $output --expected-namespace "phase10-10-acceptance-minilm-20261023" --expected-embedding-model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Batch $($batch.Name) did not pass. Saved report: $output"
            throw "Stop here; inspect the saved batch report before continuing."
        }
        Write-Host "Saved: $output"
    }

A batch can fail because of a transport/quota problem or because an HTTP-success
answer failed its quality checks. These are different recovery paths.

For transport failures, identify only the cases whose result status is not
success, then rerun those IDs with a new report path. Example:

    $retryCaseIds = "Q03,Q04"
    $retryOutput = "evaluation_reports/phase11_6_batch_02_retry_01.json"
    & $python -m scripts.chat_answer_source_audit --suite phase11_0 --case-ids $retryCaseIds --output $retryOutput --expected-namespace "phase10-10-acceptance-minilm-20261023" --expected-embedding-model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

The example case IDs above are placeholders; use only the actual transport-failed
IDs in your saved report. Include both the original report and every retry report
in the final consolidation. The consolidator permits a later success to replace
an earlier transport failure, but will reject two HTTP-success attempts for the
same case.

For an HTTP-success answer-quality failure, do not simply rerun the same case
and choose whichever model answer passes. Review the failed checks and trace,
make a targeted code/annotation correction only if justified, then restart from
a clean commit. Reports produced by different commits cannot be consolidated.


If a batch has transport failures from provider quota, wait until provider
quota is available and rerun only those failed case IDs into a new report.
Include both the original failed report and retry report during consolidation;
the duplicate policy replaces only the prior transport failure with the later
successful response.

Do not blindly rerun cases that returned HTTP 200 but failed an answer-quality
check. Diagnose those cases first. If a code change is required, use a new clean
runner commit and keep reports from different commits separate.

## 6. Consolidation command

After the nine original batch reports and any transport-retry reports are saved,
run this command. Add any retry report paths to the list; do not remove the
original reports that contain the first failed attempts.

    $ErrorActionPreference = "Stop"
    Set-Location -LiteralPath "C:\Users\Muaviz Khan\OneDrive\Desktop\All Projects\Youtube_Video_ChatSystem"
    $python = ".\.venv\Scripts\python.exe"
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $output = "evaluation_reports/phase11_6_full_registry_consolidated_$timestamp.json"
    $expectedRunnerCommit = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $expectedRunnerCommit.Length -ne 40) { throw "Could not verify the current Git commit." }

    & $python -m scripts.merge_chat_answer_source_audit_reports --reports evaluation_reports/phase11_6_batch_01.json evaluation_reports/phase11_6_batch_02.json evaluation_reports/phase11_6_batch_03.json evaluation_reports/phase11_6_batch_04.json evaluation_reports/phase11_6_batch_05.json evaluation_reports/phase11_6_batch_06.json evaluation_reports/phase11_6_batch_07.json evaluation_reports/phase11_6_batch_08.json evaluation_reports/phase11_6_batch_09.json --output $output --expected-runner-commit $expectedRunnerCommit

    Write-Host "Consolidated report: $output"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Acceptance is incomplete or failed. Inspect missing, transport-failed, and quality-failed IDs in the report."
        throw "Phase 11.6 full-registry acceptance did not pass."
    }

The batch filenames above are a canonical naming convention. Timestamp suffixes
may be added as long as the exact paths are supplied to the consolidator.

## 7. Acceptance gate

Phase 11.6 passes only when the consolidated report confirms:
- 43 unique registered cases attempted, with zero missing or duplicate cases.
- Every case returned a successful HTTP response.
- All 43 per-case answer/source quality checks passed.
- One consistent effective server configuration across all cases.
- Effective configuration exactly matches the frozen settings.
- Complete five-video representative coverage and full-registry accounting.
- No timestamp/source-provenance, encoding, language, or context-budget failures.

The report's cases_succeeded remains an HTTP/API count. It is not a semantic
quality score. quality_passed is the release gate.

## 8. Explicit non-goals

- No live semantic evidence-assessor integration. The Phase 11.0B/11.0C
  documentation still requires independently adjudicated development and
  held-out labels before production integration.
- No change to generation answer thresholds or curated evidence annotations.
- No retrieval, MMR, chunking, embedding, reranker, citation, timestamp, or
  generation-context budget changes.
- No production API or Chrome extension changes.
- No Urdu support claim; Urdu remains blocked pending verified
  publisher-uploaded Urdu-script captions.
