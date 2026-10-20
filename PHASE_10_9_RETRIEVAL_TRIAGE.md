# Phase 10.9 — Retrieval reliability and failure diagnosis

## Objective

Reach a reliable, maintainable level of YouTube transcript question answering without open-ended fine-tuning. Phase 10.9 focuses on reproducibly locating the point where required evidence is lost, then applying only corrections supported by the trace.

This PR adds a targeted runner around the repository's existing retrieval diagnostic machinery. It does not change production retrieval, generation, prompts, embeddings, ranking, thresholds, citation policy, or index format.

## Known investigation targets

These targets are based on the latest recorded five-video evaluation. They are diagnostic priorities, not hard-coded expected outcomes.

| Video | Target questions | Prior observation | First check |
|---|---|---|---|
| Kingfisher interview — `MdeQMVBuGgY` | `Q03` | Production context missed a required Kingfisher Red evidence group | Confirm gold phrase exists in the actual index, then trace raw FAISS → MMR → fusion → anchors → context |
| Agentic engineering workflow — `A-JBaNvv3Tk` | `AI01` | At least one required evidence group was missing | Check every required group independently and identify the first stage where its candidates disappear |
| Arabic silence video — `fAd2nKfrDB8` | `AR01`–`AR04` | Earlier report showed zero evidence for these evaluated cases | Verify index state and actual transcript language first; distinguish a wrong/stale index from a retrieval miss |

The Urdu benchmark is explicitly blocked pending verification of a genuine Urdu-script caption track. We must not fabricate a benchmark by treating an unverified translation as source truth.

## Exact order of execution

### 1. Freeze and record the baseline

This branch is based on main commit `4a3d636` (Phase 10.8.8). The Phase 10.9 changes in this PR are diagnostic-only, so the production retrieval path under test is unchanged. Preserve the old five-video JSON report. Its `experiment_metadata` is the reference for the embedding model, index namespace, Python version, Git revision, and effective retrieval settings. New reports will record the Phase 10.9 branch commit as their current Git provenance.

Inspect the prior report before setting any namespace or model override:

```powershell
$prior = Get-Content ".\evaluation_reports\phase10_4_multivideo_topk7_distance150_lambda070_20261017.json" -Raw | ConvertFrom-Json
$prior.experiment_metadata | ConvertTo-Json -Depth 8
```

For comparison with that recorded run, the known key settings were `RAG_TOP_K=7`, `RAG_MAX_DISTANCE=1.50`, and `RAG_MMR_LAMBDA=0.70`. The following restores the embedding model, namespace, chunking settings, and all retrieval knobs recorded in the old report so unrelated shell or `.env` overrides do not silently change the baseline.

```powershell
$meta = $prior.experiment_metadata
$env:HF_EMBEDDING_MODEL = [string]$meta.embedding_model
$env:RAG_INDEX_NAMESPACE = [string]$meta.index_namespace
$env:RAG_CHUNK_SIZE = [string]$meta.index_config.chunk_size
$env:RAG_CHUNK_OVERLAP = [string]$meta.index_config.chunk_overlap

$envMap = @{
    top_k = "RAG_TOP_K"
    fetch_k = "RAG_MMR_FETCH_K"
    lambda_mult = "RAG_MMR_LAMBDA"
    max_distance = "RAG_MAX_DISTANCE"
    recovery_max_distance = "RAG_RECOVERY_MAX_DISTANCE"
    recovery_top_k = "RAG_RECOVERY_TOP_K"
    recovery_fetch_k = "RAG_RECOVERY_FETCH_K"
    rrf_k = "RAG_RRF_K"
    dense_lexical_rrf_weight = "RAG_DENSE_LEXICAL_RRF_WEIGHT"
    context_expansion_chunks = "RAG_CONTEXT_EXPANSION_CHUNKS"
    context_max_chunks = "RAG_CONTEXT_MAX_CHUNKS"
    dense_semantic_k = "RAG_DENSE_SEMANTIC_K"
    dense_semantic_fetch_k = "RAG_DENSE_SEMANTIC_FETCH_K"
    dense_anchor_limit = "RAG_DENSE_ANCHOR_LIMIT"
    dense_anchor_min_chunk_gap = "RAG_DENSE_ANCHOR_MIN_CHUNK_GAP"
    dense_context_max_chunks = "RAG_DENSE_CONTEXT_MAX_CHUNKS"
    dense_facet_rerank_weight = "RAG_DENSE_FACET_RERANK_WEIGHT"
    rerank_enabled = "RAG_RERANK_ENABLED"
    rerank_model = "RAG_RERANK_MODEL"
    rerank_candidate_k = "RAG_RERANK_CANDIDATE_K"
    rerank_batch_size = "RAG_RERANK_BATCH_SIZE"
    rerank_max_length = "RAG_RERANK_MAX_LENGTH"
}
foreach ($key in $envMap.Keys) {
    $setting = $meta.effective_retrieval_config.PSObject.Properties[$key]
    if ($null -eq $setting) {
        throw "Prior report is missing effective retrieval setting: $key"
    }
    [Environment]::SetEnvironmentVariable($envMap[$key], [string]$setting.Value, "Process")
}

if ([int]$meta.effective_retrieval_config.top_k -ne 7 -or
    [double]$meta.effective_retrieval_config.max_distance -ne 1.50 -or
    [double]$meta.effective_retrieval_config.lambda_mult -ne 0.70) {
    throw "The selected prior report does not contain the documented top_k=7, max_distance=1.50, lambda=0.70 baseline."
}

$env:PYTHONIOENCODING = "utf-8"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
```

If the prior report is unavailable, its metadata is incomplete, or the matching index is missing, stop and label the next run a new baseline rather than claiming a direct comparison.

From the repository root, after matching the configuration:

```powershell
$baseline = ".\evaluation_reports\phase10_9_multivideo_baseline_$stamp.json"

.\.venv\Scripts\python.exe -m Backend.rag_system.multi_video_evaluation --output $baseline
$baselineExitCode = $LASTEXITCODE
Write-Host "Baseline report: $baseline"
Write-Host "Exit code: $baselineExitCode"
```

A non-zero exit code is useful benchmark evidence and is not a script crash by itself. Preserve the report either way.

Before interpreting quality results, check the report's five selected videos, evaluation status, per-video transcript language, embedding model, index namespace, and effective settings. Do not rebuild any index merely to make the run proceed.

### 2. Run the focused stage-by-stage triage

```powershell
$triage = ".\evaluation_reports\phase10_9_retrieval_triage_$stamp.json"

.\.venv\Scripts\python.exe -m scripts.phase10_9_retrieval_triage --output $triage
$triageExitCode = $LASTEXITCODE
Write-Host "Triage report: $triage"
Write-Host "Exit code: $triageExitCode"
```

This command uses the existing gold-evidence retrieval trace and captures:

- Git/runtime and effective retrieval configuration provenance.
- SHA-256 hashes and sizes for each target's `index.faiss`, `index.pkl`, and `metadata.json`, plus metadata schema, transcript language code, chunk count, and index state. Index files are hashed with streaming reads, and full transcript text is not copied into the snapshot.
- Candidate membership/rank evidence from raw FAISS, semantic MMR, semantic fusion, lexical retrieval, hybrid fusion, facet reranking, anchor selection, and production context.
- The repository's existing group/candidate diagnosis and a concise stage-presence summary.

The runner does not generate final chatbot answers, modify indexes, or change retrieval behavior. It deliberately blocks a trace when the index is missing/stale/invalid or its recorded transcript language conflicts with the registered benchmark. Such a result means the diagnostic run is not comparable yet; it is not a retrieval-quality conclusion.

To rerun one target only:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase10_9_retrieval_triage --videos fAd2nKfrDB8 --output ".\evaluation_reports\phase10_9_arabic_triage_$stamp.json"
```

### 3. Classify the first evidence-loss point

For each missing gold group, use this order:

1. **Gold evidence absent from the index:** candidate count is zero. Check the registered evidence phrase against the actual transcript and index language. Decide whether the index is incomplete or the annotation needs review before touching ranking.
2. **Indexed evidence not surfaced by raw semantic candidates:** inspect per-query raw FAISS ranks and distances. Only then consider query formulation, embedding coverage, or candidate-fetch limits.
3. **Raw candidate exists but later rankings lose it:** compare MMR, semantic/lexical fusion, hybrid fusion, facet ranking, and anchor selection. Identify the first transition at which all valid gold candidates disappear.
4. **Anchor selected but evidence absent from context:** inspect context expansion and candidate/chunk identity. Do not assume this is a ranking issue.
5. **Evidence is present in production context but the live answer fails:** that is a generation/grounding issue, not a retrieval fix. Use the separate answer/source audit and transcript citations to verify what the model received.

A stage label is a hypothesis from the trace, not permission to change a parameter. Validate the candidate text, timestamp, video identity, and gold annotation before assigning a root cause.

### 4. Choose the smallest safe correction

For every proposed fix, record the failing question, missing gold group, source chunk IDs, last stage where evidence is present, first stage where it is lost, root cause, minimal change, and a regression test.

Allowed fixes are narrowly scoped to the proven problem. Do not raise retrieval/context budgets, relax distance gates, switch embedding models, adjust MMR/RRF weights, reindex, or change prompts solely to make scores rise. A change to one stage should not silently compensate for a failure elsewhere.

If the index lacks the correct language or is stale, stop and resolve the index/comparability issue separately. Do not overwrite the known baseline index during diagnosis.

### 5. Regression and release gates

Before calling Phase 10.9 complete:

- All targeted traces are reproducible and have no blocked/error statuses, or every remaining blocker is explicitly documented with evidence.
- Every claimed root cause is linked to trace evidence and a focused automated regression test.
- The full automated test suite and GitHub Actions CI pass.
- The same five-video evaluation is rerun with the same index namespace, embedding model, and effective retrieval settings.
- Previously passing English and Hindi videos remain passing; the three targets either improve at the diagnosed stage without regressions or have a documented, evidence-backed residual blocker.
- A fresh live answer/source audit is run for any change that can affect generated answers, evidence grounding, or timestamps. Retrieval benchmark success is not substituted for answer-quality acceptance.
- The PR description includes before/after per-video and per-question results and states any limitations (including Urdu source verification).

## What could go wrong, and how we avoid it

- **Different index or configuration makes the comparison misleading:** compare provenance and the recorded per-video transcript language before interpreting scores.
- **An index state check blocks the run:** preserve the report and diagnose namespace, metadata and index availability first; do not blindly rebuild.
- **Gold phrases do not actually match the current captions:** manually verify the phrase and source before labeling it a retrieval defect.
- **Changing retrieval improves one example but hurts others:** require the complete five-video regression run, not only the targeted question.
- **One good HTTP response is mistaken for semantic quality:** keep retrieval metrics and live answer/source quality as separate release gates.
- **Diagnostics are mistaken for a quality pass:** `diagnostics_completed` means traces were collected only; it is not an answer-quality or benchmark pass.

## Scope boundary and follow-up work

This first PR only establishes the reproducible diagnostic runbook and runner. It intentionally does not try to close the underlying retrieval failures without the local index traces. After those reports are reviewed, make the minimal validated fixes in this Phase 10.9 branch/PR (or a narrowly scoped follow-up PR if the evidence shows independent defects).

At the close of Phase 10, return to the earlier phases for a consolidated pass over remaining cross-phase issues. Do not expand this phase into that full-system review prematurely.
