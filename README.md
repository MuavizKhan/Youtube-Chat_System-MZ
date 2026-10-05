# YouTube Video Chat System

A Chrome extension and FastAPI backend that let a user ask questions about a YouTube video's transcript. Answers are grounded in retrieved transcript chunks, with timestamped source segments returned separately for navigation.

## Product flow

1. The user opens a YouTube video.
2. The extension detects the video ID and submits index preparation in the background while keeping the chat panel closed.
3. The backend creates a durable per-video index job, and one worker claims the job through SQLite lease coordination.
4. The worker fetches the transcript, creates timestamp-aware chunks and embeddings, and persists a per-video FAISS index.
5. The extension polls the durable status endpoint and shows whether the video is queued, preparing, ready, or failed.
6. The user opens the chat and asks questions. Chat only uses an already-valid index; it does not build an index as part of the question request.
7. Answers include source segments with start/end timestamps.

## Architecture

```text
YouTube page
  └─ Chrome content script
       └─ Manifest V3 service worker
            ├─ POST /index       → prepare/reuse/rebuild index
            ├─ GET /index/{id}   → inspect index status
            └─ POST /chat        → retrieve + generate answer

FastAPI
  ├─ index_service.py  → job dispatch, recovery, worker lifecycle
  ├─ index_jobs.py     → durable SQLite state + worker leases
  ├─ indexing.py       → lifecycle policy, validation, safe persistence
  ├─ retrieval.py      → FAISS loading and retrieval
  └─ chain.py          → grounded answer generation
```

## API contract

### `POST /index`

Request:

```json
{
  "video_id": "Gfr50f6ZBvo"
}
```

Accepts a raw 11-character YouTube video ID or a supported YouTube URL. If the index is already valid, the endpoint returns `200` immediately. Otherwise it creates or reuses a durable background job and returns `202` without waiting for transcript/embedding work to finish.

Queued response (HTTP 202):

```json
{
  "video_id": "Gfr50f6ZBvo",
  "state": "queued",
  "action": "create",
  "ready": false,
  "job_id": "7c7d9d2d..." 
}
```

Ready response:

```json
{
  "video_id": "Gfr50f6ZBvo",
  "state": "ready",
  "action": "create",
  "ready": true,
  "job_id": "7c7d9d2d..."
}
```

The action is one of `create`, `reuse`, or `rebuild`. Queued/building responses include a durable `job_id` for request tracing; clients should continue using `GET /index/{video_id}` for readiness polling.

### `GET /index/{video_id}`

Returns the current state: `missing`, `queued`, `building`, `ready`, `invalid`, `stale`, or `failed`, plus the recommended lifecycle action, a `ready` boolean, and the active/latest `job_id` when one exists.

`queued`, `building`, and `failed` are persisted in SQLite, so multiple FastAPI worker processes on the same shared filesystem observe the same job state. A worker lease expires and can be reclaimed after a process failure.

### `POST /chat`

Request:

```json
{
  "video_id": "Gfr50f6ZBvo",
  "question": "What is the main point of this video?"
}
```

Chat requires the index to be ready. If preparation is queued or running, the API returns `409 index_not_ready`; if preparation has failed, it returns `503 index_preparation_failed`; missing/invalid/stale indexes return `404 video_not_indexed` until `POST /index` is requested.

## Local development

### Backend

Create a project-root `.env` based on [`.env.example`](.env.example), including the Hugging Face settings required by the configured inference provider.

Install dependencies in your virtual environment, then run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn Backend.app:app --reload
```

Useful endpoints:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/ready`
- `http://127.0.0.1:8000/docs`

### Chrome extension

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Choose **Load unpacked** and select the `chrome-extension` directory.
4. Start the local backend.
5. Open a YouTube video. The extension prepares its index while the chat remains closed.

The extension currently targets the local development backend at `http://127.0.0.1:8000`. The backend URL is centralized in `chrome-extension/config.js`; change that value and review `manifest.json` host permissions before any hosted release.

### Containerized backend

Phase 7 adds a single-instance Docker baseline:

```powershell
docker build -t youtube-video-chat-backend .
docker run --rm -p 8000:8000 --env-file .env youtube-video-chat-backend
```

The image does not contain secrets, SQLite state, or FAISS indexes. Persist the backend's vector-store directory and job database outside the container for any restart-safe deployment.

## Phase 8: RAG quality and source UX

Phase 8 begins tightening the user-facing trust contract around retrieved evidence:

- The user-facing answer remains grounded in retrieved transcript evidence, with accidental internal source scaffolding stripped defensively.
- Model output is defensively sanitized so accidental internal source labels do not leak into the user-facing answer.
- Video sources are shown as distinct, labeled controls with clean timestamps for easier evidence navigation.
- Regression tests cover the new grounding contract and output sanitization.

The focus of this phase is retrieval quality, grounded answers, and evidence usability before moving into broader deployment architecture.

## Phase 8 — Context-aware retrieval

The next Phase 8 slice expands strong retrieval hits with a small, chronological neighborhood of transcript chunks. The expansion is bounded and configurable so follow-up or multi-part questions can receive nearby evidence without allowing prompt/context growth to become unbounded.

- RAG_CONTEXT_EXPANSION_CHUNKS controls the number of adjacent chunks added on each side of a retrieved anchor (default 1).
- RAG_CONTEXT_MAX_CHUNKS caps the final retrieved context (default 12).
- Anchor chunks are preserved, duplicates are removed, and the final context is returned in chronological order.

This improves continuity for questions that depend on what was said immediately before or after a retrieved passage while keeping the existing FAISS/MMR/lexical retrieval strategy intact.

## Phase 8 — Answer-quality evaluation

The next Phase 8 slice strengthens the deterministic RAG evaluation harness. The live end-to-end evaluator now checks not only the answer text but also the evidence provenance returned by the RAG chain.

- Answerable cases must have retrieved evidence and at least one valid source segment.
- Unanswerable cases must not return retrieved evidence or source segments.
- Source counts, video identity, timestamps, and duration metadata are validated.
- These checks run without another model call, so CI can enforce the contract without consuming inference quota.

The evaluator remains a regression harness rather than a claim of semantic faithfulness; human or model-based evaluation is still needed for deeper answer correctness.

## Phase 8 — Retrieval quality benchmarking

The next slice turns the retrieval regression suite into an explicit quality benchmark with machine-readable metrics and hard thresholds.

- must_retrieve_recall measures how often the 11 required retrieval cases return evidence.
- must_not_retrieve_rejection_rate measures whether the unrelated control case stays empty.
- strict_pass_rate combines the strict positive and negative retrieval contracts.
- strict_max_best_distance records the worst best-match distance for required retrieval cases and keeps it within the existing MAX_DISTANCE contract.
- The JSON evaluation summary includes benchmark metrics, thresholds, and explicit benchmark failures so regressions are visible in CI or manual evaluation output.
- The benchmark remains deterministic and model-free. It does not claim semantic relevance without labeled ground-truth chunks or human/model judgments.

The live evaluator can be run with: python -m Backend.rag_system.evaluation --json

A non-zero exit code indicates either a per-case retrieval regression or a failed benchmark threshold.


## Phase 8 — Gold-evidence retrieval evaluation

This slice moves the retrieval benchmark from "did anything come back?" to "did the retrieved evidence contain the concepts required by the question?"

- `Backend/rag_system/gold_evidence.py` stores human-curated evidence groups for Q01-Q10 using transcript phrases rather than chunk IDs, so the annotations survive chunking changes.
- The annotations were calibrated against the available transcript text for the benchmark video; alternatives are included where automatic-transcript wording varies. The reference transcript used during calibration is the [published transcript copy](https://youtubetotranscript.com/transcript?current_language_code=en&v=MdeQMVBuGgY).
- Each evidence group may contain alternative phrases for the same concept, and each question defines the minimum evidence-group coverage required to pass.
- The **raw-anchor gold benchmark** keeps context expansion disabled and reports Hit@K, evidence-group coverage, MRR, and Precision@K. It is intentionally diagnostic: it tells us how much of the required evidence is found by the initial retrieval anchors.
- The **production-context gold benchmark** scores the final expanded context used by the answer-generation pipeline. It reports evidence presence, mean evidence-group coverage, mean relevance ratio, and case pass rate. This is the evaluator's quality gate because the production system intentionally uses bounded adjacent-context expansion.
- Q11 is evaluated separately by the timestamp-grounded benchmark in Part 6.
- Both gold benchmarks are deterministic and model-free; neither is a semantic-faithfulness judge.

The JSON evaluation summary exposes both `gold_benchmark` (raw-anchor diagnostic) and `gold_context_benchmark` (production-context gate). The evaluator exits non-zero when the production-context gold benchmark or another hard benchmark violates its thresholds.


## Phase 8 — Retrieval fusion improvement

The observed live benchmark showed that semantic retrieval could find the right section only after adjacent-context expansion, while lexical evidence was appended behind semantic results even when it contained exact transcript terminology. This follow-up adds Reciprocal Rank Fusion (RRF) to the semantic + lexical retrieval merge:

- semantic and lexical score scales are not compared directly;
- each branch contributes according to rank using configurable `RAG_RRF_K` (default `60`);
- chunks supported by both semantic and lexical retrieval receive combined rank evidence;
- the existing FAISS distance contract remains unchanged for downstream gating;
- production context expansion remains bounded and unchanged.

This addresses a real ranking weakness rather than relaxing the gold thresholds merely to make the benchmark green.

## Phase 8 — Query-aware hybrid retrieval correction

The live benchmark showed a second retrieval weakness after RRF was introduced: the full conversational question could dilute the embedding signal with speaker attribution and question boilerplate. At the same time, lexical matching treated those generic words as evidence.

The retrieval path now:

- preserves the original question;
- creates a deterministic content-focused query variant;
- retrieves both semantic variants and rank-fuses them with RRF;
- uses content-aware lexical retrieval with lightweight inverse-document-frequency weighting;
- gives stronger weight to rare domain terms and adjacent phrase matches;
- keeps lexical fallback conservative so weak common-word matches do not bypass the semantic evidence gate.

This is a general retrieval improvement. It does not contain benchmark-specific answer phrases, lower evaluation thresholds, or change the production context limit.

The distinction is important: the original query preserves conversational intent, while the focus query improves recall for transcript wording when the same intent is expressed with different language.

## Phase 8 — Adaptive retrieval and evidence-gap diagnostics

The benchmark showed that explanation and multi-part questions can require evidence from several transcript regions. The retriever now uses a larger semantic/lexical candidate budget for evidence-dense question shapes while keeping the final context bounded by RAG_CONTEXT_MAX_CHUNKS.

The evaluator also exposes a non-gating gold-evidence diagnostic. For every curated evidence group it reports whether an exact annotated phrase exists anywhere in the indexed transcript and whether any matching chunk reached the production context. This separates:

- annotation/transcript mismatch (no matching chunk exists in the index);
- retrieval miss (matching chunks exist but were not retrieved);
- context-selection miss (matching chunks were retrieved initially but did not survive final context selection).

This diagnostic is intentionally separate from the quality gate and does not change benchmark pass/fail behavior.

## Phase 8 — Evidence-diversified anchor selection

The evidence-gap diagnostics showed that the remaining failures were often retrieval-selection failures: the indexed transcript contained the required evidence, but a bounded anchor budget could concentrate on neighboring chunks from one transcript region. For evidence-dense questions, the retriever now selects anchors in two passes:

- first prefer high-ranked chunks separated by a small transcript chunk gap, so multiple regions can survive the anchor budget;
- then fill any remaining slots using the original ranking, so compact evidence regions are not discarded;
- simple factual questions keep the existing ranking behavior;
- the final RAG_CONTEXT_MAX_CHUNKS bound and evidence gates remain unchanged.

This change targets evidence coverage without using benchmark-specific phrases or question IDs. The diagnostics remain available to verify whether the diversified anchors reach the required evidence groups.


## Phase 8 — End-to-end retrieval diagnostics

The retrieval-gap diagnostics showed that the remaining gold-evidence failures could occur at several different stages. The diagnostic evaluator now traces selected gold chunks through the complete production retrieval path:

- raw FAISS candidates before MMR;
- per-query semantic MMR results;
- semantic-variant RRF;
- lexical retrieval;
- semantic + lexical hybrid RRF;
- diversified anchor selection;
- final context expansion.

For each gold candidate it reports stage ranks and classifies the first observed failure as candidate generation, MMR, fusion, anchor selection, or context selection. This instrumentation does not change production retrieval or benchmark thresholds.

Run a focused diagnostic for the benchmark questions with:

python -m Backend.rag_system.evaluation MdeQMVBuGgY --diagnose Q04 Q05 --json

The focused JSON output can be redirected to a small file for inspection without scrolling through the full regression benchmark.


## Phase 8 — Timestamp-grounded temporal evaluation

Part 6 adds a temporal evidence benchmark for Q11, which asks where Kingfisher Airlines is discussed and what is covered in those sections.

- `Backend/rag_system/temporal_evidence.py` defines chapter-level gold windows for the benchmark video `MdeQMVBuGgY`: **01:07:20–01:35:00** (“Rise & Fall of Kingfisher Airlines”) and **01:59:31–02:16:12** (“Turmoil at Kingfisher Airlines”).
- These are coarse chapter intervals taken from the [published video chapter list](https://socialcounts.org/youtube-video-live-view-count/MdeQMVBuGgY), not word-level or manually verified sentence boundaries. A retrieved chunk is considered temporally relevant only when its timestamp interval overlaps a gold window and its transcript text matches that section’s curated topic phrases.
- The benchmark evaluates raw top-k retrieval anchors with context expansion disabled, and reports temporal Hit@K, section-window coverage, MRR, timestamp-bound validity, and source-video identity validity.
- Missing, non-finite, negative, reversed/zero-length timestamps and sources from a different video cannot earn temporal relevance. The evaluator fails if any retrieved top-k result has invalid timestamp metadata or the wrong video identity.
- Video-specific text and temporal gold benchmarks run only for `MdeQMVBuGgY`. The Q01–Q13 regression questions are also written for this benchmark video; use a matching question set before interpreting those regression results for a different video.
- This is a deterministic section-level retrieval benchmark. It does not claim word-level timestamp precision, answer faithfulness, or that the answer's prose accurately summarizes the section.

The machine-readable JSON summary includes `temporal_benchmark`; the evaluator exits non-zero if its thresholds fail. Real-video retrieval quality still needs a smoke run against the indexed benchmark video and the configured embedding environment.

## Production hardening

Phase 7 adds several deployment safeguards without requiring a paid service:

- Explicit trusted-host validation through `TRUSTED_HOSTS`.
- A configurable request-body limit through `MAX_REQUEST_BODY_BYTES`.
- Shared CORS and rate-limit configuration remains required for production.
- The extension backend URL is centralized in one non-secret configuration file.
- The Docker image provides a repeatable FastAPI runtime and HTTP health check.

Client-side extension configuration is not a secret. Authentication/identity management for a public multi-user deployment is intentionally still a later phase.

## Tests and CI

Run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

CI checks dependency consistency, compiles the Python source, validates Chrome extension JavaScript syntax, and runs pytest.

## Current operational boundaries

- FAISS indexes are stored on local disk under the backend vector-store directory.
- Durable index jobs are stored in SQLite and coordinate multiple FastAPI workers that share the same filesystem.
- The worker lease, SQLite database, and FAISS directory must all be on shared persistent storage for multi-worker safety.
- Job IDs are diagnostic/request-tracing identifiers; readiness remains keyed by video ID so the extension can safely recover from navigation and repeated requests. Separate containers need an external database and shared object/storage layer before this design can span instances.
- Index preparation is asynchronous from the HTTP client's perspective. Long-running distributed queues, cancellation, autoscaling, external job brokers, and cross-container storage are deployment architecture work.
- The extension currently targets local development. A public release still needs a hosted HTTPS backend, production CORS/host-permission settings, authentication/abuse controls, privacy disclosures, and end-to-end release validation.
- Clickable source timestamps currently seek the in-page YouTube player. A stronger full-view timestamp deep-link experience remains a planned UX phase.

## Project quality policy

Changes should be made on a phase branch, tested in CI, reviewed in a pull request, and merged only when required checks pass. Passing automated tests is necessary but does not replace testing against real YouTube videos, the configured model provider, and the intended deployment environment.
