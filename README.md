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

- Identity and participant questions must be answered only when the transcript explicitly establishes the identity.
- Model output is defensively sanitized so accidental internal source labels do not leak into the user-facing answer.
- Video sources are shown as distinct, labeled controls with clean timestamps for easier evidence navigation.
- Regression tests cover the new grounding contract and output sanitization.

The focus of this phase is retrieval quality, grounded answers, and evidence usability before moving into broader deployment architecture.

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
