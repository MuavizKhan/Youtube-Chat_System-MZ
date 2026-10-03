# YouTube Video Chat System

A Chrome extension and FastAPI backend that let a user ask questions about a YouTube video's transcript. Answers are grounded in retrieved transcript chunks, with timestamped source segments returned separately for navigation.

## Product flow

1. The user opens a YouTube video.
2. The extension detects the video ID and queues idempotent index preparation in the background while keeping the chat panel closed.
3. The backend persists a per-video job in SQLite and a worker fetches the transcript, creates timestamp-aware chunks and embeddings, and persists the FAISS index.
4. The extension polls durable readiness state and shows whether the video is queued, preparing, ready, or failed.
5. The user opens the chat and asks questions. Chat only uses an already-valid index; it never builds an index as part of a question request.
6. Answers include source segments with start/end timestamps.

## Architecture

```text
YouTube page
  └─ Chrome content script
       └─ Manifest V3 service worker
            ├─ POST /index       → enqueue/reuse preparation job
            ├─ GET /index/{id}   → inspect durable job/index status
            └─ POST /chat        → retrieve + generate answer

FastAPI
  ├─ index_service.py
  ├─ index_jobs.py      → durable SQLite queue + worker coordination
  ├─ indexing.py        → lifecycle policy, validation, safe persistence
  ├─ retrieval.py       → FAISS loading and retrieval
  └─ chain.py           → grounded answer generation
```

## API contract

### `POST /index`

Request:

```json
{
  "video_id": "Gfr50f6ZBvo"
}
```

Accepts a raw 11-character YouTube video ID or a supported YouTube URL. The endpoint is asynchronous: it creates or reuses an idempotent preparation job and returns immediately.

When work is required, the response is HTTP 202:

```json
{
  "video_id": "Gfr50f6ZBvo",
  "state": "queued",
  "action": "create",
  "ready": false,
  "job_id": "..."
}
```

When the index is already valid, the response is HTTP 200 with `state=ready`, `action=reuse`, and `ready=true`.

The lifecycle action is one of `create`, `reuse`, or `rebuild`. Active job states are `queued` and `building`.

### `GET /index/{video_id}`

Returns the durable preparation/index state: `missing`, `invalid`, `stale`, `queued`, `building`, `ready`, or `failed`.

Queued/building state includes the `job_id`. A failed job is exposed as `state=failed`, `action=retry`.

Index jobs are persisted in SQLite and are coordinated transactionally between API processes that share the same database file. A stale building job is automatically returned to the queue after the configured timeout, allowing recovery from a worker crash.

### `POST /chat`

Request:

```json
{
  "video_id": "Gfr50f6ZBvo",
  "question": "What is the main point of this video?"
}
```

Chat requires the index to be ready. If it is missing, invalid, stale, queued, or building, the API returns HTTP 409 with `error=index_not_ready` and the current state/job ID where available. Clients should call `POST /index` when preparation is needed, then poll `GET /index/{video_id}` until the state is `ready`.

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

The extension currently targets the local development backend at `http://127.0.0.1:8000`. Change this configuration and review host permissions before any hosted release.

## Tests and CI

Run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

CI checks dependency consistency, compiles the Python source, validates Chrome extension JavaScript syntax, and runs pytest.

## Current operational boundaries

- FAISS indexes are stored on local disk under the backend vector-store directory.
- Index job state is durable in SQLite and workers coordinate across API processes that share the same SQLite file. This is suitable for a single host/shared filesystem deployment; a multi-container deployment should move the job store to a managed shared database/queue.
- Index preparation is asynchronous and idempotent. Durable queueing and crash recovery are implemented; cancellation, distributed queueing, and multi-instance shared storage remain deployment architecture work.
- The extension currently targets local development. A public release still needs a hosted HTTPS backend, production CORS/host-permission settings, authentication/abuse controls, privacy disclosures, and end-to-end release validation.
- Clickable source timestamps currently seek the in-page YouTube player. A stronger full-view timestamp deep-link experience remains a planned UX phase.

## Project quality policy

Changes should be made on a phase branch, tested in CI, reviewed in a pull request, and merged only when required checks pass. Passing automated tests is necessary but does not replace testing against real YouTube videos, the configured model provider, and the intended deployment environment.
