# Phase 11.0 — Unseen-video generalisation

This phase follows the first transcript-failure milestone in PR #143. It does not
change the accepted Phase 10.11 retrieval settings, embedding model, citation
policy, or five-video index namespace.

## Live answer/source audit

The original phase10_11 suite remains the frozen 16-case representative
regression gate. The new phase11_0 suite runs all 43 registered questions:
the same 16 regression cases plus the 27 cases that were not previously run
live. It has a separate report ID and output path. A passing Phase 11.0 report
requires every registered question to have been attempted exactly once, every
request to succeed, the expected retrieval configuration to match, and all
answer/source/timestamp/language checks to pass.

Run the full suite only when the configured model-provider quota can support
43 live chat requests:

~~~powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "C:\Users\Muaviz Khan\OneDrive\Desktop\All Projects\Youtube_Video_ChatSystem"

.\.venv\Scripts\python.exe -m scripts.chat_answer_source_audit --suite phase11_0 --output evaluation_reports/phase11_0_full_registry_20261030.json
~~~

Run the backend with the accepted Phase 10.11 index namespace and confirm its
startup/effective configuration before executing the audit. Do not point this
suite at the Phase 11.0 smoke namespace.

A selected --case-ids run is diagnostic only: it writes a subset report and
disables the complete-registry gate. It cannot close the phase.

## Unseen-video indexing smoke

The smoke runner uses a separate namespace so it cannot overwrite the frozen
pilot indexes. Both the backend and the runner process must be started with
the same environment variable:

~~~powershell
$env:RAG_INDEX_NAMESPACE = "phase11-0-unseen-video-smoke"
.\.venv\Scripts\python.exe -m uvicorn Backend.app:app --reload --host 127.0.0.1 --port 8000
~~~

In a second PowerShell terminal:

~~~powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "C:\Users\Muaviz Khan\OneDrive\Desktop\All Projects\Youtube_Video_ChatSystem"
$env:RAG_INDEX_NAMESPACE = "phase11-0-unseen-video-smoke"

.\.venv\Scripts\python.exe scripts/phase11_0_unseen_video_smoke.py
~~~

The default manifest includes two videos outside the five-video benchmark
registry (XGLx-Anjxvg and jzjHxh9nPkA) and one known publisher-uploaded
Arabic-caption control (fAd2nKfrDB8). The caption control is not counted as an
unseen video.

The runner checks:
- index readiness and persisted FAISS/metadata files;
- selected caption language and auto-generated vs publisher-uploaded variant;
- unique transcript segment/character counts and timestamp span;
- a second POST /index returning ready/reuse without changing persisted
  index-file hashes;
- configured coverage for short and long caption timelines.

The manifest intentionally requires both short and long transcript timeline
classes. If the current videos do not cover both, the report lists the missing
class. Add a verified short- or long-timeline video to
evaluation_data/phase11_0_unseen_video_smoke.json rather than weakening the
coverage requirement. The timeline span is derived from caption timestamps; it
is not represented as the video's exact media duration.

For a real video with a verified unavailable/no-caption condition, add a case
with expected_outcome: "failed" and the corresponding stable
expected_error_code (VIDEO_UNAVAILABLE or NO_SUPPORTED_TRANSCRIPT).
Do not invent video IDs or assume an error category without confirming it.
The unit suite also exercises all four stable transcript-failure codes through
the durable index-job failure path.

## Index metadata compatibility

The new transcript statistics are additive metadata. Legacy FAISS indexes
without the statistics remain valid and loadable; malformed or partially
written statistics are rejected. Run the smoke checks only in the isolated
Phase 11.0 namespace so pilot index contents are not rebuilt or overwritten.

## Completion criteria

Phase 11.0 is complete only after:
1. the full 43-question live audit passes with complete registry accounting;
2. at least two unregistered videos pass the isolated indexing smoke;
3. automatic and publisher-uploaded caption tracks are both observed;
4. short and long transcript timeline coverage is observed;
5. repeated index requests prove reuse without rebuilding;
6. transcript-disabled/unavailable failure behavior remains covered by tests or
   a verified live negative case; and
7. the Phase 10.11 report and retrieval configuration remain unchanged.

Urdu remains blocked pending verification against an actual publisher-uploaded
Urdu-script caption track. This phase does not claim Urdu support.
