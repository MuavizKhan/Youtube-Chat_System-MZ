# Browser-assisted YouTube caption ingestion

## Why this path exists

The backend's direct youtube-transcript-api requests can be blocked by YouTube at the backend's public IP. Retrying the same server request for every video does not fix an IP-level restriction.

The extension now reads caption tracks exposed by the YouTube player in the active browser tab. The page-world script obtains the current player response, orders caption tracks by English, Hindi, Urdu, and Arabic preference (then other available languages), tries up to six tracks, fetches their timed-text payload with the user's browser session, and converts JSON3/XML captions to timestamped segments.

The extension submits those bounded segments to POST /index/browser-transcript. The backend validates and stages the transcript, then dispatches the existing durable index worker. The worker consumes the staged transcript through the same get_transcript -> documents -> timestamped chunks -> embeddings -> FAISS flow. Retrieval and answer generation are unchanged.

## Fallback behavior

If the browser cannot expose/fetch a caption track, the extension retains the existing POST /index path. The backend youtube-transcript-api remains a fallback. IpBlocked and RequestBlocked become an explicit YOUTUBE_IP_BLOCKED failure instead of a generic message.

The official YouTube Data API captions.list and captions.download methods are OAuth-authorized management methods; downloading a caption track requires permission to edit the video. They do not provide arbitrary public-video transcript downloads for an ordinary API key.

## Limits

The browser-caption upload is limited to 16 MiB, 100,000 segments, and 1.5 million caption characters. The backend stages the transcript under data/browser_transcripts/, consumes it once, and that directory is ignored by Git. The normal request-body limit for chat and feedback remains unchanged.

This does not make private, unavailable, age-gated, browser-blocked, or no-caption videos transcribable. A separate audio-to-text fallback would be required for those cases. This change improves coverage for public videos whose caption tracks are available in the YouTube player, including auto-generated tracks.

## Manual verification

1. Load the updated unpacked extension after restarting the backend.
2. Open a public YouTube video with captions (including auto-generated captions).
3. Confirm browser caption ingestion creates an index and status becomes ready.
4. Ask a question and confirm timestamped sources still work.
5. Test a video without a caption track and verify the backend fallback reports the actual failure category.
6. Check data/browser_transcripts/ is empty after successful indexing.
