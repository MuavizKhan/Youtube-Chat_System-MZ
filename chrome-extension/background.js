/**
 * background.js
 *
 * Manifest V3 service worker.
 *
 * Responsibilities:
 * 1. Toggle the YouTube AI chat panel when the extension icon
 *    is clicked.
 * 2. Forward chat requests from content.js to FastAPI.
 * 3. Validate requests and video identity.
 * 4. Handle backend/network errors.
 * 5. Return the complete RAG response, including sources.
 */


// ============================================================
// CONFIGURATION
// ============================================================

const BACKEND_BASE_URL =
    "http://127.0.0.1:8000";

const BACKEND_URL =
    `${BACKEND_BASE_URL}/chat`;

const INDEX_URL =
    `${BACKEND_BASE_URL}/index`;

const INDEX_STATUS_URL =
    `${BACKEND_BASE_URL}/index`;

const REQUEST_TIMEOUT_MS =
    15000;


// ============================================================
// 1. EXTENSION ACTION → TOGGLE CHAT
// ============================================================

chrome.action.onClicked.addListener(
    async (tab) => {

        const tabId =
            tab?.id;

        if (
            typeof tabId !== "number"
        ) {
            return;
        }

        const tabUrl =
            tab?.url || "";

        // Only operate on YouTube pages.
        if (
            !isYouTubeUrl(tabUrl)
        ) {
            return;
        }

        try {

            await chrome.tabs.sendMessage(
                tabId,
                {
                    type: "toggle_chat"
                }
            );

        } catch (error) {

            console.error(
                "Could not communicate with YouTube page:",
                error
            );
        }
    }
);


// ============================================================
// 2. CONTENT SCRIPT → BACKEND
// ============================================================

chrome.runtime.onMessage.addListener(
    (
        message,
        sender,
        sendResponse
    ) => {

        if (!message) {
            return;
        }

        if (message.type === "prepare_index") {
            const videoId = normalizeString(message.video_id);
            const senderUrl = sender?.tab?.url || "";
            const pageVideoId = extractVideoIdFromUrl(senderUrl);

            if (!isYouTubeUrl(senderUrl) || !videoId || pageVideoId !== videoId) {
                sendResponse({
                    success: false,
                    error: "Index preparation must originate from the current YouTube video."
                });
                return;
            }

            handleIndexPreparation(videoId, sendResponse);
            return true;
        }

        if (message.type === "index_status") {
            const videoId = normalizeString(message.video_id);
            const senderUrl = sender?.tab?.url || "";
            const pageVideoId = extractVideoIdFromUrl(senderUrl);

            if (!isYouTubeUrl(senderUrl) || !videoId || pageVideoId !== videoId) {
                sendResponse({
                    success: false,
                    error: "Index status requests must originate from the current YouTube video."
                });
                return;
            }

            handleIndexStatus(videoId, sendResponse);
            return true;
        }

        if (message.type !== "chat") {
            return;
        }

        const videoId =
            normalizeString(
                message.video_id
            );

        const question =
            normalizeString(
                message.question
            );

        // ----------------------------------------------------
        // Sender information
        // ----------------------------------------------------

        const senderUrl =
            sender?.tab?.url || "";

        // ----------------------------------------------------
        // Validate sender
        // ----------------------------------------------------

        if (
            !isYouTubeUrl(senderUrl)
        ) {

            sendResponse({
                success: false,
                error:
                    "Chat requests must originate from a YouTube page."
            });

            return;
        }

        // ----------------------------------------------------
        // Validate request
        // ----------------------------------------------------

        if (!videoId) {

            sendResponse({
                success: false,
                error:
                    "No YouTube video ID was found."
            });

            return;
        }

        if (!question) {

            sendResponse({
                success: false,
                error:
                    "Question cannot be empty."
            });

            return;
        }

        // ----------------------------------------------------
        // Verify the video has not changed
        // ----------------------------------------------------

        const pageVideoId =
            extractVideoIdFromUrl(
                senderUrl
            );

        if (
            pageVideoId &&
            pageVideoId !== videoId
        ) {

            sendResponse({
                success: false,
                error:
                    "The video changed before the request was processed."
            });

            return;
        }

        // ----------------------------------------------------
        // Run async backend request
        // ----------------------------------------------------

        handleChatRequest(
            {
                videoId,
                question
            },
            sendResponse
        );

        // Important:
        // Keep the message channel open while the async
        // FastAPI request is running.
        return true;
    }
);


// ============================================================
 // 3. PREPARE VIDEO INDEX
 // ============================================================

async function handleIndexPreparation(videoId, sendResponse) {
    const controller = new AbortController();
    const timeoutId = setTimeout(
        () => controller.abort(),
        REQUEST_TIMEOUT_MS
    );

    try {
        const response = await fetch(INDEX_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({ video_id: videoId }),
            signal: controller.signal
        });

        const data = await parseResponse(response);

        if (!response.ok) {
            throw new Error(
                extractBackendError(data) ||
                `Index request failed (HTTP ${response.status}).`
            );
        }

        if (!data || typeof data !== "object") {
            throw new Error("The backend returned an invalid index response.");
        }

        sendResponse({
            success: true,
            video_id: data.video_id,
            state: data.state,
            action: data.action,
            ready: data.ready === true
        });
    } catch (error) {
        sendResponse({
            success: false,
            error: error?.name === "AbortError"
                ? "The index service did not respond. Reopen the chat to retry."
                : (error instanceof Error && error.message
                    ? error.message
                    : "Unable to contact the index service.")
        });
    } finally {
        clearTimeout(timeoutId);
    }
}


async function handleIndexStatus(videoId, sendResponse) {
    const controller = new AbortController();
    const timeoutId = setTimeout(
        () => controller.abort(),
        REQUEST_TIMEOUT_MS
    );

    try {
        const response = await fetch(
            `${INDEX_STATUS_URL}/${encodeURIComponent(videoId)}`,
            {
                method: "GET",
                headers: {
                    "Accept": "application/json"
                },
                signal: controller.signal
            }
        );

        const data = await parseResponse(response);

        if (!response.ok) {
            throw new Error(
                extractBackendError(data) ||
                `Index status failed (HTTP ${response.status}).`
            );
        }

        sendResponse({
            success: true,
            video_id: data?.video_id || videoId,
            state: data?.state || "unknown",
            action: data?.action || "unknown",
            ready: data?.ready === true
        });
    } catch (error) {
        sendResponse({
            success: false,
            error: error?.name === "AbortError"
                ? "Checking video readiness timed out."
                : (error instanceof Error && error.message
                    ? error.message
                    : "Unable to check video readiness.")
        });
    } finally {
        clearTimeout(timeoutId);
    }
}

// ============================================================
// 3. HANDLE CHAT REQUEST
// ============================================================

async function handleChatRequest(
    {
        videoId,
        question
    },
    sendResponse
) {

    const controller =
        new AbortController();

    const timeoutId =
        setTimeout(
            () => {
                controller.abort();
            },
            REQUEST_TIMEOUT_MS
        );

    try {

        const response =
            await fetch(
                BACKEND_URL,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",

                        "Accept":
                            "application/json"
                    },

                    body: JSON.stringify({
                        video_id:
                            videoId,

                        question:
                            question
                    }),

                    signal:
                        controller.signal
                }
            );

        // ----------------------------------------------------
        // Parse response
        // ----------------------------------------------------

        const data =
            await parseResponse(
                response
            );

        // ----------------------------------------------------
        // HTTP error
        // ----------------------------------------------------

        if (!response.ok) {

            throw new Error(
                extractBackendError(data) ||
                `Backend returned status ${response.status}.`
            );
        }

        // ----------------------------------------------------
        // Validate backend response
        // ----------------------------------------------------

        if (
            !data ||
            typeof data !== "object"
        ) {

            throw new Error(
                "The backend returned an invalid response."
            );
        }

        const answer =
            typeof data.answer === "string"
                ? data.answer.trim()
                : "";

        if (!answer) {

            throw new Error(
                "The backend returned an empty answer."
            );
        }

        // ----------------------------------------------------
        // Return complete RAG response
        // ----------------------------------------------------

        sendResponse({

            success:
                true,

            answer:
                answer,

            video_id:
                typeof data.video_id === "string"
                    ? data.video_id
                    : videoId,

            sources:
                Array.isArray(data.sources)
                    ? data.sources
                    : [],

            retrieved_chunks:
                normalizeNumber(
                    data.retrieved_chunks
                ),

            source_segments:
                normalizeNumber(
                    data.source_segments
                ),

            model:
                typeof data.model === "string"
                    ? data.model
                    : "",

            retrieval_method:
                typeof data.retrieval_method === "string"
                    ? data.retrieval_method
                    : "",

            retrieval_config:
                data.retrieval_config &&
                typeof data.retrieval_config === "object"
                    ? data.retrieval_config
                    : {}
        });

    } catch (error) {

        console.error(
            "YouTube AI backend request failed:",
            error
        );

        let errorMessage =
            "Unable to contact the backend.";

        if (
            error?.name === "AbortError"
        ) {

            errorMessage =
                "The backend request timed out. Please try again.";

        } else if (
            error instanceof Error &&
            error.message
        ) {

            errorMessage =
                error.message;
        }

        sendResponse({

            success:
                false,

            error:
                errorMessage
        });

    } finally {

        clearTimeout(
            timeoutId
        );
    }
}


// ============================================================
// 4. RESPONSE HELPERS
// ============================================================

async function parseResponse(
    response
) {

    const contentType =
        response.headers.get(
            "content-type"
        ) || "";

    if (
        contentType.includes(
            "application/json"
        )
    ) {

        return response.json();
    }

    const text =
        await response.text();

    if (!text.trim()) {
        return null;
    }

    return {
        detail:
            text.trim()
    };
}


function extractBackendError(
    data
) {

    if (
        !data ||
        typeof data !== "object"
    ) {
        return "";
    }

    if (typeof data.detail === "string") {
        return data.detail;
    }

    if (
        data.detail &&
        typeof data.detail === "object" &&
        typeof data.detail.message === "string"
    ) {
        return data.detail.message;
    }

    if (typeof data.message === "string") {
        return data.message;
    }

    if (typeof data.error === "string") {
        return data.error;
    }

    return "";
}


// ============================================================
// 5. VALIDATION HELPERS
// ============================================================

function normalizeString(
    value
) {

    if (
        typeof value !== "string"
    ) {
        return "";
    }

    return value.trim();
}


function normalizeNumber(
    value
) {

    const number =
        Number(value);

    return Number.isFinite(number)
        ? number
        : 0;
}


function isYouTubeUrl(
    value
) {

    if (
        typeof value !== "string" ||
        !value
    ) {
        return false;
    }

    try {

        const url =
            new URL(value);

        return (
            url.protocol === "https:" &&
            (
                url.hostname ===
                    "www.youtube.com" ||
                url.hostname ===
                    "youtube.com"
            )
        );

    } catch {

        return false;
    }
}


function extractVideoIdFromUrl(
    value
) {

    if (
        !isYouTubeUrl(value)
    ) {
        return null;
    }

    try {

        const url =
            new URL(value);

        // Standard watch URL:
        // /watch?v=VIDEO_ID

        const watchVideoId =
            url.searchParams.get("v");

        if (watchVideoId) {

            return watchVideoId;
        }

        // Supported paths:
        // /shorts/VIDEO_ID
        // /embed/VIDEO_ID
        // /live/VIDEO_ID

        const pathParts =
            url.pathname
                .split("/")
                .filter(Boolean);

        if (
            pathParts.length >= 2 &&
            (
                pathParts[0] ===
                    "shorts" ||
                pathParts[0] ===
                    "embed" ||
                pathParts[0] ===
                    "live"
            )
        ) {

            return pathParts[1];
        }

        return null;

    } catch {

        return null;
    }
}