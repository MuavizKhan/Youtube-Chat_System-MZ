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

importScripts("config.js");

// ============================================================
// CONFIGURATION
// ============================================================


const BACKEND_URL =
    `${BACKEND_BASE_URL}/chat`;

const INDEX_URL =
    `${BACKEND_BASE_URL}/index`;

const INDEX_STATUS_URL =
    `${BACKEND_BASE_URL}/index`;

const INDEX_BROWSER_TRANSCRIPT_URL =
    `${BACKEND_BASE_URL}/index/browser-transcript`;

const FEEDBACK_URL =
    `${BACKEND_BASE_URL}/feedback`;

const REQUEST_TIMEOUT_MS =
    15000;

// Chat may include retrieval plus several model calls. Keep index requests
// on the shorter timeout, but allow a normal slow generation to finish.
const CHAT_REQUEST_TIMEOUT_MS =
    90000;

const CONTENT_SCRIPT_MARKER =
    "data-youtube-ai-chat-content-version";

async function injectFreshYouTubeContentScript(tabId) {
    if (typeof tabId !== "number" || !chrome.scripting?.executeScript) {
        return false;
    }

    try {
        // Remove only this extension's old UI and marker. An orphaned content
        // script cannot reconnect to a newly loaded extension runtime.
        await chrome.scripting.executeScript({
            target: { tabId },
            world: "ISOLATED",
            func: (markerName) => {
                document.documentElement?.removeAttribute(markerName);
                document.getElementById("youtube-ai-chat-launcher")?.remove();
                document.getElementById("youtube-ai-chat")?.remove();
            },
            args: [CONTENT_SCRIPT_MARKER]
        });

        await chrome.scripting.insertCSS({
            target: { tabId },
            files: ["style.css"]
        });

        await chrome.scripting.executeScript({
            target: { tabId },
            world: "ISOLATED",
            files: ["content.js"]
        });

        console.info(
            "YouTube AI content script reattached after extension lifecycle change.",
            { tabId }
        );
        return true;
    } catch (error) {
        console.warn(
            "Could not reattach the YouTube AI content script.",
            { tabId, error: String(error?.message || error) }
        );
        return false;
    }
}

async function ensureYouTubeContentScript(tabId) {
    try {
        const response = await chrome.tabs.sendMessage(tabId, {
            type: "extension_ping"
        });
        if (response?.success === true) {
            return true;
        }
    } catch (_error) {
        // No live content script is listening; inject a fresh isolated world.
    }

    return injectFreshYouTubeContentScript(tabId);
}

async function reconnectOpenYouTubeTabs() {
    const tabs = await chrome.tabs.query({
        url: ["https://www.youtube.com/*"]
    });

    await Promise.all(
        tabs
            .filter((tab) => typeof tab.id === "number")
            .map((tab) => ensureYouTubeContentScript(tab.id))
    );
}

// Chrome leaves an already-injected content script orphaned when an
// extension is reloaded or updated. Reattach a current script to open
// YouTube tabs so users do not have to guess which page needs refreshing.
chrome.runtime.onInstalled.addListener((details) => {
    if (details.reason !== "install" && details.reason !== "update") {
        return;
    }

    void reconnectOpenYouTubeTabs().catch((error) => {
        console.warn(
            "Could not reconnect existing YouTube tabs after extension update.",
            error
        );
    });
});

// Chrome 157+ also reports manual reloads and re-enables through this event.
// Keep it optional so older supported Chrome versions retain onInstalled
// recovery without referencing an unavailable API.
if (chrome.runtime.onExtensionLoaded?.addListener) {
    chrome.runtime.onExtensionLoaded.addListener(() => {
        void reconnectOpenYouTubeTabs().catch((error) => {
            console.warn(
                "Could not reconnect YouTube tabs after the extension was loaded.",
                error
            );
        });
    });
}



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

        let ping = null;
        try {
            ping = await chrome.tabs.sendMessage(tabId, {
                type: "extension_ping"
            });
        } catch (_error) {
            // The tab may still have a content script from an invalidated
            // extension context.
        }

        if (ping?.success !== true) {
            const recovered = await injectFreshYouTubeContentScript(tabId);
            if (!recovered) {
                return;
            }
        }

        try {
            await chrome.tabs.sendMessage(tabId, {
                type: "toggle_chat"
            });
        } catch (error) {
            console.warn(
                "YouTube AI content script did not respond after recovery.",
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

            handleIndexPreparation(videoId, sender?.tab?.id, sendResponse);
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

        if (message.type === "feedback") {
            const senderUrl = sender?.tab?.url || "";
            const feedback = message.feedback && typeof message.feedback === "object"
                ? message.feedback
                : null;
            const videoId = normalizeString(feedback?.video_id);
            const pageVideoId = extractVideoIdFromUrl(senderUrl);

            if (!isYouTubeUrl(senderUrl) || !feedback || !videoId || pageVideoId !== videoId) {
                sendResponse({
                    success: false,
                    error: "Feedback must originate from the YouTube video being rated."
                });
                return;
            }

            handleFeedbackSubmission(feedback, sendResponse);
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

        const history =
            normalizeConversationHistory(
                message.history
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
                question,
                history
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

async function fetchBrowserTranscriptFromTab(tabId, videoId) {
    if (typeof tabId !== "number" || !chrome.scripting?.executeScript) {
        throw new Error("Browser caption capture is unavailable in this extension context.");
    }

    const results = await chrome.scripting.executeScript({
        target: { tabId },
        world: "MAIN",
        args: [videoId],
        func: async (expectedVideoId) => {
            const currentVideoId = () => {
                try {
                    const pageUrl = new URL(window.location.href);
                    const queryId = pageUrl.searchParams.get("v");
                    if (queryId) {
                        return queryId;
                    }
                    const pathMatch = pageUrl.pathname.match(/\/(?:shorts|embed)\/([A-Za-z0-9_-]{11})/);
                    return pathMatch ? pathMatch[1] : null;
                } catch (_error) {
                    return null;
                }
            };

            if (currentVideoId() !== expectedVideoId) {
                return {
                    ok: false,
                    code: "VIDEO_CHANGED",
                    message: "The YouTube video changed before captions could be read."
                };
            }

            const parsePlayerResponse = (candidate) => {
                if (!candidate) {
                    return null;
                }
                if (typeof candidate === "string") {
                    try {
                        candidate = JSON.parse(candidate);
                    } catch (_error) {
                        return null;
                    }
                }
                if (typeof candidate !== "object") {
                    return null;
                }
                const playerVideoId = candidate.videoDetails?.videoId;
                if (playerVideoId && playerVideoId !== expectedVideoId) {
                    return null;
                }
                return candidate;
            };

            const readPlayerResponse = () => {
                const candidates = [];
                try {
                    const player = document.getElementById("movie_player");
                    if (player && typeof player.getPlayerResponse === "function") {
                        candidates.push(player.getPlayerResponse());
                    }
                } catch (_error) {
                    // Use the global player response fallback.
                }
                candidates.push(window.ytInitialPlayerResponse);
                try {
                    if (window.ytcfg && typeof window.ytcfg.get === "function") {
                        candidates.push(window.ytcfg.get("PLAYER_RESPONSE"));
                    }
                } catch (_error) {
                    // Some YouTube builds do not expose PLAYER_RESPONSE through ytcfg.
                }
                candidates.push(window.ytcfg?.data_?.PLAYER_RESPONSE);
                candidates.push(window.ytplayer?.config?.args?.player_response);

                for (const candidate of candidates) {
                    const parsed = parsePlayerResponse(candidate);
                    if (parsed) {
                        return parsed;
                    }
                }
                return null;
            };

            let playerResponse = null;
            let captionTracks = [];
            for (let attempt = 0; attempt < 12; attempt += 1) {
                playerResponse = readPlayerResponse();
                captionTracks = playerResponse?.captions
                    ?.playerCaptionsTracklistRenderer?.captionTracks || [];
                if (captionTracks.length > 0) {
                    break;
                }
                await new Promise((resolve) => setTimeout(resolve, 400));
            }

            if (currentVideoId() !== expectedVideoId) {
                return {
                    ok: false,
                    code: "VIDEO_CHANGED",
                    message: "The YouTube video changed while captions were loading."
                };
            }

            const languagePriority = ["en", "hi", "ur", "ar"];
            const languageBase = (track) =>
                String(track.languageCode || "").toLowerCase().split("-")[0];
            const orderedTracks = [...captionTracks].sort((first, second) => {
                const firstLanguage = languagePriority.indexOf(languageBase(first));
                const secondLanguage = languagePriority.indexOf(languageBase(second));
                const firstRank = firstLanguage < 0 ? languagePriority.length : firstLanguage;
                const secondRank = secondLanguage < 0 ? languagePriority.length : secondLanguage;
                if (firstRank !== secondRank) {
                    return firstRank - secondRank;
                }
                const firstGenerated = first.kind === "asr" ? 1 : 0;
                const secondGenerated = second.kind === "asr" ? 1 : 0;
                return firstGenerated - secondGenerated;
            });

            const captionText = (track) => {
                const name = track.name;
                if (typeof name === "string") {
                    return name;
                }
                if (name?.simpleText) {
                    return name.simpleText;
                }
                if (Array.isArray(name?.runs)) {
                    return name.runs.map((run) => run.text || "").join("");
                }
                return track.languageCode || "Unknown";
            };

            const parseJson3 = (payload) => {
                if (!Array.isArray(payload?.events)) {
                    return [];
                }
                const snippets = [];
                for (const event of payload.events) {
                    if (!Array.isArray(event.segs)) {
                        continue;
                    }
                    const text = event.segs
                        .map((segment) => segment?.utf8 || "")
                        .join("")
                        .replace(/\s+/g, " ")
                        .trim();
                    const startMs = Number(event.tStartMs);
                    const durationMs = Number(event.dDurationMs || 0);
                    if (!text || !Number.isFinite(startMs) || startMs < 0) {
                        continue;
                    }
                    snippets.push({
                        text,
                        start: startMs / 1000,
                        duration: Number.isFinite(durationMs) && durationMs > 0
                            ? durationMs / 1000
                            : 0
                    });
                }
                return snippets;
            };

            const parseXml = (raw) => {
                const documentNode = new DOMParser().parseFromString(raw, "application/xml");
                if (documentNode.querySelector("parsererror")) {
                    return [];
                }
                return Array.from(documentNode.getElementsByTagName("text"))
                    .map((node) => ({
                        text: (node.textContent || "").replace(/\s+/g, " ").trim(),
                        start: Number(node.getAttribute("start")),
                        duration: Number(node.getAttribute("dur") || 0)
                    }))
                    .filter((item) =>
                        item.text &&
                        Number.isFinite(item.start) &&
                        item.start >= 0 &&
                        Number.isFinite(item.duration) &&
                        item.duration >= 0
                    );
            };

            // YouTube's timed-text endpoint can return HTTP 200 with an empty
            // body for exp=xpe tracks because it expects a player-issued PO token.
            // The native transcript panel already has YouTube's own authenticated
            // request path, so use the rendered segments as a browser-side fallback.
            const captureTranscriptFromPanel = async () => {
                const segmentSelector =
                    "transcript-segment-view-model, ytd-transcript-segment-renderer";
                const expandedPanel = () => {
                    const panels = Array.from(
                        document.querySelectorAll("ytd-engagement-panel-section-list-renderer")
                    ).filter((panel) =>
                        panel.getAttribute("visibility") === "ENGAGEMENT_PANEL_VISIBILITY_EXPANDED"
                    );
                    return panels.find((panel) => panel.querySelector(segmentSelector)) ||
                        panels.find((panel) =>
                            /transcript/i.test(panel.getAttribute("target-id") || "")
                        ) ||
                        null;
                };
                const panelHasSegments = (panel) =>
                    Boolean(panel && panel.querySelector(segmentSelector));
                const pageStillMatches = () => {
                    if (currentVideoId() !== expectedVideoId) {
                        return false;
                    }
                    const renderedId = document.querySelector("ytd-watch-flexy")
                        ?.getAttribute("video-id");
                    return !renderedId || renderedId === expectedVideoId;
                };
                const parseTimestampSeconds = (value) => {
                    const text = String(value || "").trim();
                    const hours = text.match(/(\d+)\s*(?:h|hours?)/i);
                    const minutes = text.match(/(\d+)\s*(?:m|min|minutes?)/i);
                    const seconds = text.match(/(\d+)\s*(?:s|sec|seconds?)/i);
                    if (hours || minutes || seconds) {
                        return Number(hours?.[1] || 0) * 3600 +
                            Number(minutes?.[1] || 0) * 60 +
                            Number(seconds?.[1] || 0);
                    }
                    const clock = text.match(/\d+(?::\d{1,2}){1,2}/);
                    if (!clock) {
                        return null;
                    }
                    const parts = clock[0].split(":").map(Number);
                    if (parts.some((part) => !Number.isFinite(part))) {
                        return null;
                    }
                    return parts.length === 3
                        ? parts[0] * 3600 + parts[1] * 60 + parts[2]
                        : parts[0] * 60 + parts[1];
                };
                const readSegment = (segment) => {
                    const isModern =
                        segment.tagName.toLowerCase() === "transcript-segment-view-model";
                    const timestampElement = isModern
                        ? (
                            segment.querySelector(".ytwTranscriptSegmentViewModelTimestamp") ||
                            Array.from(segment.querySelectorAll("[class*='Timestamp']"))
                                .find((element) => !/A11y/i.test(String(element.className || "")))
                        )
                        : segment.querySelector(".segment-timestamp");
                    const textElement = isModern
                        ? segment.querySelector(".ytAttributedStringHost, [role='text']")
                        : segment.querySelector(".segment-text") ||
                            segment.querySelector("yt-formatted-string");
                    let text = "";
                    if (textElement) {
                        text = textElement.textContent || "";
                    } else {
                        const clone = segment.cloneNode(true);
                        clone.querySelectorAll(
                            "script, style, template, [class*='Timestamp']"
                        ).forEach((element) => element.remove());
                        text = clone.textContent || "";
                    }
                    const timestampText = timestampElement
                        ? (timestampElement.textContent || timestampElement.getAttribute("aria-label") || "")
                        : "";
                    return {
                        text: String(text).replace(/\s+/g, " ").trim(),
                        start: parseTimestampSeconds(timestampText)
                    };
                };

                let panel = expandedPanel();
                let openedPanel = false;
                if (!panelHasSegments(panel)) {
                    let showButton = document.querySelector(
                        "ytd-video-description-transcript-section-renderer button, " +
                        "ytd-video-description-transcript-section-renderer [role='button']"
                    );
                    if (!showButton) {
                        const expandDescription = document.querySelector(
                            "ytd-text-inline-expander#description-inline-expander #expand, " +
                            "ytd-text-inline-expander #expand"
                        );
                        if (expandDescription) {
                            expandDescription.click();
                            await new Promise((resolve) => setTimeout(resolve, 350));
                            showButton = document.querySelector(
                                "ytd-video-description-transcript-section-renderer button, " +
                                "ytd-video-description-transcript-section-renderer [role='button']"
                            );
                        }
                    }
                    if (!showButton) {
                        showButton = Array.from(
                            document.querySelectorAll("button, [role='button']")
                        ).find((element) =>
                            /show transcript/i.test(
                                (element.getAttribute("aria-label") || "") + " " +
                                (element.getAttribute("title") || "") + " " +
                                (element.textContent || "")
                            )
                        );
                    }
                    if (showButton) {
                        showButton.click();
                        openedPanel = !panel;
                    }
                }

                let previousSignature = "";
                let stableChecks = 0;
                let sawPanel = Boolean(panel);
                for (let waited = 0; waited < 20000; waited += 250) {
                    await new Promise((resolve) => setTimeout(resolve, 250));
                    if (!pageStillMatches()) {
                        return {
                            ok: false,
                            code: "VIDEO_CHANGED",
                            message: "The YouTube page changed while its transcript panel was loading."
                        };
                    }

                    panel = expandedPanel();
                    if (panel) {
                        sawPanel = true;
                    }
                    const elements = panel
                        ? Array.from(panel.querySelectorAll(segmentSelector))
                        : [];
                    const segments = elements
                        .map(readSegment)
                        .filter((segment) =>
                            segment.text &&
                            Number.isFinite(segment.start) &&
                            segment.start >= 0
                        );
                    const loading = Boolean(panel && panel.querySelector(
                        "ytd-continuation-item-renderer, yt-content-loading-renderer"
                    ));
                    const last = segments[segments.length - 1];
                    const signature = segments.length
                        ? [
                            segments.length,
                            segments[0].start,
                            last.start,
                            last.text.slice(0, 100),
                            segments.reduce((sum, item) => sum + item.text.length, 0)
                        ].join("|")
                        : "";

                    if (signature && !loading && signature === previousSignature) {
                        stableChecks += 1;
                    } else {
                        stableChecks = 0;
                    }
                    previousSignature = signature;

                    if (stableChecks >= 4) {
                        for (let index = 1; index < segments.length; index += 1) {
                            if (segments[index].start < segments[index - 1].start) {
                                return {
                                    ok: false,
                                    code: "TRANSCRIPT_PANEL_INCONSISTENT",
                                    message: "YouTube's transcript panel returned timestamps out of order."
                                };
                            }
                        }

                        const totalCharacters = segments.reduce(
                            (total, item) => total + item.text.length,
                            0
                        );
                        if (
                            segments.length > 100000 ||
                            totalCharacters > 1500000
                        ) {
                            return {
                                ok: false,
                                code: "CAPTION_LIMIT_EXCEEDED",
                                message: "This video's transcript exceeds the current indexing limit."
                            };
                        }

                        const snippets = segments.map((segment, index) => ({
                            text: segment.text,
                            start: segment.start,
                            duration: index + 1 < segments.length
                                ? Math.max(0, segments[index + 1].start - segment.start)
                                : 0
                        }));

                        const panelLanguage = panel.querySelector(
                            "ytd-transcript-footer-renderer #label-text"
                        )?.textContent?.trim() || "";
                        const selectedTrack = orderedTracks.find((track) => {
                            const label = captionText(track).toLowerCase();
                            const panelLabel = panelLanguage.toLowerCase();
                            return panelLabel &&
                                (panelLabel.includes(label) || label.includes(panelLabel));
                        }) || orderedTracks[0] || null;
                        const closeButton = panel.querySelector(
                            "yt-icon-button#visibility-button button, " +
                            "button[aria-label*='Close' i], [role='button'][aria-label*='Close' i]"
                        );

                        if (openedPanel && closeButton) {
                            closeButton.click();
                        }

                        return {
                            ok: true,
                            video_id: expectedVideoId,
                            language: panelLanguage ||
                                (selectedTrack ? captionText(selectedTrack) : "YouTube transcript"),
                            language_code: selectedTrack?.languageCode || "und",
                            is_generated: selectedTrack?.kind === "asr",
                            snippets,
                            source: "youtube_transcript_panel"
                        };
                    }
                }

                return {
                    ok: false,
                    code: sawPanel ? "TRANSCRIPT_PANEL_INCOMPLETE" : "NO_TRANSCRIPT_PANEL",
                    message: sawPanel
                        ? "YouTube opened the transcript panel but its timed segments did not finish loading."
                        : "YouTube did not expose its built-in transcript panel for this video."
                };
            };

            // Try up to six highest-priority tracks so one unavailable track does not
            // prevent the remaining preferred languages and fallbacks from being tried.
            const attempts = orderedTracks.slice(0, 6);
            const failures = [];
            for (const track of attempts) {
                if (!track.baseUrl) {
                    continue;
                }
                try {
                    const captionUrl = new URL(track.baseUrl, window.location.origin);
                    if (captionUrl.searchParams.get("exp") === "xpe") {
                        failures.push(
                            "This caption URL requires YouTube's runtime proof-of-origin token (exp=xpe)."
                        );
                        continue;
                    }
                    captionUrl.searchParams.set("fmt", "json3");
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 8000);
                    let response;
                    try {
                        response = await fetch(captionUrl.toString(), {
                            method: "GET",
                            credentials: "include",
                            cache: "no-store",
                            signal: controller.signal
                        });
                    } finally {
                        clearTimeout(timeoutId);
                    }

                    if (!response.ok) {
                        failures.push("HTTP " + response.status);
                        continue;
                    }

                    const raw = await response.text();
                    let snippets = [];
                    try {
                        snippets = parseJson3(JSON.parse(raw));
                    } catch (_error) {
                        snippets = parseXml(raw);
                    }

                    if (!snippets.length) {
                        failures.push("The caption track returned no timed segments.");
                        continue;
                    }

                    const totalCharacters = snippets.reduce(
                        (total, item) => total + item.text.length,
                        0
                    );
                    if (snippets.length > 100000 || totalCharacters > 1500000) {
                        return {
                            ok: false,
                            code: "CAPTION_LIMIT_EXCEEDED",
                            message: "This video's caption track is larger than the current indexing limit."
                        };
                    }

                    if (currentVideoId() !== expectedVideoId) {
                        return {
                            ok: false,
                            code: "VIDEO_CHANGED",
                            message: "The YouTube video changed while captions were being retrieved."
                        };
                    }

                    const videoIdFromPlayer = playerResponse?.videoDetails?.videoId;
                    if (videoIdFromPlayer && videoIdFromPlayer !== expectedVideoId) {
                        return {
                            ok: false,
                            code: "VIDEO_CHANGED",
                            message: "The YouTube player changed videos while captions were being retrieved."
                        };
                    }

                    return {
                        ok: true,
                        video_id: expectedVideoId,
                        language: captionText(track),
                        language_code: track.languageCode || "und",
                        is_generated: track.kind === "asr",
                        snippets
                    };
                } catch (error) {
                    failures.push(error?.name === "AbortError"
                        ? "caption request timed out"
                        : String(error?.message || error));
                }
            }

            // Prefer YouTube's own rendered transcript over returning a generic
            // failure. It uses the same authenticated player flow that works in
            // the page even when direct timed-text URLs are proof-token gated.
            const panelCapture = await captureTranscriptFromPanel();
            if (panelCapture.ok) {
                return panelCapture;
            }
            failures.push(panelCapture.message);

            return {
                ok: false,
                code: panelCapture.code || (
                    captionTracks.length ? "CAPTION_FETCH_FAILED" : "NO_CAPTION_TRACKS"
                ),
                message: failures.length
                    ? "Browser caption acquisition failed. " + failures.join("; ")
                    : "The YouTube player did not expose a usable caption source."
            };
        }
    });

    const mainFrameResult = results?.find((entry) => entry.frameId === 0) || results?.[0];
    if (!mainFrameResult) {
        const error = new Error("Chrome did not return a result for the YouTube page caption-capture script.");
        error.errorCode = "BROWSER_SCRIPT_NOT_EXECUTED";
        throw error;
    }
    if (mainFrameResult.error) {
        const error = new Error(
            String(mainFrameResult.error?.message || mainFrameResult.error)
        );
        error.errorCode = "BROWSER_SCRIPT_EXECUTION_FAILED";
        throw error;
    }

    const capture = mainFrameResult.result;
    if (!capture || capture.ok !== true) {
        const error = new Error(
            capture?.message || "The YouTube player did not provide captions."
        );
        error.errorCode = capture?.code || "BROWSER_CAPTURE_FAILED";
        throw error;
    }
    if (capture.video_id !== videoId) {
        const error = new Error("The browser returned captions for a different YouTube video.");
        error.errorCode = "BROWSER_VIDEO_ID_MISMATCH";
        throw error;
    }

    return {
        video_id: capture.video_id,
        language: capture.language,
        language_code: capture.language_code,
        is_generated: capture.is_generated,
        snippets: capture.snippets
    };
}


async function fetchIndexStatusForPreparation(videoId) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 7000);
    try {
        const response = await fetch(
            INDEX_STATUS_URL + "/" + encodeURIComponent(videoId),
            {
                method: "GET",
                headers: { "Accept": "application/json" },
                signal: controller.signal
            }
        );
        const data = await parseResponse(response);
        if (!response.ok) {
            return null;
        }
        return data;
    } finally {
        clearTimeout(timeoutId);
    }
}


async function postBrowserTranscript(transcript) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000);
    try {
        const response = await fetch(INDEX_BROWSER_TRANSCRIPT_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(transcript),
            signal: controller.signal
        });
        const data = await parseResponse(response);
        if (!response.ok) {
            throw new Error(
                data?.error_message ||
                extractBackendError(data) ||
                ("Browser transcript upload failed (HTTP " + response.status + ").")
            );
        }
        return data;
    } finally {
        clearTimeout(timeoutId);
    }
}


async function handleIndexPreparation(videoId, tabId, sendResponse) {
    // Avoid fetching the same captions again when an index is already ready
    // or a durable worker is already preparing this video.
    try {
        const current = await fetchIndexStatusForPreparation(videoId);
        if (
            current &&
            (
                current.ready === true ||
                current.state === "ready"
            )
        ) {
            sendResponse({
                success: true,
                video_id: current.video_id || videoId,
                state: current.state,
                action: current.action,
                ready: current.ready === true,
                error_code: current.error_code || null,
                error_message: current.error_message || null
            });
            return;
        }
    } catch (_error) {
        // Continue to browser capture; a later upload or fallback will report
        // a backend connectivity error if the local service is unavailable.
    }

    // The browser tab uses the same YouTube session and network path that
    // loaded the video. This is the primary transcript source.
    let browserFailure = null;
    let browserFailureStage = "capture";
    try {
        const transcript = await fetchBrowserTranscriptFromTab(tabId, videoId);
        browserFailureStage = "upload";
        const data = await postBrowserTranscript(transcript);
        sendResponse({
            success: true,
            video_id: data.video_id || videoId,
            state: data.state || "queued",
            action: data.action || "create",
            ready: data.ready === true,
            error_code: data.error_code || null,
            error_message: data.error_message || null
        });
        return;
    } catch (browserError) {
        browserFailure = {
            stage: browserFailureStage,
            code: browserError?.errorCode || (
                browserFailureStage === "upload"
                    ? "BROWSER_TRANSCRIPT_UPLOAD_FAILED"
                    : "BROWSER_CAPTURE_FAILED"
            ),
            message: String(browserError?.message || browserError || "Unknown browser caption failure.")
        };
        console.warn(
            "Browser caption capture was unavailable; falling back to server transcript acquisition:",
            {
                videoId,
                tabId,
                browserErrorCode: browserFailure.code,
                browserErrorMessage: browserFailure.message
            }
        );
    }

    // Keep the existing transcript API as a fallback when the active player
    // does not expose a usable caption track.
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
            const backendMessage = data?.error_message ||
                extractBackendError(data) ||
                ("Index request failed (HTTP " + response.status + ").");
            const error = new Error(backendMessage);
            error.errorCode = data?.error_code || null;
            throw error;
        }

        if (!data || typeof data !== "object") {
            throw new Error("The backend returned an invalid index response.");
        }

        sendResponse({
            success: true,
            video_id: data.video_id,
            state: data.state,
            action: data.action,
            ready: data.ready === true,
            error_code: data.error_code || null,
            error_message: data.error_message || null,
            browser_diagnostic: browserFailure
        });
    } catch (error) {
        const serverErrorMessage = error?.name === "AbortError"
            ? "The index service did not respond. Reopen the chat to retry."
            : (error instanceof Error && error.message
                ? error.message
                : "Unable to contact the index service.");
        const browserDiagnostic = browserFailure
            ? "\n\nBrowser transcript " + browserFailure.stage + " failed (" +
                browserFailure.code + "): " + browserFailure.message
            : "";
        sendResponse({
            success: false,
            error_code: error?.errorCode || null,
            error: serverErrorMessage + browserDiagnostic
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
            ready: data?.ready === true,
            error_code: data?.error_code || null,
            error_message: data?.error_message || null
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
// 3. HANDLE ANSWER FEEDBACK
// ============================================================

async function handleFeedbackSubmission(feedback, sendResponse) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
        const response = await fetch(FEEDBACK_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(feedback),
            signal: controller.signal
        });
        const data = await parseResponse(response);

        if (!response.ok) {
            throw new Error(
                extractBackendError(data) ||
                `Feedback submission failed (HTTP ${response.status}).`
            );
        }

        sendResponse({
            success: true,
            feedback_id: typeof data?.feedback_id === "string" ? data.feedback_id : null,
            status: typeof data?.status === "string" ? data.status : "received"
        });
    } catch (error) {
        sendResponse({
            success: false,
            error: error?.name === "AbortError"
                ? "Feedback submission timed out. Please retry."
                : (error instanceof Error && error.message
                    ? error.message
                    : "Unable to submit feedback.")
        });
    } finally {
        clearTimeout(timeoutId);
    }
}


// ============================================================
// 4. HANDLE CHAT REQUEST
// ============================================================


async function handleChatRequest(
    {
        videoId,
        question,
        history
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
            CHAT_REQUEST_TIMEOUT_MS
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
                            question,

                        conversation_history:
                            history
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

            request_id:
                response.headers.get("X-Request-ID") || null,

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


const MAX_CONVERSATION_TURN_CHARS = 1000;


function normalizeConversationHistory(
    value
) {
    if (!Array.isArray(value)) {
        return [];
    }

    return value
        .slice(-6)
        .map(
            (turn) => ({
                role:
                    turn?.role === "assistant"
                        ? "assistant"
                        : "user",
                // Match ChatRequest's per-turn 1000-character limit so a
                // long assistant answer cannot make every follow-up fail validation.
                content:
                    normalizeString(
                        turn?.content
                    ).slice(0, MAX_CONVERSATION_TURN_CHARS)
            })
        )
        .filter(
            (turn) => turn.content
        );
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