/**
 * content.js
 *
 * YouTube page integration.
 *
 * Responsibilities:
 * 1. Detect the current YouTube video.
 * 2. Keep the chat closed by default.
 * 3. Open/close the chat panel.
 * 4. Send user questions to the background service worker.
 * 5. Display the RAG answer.
 * 6. Display clickable source timestamps.
 */


// ============================================================
// STATE
// ============================================================

let sidebar = null;

let launcher = null;

let chatBody = null;

let chatInput = null;

let sendButton = null;

let sourcesContainer = null;

let isRequestInProgress = false;

let activeRequestId = 0;

const MAX_CONVERSATION_HISTORY = 6;

let conversationHistory = [];

// Keep all evidence segments surfaced during the current video's
// conversation so a later question does not replace earlier timestamps.
let sourceHistory = [];

let lastVideoId = null;

let indexStatus = "unknown";

let indexStatusVideoId = null;

let indexPreparationInProgressFor = null;

let indexPollTimer = null;

let indexPollVideoId = null;

let indexPollAttempts = 0;

const INDEX_POLL_INTERVAL_MS = 2000;

const INDEX_POLL_MAX_ATTEMPTS = 90;

// Floating launcher and panel geometry. The launcher remains visible while
// the chat is open so it can always be repositioned.
const LAUNCHER_SIZE = 60;
const VIEWPORT_MARGIN = 12;
const CHAT_PANEL_GAP = 12;
const MIN_USABLE_PANEL_WIDTH = 280;
const MIN_USABLE_PANEL_HEIGHT = 260;
const MAX_CHAT_PANEL_WIDTH = 380;
const MAX_CHAT_PANEL_HEIGHT = 560;
const LAUNCHER_POSITION_KEY = "youtube-ai-chat-position-v1";

let launcherDragState = null;
let suppressLauncherClick = false;

// ============================================================
// 1. YOUTUBE VIDEO ID
// ============================================================

function getVideoId() {

    const url = new URL(
        window.location.href
    );


    // Standard watch URL:
    // youtube.com/watch?v=VIDEO_ID

    const watchVideoId =
        url.searchParams.get("v");


    if (watchVideoId) {

        return watchVideoId;
    }


    // Supported path formats:
    //
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
            pathParts[0] === "shorts" ||
            pathParts[0] === "embed" ||
            pathParts[0] === "live"
        )
    ) {

        return pathParts[1];
    }


    return null;
}


// ============================================================
// 2. PAGE VALIDATION
// ============================================================

function isVideoPage() {

    return Boolean(
        getVideoId()
    );
}


// ============================================================
// 3. TIMESTAMP FORMAT
// ============================================================

function formatTimestamp(
    seconds
) {

    const totalSeconds =
        Math.max(
            0,
            Math.floor(
                Number(seconds) || 0
            )
        );


    const hours =
        Math.floor(
            totalSeconds / 3600
        );


    const minutes =
        Math.floor(
            (totalSeconds % 3600) / 60
        );


    const remainingSeconds =
        totalSeconds % 60;


    if (hours > 0) {

        return (
            `${String(hours).padStart(2, "0")}:` +
            `${String(minutes).padStart(2, "0")}:` +
            `${String(remainingSeconds).padStart(2, "0")}`
        );
    }


    return (
        `${String(minutes).padStart(2, "0")}:` +
        `${String(remainingSeconds).padStart(2, "0")}`
    );
}


// ============================================================
// 4. FLOATING LAUNCHER: POSITION, DRAG, AND VIEWPORT SAFETY
// ============================================================

function clampNumber(value, minimum, maximum) {
    return Math.min(Math.max(value, minimum), Math.max(minimum, maximum));
}


function readSavedLauncherPosition() {
    try {
        const raw = window.localStorage.getItem(LAUNCHER_POSITION_KEY);
        if (!raw) {
            return null;
        }

        const saved = JSON.parse(raw);
        if (
            saved &&
            Number.isFinite(saved.left) &&
            Number.isFinite(saved.top)
        ) {
            return saved;
        }
    } catch (error) {
        // Storage can be unavailable in private/restricted browsing contexts.
    }

    return null;
}


function setLauncherPosition(left, top, persist = false) {
    if (!launcher) {
        return;
    }

    const maxLeft = Math.max(
        VIEWPORT_MARGIN,
        window.innerWidth - LAUNCHER_SIZE - VIEWPORT_MARGIN
    );
    const maxTop = Math.max(
        VIEWPORT_MARGIN,
        window.innerHeight - LAUNCHER_SIZE - VIEWPORT_MARGIN
    );

    const position = {
        left: Math.round(clampNumber(left, VIEWPORT_MARGIN, maxLeft)),
        top: Math.round(clampNumber(top, VIEWPORT_MARGIN, maxTop))
    };

    launcher.style.left = position.left + "px";
    launcher.style.top = position.top + "px";
    launcher.style.right = "auto";
    launcher.style.bottom = "auto";

    if (persist) {
        try {
            window.localStorage.setItem(
                LAUNCHER_POSITION_KEY,
                JSON.stringify(position)
            );
        } catch (error) {
            // Position still works for this page even if persistence is blocked.
        }
    }

    if (sidebar && !sidebar.hidden) {
        positionChatPanel();
    }
}


function positionChatPanel() {
    if (!sidebar || sidebar.hidden || !launcher) {
        return;
    }

    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    // Do not leave a tiny, unusable panel open on genuinely small viewports.
    // Normal edge positions are handled by clamping the panel back on-screen.
    if (
        viewportWidth < MIN_USABLE_PANEL_WIDTH ||
        viewportHeight < MIN_USABLE_PANEL_HEIGHT + 2 * VIEWPORT_MARGIN
    ) {
        closeChat();
        return;
    }

    const panelWidth = Math.min(
        MAX_CHAT_PANEL_WIDTH,
        viewportWidth - 2 * VIEWPORT_MARGIN
    );

    const launcherRect = launcher.getBoundingClientRect();

    // The panel opens above the floating launcher. As the launcher is dragged
    // toward the top edge, the panel progressively shrinks; when it cannot
    // retain a readable minimum height, the panel closes automatically.
    const availableHeight = (
        launcherRect.top - CHAT_PANEL_GAP - VIEWPORT_MARGIN
    );
    const panelHeight = Math.min(
        MAX_CHAT_PANEL_HEIGHT,
        viewportHeight - 2 * VIEWPORT_MARGIN,
        availableHeight
    );

    if (
        panelWidth < MIN_USABLE_PANEL_WIDTH ||
        panelHeight < MIN_USABLE_PANEL_HEIGHT
    ) {
        closeChat();
        return;
    }

    sidebar.style.width = Math.floor(panelWidth) + "px";
    sidebar.style.height = Math.floor(panelHeight) + "px";
    sidebar.style.maxWidth = Math.floor(viewportWidth - 2 * VIEWPORT_MARGIN) + "px";
    sidebar.style.maxHeight = Math.floor(viewportHeight - 2 * VIEWPORT_MARGIN) + "px";
    sidebar.style.minWidth = "0";
    sidebar.style.minHeight = "0";
    sidebar.style.resize = "none";

    const launcherIsOnRight = (
        launcherRect.left + launcherRect.width / 2 >= viewportWidth / 2
    );

    const preferredLeft = launcherIsOnRight
        ? launcherRect.left - CHAT_PANEL_GAP - panelWidth
        : launcherRect.right + CHAT_PANEL_GAP;

    const left = clampNumber(
        preferredLeft,
        VIEWPORT_MARGIN,
        viewportWidth - panelWidth - VIEWPORT_MARGIN
    );

    // The panel's bottom stays above the launcher, while its height is based
    // on the available viewport space. The chat body scrolls independently.
    const top = launcherRect.top - CHAT_PANEL_GAP - panelHeight;

    sidebar.style.left = Math.round(left) + "px";
    sidebar.style.top = Math.round(top) + "px";
    sidebar.style.right = "auto";
    sidebar.style.bottom = "auto";
}


function updateLauncherAccessibility() {
    if (!launcher) {
        return;
    }

    const chatIsOpen = Boolean(sidebar && !sidebar.hidden);
    const label = chatIsOpen
        ? "Close YouTube AI Chat"
        : "Open YouTube AI Chat";

    launcher.title = label;
    launcher.setAttribute("aria-label", label);
    launcher.setAttribute("aria-expanded", String(chatIsOpen));
}


function installLauncherDragging() {
    if (!launcher) {
        return;
    }

    launcher.addEventListener("pointerdown", (event) => {
        if (event.pointerType === "mouse" && event.button !== 0) {
            return;
        }

        const rect = launcher.getBoundingClientRect();
        launcherDragState = {
            pointerId: event.pointerId,
            startX: event.clientX,
            startY: event.clientY,
            originLeft: rect.left,
            originTop: rect.top,
            moved: false
        };

        try {
            launcher.setPointerCapture(event.pointerId);
        } catch (error) {
            // Pointer capture may be unavailable in a restricted WebView.
        }
    });

    launcher.addEventListener("pointermove", (event) => {
        if (
            !launcherDragState ||
            event.pointerId !== launcherDragState.pointerId
        ) {
            return;
        }

        const deltaX = event.clientX - launcherDragState.startX;
        const deltaY = event.clientY - launcherDragState.startY;

        if (!launcherDragState.moved && Math.hypot(deltaX, deltaY) < 5) {
            return;
        }

        launcherDragState.moved = true;
        launcher.classList.add("is-dragging");
        event.preventDefault();

        setLauncherPosition(
            launcherDragState.originLeft + deltaX,
            launcherDragState.originTop + deltaY
        );
    });

    const finishDrag = (event) => {
        if (
            !launcherDragState ||
            event.pointerId !== launcherDragState.pointerId
        ) {
            return;
        }

        const wasDragged = launcherDragState.moved;
        launcherDragState = null;
        launcher.classList.remove("is-dragging");

        if (wasDragged) {
            const rect = launcher.getBoundingClientRect();
            setLauncherPosition(rect.left, rect.top, true);

            // A drag can generate a synthetic click. Suppress it so moving the
            // icon never accidentally opens or closes the chat.
            suppressLauncherClick = true;
            window.setTimeout(() => {
                suppressLauncherClick = false;
            }, 0);
        }

        try {
            if (launcher.hasPointerCapture(event.pointerId)) {
                launcher.releasePointerCapture(event.pointerId);
            }
        } catch (error) {
            // Pointer capture may already have been released by the browser.
        }
    };

    launcher.addEventListener("pointerup", finishDrag);
    launcher.addEventListener("pointercancel", finishDrag);
}


function createLauncher() {
    if (launcher) {
        return;
    }

    launcher = document.createElement("button");
    launcher.id = "youtube-ai-chat-launcher";
    launcher.type = "button";
    launcher.setAttribute("aria-controls", "youtube-ai-chat");
    launcher.setAttribute("aria-expanded", "false");
    launcher.draggable = false;

    const avatar = document.createElement("img");
    avatar.className = "chatbot-avatar";
    avatar.src = chrome.runtime.getURL("assets/chatbot-avatar.jpg");
    avatar.alt = "";
    avatar.draggable = false;
    avatar.setAttribute("aria-hidden", "true");
    launcher.appendChild(avatar);

    launcher.addEventListener("click", (event) => {
        if (suppressLauncherClick) {
            event.preventDefault();
            event.stopPropagation();
            suppressLauncherClick = false;
            return;
        }

        toggleChat();
    });

    document.body.appendChild(launcher);
    installLauncherDragging();
    updateLauncherAccessibility();

    const savedPosition = readSavedLauncherPosition();
    setLauncherPosition(
        savedPosition ? savedPosition.left : window.innerWidth - LAUNCHER_SIZE - 20,
        savedPosition ? savedPosition.top : window.innerHeight - LAUNCHER_SIZE - 20
    );

    window.addEventListener("resize", () => {
        if (!launcher) {
            return;
        }

        const rect = launcher.getBoundingClientRect();
        setLauncherPosition(rect.left, rect.top);
        positionChatPanel();
    });
}


// ============================================================
// 5. CREATE SIDEBAR
// ============================================================

function createSidebar() {

    if (sidebar) {
        return;
    }


    sidebar =
        document.createElement("aside");


    sidebar.id =
        "youtube-ai-chat";


    sidebar.setAttribute(
        "aria-label",
        "YouTube AI Chat"
    );


    sidebar.hidden = true;


    sidebar.innerHTML = `

        <div class="chat-header">

            <div class="chat-title">
                <span class="chat-title-avatar">
                    <img
                        src="${chrome.runtime.getURL("assets/chatbot-avatar.jpg")}"
                        alt=""
                        draggable="false"
                        aria-hidden="true"
                    />
                </span>
                <span>YouTube AI</span>
            </div>

            <button
                type="button"
                id="chat-close"
                class="chat-close"
                aria-label="Close chat"
                title="Close"
            >
                ×
            </button>

        </div>


        <div
            class="chat-body"
            id="chat-body"
        >

            <div class="welcome-message">

    <strong>
        Ask anything about this video.
    </strong>

    <span>
        Answers are generated from the video's transcript.
    </span>

</div>

            <div
                id="index-status"
                class="index-status"
                role="status"
                aria-live="polite"
            >
                Preparing this video...
            </div>

        </div>


        <div
            id="chat-sources"
            class="chat-sources"
            hidden
        ></div>


        <div class="chat-input-area">

            <input
                type="text"
                id="chat-input"
                placeholder="Ask a question..."
                autocomplete="off"
                maxlength="1000"
            />

            <button
                type="button"
                id="chat-send"
            >
                Send
            </button>

        </div>
    `;


    document.body.appendChild(
        sidebar
    );


    chatBody =
        sidebar.querySelector(
            "#chat-body"
        );


    chatInput =
        sidebar.querySelector(
            "#chat-input"
        );


    sendButton =
        sidebar.querySelector(
            "#chat-send"
        );


    sourcesContainer =
        sidebar.querySelector(
            "#chat-sources"
        );


    const closeButton =
        sidebar.querySelector(
            "#chat-close"
        );


    // --------------------------------------------------------
    // Close button
    // --------------------------------------------------------

    closeButton.addEventListener(
        "click",
        closeChat
    );


    // --------------------------------------------------------
    // Send button
    // --------------------------------------------------------

    sendButton.addEventListener(
        "click",
        sendMessage
    );


    // --------------------------------------------------------
    // Enter key
    // --------------------------------------------------------

    chatInput.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {

                event.preventDefault();

                sendMessage();
            }
        }
    );


    // --------------------------------------------------------
    // Escape closes the chat
    // --------------------------------------------------------

    document.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Escape" &&
                !sidebar.hidden
            ) {

                closeChat();
            }
        }
    );
}


// ============================================================
// 6. OPEN CHAT
// ============================================================

function updateIndexStatus(videoId, status, message = "") {
    if (videoId !== getVideoId()) {
        return;
    }

    indexStatusVideoId = videoId;
    indexStatus = status;

    const statusElement = sidebar?.querySelector("#index-status");
    if (!statusElement) {
        return;
    }

    const messages = {
        unknown: "Checking video readiness...",
        queued: "Video preparation is queued...",
        preparing: "Preparing this video. This may take a moment...",
        ready: "Video ready — ask anything about it.",
        failed: message || "Could not prepare this video. Reopen the chat to retry."
    };

    statusElement.textContent = messages[status] || messages.unknown;
    statusElement.dataset.state = status;
}


function clearIndexPolling() {
    if (indexPollTimer) {
        clearTimeout(indexPollTimer);
        indexPollTimer = null;
    }

    indexPollVideoId = null;
    indexPollAttempts = 0;
}


function pollIndexStatus(videoId) {
    if (!videoId || videoId !== getVideoId()) {
        clearIndexPolling();
        return;
    }

    if (indexPollAttempts >= INDEX_POLL_MAX_ATTEMPTS) {
        clearIndexPolling();
        updateIndexStatus(
            videoId,
            "failed",
            "Index preparation is taking longer than expected. Reopen the chat to retry."
        );
        return;
    }

    indexPollVideoId = videoId;
    indexPollAttempts += 1;

    chrome.runtime.sendMessage(
        {
            type: "index_status",
            video_id: videoId
        },
        (response) => {
            const runtimeError = chrome.runtime.lastError;

            if (videoId !== getVideoId()) {
                clearIndexPolling();
                return;
            }

            if (runtimeError || !response || response.success !== true) {
                indexPollTimer = setTimeout(
                    () => pollIndexStatus(videoId),
                    INDEX_POLL_INTERVAL_MS
                );
                return;
            }

            if (response.ready === true || response.state === "ready") {
                clearIndexPolling();
                updateIndexStatus(videoId, "ready");
                return;
            }

            if (response.state === "failed") {
                clearIndexPolling();
                updateIndexStatus(
                    videoId,
                    "failed",
                    "Could not prepare this video. Reopen the chat to retry."
                );
                return;
            }

            if (response.state === "queued") {
                updateIndexStatus(videoId, "queued");
            } else {
                updateIndexStatus(videoId, "preparing");
            }

            indexPollTimer = setTimeout(
                () => pollIndexStatus(videoId),
                INDEX_POLL_INTERVAL_MS
            );
        }
    );
}


function requestIndexPreparation(videoId) {
    if (!videoId || videoId !== getVideoId()) {
        return;
    }

    if (
        indexPreparationInProgressFor === videoId ||
        (indexStatusVideoId === videoId && indexStatus === "ready")
    ) {
        return;
    }

    clearIndexPolling();

    indexPreparationInProgressFor = videoId;
    updateIndexStatus(videoId, "preparing");

    chrome.runtime.sendMessage(
        {
            type: "prepare_index",
            video_id: videoId
        },
        (response) => {
            const runtimeError = chrome.runtime.lastError;

            if (indexPreparationInProgressFor === videoId) {
                indexPreparationInProgressFor = null;
            }

            if (videoId !== getVideoId()) {
                return;
            }

            if (runtimeError) {
                updateIndexStatus(
                    videoId,
                    "failed",
                    "The extension could not contact the backend. Reopen the chat to retry."
                );
                return;
            }

            if (!response || response.success !== true) {
                updateIndexStatus(
                    videoId,
                    "failed",
                    response?.error || "Could not start video preparation. Reopen the chat to retry."
                );
                return;
            }

            if (response.ready === true || response.state === "ready") {
                clearIndexPolling();
                updateIndexStatus(videoId, "ready");
                return;
            }

            if (response.state === "queued") {
                updateIndexStatus(videoId, "queued");
            } else {
                updateIndexStatus(videoId, "preparing");
            }

            pollIndexStatus(videoId);
        }
    );
}

function openChat() {

    if (!isVideoPage()) {

        showTemporaryLauncherMessage(
            "Open a YouTube video first."
        );

        return;
    }


    createSidebar();


    sidebar.hidden = false;
    updateLauncherAccessibility();
    positionChatPanel();


    // Capture the current video ID and retry preparation if needed.
    lastVideoId = getVideoId();

    if (indexStatusVideoId !== lastVideoId || indexStatus === "failed" || indexStatus === "unknown") {
        requestIndexPreparation(lastVideoId);
    } else {
        updateIndexStatus(lastVideoId, indexStatus);
    }

    setTimeout(
        () => {

            if (chatInput) {

                chatInput.focus();
            }

        },
        0
    );
}


// ============================================================
// 7. CLOSE CHAT
// ============================================================

function closeChat() {

    if (!sidebar) {
        return;
    }


    sidebar.hidden = true;

    if (launcher) {
        launcher.hidden = !isVideoPage();
        updateLauncherAccessibility();
    }
}


// ============================================================
// 8. TOGGLE CHAT
// ============================================================

function toggleChat() {

    if (!sidebar) {

        openChat();

        return;
    }


    if (sidebar.hidden) {

        openChat();

    } else {

        closeChat();
    }
}


// ============================================================
// 9. TEMPORARY LAUNCHER MESSAGE
// ============================================================

function showTemporaryLauncherMessage(message) {
    if (!launcher) {
        return;
    }

    launcher.title = message;
    launcher.setAttribute("aria-label", message);
    launcher.classList.add("launcher-warning");

    setTimeout(() => {
        if (launcher) {
            launcher.classList.remove("launcher-warning");
            updateLauncherAccessibility();
        }
    }, 1800);
}


// ============================================================
// 10. APPEND MESSAGE
// ============================================================

function appendMessage(text, type) {

    const message = document.createElement("div");

    message.className =
        `chat-message ${type}`;

    if (type === "assistant") {

        message.innerHTML =
            renderMarkdown(text);

    } else {

        message.textContent = text;
    }

    chatBody.appendChild(message);

    chatBody.scrollTop =
        chatBody.scrollHeight;

    return message;
}


// ============================================================
// 11. CLEAR SOURCES
// ============================================================

function clearSources() {

    if (!sourcesContainer) {
        return;
    }


    sourcesContainer.innerHTML =
        "";


    sourcesContainer.hidden =
        true;
}


function resetSourceHistory() {

    sourceHistory = [];

    clearSources();
}


function mergeSources(
    sources
) {

    if (!Array.isArray(sources)) {
        return;
    }

    sources.forEach(
        (source) => {

            const start = Number(source?.start);
            const endValue = Number(source?.end);
            const end = Number.isFinite(endValue)
                ? endValue
                : start;

            if (
                !Number.isFinite(start) ||
                start < 0 ||
                !Number.isFinite(end) ||
                end < start
            ) {
                return;
            }

            const alreadyExists =
                sourceHistory.some(
                    (existing) =>
                        existing.start === start &&
                        existing.end === end
                );

            if (!alreadyExists) {
                sourceHistory.push({
                    start,
                    end
                });
            }
        }
    );
}


// ============================================================
// 12. RENDER SOURCES
// ============================================================

function renderSources() {

    clearSources();

    if (sourceHistory.length === 0) {
        return;
    }

    const sourcesToggle =
        document.createElement("button");

    sourcesToggle.type = "button";
    sourcesToggle.className = "sources-toggle";
    sourcesToggle.textContent =
        "Video sources · " + sourceHistory.length;
    sourcesToggle.setAttribute("aria-expanded", "false");
    sourcesToggle.setAttribute("aria-controls", "chat-sources-list");

    const list =
        document.createElement("div");

    list.id = "chat-sources-list";
    list.className = "sources-list";
    list.hidden = true;

    sourceHistory.forEach(
        (source, index) => {

            const sourceButton =
                document.createElement("button");

            sourceButton.type = "button";
            sourceButton.className = "source-button";

            const start =
                Number(source.start) || 0;

            const endValue = Number(source.end);
            const end = Number.isFinite(endValue)
                ? endValue
                : start;

            const sourceId = index + 1;

            sourceButton.textContent =
                "Source " + sourceId + " · " +
                formatTimestamp(start) + " → " +
                formatTimestamp(end);

            sourceButton.title =
                "Jump to this evidence segment";

            sourceButton.setAttribute(
                "aria-label",
                "Source " + sourceId + ", " +
                formatTimestamp(start) + " to " +
                formatTimestamp(end) + ". Jump to this part of the video."
            );

            sourceButton.addEventListener(
                "click",
                () => {
                    seekToTimestamp(start);
                }
            );

            list.appendChild(sourceButton);
        }
    );

    sourcesToggle.addEventListener(
        "click",
        () => {
            const expanded =
                sourcesToggle.getAttribute("aria-expanded") !== "true";

            sourcesToggle.setAttribute(
                "aria-expanded",
                String(expanded)
            );

            list.hidden = !expanded;
        }
    );

    sourcesContainer.appendChild(sourcesToggle);
    sourcesContainer.appendChild(list);
    sourcesContainer.hidden = false;
}


// ============================================================
// 13. SEEK YOUTUBE VIDEO
// ============================================================

function seekToTimestamp(
    seconds
) {

    const video =
        document.querySelector(
            "video"
        );


    if (!video) {

        appendMessage(
            "YouTube's video player could not be found.",
            "system"
        );

        return;
    }


    const timestamp =
        Number(seconds);


    if (
        !Number.isFinite(timestamp) ||
        timestamp < 0
    ) {

        return;
    }


    video.currentTime =
        timestamp;


    video.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}


// new code adding.
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}


function renderMarkdown(text) {
    // Normalize common provider artifacts before parsing Markdown. Do not decode
    // arbitrary HTML: the text is escaped below before any formatting is added.
    const normalizedText = String(text ?? "")
        .replace(/\r\n?/g, "\n")
        .replace(/&#x20;|&#32;/gi, " ")
        .replace(/^[ \t]*\\[ \t]*$/gm, "")
        .replace(/^[ \t]*[-*][ \t]*\\?[ \t]*$/gm, "")
        // Models sometimes insert blank lines between bullets, which should not
        // create vertical gaps inside a single list.
        .replace(/\n[ \t]*\n(?=[ \t]*[-*]\s+)/g, "\n")
        .replace(/\n{3,}/g, "\n\n");

    let html = escapeHtml(normalizedText);

    // Convert escaped Markdown characters back first.
    html = html.replace(/\\\\([*_\x60])/g, "$1");

    // Remove empty Markdown bullet lines.
    html = html.replace(/^[ \t]*[-*][ \t]*$/gm, "");

    // Bold: **text**
    html = html.replace(
        /\*\*(.+?)\*\*/gs,
        "<strong>$1</strong>"
    );

    // Italic: *text*
    html = html.replace(
        /(^|[^\*])\*([^*\n]+)\*(?!\*)/g,
        "$1<em>$2</em>"
    );

    // Inline code: `code`
    html = html.replace(
        /`([^`]+)`/g,
        "<code>$1</code>"
    );

    // Bullet points
    html = html.replace(
        /^[ \t]*[-*][ \t]+(.+)$/gm,
        "<li>$1</li>"
    );

    // Group consecutive list items.
    html = html.replace(
        /(<li>.*?<\/li>(?:\s*<li>.*?<\/li>)*)/gs,
        "<ul>$1</ul>"
    );

    // Convert newlines to line breaks, then remove those introduced between
    // list elements. Each list item should occupy only its natural line height.
    html = html.replace(/\n/g, "<br>");
    html = html.replace(/<\/li>(?:<br\s*\/?>(?:\s*)?)+<li>/g, "</li><li>");
    html = html.replace(/<ul>(?:<br\s*\/?>(?:\s*)?)+/g, "<ul>");
    html = html.replace(/(?:<br\s*\/?>(?:\s*)?)+<\/ul>/g, "</ul>");

    return html;
}

// ============================================================
// REQUEST STATE RESET
// ============================================================

function resetRequestState() {
    isRequestInProgress = false;

    if (sendButton) {
        sendButton.disabled = false;
        sendButton.textContent = "Send";
    }

    if (chatInput) {
        chatInput.disabled = false;
    }
}

// ============================================================
// 14. SEND MESSAGE
// ============================================================

function sendMessage() {
    if (isRequestInProgress || !chatInput) {
        return;
    }

    const question = chatInput.value.trim();

    if (!question) {
        return;
    }

    const conversationalGreetings = new Set([
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you"
    ]);

    const normalizedQuestion = question
        .toLowerCase()
        .trim()
        .replace(/[.!?]+$/, "");

    // Keep simple conversation local.
    if (conversationalGreetings.has(normalizedQuestion)) {
        appendMessage(question, "user");
        chatInput.value = "";

        appendMessage(
            "Hi! Ask me anything about this video.",
            "assistant"
        );

        return;
    }

    const videoId = getVideoId();

    if (!videoId) {
        appendMessage(
            "Please open a YouTube video before asking a question.",
            "system"
        );
        return;
    }

    appendMessage(question, "user");
    chatInput.value = "";

    const aiMessage = appendMessage(
        "Thinking...",
        "assistant"
    );

    if (!aiMessage) {
        return;
    }

    isRequestInProgress = true;

    if (sendButton) {
        sendButton.disabled = true;
        sendButton.textContent = "...";
    }

    chatInput.disabled = true;

    const requestedVideoId = videoId;

    // Give this request a unique ID.
    // This prevents late responses from older requests
    // from modifying the current chat state.
    const requestId = ++activeRequestId;

    try {
        chrome.runtime.sendMessage(
            {
                type: "chat",
                video_id: requestedVideoId,
                question,
                history: conversationHistory.slice(
                    -MAX_CONVERSATION_HISTORY
                )
            },
            (response) => {
                const runtimeError = chrome.runtime.lastError;

                if (runtimeError) {
                    if (requestId === activeRequestId) {
                        resetRequestState();

                        aiMessage.textContent =
                            "The extension could not contact its background service.";
                    }

                    console.error(
                        "YouTube AI background error:",
                        runtimeError.message
                    );

                    return;
                }

                // Ignore responses belonging to an old request
                // or a previous YouTube video.
                if (
                    requestId !== activeRequestId ||
                    requestedVideoId !== getVideoId()
                ) {
                    return;
                }

                resetRequestState();

                if (!response || !response.success) {
                    aiMessage.textContent =
                        response?.error ||
                        "The backend could not process the request.";

                    return;
                }

                const answer =
                    typeof response.answer === "string"
                        ? response.answer.trim()
                        : "";

                aiMessage.innerHTML = renderMarkdown(
                    answer || "No answer was returned."
                );

                if (answer) {
                    conversationHistory.push(
                        {
                            role: "user",
                            content: question
                        },
                        {
                            role: "assistant",
                            content: answer
                        }
                    );

                    conversationHistory =
                        conversationHistory.slice(
                            -MAX_CONVERSATION_HISTORY
                        );
                }

                mergeSources(
                    response.sources || []
                );

                renderSources();

                if (chatBody) {
                    chatBody.scrollTop =
                        chatBody.scrollHeight;
                }
            }
        );
    } catch (error) {
        if (requestId === activeRequestId) {
            resetRequestState();

            aiMessage.textContent =
                "The extension could not send the request.";
        }

        console.error(
            "chrome.runtime.sendMessage failed:",
            error
        );
    }
}

// ============================================================
// 15. HANDLE YOUTUBE SPA NAVIGATION
// ============================================================

function handleVideoNavigation() {
    const currentVideoId = getVideoId();

    if (currentVideoId !== lastVideoId) {
        clearIndexPolling();

        lastVideoId = currentVideoId;

        // Invalidate responses from the previous video.
        activeRequestId += 1;

        resetRequestState();

        if (chatInput) {
            chatInput.value = "";
        }

        conversationHistory = [];

        resetSourceHistory();

        if (currentVideoId) {
            createLauncher();
            requestIndexPreparation(currentVideoId);

            if (chatBody) {
                chatBody.innerHTML = `
                    <div class="welcome-message">
                        <strong>
                            Ask anything about this video.
                        </strong>

                        <br>

                        <span>
                            Answers are generated from the video's transcript.
                        </span>
                    </div>
                `;
            }
        } else {
            closeChat();

            if (launcher) {
                launcher.hidden = true;
            }
        }
    }

    if (!currentVideoId) {
        closeChat();

        if (launcher) {
            launcher.hidden = true;
        }

        return;
    }

    if (launcher) {
        // Keep the launcher visible even while the chat is open so it remains
        // draggable and can always be used to close the panel.
        launcher.hidden = false;
        updateLauncherAccessibility();

        if (sidebar && !sidebar.hidden) {
            positionChatPanel();
        }
    }
}


// ============================================================
// 16. RECEIVE MESSAGES FROM SERVICE WORKER
// ============================================================

chrome.runtime.onMessage.addListener(
    (message) => {

        if (
            !message ||
            message.type !== "toggle_chat"
        ) {

            return;
        }


        toggleChat();
    }
);


// ============================================================
// 17. INITIALIZE
// ============================================================

function initialize() {

    if (!isVideoPage()) {
        return;
    }


    lastVideoId = getVideoId();

    createLauncher();

    // Prepare the transcript/index while keeping the chat closed.
    requestIndexPreparation(lastVideoId);
}


// ============================================================
// YOUTUBE SPA NAVIGATION EVENTS
// ============================================================

document.addEventListener(
    "yt-navigate-finish",
    () => {

        handleVideoNavigation();
    }
);


window.addEventListener(
    "popstate",
    () => {

        handleVideoNavigation();
    }
);


// ============================================================
// START
// ============================================================

initialize();