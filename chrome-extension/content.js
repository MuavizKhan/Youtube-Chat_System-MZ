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

let lastVideoId = null;

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
// 4. CREATE LAUNCHER
// ============================================================

function createLauncher() {

    if (launcher) {
        return;
    }


    launcher =
        document.createElement("button");


    launcher.id =
        "youtube-ai-chat-launcher";


    launcher.type =
        "button";


    launcher.textContent =
        "🤖";


    launcher.title =
        "Open YouTube AI Chat";


    launcher.setAttribute(
        "aria-label",
        "Open YouTube AI Chat"
    );


    launcher.addEventListener(
        "click",
        openChat
    );


    document.body.appendChild(
        launcher
    );
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
                <span>🤖</span>
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

function openChat() {

    if (!isVideoPage()) {

        showTemporaryLauncherMessage(
            "Open a YouTube video first."
        );

        return;
    }


    createSidebar();


    sidebar.hidden = false;

    launcher.hidden = true;


    // Capture the current video ID.
    lastVideoId =
        getVideoId();


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

        launcher.hidden = false;
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

function showTemporaryLauncherMessage(
    message
) {

    if (!launcher) {
        return;
    }


    const originalText =
        launcher.textContent;


    launcher.textContent =
        "⚠";


    launcher.title =
        message;


    setTimeout(
        () => {

            if (launcher) {

                launcher.textContent =
                    originalText;

                launcher.title =
                    "Open YouTube AI Chat";
            }

        },
        1800
    );
}


// ============================================================
// 10. APPEND MESSAGE
// ============================================================

function appendMessage(text, type) {

    const message = document.createElement("div");

    message.className =
        `youtube-ai-message ${type}`;

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


// ============================================================
// 12. RENDER SOURCES
// ============================================================

function renderSources(
    sources
) {

    clearSources();


    if (
        !Array.isArray(sources) ||
        sources.length === 0
    ) {

        return;
    }


    const title =
        document.createElement(
            "div"
        );


    title.className =
        "sources-title";


    title.textContent =
        "Video sources";


    sourcesContainer.appendChild(
        title
    );


    const list =
        document.createElement(
            "div"
        );


    list.className =
        "sources-list";


    sources.forEach(
        (source) => {

            const sourceButton =
                document.createElement(
                    "button"
                );


            sourceButton.type =
                "button";


            sourceButton.className =
                "source-button";


            const start =
                Number(
                    source.start
                ) || 0;


            const end =
                Number(
                    source.end
                ) || start;


            sourceButton.textContent =
                `${formatTimestamp(start)} → ` +
                `${formatTimestamp(end)}`;


            sourceButton.title =
                "Jump to this part of the video";


            sourceButton.addEventListener(
                "click",
                () => {

                    seekToTimestamp(
                        start
                    );
                }
            );


            list.appendChild(
                sourceButton
            );
        }
    );


    sourcesContainer.appendChild(
        list
    );


    sourcesContainer.hidden =
        false;
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
    let html = escapeHtml(text);

    // Convert escaped markdown characters back first.
    html = html.replace(/\\([*_`])/g, "$1");

    // Remove empty Markdown bullet lines.
    html = html.replace(/^\s*[-*]\s*$/gm, "");

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
        /^\s*[-*]\s+(.+)$/gm,
        "<li>$1</li>"
    );

    // Group consecutive list items.
    html = html.replace(
        /(<li>.*?<\/li>(?:\s*<li>.*?<\/li>)*)/gs,
        "<ul>$1</ul>"
    );

    // Markdown line breaks.
    html = html.replace(
        /\n/g,
        "<br>"
    );

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
        clearSources();

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
    clearSources();

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
                question
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

                renderSources(
                    response.sources || []
                );

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
        lastVideoId = currentVideoId;

        // Invalidate responses from the previous video.
        activeRequestId += 1;

        resetRequestState();

        if (chatInput) {
            chatInput.value = "";
        }

        clearSources();

        if (currentVideoId) {
            createLauncher();

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
        launcher.hidden =
            sidebar
                ? !sidebar.hidden
                : false;
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


    lastVideoId =
        getVideoId();


    createLauncher();


    // Chat intentionally remains closed.
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