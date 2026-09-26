console.log("YouTube AI Chat loaded");

function getVideoId() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get("v");
}

function createChatSidebar() {
    if (document.getElementById("youtube-ai-chat")) {
        return;
    }

    const sidebar = document.createElement("div");

    sidebar.id = "youtube-ai-chat";

    sidebar.innerHTML = `
        <div class="chat-header">
            🤖 YouTube AI
        </div>

        <div class="chat-body" id="chat-body">
            <p>Ask anything about this video.</p>
        </div>

        <div class="chat-input-area">
            <input
                type="text"
                id="chat-input"
                placeholder="Ask a question..."
            />
            <button id="chat-send">Send</button>
        </div>
    `;

    document.body.appendChild(sidebar);

    setupChat();
}

function setupChat() {

    const input = document.getElementById("chat-input");
    const sendButton = document.getElementById("chat-send");
    const chatBody = document.getElementById("chat-body");

    function sendMessage() {

        const question = input.value.trim();

        if (question === "") {
            return;
        }

        // Display user's message
        const userMessage = document.createElement("div");

        userMessage.className = "user-message";
        userMessage.textContent = question;

        chatBody.appendChild(userMessage);

        // Clear input
        input.value = "";

        // Temporary AI response
        const aiMessage = document.createElement("div");

        aiMessage.className = "ai-message";
        aiMessage.textContent =
            "This is a temporary AI response. The backend will be connected later.";

        chatBody.appendChild(aiMessage);

        // Automatically scroll to latest message
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    sendButton.addEventListener("click", sendMessage);

    input.addEventListener("keydown", function(event) {

        if (event.key === "Enter") {
            sendMessage();
        }

    });
}

console.log("Current video ID:", getVideoId());

createChatSidebar();