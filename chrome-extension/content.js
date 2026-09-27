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

            <button id="chat-send">
                Send
            </button>

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

        const videoId = getVideoId();


        // -----------------------------------
        // Display user's message
        // -----------------------------------

        const userMessage = document.createElement("div");

        userMessage.className = "user-message";

        userMessage.textContent = question;

        chatBody.appendChild(userMessage);


        // Clear input

        input.value = "";


        // -----------------------------------
        // Display loading message
        // -----------------------------------

        const aiMessage = document.createElement("div");

        aiMessage.className = "ai-message";

        aiMessage.textContent = "Thinking...";

        chatBody.appendChild(aiMessage);


        // -----------------------------------
        // Send message to background script
        // -----------------------------------

        chrome.runtime.sendMessage(
            {
                type: "chat",
                video_id: videoId,
                question: question
            },

            function(response) {

                if (chrome.runtime.lastError) {

                    console.error(
                        "Extension error:",
                        chrome.runtime.lastError.message
                    );

                    aiMessage.textContent =
                        "Unable to communicate with the extension.";

                    return;
                }


                if (!response || !response.success) {

                    aiMessage.textContent =
                        "Sorry, I couldn't connect to the backend.";

                    console.error(
                        "Backend error:",
                        response?.error
                    );

                    return;
                }


                // -----------------------------------
                // Display backend response
                // -----------------------------------

                aiMessage.textContent = response.answer;

                chatBody.scrollTop =
                    chatBody.scrollHeight;
            }
        );
    }


    // -----------------------------------
    // Send button
    // -----------------------------------

    sendButton.addEventListener(
        "click",
        sendMessage
    );


    // -----------------------------------
    // Enter key
    // -----------------------------------

    input.addEventListener(
        "keydown",
        function(event) {

            if (event.key === "Enter") {

                sendMessage();

            }

        }
    );
}


console.log(
    "Current video ID:",
    getVideoId()
);


createChatSidebar();