chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

    if (message.type === "chat") {

        fetch("http://127.0.0.1:8000/chat", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                video_id: message.video_id,
                question: message.question
            })
        })
        .then(response => {

            if (!response.ok) {
                throw new Error(
                    `Backend returned status ${response.status}`
                );
            }

            return response.json();
        })
        .then(data => {

            sendResponse({
                success: true,
                answer: data.answer
            });
        })
        .catch(error => {

            console.error("Backend error:", error);

            sendResponse({
                success: false,
                error: error.message
            });
        });

        return true;
    }
});