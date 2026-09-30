/**
 * background.js
 *
 * Manifest V3 service worker.
 *
 * Responsibilities:
 * 1. Toggle the YouTube AI chat panel when the extension icon
 *    is clicked.
 * 2. Forward chat requests from content.js to FastAPI.
 * 3. Return the complete RAG response, including sources.
 */


// ============================================================
// 1. EXTENSION ACTION → TOGGLE CHAT
// ============================================================

chrome.action.onClicked.addListener(
    async (tab) => {

        if (!tab.id) {
            return;
        }


        // Only operate on YouTube pages.
        if (
            !tab.url ||
            !tab.url.startsWith(
                "https://www.youtube.com/"
            )
        ) {

            return;
        }


        try {

            await chrome.tabs.sendMessage(
                tab.id,
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

        if (
            !message ||
            message.type !== "chat"
        ) {

            return;
        }


        const videoId = (
            message.video_id || ""
        ).trim();


        const question = (
            message.question || ""
        ).trim();


        // ----------------------------------------------------
        // Validate extension request
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
        // Call FastAPI
        // ----------------------------------------------------

        fetch(
            "http://127.0.0.1:8000/chat",
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
                })
            }
        )

        .then(
            async (response) => {

                // ------------------------------------------------
                // Backend returned an HTTP error
                // ------------------------------------------------

                if (!response.ok) {

                    let errorMessage =
                        `Backend returned status ${response.status}.`;


                    try {

                        const errorData =
                            await response.json();


                        if (
                            errorData &&
                            errorData.detail
                        ) {

                            errorMessage =
                                errorData.detail;
                        }

                    } catch {

                        // Keep the generic HTTP error.
                    }


                    throw new Error(
                        errorMessage
                    );
                }


                return response.json();
            }
        )

        .then(
            (data) => {

                sendResponse({

                    success: true,

                    answer:
                        data.answer || "",

                    video_id:
                        data.video_id || videoId,

                    sources:
                        Array.isArray(data.sources)
                            ? data.sources
                            : [],

                    retrieved_chunks:
                        data.retrieved_chunks || 0
                });
            }
        )

        .catch(
            (error) => {

                console.error(
                    "Backend error:",
                    error
                );


                sendResponse({

                    success: false,

                    error:
                        error.message ||
                        "Unable to contact the backend."
                });
            }
        );


        // Keep the message channel open while fetch runs.
        return true;
    }
);