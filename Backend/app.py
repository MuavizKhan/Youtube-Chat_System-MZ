from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()


# Allow the Chrome extension / YouTube page
# to communicate with our FastAPI backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.youtube.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    video_id: str
    question: str


@app.post("/chat")
def chat(request: ChatRequest):

    return {
        "answer": f"Test response for: {request.question}",
        "video_id": request.video_id
    }