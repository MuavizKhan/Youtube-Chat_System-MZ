from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document


video_id = "Gfr50f6ZBvo"

api = YouTubeTranscriptApi()

transcript = api.fetch(video_id)


documents = []

for snippet in transcript.snippets:

    document = Document(
        page_content=snippet.text,

        metadata={
            "video_id": transcript.video_id,
            "start": snippet.start,
            "duration": snippet.duration,
            "language": transcript.language,
            "language_code": transcript.language_code,
            "is_generated": transcript.is_generated
        }
    )

    documents.append(document)


print("Number of documents:", len(documents))

print("\nFirst document:")
print(documents[0])

print("\nSecond document:")
print(documents[1])