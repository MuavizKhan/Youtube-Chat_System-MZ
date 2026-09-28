"""
indexing.py

Offline indexing pipeline for YouTube videos.

Pipeline:

YouTube URL / Video ID
        ↓
Transcript
        ↓
LangChain Documents
        ↓
Timestamp-aware chunks
        ↓
Hugging Face embeddings
        ↓
FAISS vector store
        ↓
Local persistence
"""

import argparse
from typing import Optional

from youtube_transcript_api import (
    InvalidVideoId,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PREFERRED_LANGUAGES,
    VECTOR_STORE_ROOT,
)

from .retrieval import (
    create_embedding_model,
    extract_video_id,
)


# ============================================================
# 1. GET YOUTUBE TRANSCRIPT
# ============================================================

def get_transcript(
    video_id: str,
    languages: Optional[list[str]] = None,
):
    """
    Fetch a YouTube transcript.

    Parameters
    ----------
    video_id:
        YouTube video ID only.

    languages:
        Preferred transcript languages in priority order.
    """

    if not video_id or not video_id.strip():

        raise ValueError(
            "video_id cannot be empty."
        )


    video_id = video_id.strip()


    if languages is None:

        languages = PREFERRED_LANGUAGES


    api = YouTubeTranscriptApi()


    try:

        return api.fetch(
            video_id,
            languages=languages,
        )


    except TranscriptsDisabled as error:

        raise RuntimeError(
            f"Transcripts are disabled for video: {video_id}"
        ) from error


    except NoTranscriptFound as error:

        raise RuntimeError(
            f"No transcript found for video '{video_id}' "
            f"for languages: {languages}"
        ) from error


    except VideoUnavailable as error:

        raise RuntimeError(
            f"Video '{video_id}' is unavailable."
        ) from error


    except InvalidVideoId as error:

        raise RuntimeError(
            f"Invalid YouTube video ID: {video_id}"
        ) from error


# ============================================================
# 2. TRANSCRIPT → LANGCHAIN DOCUMENTS
# ============================================================

def transcript_to_documents(
    transcript,
) -> list[Document]:
    """
    Convert each transcript snippet into a LangChain
    Document while preserving timestamp metadata.
    """

    documents = []


    for snippet in transcript.snippets:

        text = snippet.text.strip()


        if not text:
            continue


        document = Document(

            page_content=text,

            metadata={

                "video_id":
                    transcript.video_id,

                "start":
                    float(
                        snippet.start
                    ),

                "duration":
                    float(
                        snippet.duration
                    ),

                "language":
                    transcript.language,

                "language_code":
                    transcript.language_code,

                "is_generated":
                    transcript.is_generated,

            },
        )


        documents.append(
            document
        )


    if not documents:

        raise ValueError(
            "Transcript was returned, but no usable "
            "transcript snippets were found."
        )


    return documents


# ============================================================
# 3. BUILD TRANSCRIPT TIMELINE
# ============================================================

def build_transcript_timeline(
    documents: list[Document],
):
    """
    Combine snippets into one continuous transcript and
    record character ranges for timestamp mapping.
    """

    if not documents:

        raise ValueError(
            "No documents supplied."
        )


    pieces = []

    snippet_spans = []

    current_position = 0


    for document in documents:

        text = document.page_content.strip()


        if not text:
            continue


        if pieces:

            pieces.append(" ")

            current_position += 1


        char_start = current_position


        pieces.append(
            text
        )


        current_position += len(
            text
        )


        char_end = current_position


        snippet_start = float(
            document.metadata["start"]
        )


        snippet_duration = float(
            document.metadata["duration"]
        )


        snippet_end = (
            snippet_start
            + snippet_duration
        )


        snippet_spans.append(
            {
                "char_start": char_start,
                "char_end": char_end,
                "start": snippet_start,
                "end": snippet_end,
            }
        )


    full_text = "".join(
        pieces
    )


    if not full_text:

        raise ValueError(
            "The transcript produced an empty text body."
        )


    return (
        full_text,
        snippet_spans,
    )


# ============================================================
# 4. MAP CHUNK RANGE TO TIMESTAMPS
# ============================================================

def get_timestamp_for_range(
    char_start: int,
    char_end: int,
    snippet_spans: list[dict],
):
    """
    Find transcript snippets overlapping a chunk range.

    Start = earliest overlapping snippet.
    End   = latest overlapping snippet.
    """

    overlapping_spans = []


    for span in snippet_spans:

        if (

            span["char_end"] > char_start

            and

            span["char_start"] < char_end

        ):

            overlapping_spans.append(
                span
            )


    if not overlapping_spans:

        nearest_span = min(

            snippet_spans,

            key=lambda span:
                abs(
                    span["char_start"]
                    - char_start
                ),
        )


        return (
            nearest_span["start"],
            nearest_span["end"],
        )


    start_time = min(
        span["start"]
        for span in overlapping_spans
    )


    end_time = max(
        span["end"]
        for span in overlapping_spans
    )


    return (
        start_time,
        end_time,
    )


# ============================================================
# 5. LOCATE CHUNK POSITIONS
# ============================================================

def locate_chunk_positions(
    full_text: str,
    chunks: list[str],
):
    """
    Locate each generated chunk in the original transcript.

    Returns:
        list[(start_character, end_character)]
    """

    positions = []

    previous_start = 0
    previous_chunk_length = 0


    for index, chunk in enumerate(
        chunks
    ):

        chunk_text = chunk.strip()


        if not chunk_text:

            positions.append(
                (
                    previous_start,
                    previous_start,
                )
            )

            continue


        if index == 0:

            search_start = 0

        else:

            estimated_start = (

                previous_start
                + previous_chunk_length
                - CHUNK_OVERLAP

            )


            search_start = max(

                previous_start,

                estimated_start - 100,
            )


        found_position = full_text.find(

            chunk_text,

            search_start,
        )


        if found_position == -1:

            found_position = full_text.find(

                chunk_text,

                previous_start,
            )


        if found_position == -1:

            raise RuntimeError(
                f"Could not locate chunk {index} "
                "inside the original transcript."
            )


        chunk_start = found_position

        chunk_end = (
            chunk_start
            + len(chunk_text)
        )


        positions.append(
            (
                chunk_start,
                chunk_end,
            )
        )


        previous_start = chunk_start

        previous_chunk_length = len(
            chunk_text
        )


    return positions


# ============================================================
# 6. CREATE TIMESTAMP-AWARE CHUNKS
# ============================================================

def split_into_timestamped_chunks(
    documents: list[Document],
) -> list[Document]:
    """
    Split the transcript into chunks while preserving
    approximate video timestamps.
    """

    full_text, snippet_spans = (
        build_transcript_timeline(
            documents
        )
    )


    splitter = (
        RecursiveCharacterTextSplitter(

            chunk_size=CHUNK_SIZE,

            chunk_overlap=CHUNK_OVERLAP,

            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                ", ",
                " ",
                "",
            ],

            length_function=len,

            is_separator_regex=False,
        )
    )


    raw_chunks = splitter.split_text(
        full_text
    )


    if not raw_chunks:

        raise ValueError(
            "Text splitter produced no chunks."
        )


    chunk_positions = (
        locate_chunk_positions(
            full_text,
            raw_chunks,
        )
    )


    first_document = documents[0]

    chunk_documents = []


    for index, (
        chunk_text,
        (char_start, char_end),
    ) in enumerate(

        zip(
            raw_chunks,
            chunk_positions,
        )
    ):

        start_time, end_time = (
            get_timestamp_for_range(

                char_start,
                char_end,
                snippet_spans,
            )
        )


        metadata = {

            "video_id":
                first_document.metadata[
                    "video_id"
                ],

            "language":
                first_document.metadata[
                    "language"
                ],

            "language_code":
                first_document.metadata[
                    "language_code"
                ],

            "is_generated":
                first_document.metadata[
                    "is_generated"
                ],

            "start":
                float(start_time),

            "end":
                float(end_time),

            "duration":
                max(
                    0.0,
                    float(end_time)
                    - float(start_time),
                ),

            "chunk_id":
                index,

            "chunk_length":
                len(chunk_text),

            "source":
                "youtube_transcript",
        }


        chunk_documents.append(

            Document(

                page_content=chunk_text,

                metadata=metadata,
            )
        )


    return chunk_documents


# ============================================================
# 7. CREATE FAISS VECTOR STORE
# ============================================================

def create_vector_store(
    chunks: list[Document],
):
    """
    Embed chunks and create the FAISS index.
    """

    if not chunks:

        raise ValueError(
            "Cannot create FAISS vector store because "
            "no chunks were produced."
        )


    embeddings = (
        create_embedding_model()
    )


    return FAISS.from_documents(

        chunks,

        embeddings,
    )


# ============================================================
# 8. SAVE VECTOR STORE
# ============================================================

def save_vector_store(
    vector_store,
    video_id: str,
):
    """
    Save one FAISS index per YouTube video.
    """

    VECTOR_STORE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    video_store_path = (
        VECTOR_STORE_ROOT
        / video_id
    )


    vector_store.save_local(

        str(video_store_path)
    )


    return video_store_path


# ============================================================
# 9. INSPECT CHUNKS
# ============================================================

def inspect_chunks(
    chunks: list[Document],
    number_of_chunks: int = 3,
):
    """
    Print a small sample during indexing.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "CHUNK INSPECTION"
    )

    print(
        "=" * 70
    )


    for chunk in chunks[
        :number_of_chunks
    ]:

        metadata = chunk.metadata


        print(
            f"\nChunk ID: "
            f"{metadata.get('chunk_id')}"
        )


        print(
            f"Timestamp: "
            f"{metadata.get('start', 0):.2f}s"
            f" → "
            f"{metadata.get('end', 0):.2f}s"
        )


        print(
            f"Length: "
            f"{metadata.get('chunk_length')} characters"
        )


        print(
            f"Text:\n"
            f"{chunk.page_content}"
        )


# ============================================================
# 10. COMPLETE INDEXING PIPELINE
# ============================================================

def index_video(
    video_reference: str,
    languages: Optional[list[str]] = None,
    save_index: bool = True,
):
    """
    Complete indexing pipeline.
    """

    video_id = extract_video_id(
        video_reference
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "YOUTUBE RAG INDEXING"
    )

    print(
        "=" * 70
    )

    print(
        f"\nVideo ID: {video_id}"
    )


    # --------------------------------------------------------
    # 1. Transcript
    # --------------------------------------------------------

    print(
        "\n[1/6] Fetching transcript..."
    )


    transcript = get_transcript(

        video_id,

        languages=languages,
    )


    print(
        f"Language: "
        f"{transcript.language}"
    )


    print(
        f"Language code: "
        f"{transcript.language_code}"
    )


    print(
        f"Generated: "
        f"{transcript.is_generated}"
    )


    print(
        f"Transcript snippets: "
        f"{len(transcript.snippets)}"
    )


    # --------------------------------------------------------
    # 2. Documents
    # --------------------------------------------------------

    print(
        "\n[2/6] Creating LangChain Documents..."
    )


    documents = transcript_to_documents(
        transcript
    )


    print(
        f"Documents created: "
        f"{len(documents)}"
    )


    # --------------------------------------------------------
    # 3. Chunks
    # --------------------------------------------------------

    print(
        "\n[3/6] Creating timestamp-aware chunks..."
    )


    chunks = split_into_timestamped_chunks(
        documents
    )


    print(
        f"Chunks created: "
        f"{len(chunks)}"
    )


    print(
        f"Chunk size: "
        f"{CHUNK_SIZE}"
    )


    print(
        f"Chunk overlap: "
        f"{CHUNK_OVERLAP}"
    )


    inspect_chunks(
        chunks
    )


    # --------------------------------------------------------
    # 4. Embeddings
    # --------------------------------------------------------

    print(
        "\n[4/6] Creating/loading embeddings..."
    )


    create_embedding_model()


    # --------------------------------------------------------
    # 5. FAISS
    # --------------------------------------------------------

    print(
        "\n[5/6] Creating FAISS vector store..."
    )


    vector_store = create_vector_store(
        chunks
    )


    print(
        "FAISS vector store created."
    )


    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    if save_index:

        print(
            "\n[6/6] Saving vector store..."
        )


        saved_path = save_vector_store(

            vector_store,

            video_id,
        )


        print(
            f"Saved to:\n{saved_path}"
        )

    else:

        print(
            "\n[6/6] Save skipped."
        )


    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "INDEXING COMPLETED"
    )

    print(
        "=" * 70
    )


    print(
        f"\nVideo ID: {video_id}"
    )


    print(
        f"Transcript snippets: "
        f"{len(documents)}"
    )


    print(
        f"Chunks: "
        f"{len(chunks)}"
    )


    print(
        f"Embedding model: "
        f"{create_embedding_model().__class__.__name__}"
    )


    print(
        "Vector store: FAISS"
    )


    return vector_store


# ============================================================
# 11. COMMAND-LINE ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a local FAISS index for a YouTube video."
        )
    )


    parser.add_argument(
        "video",
        help=(
            "YouTube video ID or YouTube URL."
        ),
    )


    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        default=None,
        help=(
            "Preferred transcript language. "
            "May be supplied more than once."
        ),
    )


    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Build the index without saving it.",
    )


    args = parser.parse_args()


    index_video(

        video_reference=args.video,

        languages=(
            args.languages
            if args.languages
            else PREFERRED_LANGUAGES
        ),

        save_index=not args.no_save,
    )


if __name__ == "__main__":

    main()