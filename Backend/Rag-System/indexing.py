"""
indexing.py

YouTube RAG - Indexing Pipeline

Pipeline:

YouTube Video ID
        ↓
Transcript
        ↓
Timestamp-aware transcript representation
        ↓
LangChain Documents
        ↓
Text Chunking
        ↓
Hugging Face Embeddings
        ↓
FAISS Vector Store

This file handles ONLY indexing.

It does NOT handle:
- User questions
- Retrieval
- Prompt construction
- LLM generation
- FastAPI
- Chrome extension
"""

from pathlib import Path
from typing import Optional

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    InvalidVideoId,
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION
# ============================================================

# Preferred transcript languages.
#
# For our first RAG version, English is the baseline.
#
# Later we can make language selection dynamic.
PREFERRED_LANGUAGES = ["en"]


# Initial chunking baseline.
#
# These values are experimental starting points.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# Free/local Hugging Face embedding model.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Folder where each video's FAISS index will be stored.
VECTOR_STORE_ROOT = (
    Path(__file__).resolve().parent / "vector_stores"
)


# ============================================================
# 1. GET YOUTUBE TRANSCRIPT
# ============================================================

def get_transcript(
    video_id: str,
    languages: Optional[list[str]] = None
):
    """
    Fetch a YouTube transcript.

    Parameters
    ----------
    video_id:
        YouTube video ID only.

    languages:
        Preferred transcript languages in priority order.

    Returns
    -------
    FetchedTranscript
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

        transcript = api.fetch(
            video_id,
            languages=languages
        )

        return transcript

    except TranscriptsDisabled:

        raise RuntimeError(
            f"Transcripts are disabled for video: {video_id}"
        )

    except NoTranscriptFound:

        raise RuntimeError(
            f"No transcript found for video '{video_id}' "
            f"for languages: {languages}"
        )

    except VideoUnavailable:

        raise RuntimeError(
            f"Video '{video_id}' is unavailable."
        )

    except InvalidVideoId:

        raise RuntimeError(
            f"Invalid YouTube video ID: {video_id}"
        )


# ============================================================
# 2. TRANSCRIPT → LANGCHAIN DOCUMENTS
# ============================================================

def transcript_to_documents(
    transcript
) -> list[Document]:
    """
    Convert each YouTube transcript snippet into
    a LangChain Document.

    IMPORTANT:
    We preserve the original timestamp information.
    """

    documents = []

    for snippet in transcript.snippets:

        text = snippet.text.strip()

        # Ignore empty transcript snippets.
        if not text:
            continue

        document = Document(

            page_content=text,

            metadata={
                "video_id": transcript.video_id,
                "start": float(snippet.start),
                "duration": float(snippet.duration),
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
            }
        )

        documents.append(document)

    if not documents:

        raise ValueError(
            "Transcript was returned, but no usable "
            "transcript snippets were found."
        )

    return documents


# ============================================================
# 3. BUILD CONTINUOUS TRANSCRIPT + CHARACTER TIMELINE
# ============================================================

def build_transcript_timeline(
    documents: list[Document]
):
    """
    Combine transcript snippets into one continuous text
    while keeping track of which character positions belong
    to which transcript snippet.

    Example:

        full_text
        0 -------------------------------------> N

        snippet 1: chars 0-40
        snippet 2: chars 41-90
        snippet 3: chars 91-130

    This allows us to map RAG chunks back to timestamps.
    """

    if not documents:
        raise ValueError(
            "No documents supplied."
        )

    pieces = []
    snippet_spans = []

    current_position = 0

    for index, document in enumerate(documents):

        text = document.page_content.strip()

        if not text:
            continue

        # Add a space between transcript snippets.
        if pieces:

            pieces.append(" ")

            current_position += 1

        char_start = current_position

        pieces.append(text)

        current_position += len(text)

        char_end = current_position

        snippet_start = float(
            document.metadata["start"]
        )

        snippet_duration = float(
            document.metadata["duration"]
        )

        snippet_end = (
            snippet_start + snippet_duration
        )

        snippet_spans.append(
            {
                "char_start": char_start,
                "char_end": char_end,
                "start": snippet_start,
                "end": snippet_end,
            }
        )

    full_text = "".join(pieces)

    if not full_text:

        raise ValueError(
            "The transcript produced an empty text body."
        )

    return full_text, snippet_spans


# ============================================================
# 4. MAP CHUNK CHARACTER RANGE → VIDEO TIMESTAMP
# ============================================================

def get_timestamp_for_range(
    char_start: int,
    char_end: int,
    snippet_spans: list[dict]
):
    """
    Find the transcript snippets that overlap with a
    particular character range.

    The chunk's:
        start = earliest overlapping snippet start

        end   = latest overlapping snippet end

    Because YouTube transcript snippets can overlap in time,
    we use min(start) and max(end).
    """

    overlapping_spans = []

    for span in snippet_spans:

        if (
            span["char_end"] > char_start
            and span["char_start"] < char_end
        ):

            overlapping_spans.append(span)

    # Defensive fallback.
    if not overlapping_spans:

        nearest_span = min(
            snippet_spans,
            key=lambda span: abs(
                span["char_start"] - char_start
            )
        )

        return (
            nearest_span["start"],
            nearest_span["end"]
        )

    start_time = min(
        span["start"]
        for span in overlapping_spans
    )

    end_time = max(
        span["end"]
        for span in overlapping_spans
    )

    return start_time, end_time


# ============================================================
# 5. FIND CHUNK POSITION IN ORIGINAL TRANSCRIPT
# ============================================================

def locate_chunk_positions(
    full_text: str,
    chunks: list[str]
):
    """
    Locate each generated chunk inside the original
    transcript text.

    RecursiveCharacterTextSplitter returns chunk text but
    not character offsets, so we recover the positions
    from the original continuous transcript.

    Returns:
        list of (start, end)
    """

    positions = []

    previous_start = 0
    previous_chunk_length = 0

    for index, chunk in enumerate(chunks):

        chunk_text = chunk.strip()

        if not chunk_text:

            positions.append(
                (previous_start, previous_start)
            )

            continue

        # The next chunk normally begins approximately:
        #
        # previous_start
        #     + previous_chunk_length
        #     - CHUNK_OVERLAP
        #
        # We give the search a buffer because RecursiveCharacter
        # TextSplitter may move boundaries to a separator.

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
                estimated_start - 100
            )

        found_position = full_text.find(
            chunk_text,
            search_start
        )

        # Defensive fallback if the expected location
        # wasn't found.
        if found_position == -1:

            found_position = full_text.find(
                chunk_text,
                previous_start
            )

        if found_position == -1:

            raise RuntimeError(
                f"Could not locate chunk {index} "
                f"inside the original transcript."
            )

        chunk_start = found_position

        chunk_end = (
            chunk_start + len(chunk_text)
        )

        positions.append(
            (
                chunk_start,
                chunk_end
            )
        )

        previous_start = chunk_start
        previous_chunk_length = len(chunk_text)

    return positions


# ============================================================
# 6. CREATE TIMESTAMP-AWARE CHUNKS
# ============================================================

def split_into_timestamped_chunks(
    documents: list[Document]
) -> list[Document]:
    """
    Create RAG chunks while preserving approximate
    YouTube timestamps.

    Chunking baseline:
        chunk_size = 1000
        chunk_overlap = 200
    """

    # --------------------------------------------------------
    # Build continuous transcript
    # --------------------------------------------------------

    full_text, snippet_spans = (
        build_transcript_timeline(
            documents
        )
    )


    # --------------------------------------------------------
    # Create text splitter
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(

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
            ""
        ],

        length_function=len,

        is_separator_regex=False
    )


    # --------------------------------------------------------
    # Generate raw chunk text
    # --------------------------------------------------------

    raw_chunks = splitter.split_text(
        full_text
    )

    if not raw_chunks:

        raise ValueError(
            "Text splitter produced no chunks."
        )


    # --------------------------------------------------------
    # Find character ranges
    # --------------------------------------------------------

    chunk_positions = locate_chunk_positions(
        full_text,
        raw_chunks
    )


    # --------------------------------------------------------
    # Convert chunks into Documents
    # --------------------------------------------------------

    first_document = documents[0]

    chunk_documents = []

    for index, (
        chunk_text,
        (char_start, char_end)
    ) in enumerate(
        zip(
            raw_chunks,
            chunk_positions
        )
    ):

        start_time, end_time = (
            get_timestamp_for_range(
                char_start,
                char_end,
                snippet_spans
            )
        )

        metadata = {

            "video_id": first_document.metadata[
                "video_id"
            ],

            "language": first_document.metadata[
                "language"
            ],

            "language_code": first_document.metadata[
                "language_code"
            ],

            "is_generated": first_document.metadata[
                "is_generated"
            ],

            # Timestamp-aware metadata
            "start": start_time,

            "end": end_time,

            "duration": max(
                0.0,
                end_time - start_time
            ),

            # Useful for debugging/evaluation
            "chunk_id": index,

            "chunk_length": len(chunk_text),

            "source": "youtube_transcript"
        }


        chunk_document = Document(

            page_content=chunk_text,

            metadata=metadata
        )


        chunk_documents.append(
            chunk_document
        )


    return chunk_documents


# ============================================================
# 7. CREATE HUGGING FACE EMBEDDING MODEL
# ============================================================

def create_embedding_model():
    """
    Load the Hugging Face embedding model locally.
    """

    embeddings = HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


# ============================================================
# 8. CREATE FAISS VECTOR STORE
# ============================================================

def create_vector_store(
    chunks: list[Document],
    embeddings
):
    """
    Convert chunks into embeddings and store them in FAISS.
    """

    if not chunks:

        raise ValueError(
            "Cannot create FAISS vector store because "
            "no chunks were produced."
        )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# ============================================================
# 9. SAVE FAISS VECTOR STORE
# ============================================================

def save_vector_store(
    vector_store,
    video_id: str
):
    """
    Save one FAISS index per YouTube video.
    """

    VECTOR_STORE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    video_store_path = (
        VECTOR_STORE_ROOT / video_id
    )

    vector_store.save_local(
        str(video_store_path)
    )

    return video_store_path


# ============================================================
# 10. INSPECT CHUNKS
# ============================================================

def inspect_chunks(
    chunks: list[Document],
    number_of_chunks: int = 5
):
    """
    Print a few chunks so that we can manually inspect
    chunk quality and timestamp metadata.
    """

    print("\n" + "=" * 60)
    print("CHUNK INSPECTION")
    print("=" * 60)

    for chunk in chunks[:number_of_chunks]:

        print(
            f"\n--- Chunk {chunk.metadata['chunk_id']} ---"
        )

        print(
            f"Length: "
            f"{chunk.metadata['chunk_length']}"
            f" characters"
        )

        print(
            f"Timestamp: "
            f"{chunk.metadata['start']:.2f}s"
            f" → "
            f"{chunk.metadata['end']:.2f}s"
        )

        print(
            f"Duration: "
            f"{chunk.metadata['duration']:.2f}s"
        )

        print(
            f"Metadata: {chunk.metadata}"
        )

        print(
            f"Text:\n{chunk.page_content}"
        )


# ============================================================
# 11. COMPLETE INDEXING PIPELINE
# ============================================================

def index_video(
    video_id: str,
    languages: Optional[list[str]] = None,
    save_index: bool = True
):
    """
    Complete YouTube indexing pipeline.

    Pipeline:

        Video ID
            ↓
        Transcript
            ↓
        LangChain Documents
            ↓
        Timestamp-aware chunks
            ↓
        Hugging Face embeddings
            ↓
        FAISS
            ↓
        Save locally
    """

    print("\n" + "=" * 60)
    print("STARTING YOUTUBE RAG INDEXING")
    print("=" * 60)

    print(
        f"\nVideo ID: {video_id}"
    )


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print(
        "\n[1/6] Fetching transcript..."
    )

    transcript = get_transcript(
        video_id,
        languages
    )

    print(
        f"Language: {transcript.language}"
    )

    print(
        f"Language code: "
        f"{transcript.language_code}"
    )

    print(
        f"Generated transcript: "
        f"{transcript.is_generated}"
    )

    print(
        f"Transcript snippets: "
        f"{len(transcript)}"
    )


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print(
        "\n[2/6] Converting transcript "
        "snippets to LangChain Documents..."
    )

    documents = transcript_to_documents(
        transcript
    )

    print(
        f"Documents created: "
        f"{len(documents)}"
    )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print(
        "\n[3/6] Creating timestamp-aware "
        "transcript chunks..."
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

    # Inspect first few chunks
    inspect_chunks(
        chunks,
        number_of_chunks=5
    )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print(
        "\n[4/6] Creating Hugging Face embeddings..."
    )

    embeddings = create_embedding_model()

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )


    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print(
        "\n[5/6] Creating FAISS vector store..."
    )

    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    print(
        "FAISS vector store created successfully."
    )


    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    if save_index:

        print(
            "\n[6/6] Saving FAISS vector store..."
        )

        saved_path = save_vector_store(
            vector_store,
            video_id
        )

        print(
            f"Vector store saved to:\n"
            f"{saved_path}"
        )

    else:

        print(
            "\n[6/6] Vector store saving skipped."
        )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INDEXING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nVideo ID: {video_id}"
    )

    print(
        f"Transcript snippets: "
        f"{len(documents)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        "Vector store: FAISS"
    )

    return vector_store


# ============================================================
# 12. RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    # Development test video.
    #
    # Later FastAPI will provide this dynamically.
    video_id = "Gfr50f6ZBvo"

    index_video(
        video_id=video_id,
        languages=["en"],
        save_index=True
    )