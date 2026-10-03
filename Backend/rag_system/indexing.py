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

import shutil
import uuid

import threading

import argparse
import json
from typing import Optional

from youtube_transcript_api import (
    InvalidVideoId,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from enum import Enum

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PREFERRED_LANGUAGES,
    VECTOR_STORE_ROOT,
)

from .retrieval import (
    create_embedding_model,
    extract_video_id,
    load_vector_store,
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


class IndexAction(str, Enum):
    CREATE = "create"
    REUSE = "reuse"
    REBUILD = "rebuild"


# ============================================================
# 8. BUILD INDEX METADATA
# ============================================================

class IndexState(str, Enum):
    """
    Lifecycle state of a persisted video vector store.
    """

    MISSING = "missing"
    VALID = "valid"
    INVALID = "invalid"
    STALE = "stale"

INDEX_METADATA_VERSION = 1


def get_index_action(state: IndexState) -> IndexAction:
    """
    Determine the lifecycle action for an index state.

    This function only decides what should happen.
    It does not create, reuse, delete, or rebuild an index.
    """

    if state is IndexState.MISSING:
        return IndexAction.CREATE

    if state is IndexState.VALID:
        return IndexAction.REUSE

    if state in (IndexState.INVALID, IndexState.STALE):
        return IndexAction.REBUILD

    raise ValueError(f"Unsupported index state: {state}")


def build_index_metadata(
    chunks: list[Document],
) -> dict:
    """
    Build metadata describing the configuration and source
    information used to create a FAISS index.

    Metadata is derived from the actual indexed chunks so that
    persisted index information cannot silently disagree with
    the index contents.
    """

    if not chunks:
        raise ValueError(
            "Cannot build index metadata because no chunks were produced."
        )

    first_chunk = chunks[0]
    metadata = first_chunk.metadata

    required_metadata = {
        "video_id",
        "language",
        "language_code",
        "is_generated",
    }

    missing_metadata = required_metadata - metadata.keys()

    if missing_metadata:
        missing = ", ".join(sorted(missing_metadata))

        raise ValueError(
            "Cannot build index metadata because required chunk "
            f"metadata is missing: {missing}"
        )

    video_ids = {
        chunk.metadata.get("video_id")
        for chunk in chunks
    }

    if len(video_ids) != 1:
        raise ValueError(
            "Cannot build index metadata because chunks contain "
            "multiple video IDs."
        )

    return {
        "metadata_version": INDEX_METADATA_VERSION,
        "video_id": metadata["video_id"],
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_count": len(chunks),
        "transcript_language": metadata["language"],
        "transcript_language_code": metadata["language_code"],
        "transcript_generated": bool(
            metadata["is_generated"]
        ),
    }

# private validation helper
def _validate_vector_store_contents(
    video_id: str,
    video_store_path,
) -> tuple[bool, bool]:
    """
    Validate the persisted files and metadata for one video's
    vector store.

    Returns:
        (is_valid, is_stale)

    is_valid:
        True when the persisted structure and metadata are valid.

    is_stale:
        True when the persisted index is valid but was created
        using a different indexing configuration.
    """

    required_files = (
        "index.faiss",
        "index.pkl",
        "metadata.json",
    )

    for filename in required_files:
        file_path = video_store_path / filename

        if not file_path.is_file():
            return False, False

    metadata_path = video_store_path / "metadata.json"

    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as metadata_file:
            metadata = json.load(metadata_file)

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return False, False

    if not isinstance(metadata, dict):
        return False, False

    required_metadata = {
        "metadata_version",
        "video_id",
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
        "chunk_count",
        "transcript_language",
        "transcript_language_code",
        "transcript_generated",
    }

    if not required_metadata.issubset(metadata.keys()):
        return False, False

    if type(metadata["metadata_version"]) is not int:
        return False, False

    if type(metadata["chunk_size"]) is not int:
        return False, False

    if type(metadata["chunk_overlap"]) is not int:
        return False, False

    if type(metadata["chunk_count"]) is not int:
        return False, False

    if type(metadata["transcript_generated"]) is not bool:
        return False, False

    if metadata["metadata_version"] != INDEX_METADATA_VERSION:
        return False, False

    if metadata["video_id"] != video_id:
        return False, False

    if metadata["chunk_count"] <= 0:
        return False, False

    if (
        not isinstance(
            metadata["transcript_language"],
            str,
        )
        or not metadata["transcript_language"].strip()
    ):
        return False, False

    if (
        not isinstance(
            metadata["transcript_language_code"],
            str,
        )
        or not metadata["transcript_language_code"].strip()
    ):
        return False, False

    is_stale = (
        metadata["embedding_model"] != EMBEDDING_MODEL
        or metadata["chunk_size"] != CHUNK_SIZE
        or metadata["chunk_overlap"] != CHUNK_OVERLAP
    )

    return True, is_stale


# state resolver
def get_index_state(
    video_id: str,
) -> IndexState:
    """
    Determine the lifecycle state of one video's persisted
    vector store.
    """

    if not video_id or not video_id.strip():
        return IndexState.MISSING

    video_store_path = (
        VECTOR_STORE_ROOT
        / video_id
    )

    if not video_store_path.is_dir():
        return IndexState.MISSING

    is_valid, is_stale = (
        _validate_vector_store_contents(
            video_id,
            video_store_path,
        )
    )

    if not is_valid:
        return IndexState.INVALID

    if is_stale:
        return IndexState.STALE

    return IndexState.VALID


# VALIDATE VECTOR STORE
def validate_vector_store(
    video_id: str,
) -> bool:
    """
    Validate the persisted structure and metadata of one video's
    FAISS vector store.

    Configuration drift is considered invalid by this function.
    """

    if not video_id or not video_id.strip():
        return False

    video_store_path = (
        VECTOR_STORE_ROOT
        / video_id
    )

    if not video_store_path.is_dir():
        return False

    is_valid, is_stale = (
        _validate_vector_store_contents(
            video_id,
            video_store_path,
        )
    )

    if not is_valid:
        return False

    if is_stale:
        return False

    return True


# ============================================================
# 9. SAVE VECTOR STORE
# ============================================================

def save_vector_store(
    vector_store,
    video_id: str,
    chunks: list[Document],
):
    """
    Safely persist one FAISS index per YouTube video.

    The index is first written to a staging directory. Only after
    the staged index and metadata pass validation is the staged
    directory published as the final index.

    An existing index is preserved if staging or validation fails.
    """

    if not video_id or not video_id.strip():
        raise ValueError(
            "video_id cannot be empty."
        )

    if not chunks:
        raise ValueError(
            "Cannot save vector store because no chunks were supplied."
        )

    video_id = video_id.strip()

    VECTOR_STORE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_store_path = (
        VECTOR_STORE_ROOT
        / video_id
    )

    staging_path = (
        VECTOR_STORE_ROOT
        / f".{video_id}.staging-{uuid.uuid4().hex}"
    )

    backup_path = (
        VECTOR_STORE_ROOT
        / f".{video_id}.backup-{uuid.uuid4().hex}"
    )

    try:
        # ----------------------------------------------------
        # 1. Build the complete index in isolation
        # ----------------------------------------------------

        staging_path.mkdir(
            parents=True,
            exist_ok=False,
        )

        vector_store.save_local(
            str(staging_path)
        )

        # ----------------------------------------------------
        # 2. Write metadata into the staging directory
        # ----------------------------------------------------

        metadata = build_index_metadata(
            chunks
        )

        metadata_path = (
            staging_path
            / "metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as metadata_file:

            json.dump(
                metadata,
                metadata_file,
                indent=2,
                ensure_ascii=False,
            )

            metadata_file.write("\n")

        # ----------------------------------------------------
        # 3. Validate the complete staged index
        # ----------------------------------------------------

        is_valid, is_stale = (
            _validate_vector_store_contents(
                video_id,
                staging_path,
            )
        )

        if not is_valid or is_stale:
            raise RuntimeError(
                "Staged vector store failed validation."
            )

        # ----------------------------------------------------
        # 4. Publish the staged index
        # ----------------------------------------------------

        existing_index = video_store_path.exists()
        backup_created = False

        if existing_index:
            video_store_path.rename(backup_path)
            backup_created = True

        try:
            staging_path.rename(video_store_path)
        except Exception:
            # The new index was not published.
            #
            # Restore the previous index when possible. If restoration
            # fails, deliberately leave the backup in place so the
            # original index is still recoverable.
            if (
                backup_created
                and backup_path.exists()
                and not video_store_path.exists()
            ):
                try:
                    backup_path.rename(video_store_path)
                except OSError as restore_error:
                    raise RuntimeError(
                        "Failed to publish the new vector store and "
                        f"failed to restore the previous index. "
                        f"Original index preserved at: {backup_path}"
                    ) from restore_error

            raise

        # ----------------------------------------------------
        # 5. New index published successfully
        # ----------------------------------------------------

        if backup_created and backup_path.exists():
            # Failure to remove an obsolete backup must not invalidate
            # an already successful publication.
            shutil.rmtree(
                backup_path,
                ignore_errors=True,
            )

        return video_store_path

    except Exception:
        # Remove incomplete staging data.
        if staging_path.exists():
            shutil.rmtree(
                staging_path,
                ignore_errors=True,
            )

        raise

    finally:
        # Defensive cleanup for staging only.
        #
        # IMPORTANT:
        # Never delete backup_path here. If recovery failed, that
        # backup may be the only remaining copy of the previous index.
        if staging_path.exists():
            shutil.rmtree(
                staging_path,
                ignore_errors=True,
            )

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

# create the index-building primitive
def create_index(
    video_id: str,
    languages: Optional[list[str]] = None,
):
    """
    Create and persist a FAISS index for one YouTube video.

    This function is responsible only for index creation.
    Lifecycle decisions such as whether an existing index should
    be reused, rebuilt, or treated as stale are handled separately.
    """

    if not video_id or not video_id.strip():
        raise ValueError(
            "video_id cannot be empty."
        )

    video_id = video_id.strip()

    print(
        "\n" + "=" * 70
    )

    print(
        "YOUTUBE RAG INDEX CREATION"
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
        "\n[1/5] Fetching transcript..."
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
        "\n[2/5] Creating LangChain Documents..."
    )

    documents = transcript_to_documents(
        transcript
    )

    print(
        f"Documents created: "
        f"{len(documents)}"
    )

    # --------------------------------------------------------
    # 3. Timestamp-aware chunks
    # --------------------------------------------------------

    print(
        "\n[3/5] Creating timestamp-aware chunks..."
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
    # 4. Embeddings + FAISS
    # --------------------------------------------------------

    print(
        "\n[4/5] Creating FAISS vector store..."
    )

    vector_store = create_vector_store(
        chunks
    )

    print(
        "FAISS vector store created."
    )

    # --------------------------------------------------------
    # 5. Persistence
    # --------------------------------------------------------

    print(
        "\n[5/5] Saving vector store..."
    )

    saved_path = save_vector_store(
        vector_store,
        video_id,
        chunks,
    )

    print(
        f"Saved to:\n{saved_path}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "INDEX CREATION COMPLETED"
    )

    print(
        "=" * 70
    )

    return vector_store


def clear_vector_store_cache():
    """
    Clear cached FAISS vector stores after an index changes.
    """
    load_vector_store.cache_clear()


_INDEX_LOCKS: dict[str, threading.Lock] = {}
_INDEX_LOCKS_GUARD = threading.Lock()


def _get_index_lock(video_id: str) -> threading.Lock:
    """
    Return the in-process lifecycle lock for one video ID.

    The guard protects creation of entries in the lock registry.
    """
    with _INDEX_LOCKS_GUARD:
        lock = _INDEX_LOCKS.get(video_id)

        if lock is None:
            lock = threading.Lock()
            _INDEX_LOCKS[video_id] = lock

        return lock


# ============================================================
# INDEX LIFECYCLE ORCHESTRATOR
# ============================================================

def ensure_index(
    video_id: str,
    languages: Optional[list[str]] = None,
):
    """
    Ensure that a usable FAISS index exists for one YouTube video.

    Lifecycle:

        MISSING         -> CREATE
        VALID           -> REUSE
        INVALID / STALE -> REBUILD
    """

    if not video_id or not video_id.strip():
        raise ValueError(
            "video_id cannot be empty."
        )

    video_id = video_id.strip()

    lifecycle_lock = _get_index_lock(
        video_id
    )

    with lifecycle_lock:

        # ----------------------------------------------------
        # 1. Determine state AFTER acquiring the lock
        # ----------------------------------------------------

        state = get_index_state(
            video_id
        )

        action = get_index_action(
            state
        )

        # ----------------------------------------------------
        # 2. Reuse
        # ----------------------------------------------------

        if action is IndexAction.REUSE:
            return load_vector_store(
                video_id
            )

        # ----------------------------------------------------
        # 3. Create / rebuild
        # ----------------------------------------------------

        if action in (
            IndexAction.CREATE,
            IndexAction.REBUILD,
        ):
            clear_vector_store_cache()

            create_index(
                video_id,
                languages=languages,
            )

            clear_vector_store_cache()

            # ------------------------------------------------
            # 4. Verify persisted result
            # ------------------------------------------------

            final_state = get_index_state(
                video_id
            )

            if final_state is not IndexState.VALID:
                raise RuntimeError(
                    "Index creation completed, but the persisted "
                    f"index is not valid. "
                    f"Final state: {final_state.value}"
                )

            return load_vector_store(
                video_id
            )

        raise RuntimeError(
            f"Unsupported lifecycle action: {action}"
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
            chunks,
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