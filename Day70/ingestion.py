
import hashlib
import logging
from pathlib import Path

import chromadb

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_DB_PATH,
    UPLOAD_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    IS_CLOUD
)


# ==========================================
# 1. LOGGING
# ==========================================

logger = logging.getLogger("research-assistant")


# ==========================================
# 2. EMBEDDING MODEL
# ==========================================

def get_embedding_model():

    if IS_CLOUD:

        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )


# ==========================================
# 3. PDF INGESTION FUNCTION
# ==========================================

def ingest_pdf(
    pdf_path: str,
    original_filename: str = None,
    collection_name: str = None
) -> dict:
    """
    Extract text from PDF, split it into chunks,
    generate embeddings and store them in ChromaDB.

    Supports custom collection names for safe staging.

    Uses Ollama locally and Hugging Face on Streamlit Cloud.
    """

    # --------------------------------------
    # VALIDATE PDF
    # --------------------------------------

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    if pdf_path.stat().st_size == 0:
        raise ValueError(
            "Uploaded PDF is empty."
        )

    pdf_bytes = pdf_path.read_bytes()

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError(
            "Invalid PDF file."
        )

    # --------------------------------------
    # GENERATE FILE HASH
    # --------------------------------------

    file_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    filename = (
        original_filename
        if original_filename
        else pdf_path.name
    )

    logger.info(
        "Starting PDF ingestion: %s",
        filename
    )

    # --------------------------------------
    # LOAD PDF
    # --------------------------------------

    documents = SimpleDirectoryReader(
        input_files=[str(pdf_path)]
    ).load_data()

    if not documents:
        raise ValueError(
            "No readable text found in the PDF."
        )

    for document in documents:

        document.metadata["file_name"] = filename
        document.metadata["file_hash"] = file_hash

    logger.info(
        "Loaded %d PDF pages/documents",
        len(documents)
    )

    # --------------------------------------
    # SPLIT INTO CHUNKS
    # --------------------------------------

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    nodes = splitter.get_nodes_from_documents(
        documents
    )

    if not nodes:
        raise ValueError(
            "No text chunks could be generated."
        )

    # --------------------------------------
    # CONNECT TO CHROMADB
    # --------------------------------------

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    target_collection = (
        collection_name or UPLOAD_COLLECTION_NAME
    )

    collection = client.get_or_create_collection(
        name=target_collection
    )

    # --------------------------------------
    # INITIALIZE EMBEDDING MODEL
    # --------------------------------------

    embed_model = get_embedding_model()

    # --------------------------------------
    # PREPARE TEXT AND METADATA
    # --------------------------------------

    ids = []
    texts = []
    metadatas = []

    for index, node in enumerate(nodes):

        text = node.get_content()

        if not text or not text.strip():
            continue

        metadata = dict(
            node.metadata or {}
        )

        metadata["file_name"] = filename
        metadata["file_hash"] = file_hash

        metadata["page_label"] = str(
            metadata.get(
                "page_label",
                "Unknown"
            )
        )

        # Stable chunk ID
        chunk_id = f"{file_hash}_{index}"

        ids.append(chunk_id)
        texts.append(text)
        metadatas.append(metadata)

    if not ids:
        raise ValueError(
            "No valid text chunks were generated."
        )

    # --------------------------------------
    # GENERATE EMBEDDINGS AND STORE
    # --------------------------------------

    # Process in batches to avoid oversized ChromaDB requests.
    batch_size = 500

    for start in range(0, len(texts), batch_size):

        end = start + batch_size

        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_metadatas = metadatas[start:end]

        # Same embedding interface for both environments
        batch_embeddings = embed_model.embed_documents(
            batch_texts
        )

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadatas,
            embeddings=batch_embeddings
        )

        logger.info(
            "Stored chunks %d to %d",
            start + 1,
            min(end, len(texts))
        )

    logger.info(
        "PDF ingestion completed: %d chunks stored",
        len(ids)
    )

    # --------------------------------------
    # RETURN DETAILS
    # --------------------------------------

    return {
        "file_name": filename,
        "file_hash": file_hash,
        "chunks_added": len(ids),
        "collection": target_collection
    }
