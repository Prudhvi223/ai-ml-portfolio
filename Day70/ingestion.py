
import hashlib
import logging
from pathlib import Path

import chromadb

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding

from config import (
    CHROMA_DB_PATH,
    UPLOAD_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


# ==========================================
# 1. LOGGING
# ==========================================

logger = logging.getLogger("research-assistant")


# ==========================================
# 2. PDF INGESTION FUNCTION
# ==========================================

def ingest_pdf(
    pdf_path: str,
    original_filename: str = None,
    collection_name: str = None
) -> dict:
    """
    Extract text from PDF, split it into chunks,
    generate embeddings and store them in ChromaDB.

    Supports a custom collection name for safe staging.
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
        raise ValueError("Only PDF files are supported.")

    if pdf_path.stat().st_size == 0:
        raise ValueError("Uploaded PDF is empty.")

    pdf_bytes = pdf_path.read_bytes()

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF file.")

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

    # Add original filename and file hash
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

    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL
    )

    # --------------------------------------
    # GENERATE EMBEDDINGS
    # --------------------------------------

    ids = []
    texts = []
    metadatas = []
    embeddings = []

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
            metadata.get("page_label", "Unknown")
        )

        # Stable chunk ID
        chunk_id = f"{file_hash}_{index}"

        # Generate embedding
        embedding = embed_model.get_text_embedding(
            text
        )

        ids.append(chunk_id)
        texts.append(text)
        metadatas.append(metadata)
        embeddings.append(embedding)

    if not ids:
        raise ValueError(
            "No valid text chunks were generated."
        )

    # --------------------------------------
    # STORE IN CHROMADB
    # --------------------------------------

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
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
