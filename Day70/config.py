
import os
from pathlib import Path


# ==========================================
# 1. BASE DIRECTORY
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================
# 2. ENVIRONMENT SETTINGS
# ==========================================

def get_setting(name, default=None):

    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


DEPLOYMENT_MODE = str(
    get_setting("DEPLOYMENT_MODE", "local")
).lower()

IS_CLOUD = DEPLOYMENT_MODE == "cloud"


# ==========================================
# 3. CHROMADB CONFIGURATION
# ==========================================

if IS_CLOUD:

    CHROMA_DB_PATH = Path("/tmp/day70_chroma_db")

else:

    CHROMA_DB_PATH = BASE_DIR / "Day61" / "chroma_db"


COLLECTION_NAME = "pdf_knowledge"

UPLOAD_COLLECTION_NAME = "day70_uploaded_pdfs"


# ==========================================
# 4. LLM CONFIGURATION
# ==========================================

LOCAL_LLM_MODEL = "llama3.2:1b"

GROQ_MODEL = get_setting(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
)

LLM_MODEL = (
    GROQ_MODEL if IS_CLOUD else LOCAL_LLM_MODEL
)


# ==========================================
# 5. EMBEDDING CONFIGURATION
# ==========================================

LOCAL_EMBEDDING_MODEL = "nomic-embed-text"

CLOUD_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_MODEL = (
    CLOUD_EMBEDDING_MODEL
    if IS_CLOUD
    else LOCAL_EMBEDDING_MODEL
)


# ==========================================
# 6. RETRIEVAL CONFIGURATION
# ==========================================

TOP_K = 3


# ==========================================
# 7. DOCUMENT CHUNKING
# ==========================================

CHUNK_SIZE = 300

CHUNK_OVERLAP = 50


# ==========================================
# 8. APPLICATION CONFIGURATION
# ==========================================

APP_TITLE = "AI Research Assistant"
