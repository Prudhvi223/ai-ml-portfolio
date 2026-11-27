
from pathlib import Path

# ==========================================
# 1. PROJECT DIRECTORY
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================
# 2. CHROMADB CONFIGURATION
# ==========================================

# Existing Day 61/63 ChromaDB location
CHROMA_DB_PATH = BASE_DIR / "Day61" / "chroma_db"

# Existing collection (do not overwrite)
COLLECTION_NAME = "pdf_knowledge"

# Separate collection for Day 70 uploaded PDFs
UPLOAD_COLLECTION_NAME = "day70_uploaded_pdfs"


# ==========================================
# 3. LOCAL OLLAMA MODELS
# ==========================================

LLM_MODEL = "llama3.2:1b"

EMBEDDING_MODEL = "nomic-embed-text"


# ==========================================
# 4. RETRIEVAL CONFIGURATION
# ==========================================

TOP_K = 3

CHUNK_SIZE = 300

CHUNK_OVERLAP = 50


# ==========================================
# 5. APPLICATION CONFIGURATION
# ==========================================

APP_TITLE = "AI Research Assistant"
