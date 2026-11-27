from ingestion import ingest_pdf
from graph import research_graph, refresh_retriever
from config import (
    CHROMA_DB_PATH,
    UPLOAD_COLLECTION_NAME,
    APP_TITLE,
    IS_CLOUD,
    LLM_MODEL,
    EMBEDDING_MODEL
)
import hashlib
import logging
import tempfile
from pathlib import Path
import chromadb
import streamlit as st
import nltk
import os
import stat

# Configure a private, writable NLTK directory
NLTK_DATA_DIR = "/tmp/nltk_data"

# Restrict permissions for newly created files and folders
os.umask(0o077)

os.makedirs(NLTK_DATA_DIR, exist_ok=True)
os.chmod(NLTK_DATA_DIR, stat.S_IRWXU)


if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# Download stopwords if missing
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    success = nltk.download(
        "stopwords",
        download_dir=NLTK_DATA_DIR,
        quiet=True,
        force=True
    )

    if not success:
        raise RuntimeError("Failed to download NLTK stopwords.")

    nltk.data.find("corpora/stopwords")


# ==========================================
# 1. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


# ==========================================
# 2. IMPORT PROJECT MODULES
# ==========================================


# ==========================================
# 3. LOGGING
# ==========================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("research-assistant")


# ==========================================
# 4. SAFE PDF REPLACEMENT
# ==========================================

def replace_uploaded_pdf(pdf_path, original_filename):
    """
    Ingest a new PDF into a staging collection.

    Replace the active collection only after successful ingestion.
    Restore the previous collection if replacement fails.
    """

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    staging_name = f"{UPLOAD_COLLECTION_NAME}_staging"

    # Remove leftover staging collection
    try:
        client.delete_collection(name=staging_name)
    except Exception:
        pass

    try:

        # ----------------------------------
        # STEP 1: INGEST INTO STAGING
        # ----------------------------------

        result = ingest_pdf(
            pdf_path=pdf_path,
            original_filename=original_filename,
            collection_name=staging_name
        )

        staging = client.get_collection(
            name=staging_name
        )

        if staging.count() == 0:
            raise ValueError(
                "Staging collection is empty."
            )

        # ----------------------------------
        # STEP 2: READ STAGED DATA
        # ----------------------------------

        staged_data = staging.get(
            include=[
                "documents",
                "metadatas",
                "embeddings"
            ]
        )

        if not staged_data["ids"]:
            raise ValueError(
                "No PDF chunks were generated."
            )

        # ----------------------------------
        # STEP 3: BACK UP ACTIVE COLLECTION
        # ----------------------------------

        backup_data = None

        try:

            active = client.get_collection(
                name=UPLOAD_COLLECTION_NAME
            )

            backup_data = active.get(
                include=[
                    "documents",
                    "metadatas",
                    "embeddings"
                ]
            )

        except Exception:

            logger.info(
                "No previous PDF collection found."
            )

        # ----------------------------------
        # STEP 4: REPLACE ACTIVE COLLECTION
        # ----------------------------------

        try:

            try:
                client.delete_collection(
                    name=UPLOAD_COLLECTION_NAME
                )
            except Exception:
                pass

            active = client.create_collection(
                name=UPLOAD_COLLECTION_NAME
            )

            active.upsert(
                ids=staged_data["ids"],
                documents=staged_data["documents"],
                metadatas=staged_data["metadatas"],
                embeddings=staged_data["embeddings"]
            )

            # Verify replacement
            if active.count() != len(staged_data["ids"]):
                raise ValueError(
                    "Active collection verification failed."
                )

            refresh_retriever()

        except Exception:

            logger.exception(
                "PDF replacement failed. Restoring previous data."
            )

            # ----------------------------------
            # STEP 5: RESTORE PREVIOUS DATA
            # ----------------------------------

            try:
                client.delete_collection(
                    name=UPLOAD_COLLECTION_NAME
                )
            except Exception:
                pass

            if backup_data and backup_data["ids"]:

                restored = client.create_collection(
                    name=UPLOAD_COLLECTION_NAME
                )

                restored.upsert(
                    ids=backup_data["ids"],
                    documents=backup_data["documents"],
                    metadatas=backup_data["metadatas"],
                    embeddings=backup_data["embeddings"]
                )

            try:
                refresh_retriever()
            except Exception:
                logger.exception(
                    "Could not refresh retriever after restoration."
                )

            raise

        # ----------------------------------
        # STEP 6: CLEAN UP STAGING
        # ----------------------------------

        try:

            client.delete_collection(
                name=staging_name
            )

        except Exception:

            logger.warning(
                "Could not remove staging collection."
            )

        return result

    except Exception:

        try:
            client.delete_collection(
                name=staging_name
            )
        except Exception:
            pass

        raise


# ==========================================
# 5. APPLICATION HEADER
# ==========================================

st.title("🔬 AI Research Assistant")

st.caption(
    "Agentic RAG | LangGraph | Hybrid Search | "
    "ChromaDB | Web Search"
)

st.markdown(
    "Ask a question and let the AI research assistant "
    "decide whether to search your uploaded PDF, "
    "the web, or both."
)


# ==========================================
# 6. LOAD GRAPH
# ==========================================

@st.cache_resource
def load_graph():
    return research_graph


graph = load_graph()


# ==========================================
# 7. SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_pdf_hash" not in st.session_state:
    st.session_state.uploaded_pdf_hash = None

if "uploaded_pdf_name" not in st.session_state:
    st.session_state.uploaded_pdf_name = None

if "uploaded_pdf_chunks" not in st.session_state:
    st.session_state.uploaded_pdf_chunks = 0


# ==========================================
# 8. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Research Settings")

    # --------------------------------------
    # PDF UPLOAD
    # --------------------------------------

    st.subheader("📚 Upload Research PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload a PDF to use as your research knowledge base."
    )

    if uploaded_file is not None:

        pdf_bytes = uploaded_file.getvalue()

        # Validate PDF signature
        if not pdf_bytes.startswith(b"%PDF-"):

            st.error(
                "The uploaded file does not appear to be a valid PDF."
            )

        else:

            file_hash = hashlib.sha256(
                pdf_bytes
            ).hexdigest()

            if file_hash != st.session_state.uploaded_pdf_hash:

                temp_path = None

                try:

                    with st.spinner(
                        "Processing PDF and generating embeddings..."
                    ):

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf"
                        ) as temp_file:

                            temp_file.write(pdf_bytes)
                            temp_path = temp_file.name

                        ingestion_result = replace_uploaded_pdf(
                            pdf_path=temp_path,
                            original_filename=uploaded_file.name
                        )

                        st.session_state.uploaded_pdf_hash = file_hash

                        st.session_state.uploaded_pdf_name = (
                            uploaded_file.name
                        )

                        st.session_state.uploaded_pdf_chunks = (
                            ingestion_result["chunks_added"]
                        )

                    st.success(
                        "PDF uploaded successfully!"
                    )

                    st.rerun()

                except Exception as e:

                    logger.exception(
                        "PDF ingestion failed"
                    )

                    st.error(
                        f"PDF processing failed: {str(e)}"
                    )

                finally:

                    if temp_path and Path(temp_path).exists():
                        Path(temp_path).unlink()

            else:

                st.info(
                    "This PDF is already uploaded."
                )

    # --------------------------------------
    # UPLOADED PDF STATUS
    # --------------------------------------

    st.divider()

    st.markdown("**Current PDF Knowledge Base**")

    if st.session_state.uploaded_pdf_name:

        st.success(
            f"📄 {st.session_state.uploaded_pdf_name}"
        )

        st.write(
            f"Chunks indexed: {st.session_state.uploaded_pdf_chunks}"
        )

    else:

        st.info(
            "No PDF uploaded yet."
        )

    st.divider()

    # --------------------------------------
    # AVAILABLE CAPABILITIES
    # --------------------------------------

    st.markdown("**Available capabilities**")

    st.success("📚 PDF Knowledge Base")
    st.success("🌐 External Web Search")
    st.success("🤖 Agentic Research Planning")
    st.success("📝 Report Generation")

    st.divider()

    # --------------------------------------
    # MODEL INFORMATION
    # --------------------------------------

    st.markdown("**LLM Provider**")

    if IS_CLOUD:
        st.write(f"Groq: {LLM_MODEL}")
    else:
        st.write(f"Ollama: {LLM_MODEL}")

    st.markdown("**Embedding Model**")
    st.write(EMBEDDING_MODEL)

    st.markdown("**Vector Database**")
    st.write("ChromaDB")

    st.markdown("**Orchestration**")
    st.write("LangGraph")

    if IS_CLOUD:

        st.info(
            "Cloud mode: uploaded PDFs are stored in temporary "
            "application storage. You may need to upload them "
            "again after an application restart."
        )

    st.divider()

    # --------------------------------------
    # CLEAR CHAT
    # --------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()


# ==========================================
# 9. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# 10. USER INPUT
# ==========================================

question = st.chat_input(
    "Ask your research question..."
)


# ==========================================
# 11. EXECUTE RESEARCH WORKFLOW
# ==========================================

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "🤖 Planning research and selecting sources..."
            ):

                initial_state = {
                    "question": question,
                    "rag_context": "",
                    "web_context": "",
                    "rag_sources": [],
                    "web_sources": [],
                    "sources": [],
                    "errors": [],
                    "rag_errors": [],
                    "web_errors": []
                }

                result = graph.invoke(initial_state)

            # ----------------------------------
            # ACTUAL RESEARCH SOURCES USED
            # ----------------------------------

            st.subheader("📋 Research Sources Used")

            col1, col2 = st.columns(2)

            with col1:

                if result.get("rag_context", "").strip():

                    st.success(
                        "📚 PDF Search: Used"
                    )

                elif result.get("search_pdf"):

                    st.warning(
                        "📚 PDF Search: Attempted, no relevant evidence"
                    )

                else:

                    st.info(
                        "📚 PDF Search: Not used"
                    )

            with col2:

                if result.get("web_context", "").strip():

                    st.success(
                        "🌐 Web Search: Used"
                    )

                elif result.get("search_web"):

                    st.warning(
                        "🌐 Web Search: Attempted, no results"
                    )

                else:

                    st.info(
                        "🌐 Web Search: Not used"
                    )

            # ----------------------------------
            # FINAL REPORT
            # ----------------------------------

            st.divider()

            st.subheader("📄 Final Research Report")

            report = result.get(
                "report",
                "No report generated."
            )

            st.markdown(report)

            # ----------------------------------
            # SOURCES
            # ----------------------------------

            with st.expander("📚 View Sources"):

                sources = result.get("sources", [])

                if sources:

                    for index, source in enumerate(
                        sources,
                        start=1
                    ):

                        st.write(
                            f"{index}. {source}"
                        )

                else:

                    st.warning(
                        "No sources were retrieved."
                    )

            # ----------------------------------
            # RETRIEVED CONTEXT
            # ----------------------------------

            with st.expander("🔍 View Retrieved Context"):

                st.markdown("**PDF Context**")

                rag_context = result.get(
                    "rag_context",
                    ""
                )

                st.write(
                    rag_context
                    if rag_context
                    else "No relevant PDF context retrieved."
                )

                st.divider()

                st.markdown("**Web Context**")

                web_context = result.get(
                    "web_context",
                    ""
                )

                st.write(
                    web_context
                    if web_context
                    else "No web context retrieved."
                )

            # ----------------------------------
            # WORKFLOW WARNINGS
            # ----------------------------------

            errors = result.get("errors", [])

            if errors:

                with st.expander("⚠️ Workflow Warnings"):

                    for error in errors:
                        st.warning(error)

            # ----------------------------------
            # SAVE CHAT
            # ----------------------------------

            st.session_state.messages.append({
                "role": "assistant",
                "content": report
            })

        except Exception as e:

            logger.exception(
                "Research workflow failed"
            )

            error_message = (
                f"Research workflow failed: {str(e)}"
            )

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })
