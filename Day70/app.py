# ==========================================
# 1. STANDARD LIBRARY IMPORTS
# ==========================================

from graph import research_graph, refresh_retriever
from ingestion import ingest_pdf
from config import (
    CHROMA_DB_PATH,
    UPLOAD_COLLECTION_NAME,
    APP_TITLE,
    IS_CLOUD,
    LLM_MODEL,
    EMBEDDING_MODEL
)
import os
import stat
import logging
import tempfile
import hashlib
from pathlib import Path

import nltk
import streamlit as st
import chromadb


# ==========================================
# 2. NLTK CLOUD-SAFE INITIALIZATION
# IMPORTANT: Run BEFORE importing graph
# or ingestion modules.
# ==========================================

NLTK_DATA_DIR = Path.home() / ".cache" / "nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)

try:
    NLTK_DATA_DIR.chmod(stat.S_IRWXU)
except OSError:
    pass

os.umask(0o077)

os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)

private_nltk_path = str(NLTK_DATA_DIR)

if private_nltk_path in nltk.data.path:
    nltk.data.path.remove(private_nltk_path)

nltk.data.path.insert(0, private_nltk_path)

try:
    nltk.data.find(
        "corpora/stopwords",
        paths=[private_nltk_path]
    )

except LookupError:

    downloaded = nltk.download(
        "stopwords",
        download_dir=private_nltk_path,
        quiet=True
    )

    if not downloaded:
        raise RuntimeError(
            "Could not download NLTK stopwords. "
            "Check Streamlit Cloud network access."
        )

    nltk.data.find(
        "corpora/stopwords",
        paths=[private_nltk_path]
    )


# ==========================================
# 3. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


# ==========================================
# 4. PROJECT IMPORTS
# ==========================================


# ==========================================
# 5. LOGGING
# ==========================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("research-assistant")


# ==========================================
# 6. SAFE PDF REPLACEMENT
# ==========================================

def replace_uploaded_pdf(pdf_path, original_filename):
    """
    Ingest a new PDF into a staging collection.

    Preserve the active ChromaDB collection identity.
    Replace its records only after successful ingestion.
    Restore previous records if replacement fails.
    """

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    staging_name = f"{UPLOAD_COLLECTION_NAME}_staging"

    # --------------------------------------
    # REMOVE OLD STAGING COLLECTION
    # --------------------------------------

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

        active = client.get_or_create_collection(
            name=UPLOAD_COLLECTION_NAME
        )

        backup_data = active.get(
            include=[
                "documents",
                "metadatas",
                "embeddings"
            ]
        )

        # ----------------------------------
        # STEP 4: REPLACE ACTIVE RECORDS
        # IMPORTANT:
        # Do NOT delete the collection itself.
        # ----------------------------------

        try:

            old_ids = backup_data.get("ids", [])

            # Delete existing records only.
            # Keep the collection ID unchanged.

            if old_ids:
                active.delete(ids=old_ids)

            # Insert newly staged records.

            active.upsert(
                ids=staged_data["ids"],
                documents=staged_data["documents"],
                metadatas=staged_data["metadatas"],
                embeddings=staged_data["embeddings"]
            )

            # Verify replacement.

            if active.count() != len(staged_data["ids"]):
                raise ValueError(
                    "Active collection verification failed."
                )

            # Refresh the retriever after replacement.

            refresh_retriever()

            logger.info(
                "PDF collection replaced successfully. "
                "Chunks: %s",
                active.count()
            )

        except Exception:

            logger.exception(
                "PDF replacement failed. Restoring previous data."
            )

            # ----------------------------------
            # STEP 5: RESTORE PREVIOUS RECORDS
            # ----------------------------------

            try:

                current_data = active.get()

                current_ids = current_data.get("ids", [])

                if current_ids:
                    active.delete(ids=current_ids)

                if backup_data and backup_data.get("ids"):

                    active.upsert(
                        ids=backup_data["ids"],
                        documents=backup_data["documents"],
                        metadatas=backup_data["metadatas"],
                        embeddings=backup_data["embeddings"]
                    )

                refresh_retriever()

                logger.info(
                    "Previous PDF data restored successfully."
                )

            except Exception:

                logger.exception(
                    "Could not restore previous PDF data."
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
# 7. APPLICATION HEADER
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
# 8. LOAD GRAPH
# ==========================================

@st.cache_resource
def load_graph():
    return research_graph


graph = load_graph()


# ==========================================
# 9. SESSION STATE
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
# 10. SIDEBAR
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

        # Validate PDF signature.

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

                        # Save uploaded PDF temporarily.

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf"
                        ) as temp_file:

                            temp_file.write(pdf_bytes)
                            temp_path = temp_file.name

                        # Safely replace the existing PDF.

                        ingestion_result = replace_uploaded_pdf(
                            pdf_path=temp_path,
                            original_filename=uploaded_file.name
                        )

                        # Save upload information.

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
# 11. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# 12. USER INPUT
# ==========================================

question = st.chat_input(
    "Ask your research question..."
)


# ==========================================
# 13. EXECUTE RESEARCH WORKFLOW
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
            # RESEARCH SOURCES USED
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
