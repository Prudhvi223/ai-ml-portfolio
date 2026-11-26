
import streamlit as st
import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# 1. Streamlit page
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚"
)

st.title("📚 PDF Question Answering System")
st.write("Ask questions about your indexed PDF documents.")


# 2. Connect to ChromaDB
@st.cache_resource
def load_query_engine():

    client = chromadb.PersistentClient(
        path="Day61/chroma_db"
    )

    collection = client.get_collection(
        name="pdf_knowledge"
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text"
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    llm = Ollama(
        model="llama3.2:1b",
        request_timeout=120.0
    )

    return index.as_query_engine(
        llm=llm,
        similarity_top_k=3
    )


# 3. Load query engine
try:
    query_engine = load_query_engine()
    st.success("Knowledge base connected!")
except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.stop()


# 4. Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# 5. Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 6. Accept user question
if question := st.chat_input("Ask something about your PDF..."):

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # 7. Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching your PDF..."):
            try:
                response = query_engine.query(question)

                answer = str(response)
                st.markdown(answer)

                # Show retrieved sources
                with st.expander("View retrieved sources"):
                    for i, source in enumerate(response.source_nodes):
                        st.write(f"Source {i+1}")
                        st.write(source.text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error generating answer: {e}")
