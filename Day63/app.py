
import streamlit as st
import chromadb

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 1. Streamlit page
st.set_page_config(
    page_title="Project 4 - RAG Assistant",
    page_icon="📚"
)

st.title("📚 AI PDF Assistant")
st.caption("RAG application using LangChain, ChromaDB and Ollama")


# 2. Load the RAG pipeline
@st.cache_resource
def load_rag_chain():

    # Connect to existing ChromaDB
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_store = Chroma(
        collection_name="pdf_knowledge",
        persist_directory="Day61/chroma_db",
        embedding_function=embeddings
    )

    # Create retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # Configure LLM
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the question using only the context below.

    If the answer is not present in the context,
    say that you cannot find it in the PDF.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    # Convert retrieved documents into text
    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    # Build LangChain workflow
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# 3. Initialize pipeline
try:
    rag_chain, retriever = load_rag_chain()
    st.success("RAG pipeline connected successfully!")

except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()


# 4. Initialize chat history
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
        with st.spinner("Searching PDF and generating answer..."):

            try:
                answer = rag_chain.invoke(question)

                st.markdown(answer)

                # Display retrieved sources
                with st.expander("View retrieved sources"):
                    source_docs = retriever.invoke(question)

                    for i, doc in enumerate(source_docs):
                        st.write(f"Source {i + 1}")
                        st.write(doc.page_content)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error: {e}")
