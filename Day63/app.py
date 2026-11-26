
import streamlit as st
import chromadb
import re

from rank_bm25 import BM25Okapi

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Advanced RAG Assistant")

st.caption(
    "Hybrid Search + Reranking + LangChain + Ollama"
)


# ==========================================
# 2. LOAD DATABASE AND MODELS
# ==========================================

@st.cache_resource
def load_resources():

    # Connect to existing ChromaDB
    client = chromadb.PersistentClient(
        path="Day61/chroma_db"
    )

    collection = client.get_collection(
        name="pdf_knowledge"
    )

    # Get stored documents
    stored = collection.get(
        include=["documents"]
    )

    documents = stored["documents"]
    ids = stored["ids"]

    if not documents:
        raise ValueError(
            "No documents found. Run Day61/ingest.py first."
        )

    # Embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # LLM
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0
    )

    # Prepare BM25 keyword search
    tokenized_docs = [
        re.findall(r"\w+", doc.lower())
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    return (
        collection,
        documents,
        ids,
        embeddings,
        llm,
        bm25
    )


try:

    (
        collection,
        documents,
        ids,
        embeddings,
        llm,
        bm25
    ) = load_resources()

    st.success("Advanced RAG pipeline connected!")

except Exception as e:

    st.error(f"Pipeline error: {e}")
    st.stop()


# ==========================================
# 3. HYBRID SEARCH
# ==========================================

def hybrid_search(question):

    # BM25 keyword search
    bm25_scores = bm25.get_scores(
        re.findall(r"\w+", question.lower())
    )

    bm25_ranked = sorted(
        range(len(documents)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    # Vector semantic search
    query_embedding = embeddings.embed_query(
        question
    )

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, len(documents))
    )

    vector_ids = vector_results["ids"][0]

    # Reciprocal Rank Fusion
    rrf_scores = {}

    # Add BM25 rankings
    for rank, doc_index in enumerate(
        bm25_ranked,
        start=1
    ):

        doc_id = ids[doc_index]

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1 / (60 + rank)
        )

    # Add vector rankings
    for rank, doc_id in enumerate(
        vector_ids,
        start=1
    ):

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1 / (60 + rank)
        )

    # Sort using combined scores
    hybrid_ranked = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    # Map IDs to documents
    id_to_document = dict(
        zip(ids, documents)
    )

    candidates = [
        id_to_document[doc_id]
        for doc_id in hybrid_ranked[:3]
    ]

    return candidates


# ==========================================
# 4. RERANKING
# ==========================================

def rerank_documents(question, candidates):

    candidate_text = "\n\n".join(
        f"Chunk {i+1}: {doc}"
        for i, doc in enumerate(candidates)
    )

    rerank_prompt = f"""
You are a document relevance evaluator.

Question:
{question}

Retrieved chunks:
{candidate_text}

Select the chunk that contains the most useful
information for answering the question.

Consider meaning, not only exact keyword matches.

Return ONLY the selected chunk number.
"""

    response = llm.invoke(
        rerank_prompt
    )

    result = response.content.strip()

    match = re.search(
        r"\b([1-3])\b",
        result
    )

    if match:

        selected_index = int(
            match.group(1)
        ) - 1

        if selected_index < len(candidates):
            return candidates[selected_index]

    return candidates[0]


# ==========================================
# 5. LANGCHAIN ANSWER GENERATION
# ==========================================

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI tutor.

Answer the user's question clearly using the
provided context.

Instructions:

1. Use all relevant information from the context.
2. Combine information from multiple sentences
   when necessary.
3. Explain concepts in simple language.
4. Give a direct answer to the question.
5. Do not say information is missing if the context
   contains enough information to answer.
6. If the context genuinely does not contain the
   answer, clearly say that it is unavailable.
7. Do not invent facts that are not supported
   by the context.

Context:
{context}

Question:
{question}

Answer:
""")

answer_chain = (
    prompt
    | llm
    | StrOutputParser()
)


# ==========================================
# 6. ANSWER EVALUATION
# ==========================================

def evaluate_answer(question, context, answer):

    evaluation_prompt = f"""
Evaluate this RAG-generated answer.

Question:
{question}

Retrieved context:
{context}

Generated answer:
{answer}

Evaluate these two aspects:

1. Faithfulness:
Is the answer supported by the retrieved context?

2. Answer relevancy:
Does the answer directly address the question?

Give each a score from 1 to 5.

Explain both scores briefly.

Do not reward unsupported claims.
"""

    result = llm.invoke(
        evaluation_prompt
    )

    return result.content


# ==========================================
# 7. CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ==========================================
# 8. USER QUESTION
# ==========================================

if question := st.chat_input(
    "Ask something about your PDF..."
):

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    # Assistant response
    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching documents..."
            ):

                # Hybrid retrieval
                candidates = hybrid_search(
                    question
                )

            with st.spinner(
                "Reranking relevant chunks..."
            ):

                # Reranking
                best_chunk = rerank_documents(
                    question,
                    candidates
                )

            with st.spinner(
                "Generating answer..."
            ):

                # LangChain generation
                answer = answer_chain.invoke({
                    "context": best_chunk,
                    "question": question
                })

            st.markdown(answer)

            # Show hybrid search results
            with st.expander(
                "View hybrid search results"
            ):

                for i, chunk in enumerate(
                    candidates
                ):

                    st.write(
                        f"Candidate {i+1}"
                    )

                    st.write(chunk)

            # Show reranked chunk
            with st.expander(
                "View reranked best chunk"
            ):

                st.write(best_chunk)

            # Evaluate answer
            with st.expander(
                "View answer evaluation"
            ):

                with st.spinner(
                    "Evaluating answer..."
                ):

                    evaluation = evaluate_answer(
                        question,
                        best_chunk,
                        answer
                    )

                st.write(evaluation)

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            st.error(
                f"Error generating answer: {e}"
            )
