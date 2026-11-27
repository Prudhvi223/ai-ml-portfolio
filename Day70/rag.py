
import re
import chromadb

from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings

from config import (
    CHROMA_DB_PATH,
    UPLOAD_COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K
)


class RAGRetriever:

    def __init__(self):

        # 1. Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH)
        )

        # 2. Connect to Day 70 upload collection
        self.collection = self.client.get_or_create_collection(
            name=UPLOAD_COLLECTION_NAME
        )

        # 3. Load stored documents and metadata
        stored = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = stored["documents"] or []
        self.ids = stored["ids"] or []
        self.metadatas = stored["metadatas"] or []

        # 4. Load embedding model
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL
        )

        # 5. Prepare BM25 keyword search
        if self.documents:

            tokenized_docs = [
                re.findall(r"\w+", doc.lower())
                for doc in self.documents
            ]

            self.bm25 = BM25Okapi(tokenized_docs)

        else:
            self.bm25 = None

        # 6. Map document IDs to content and metadata
        self.document_map = {
            doc_id: {
                "text": doc,
                "metadata": metadata or {}
            }
            for doc_id, doc, metadata in zip(
                self.ids,
                self.documents,
                self.metadatas
            )
        }

    def retrieve(self, question):

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not self.documents:
            raise ValueError(
                "No uploaded PDF found. Please upload a PDF first."
            )

        # -----------------------------
        # 1. BM25 keyword search
        # -----------------------------

        query_tokens = re.findall(
            r"\w+",
            question.lower()
        )

        bm25_scores = self.bm25.get_scores(
            query_tokens
        )

        bm25_ranked = sorted(
            range(len(self.documents)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )

        # -----------------------------
        # 2. Vector semantic search
        # -----------------------------

        query_embedding = self.embeddings.embed_query(
            question
        )

        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(TOP_K, len(self.documents)),
            include=["distances"]
        )

        vector_ids = vector_results["ids"][0]

        # -----------------------------
        # 3. Reciprocal Rank Fusion
        # -----------------------------

        rrf_scores = {}

        # BM25 rankings
        for rank, doc_index in enumerate(
            bm25_ranked,
            start=1
        ):

            doc_id = self.ids[doc_index]

            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0)
                + 1 / (60 + rank)
            )

        # Vector rankings
        for rank, doc_id in enumerate(
            vector_ids,
            start=1
        ):

            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0)
                + 1 / (60 + rank)
            )

        # -----------------------------
        # 4. Combine rankings
        # -----------------------------

        ranked_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )

        results = []

        for doc_id in ranked_ids[:TOP_K]:

            item = self.document_map[doc_id]
            metadata = item["metadata"]

            results.append({
                "id": doc_id,
                "text": item["text"],
                "source": metadata.get(
                    "file_name", "Unknown source"
                ),
                "page": metadata.get(
                    "page_label", "Unknown page"
                ),
                "score": rrf_scores[doc_id]
            })

        return results
