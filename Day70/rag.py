
import re
import logging
import threading

import chromadb

from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_DB_PATH,
    UPLOAD_COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    IS_CLOUD
)

logger = logging.getLogger("research-assistant.rag")


class RAGRetriever:

    STOPWORDS = {
        "the", "is", "are", "was", "were", "be", "been",
        "being", "a", "an", "and", "or", "but", "in", "on",
        "at", "to", "from", "for", "of", "with", "by",
        "about", "into", "through", "what", "which", "who",
        "when", "where", "why", "how", "explain", "describe",
        "discuss", "tell", "show", "give", "find", "compare",
        "this", "that", "these", "those", "my", "your",
        "their", "its", "it", "i", "we", "you", "me",
        "can", "could", "would", "should", "do", "does",
        "did", "have", "has", "had", "will", "shall",
        "each", "every", "simple", "language", "step",
        "steps", "using", "used", "use", "based", "pdf",
        "uploaded", "document", "documents", "file",
        "only", "include", "relevant", "page", "number",
        "numbers", "according", "please", "main", "important",
        "information", "provide", "provided", "summarize",
        "works", "work", "system", "systems"
    }

    TERM_ALIASES = {
        "rag": {"retrieval", "augmented", "generation"},
        "ai": {"artificial", "intelligence"},
        "ml": {"machine", "learning"},
        "nlp": {"natural", "language", "processing"},
        "llm": {"large", "language", "model"},
        "llms": {"large", "language", "models"},
        "db": {"database"},
        "nn": {"neural", "network"},
    }

    # ==========================================
    # 1. INITIALIZATION
    # ==========================================

    def __init__(self):

        self._lock = threading.RLock()

        # Create ONE persistent client.
        # Do not recreate this client on every retrieval.
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH)
        )

        self.collection = None

        self.documents = []
        self.ids = []
        self.metadatas = []
        self.document_map = {}
        self.bm25 = None

        # Initialize embedding model only once.
        if IS_CLOUD:

            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": True
                }
            )

        else:

            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL
            )

        self.refresh_collection()

    # ==========================================
    # 2. REFRESH COLLECTION AND DOCUMENT CACHE
    # ==========================================

    def refresh_collection(self):
        """
        Reacquire the active collection by name.

        Reload documents, IDs, and metadata so that
        the BM25 index and vector collection stay
        synchronized.
        """

        with self._lock:

            # IMPORTANT:
            # Reuse the existing PersistentClient.
            # Do not create another client here.

            self.collection = self.client.get_or_create_collection(
                name=UPLOAD_COLLECTION_NAME
            )

            # Read the current collection contents.
            stored = self.collection.get(
                include=["documents", "metadatas"]
            )

            self.documents = stored.get("documents") or []
            self.ids = stored.get("ids") or []
            self.metadatas = stored.get("metadatas") or []

            # Rebuild document lookup.
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

            # Rebuild BM25 index.
            if self.documents:

                tokenized_docs = [
                    self.tokenize(doc)
                    for doc in self.documents
                ]

                self.bm25 = BM25Okapi(tokenized_docs)

            else:

                self.bm25 = None

            # Diagnostic logging.
            logger.info(
                "CHROMA REFRESH | Collection=%s | ID=%s | Chunks=%d",
                UPLOAD_COLLECTION_NAME,
                self.collection.id,
                len(self.documents)
            )

            logger.info(
                "CHROMA PATH: %s",
                CHROMA_DB_PATH
            )

            return len(self.documents)

    # ==========================================
    # 3. GET CURRENT COLLECTION BEFORE QUERY
    # ==========================================

    def get_live_collection(self):
        """
        Obtain a fresh collection handle using the
        collection name immediately before querying.

        This avoids relying only on an old cached
        collection object.
        """

        with self._lock:

            live_collection = self.client.get_collection(
                name=UPLOAD_COLLECTION_NAME
            )

            live_count = live_collection.count()

            logger.info(
                "CHROMA LIVE CHECK | Name=%s | ID=%s | Count=%d",
                UPLOAD_COLLECTION_NAME,
                live_collection.id,
                live_count
            )

            self.collection = live_collection

            return live_collection

    # ==========================================
    # 4. TOKEN NORMALIZATION
    # ==========================================

    def normalize_token(self, token):

        if len(token) > 5 and token.endswith("ing"):

            token = token[:-3]

        elif len(token) > 4 and token.endswith("ed"):

            token = token[:-2]

        elif len(token) > 4 and token.endswith("s"):

            token = token[:-1]

        return token

    # ==========================================
    # 5. TOKENIZATION
    # ==========================================

    def tokenize(self, text):

        tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

        return [
            self.normalize_token(token)
            for token in tokens
            if len(token) >= 2
        ]

    # ==========================================
    # 6. EXTRACT QUERY KEYWORDS
    # ==========================================

    def extract_keywords(self, text):

        tokens = self.tokenize(text)

        keywords = set()

        for token in tokens:

            if token in self.STOPWORDS:
                continue

            keywords.add(token)

            aliases = self.TERM_ALIASES.get(
                token,
                set()
            )

            for alias in aliases:

                keywords.add(
                    self.normalize_token(alias)
                )

        return keywords

    # ==========================================
    # 7. SPLIT DOCUMENT INTO SENTENCES
    # ==========================================

    def get_explanatory_sentences(self, document):

        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            document
        )

        explanatory_sentences = []

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if sentence.endswith("?"):
                continue

            tokens = self.tokenize(sentence)

            if len(tokens) < 5:
                continue

            explanatory_sentences.append(sentence)

        return explanatory_sentences

    # ==========================================
    # 8. CHECK DOCUMENT RELEVANCE
    # ==========================================

    def is_relevant(self, question, document):

        question_keywords = self.extract_keywords(
            question
        )

        if not question_keywords:
            return True

        explanatory_sentences = (
            self.get_explanatory_sentences(document)
        )

        if not explanatory_sentences:
            return False

        document_keywords = set()

        for sentence in explanatory_sentences:

            document_keywords.update(
                self.extract_keywords(sentence)
            )

        matching_keywords = (
            question_keywords & document_keywords
        )

        match_count = len(matching_keywords)

        minimum_matches = min(
            2,
            len(question_keywords)
        )

        overlap_ratio = (
            match_count / len(question_keywords)
        )

        strong_sentence_match = False

        for sentence in explanatory_sentences:

            sentence_keywords = self.extract_keywords(
                sentence
            )

            sentence_matches = (
                question_keywords & sentence_keywords
            )

            if len(sentence_matches) >= 2:

                strong_sentence_match = True
                break

        relevant = (
            match_count >= minimum_matches
            and overlap_ratio >= 0.12
            and strong_sentence_match
        )

        logger.debug(
            "RELEVANCE | Matches=%d | Ratio=%.3f | "
            "Strong sentence=%s | Relevant=%s",
            match_count,
            overlap_ratio,
            strong_sentence_match,
            relevant
        )

        return relevant

    # ==========================================
    # 9. INTERNAL HYBRID RETRIEVAL
    # ==========================================

    def _retrieve_from_current_collection(self, question):

        if not self.documents:

            raise ValueError(
                "No uploaded PDF found. Please upload a PDF first."
            )

        # --------------------------------------
        # STEP 1: BM25 KEYWORD SEARCH
        # --------------------------------------

        bm25_ranked = []

        if self.bm25 is not None:

            query_tokens = self.tokenize(question)

            query_tokens = list(
                set(query_tokens)
                | self.extract_keywords(question)
            )

            bm25_scores = self.bm25.get_scores(
                query_tokens
            )

            bm25_ranked = sorted(
                [
                    i for i, score in enumerate(bm25_scores)
                    if score > 0
                ],
                key=lambda i: bm25_scores[i],
                reverse=True
            )

        # --------------------------------------
        # STEP 2: VECTOR SEMANTIC SEARCH
        # --------------------------------------

        query_embedding = self.embeddings.embed_query(
            question
        )

        candidate_count = min(
            max(TOP_K * 5, 15),
            len(self.documents)
        )

        # IMPORTANT:
        # Obtain a fresh collection handle immediately
        # before executing the vector query.

        live_collection = self.get_live_collection()

        live_count = live_collection.count()

        if live_count == 0:

            raise ValueError(
                "The ChromaDB collection is empty. "
                "Please upload the PDF again."
            )

        candidate_count = min(
            candidate_count,
            live_count
        )

        logger.info(
            "VECTOR QUERY | Collection ID=%s | Candidates=%d",
            live_collection.id,
            candidate_count
        )

        vector_results = live_collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_count,
            include=["distances"]
        )

        vector_ids = (
            vector_results["ids"][0]
            if vector_results.get("ids")
            else []
        )

        # --------------------------------------
        # STEP 3: RECIPROCAL RANK FUSION
        # --------------------------------------

        rrf_scores = {}

        # BM25 rankings.
        for rank, doc_index in enumerate(
            bm25_ranked,
            start=1
        ):

            doc_id = self.ids[doc_index]

            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0)
                + 1 / (60 + rank)
            )

        # Vector rankings.
        for rank, doc_id in enumerate(
            vector_ids,
            start=1
        ):

            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0)
                + 1 / (60 + rank)
            )

        # --------------------------------------
        # STEP 4: COMBINE RANKINGS
        # --------------------------------------

        ranked_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )

        # --------------------------------------
        # STEP 5: RELEVANCE FILTER
        # --------------------------------------

        relevant_results = []

        for doc_id in ranked_ids:

            item = self.document_map.get(doc_id)

            if not item:
                continue

            document_text = item["text"]

            if not self.is_relevant(
                question,
                document_text
            ):
                continue

            metadata = item["metadata"]

            relevant_results.append({

                "id": doc_id,

                "text": document_text,

                "source": metadata.get(
                    "file_name",
                    "Unknown source"
                ),

                "page": metadata.get(
                    "page_label",
                    "Unknown page"
                ),

                "score": rrf_scores[doc_id],

                "relevant": True

            })

            if len(relevant_results) >= TOP_K:
                break

        # --------------------------------------
        # STEP 6: RETURN RESULTS
        # --------------------------------------

        logger.info(
            "HYBRID RETRIEVAL COMPLETE | "
            "BM25=%d | Vector=%d | Relevant=%d",
            len(bm25_ranked),
            len(vector_ids),
            len(relevant_results)
        )

        return relevant_results

    # ==========================================
    # 10. PUBLIC RETRIEVAL METHOD
    # ==========================================

    def retrieve(self, question):

        if not question or not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        question = question.strip()

        with self._lock:

            # Always refresh the collection and document cache.
            self.refresh_collection()

            if not self.documents:

                raise ValueError(
                    "No uploaded PDF found. Please upload a PDF first."
                )

            try:

                return self._retrieve_from_current_collection(
                    question
                )

            except Exception as first_error:

                error_message = str(first_error).lower()

                missing_collection = (
                    "does not exist" in error_message
                    or "not found" in error_message
                )

                if not missing_collection:

                    logger.exception(
                        "PDF retrieval failed for a non-collection error."
                    )

                    raise

                logger.warning(
                    "STALE COLLECTION DETECTED. "
                    "Refreshing collection and retrying once."
                )

                # Refresh collection and document cache.
                self.refresh_collection()

                if not self.documents:

                    raise ValueError(
                        "The ChromaDB collection exists but "
                        "contains no uploaded PDF chunks."
                    ) from first_error

                # Retry once using the newly acquired collection.
                try:

                    return self._retrieve_from_current_collection(
                        question
                    )

                except Exception:

                    logger.exception(
                        "PDF retrieval failed again after collection refresh."
                    )

                    raise
