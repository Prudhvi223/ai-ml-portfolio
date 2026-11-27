
import re
import logging

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

    # ==========================================
    # 1. STOPWORDS
    # ==========================================

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
        "works", "work", "system", "systems", "explain"
    }

    # ==========================================
    # 2. TECHNICAL TERM ALIASES
    # ==========================================

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

    def __init__(self):

        # --------------------------------------
        # CONNECT TO CHROMADB
        # --------------------------------------

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name=UPLOAD_COLLECTION_NAME
        )

        # --------------------------------------
        # LOAD DOCUMENTS AND METADATA
        # --------------------------------------

        stored = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = stored.get("documents") or []
        self.ids = stored.get("ids") or []
        self.metadatas = stored.get("metadatas") or []

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

        # --------------------------------------
        # INITIALIZE EMBEDDING MODEL
        # --------------------------------------

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

        # --------------------------------------
        # INITIALIZE BM25
        # --------------------------------------

        if self.documents:

            tokenized_docs = [
                self.tokenize(doc)
                for doc in self.documents
            ]

            self.bm25 = BM25Okapi(tokenized_docs)

        else:

            self.bm25 = None

        logger.info(
            "RAG initialized with %d document chunks.",
            len(self.documents)
        )

    # ==========================================
    # 3. TOKEN NORMALIZATION
    # ==========================================

    def normalize_token(self, token):
        """
        Normalize simple plural and verb forms.

        Examples:
        embeddings -> embedd
        embedding  -> embedd
        databases  -> database
        chunks     -> chunk
        chunking   -> chunk
        """

        if len(token) > 5 and token.endswith("ing"):

            token = token[:-3]

        elif len(token) > 4 and token.endswith("ed"):

            token = token[:-2]

        elif len(token) > 4 and token.endswith("s"):

            token = token[:-1]

        return token

    # ==========================================
    # 4. TOKENIZE TEXT
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
    # 5. EXTRACT QUERY KEYWORDS
    # ==========================================

    def extract_keywords(self, text):

        tokens = self.tokenize(text)

        keywords = set()

        for token in tokens:

            if token in self.STOPWORDS:
                continue

            keywords.add(token)

            # Expand recognized abbreviations.
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
    # 6. SPLIT DOCUMENT INTO SENTENCES
    # ==========================================

    def get_explanatory_sentences(self, document):
        """
        Exclude question-only sentences.

        This helps distinguish an explanatory paragraph
        from a list of example questions mentioning a topic.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            document
        )

        explanatory_sentences = []

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # Ignore sentences that are questions.
            if sentence.endswith("?"):
                continue

            tokens = self.tokenize(sentence)

            # Ignore very short fragments and headings.
            if len(tokens) < 5:
                continue

            explanatory_sentences.append(sentence)

        return explanatory_sentences

    # ==========================================
    # 7. CHECK DOCUMENT RELEVANCE
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

        # Only use explanatory sentences for relevance.
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

        # At least one explanatory sentence must
        # contain two or more query concepts.
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
            "Relevance: matches=%d, ratio=%.3f, "
            "strong_sentence=%s, relevant=%s",
            match_count,
            overlap_ratio,
            strong_sentence_match,
            relevant
        )

        return relevant

    # ==========================================
    # 8. HYBRID RETRIEVAL
    # ==========================================

    def retrieve(self, question):

        if not question or not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        if not self.documents:

            raise ValueError(
                "No uploaded PDF found. Please upload a PDF first."
            )

        question = question.strip()

        # --------------------------------------
        # STEP 1: BM25 KEYWORD SEARCH
        # --------------------------------------

        bm25_ranked = []

        if self.bm25 is not None:

            query_tokens = self.tokenize(question)

            # Add recognized technical aliases.
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

        vector_results = self.collection.query(
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

        if not relevant_results:

            logger.warning(
                "No relevant PDF chunks found for: %s",
                question
            )

            return []

        logger.info(
            "Retrieved %d relevant PDF chunks.",
            len(relevant_results)
        )

        return relevant_results
