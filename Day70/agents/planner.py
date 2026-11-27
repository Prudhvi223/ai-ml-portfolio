
import logging
import re

import chromadb

from json_repair import repair_json
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import (
    CHROMA_DB_PATH,
    UPLOAD_COLLECTION_NAME
)

from llm_factory import get_chat_model


# ==========================================
# 1. LOGGING
# ==========================================

logger = logging.getLogger("research-assistant.planner")


# ==========================================
# 2. PYDANTIC MODELS
# ==========================================

class SubQuestion(BaseModel):
    title: str = Field(
        description="Short title of the research question"
    )

    description: str = Field(
        default="",
        description="Explanation of what should be researched"
    )


class ResearchPlan(BaseModel):
    search_pdf: bool = Field(
        description="Whether the PDF knowledge base should be searched"
    )

    search_web: bool = Field(
        description="Whether external web search is required"
    )

    subquestions: list[SubQuestion] = Field(
        description="Smaller questions needed to research the topic"
    )

    research_goal: str = Field(
        description="Overall objective of the research"
    )


# ==========================================
# 3. INITIALIZE LLM
# ==========================================

llm = get_chat_model()


# ==========================================
# 4. PLANNING PROMPT
# ==========================================

planner_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are an intelligent research planning agent.

Understand the user's research question and create
a structured research plan.

Generate:

1. A clear research goal.
2. Two to five research subquestions.
3. A title and description for every subquestion.

Return a valid JSON object containing:

- research_goal: A string
- subquestions: A list of objects
- Each object must contain title and description

Example JSON structure:

{{
    "research_goal": "Compare two technologies",
    "subquestions": [
        {{
            "title": "What is technology A?",
            "description": "Explain technology A."
        }},
        {{
            "title": "What is technology B?",
            "description": "Explain technology B."
        }}
    ]
}}

IMPORTANT:

- Do not include search_pdf or search_web.
- Do not include Markdown code fences.
- Return valid JSON only.
- Do not add explanations outside the JSON.
"""
    ),

    (
        "human",
        "Research question: {question}"
    )

])

planner_chain = planner_prompt | llm


# ==========================================
# 5. CHECK PDF AVAILABILITY
# ==========================================

def is_pdf_available():

    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH)
        )

        collection = client.get_collection(
            name=UPLOAD_COLLECTION_NAME
        )

        available = collection.count() > 0

        logger.info(
            "PDF availability: %s",
            available
        )

        return available

    except Exception as e:

        logger.info(
            "PDF collection not available: %s",
            e
        )

        return False


# ==========================================
# 6. SOURCE SELECTION
# ==========================================

def decide_sources(
    question: str,
    pdf_available=None
):

    if not question or not question.strip():
        raise ValueError(
            "Research question cannot be empty."
        )

    q = question.lower().strip()
    q = re.sub(r"\s+", " ", q)

    if pdf_available is None:
        pdf_available = is_pdf_available()

    # --------------------------------------
    # PDF EXCLUSION PHRASES
    # --------------------------------------

    pdf_exclusion_phrases = [

        "do not use my uploaded pdf",
        "don't use my uploaded pdf",
        "do not use the uploaded pdf",
        "don't use the uploaded pdf",
        "do not use the pdf",
        "don't use the pdf",
        "do not use pdf",
        "don't use pdf",
        "do not search the pdf",
        "don't search the pdf",
        "without using the pdf",
        "without using my pdf",
        "without the pdf",
        "no pdf search",
        "do not use my uploaded document",
        "don't use my uploaded document",
        "do not use the uploaded document",
        "don't use the uploaded document",
        "do not use the document",
        "don't use the document",
        "without using the document",
        "no document search"

    ]

    pdf_forbidden = any(
        phrase in q
        for phrase in pdf_exclusion_phrases
    )

    # --------------------------------------
    # WEB EXCLUSION PHRASES
    # --------------------------------------

    web_exclusion_phrases = [

        "do not search the web",
        "don't search the web",
        "do not search web",
        "don't search web",
        "without using the web",
        "without web search",
        "no web search",
        "do not use the web",
        "don't use the web",
        "do not use web",
        "don't use web",
        "without internet",
        "do not use internet",
        "don't use internet",
        "do not search online",
        "don't search online",
        "no internet search"

    ]

    web_forbidden = any(
        phrase in q
        for phrase in web_exclusion_phrases
    )

    # --------------------------------------
    # PDF INDICATORS
    # --------------------------------------

    pdf_patterns = [

        r"\bpdf\b",
        r"\bdocument\b",
        r"\bdocuments\b",
        r"\buploaded file\b",
        r"\buploaded files\b",
        r"\buploaded document\b",
        r"\buploaded documents\b",
        r"\bmy file\b",
        r"\bmy files\b",
        r"\bmy document\b",
        r"\bmy documents\b",
        r"\bthis document\b",
        r"\bthe document\b",
        r"\bthis pdf\b",
        r"\bthe pdf\b",
        r"\bknowledge base\b",
        r"\binternal document\b",
        r"\bmy uploaded\b",
        r"\baccording to\b.*\b(document|pdf|file)\b",
        r"\bin the\b.*\b(document|pdf|file)\b",
        r"\bfrom the\b.*\b(document|pdf|file)\b"

    ]

    # --------------------------------------
    # WEB INDICATORS
    # --------------------------------------

    web_patterns = [

        r"\bweb\b",
        r"\binternet\b",
        r"\bonline\b",
        r"\blatest\b",
        r"\brecent\b",
        r"\bcurrent\b",
        r"\btoday\b",
        r"\bnews\b",
        r"\bup to date\b",
        r"\bexternal research\b",
        r"\bexternal sources\b",
        r"\bsearch online\b",
        r"\bsearch the web\b"

    ]

    # --------------------------------------
    # DETECT POSITIVE INTENT
    # --------------------------------------

    explicit_pdf = (
        any(
            re.search(pattern, q)
            for pattern in pdf_patterns
        )
        and not pdf_forbidden
    )

    explicit_web = (
        any(
            re.search(pattern, q)
            for pattern in web_patterns
        )
        and not web_forbidden
    )

    # --------------------------------------
    # FINAL SOURCE DECISION
    # --------------------------------------

    # User explicitly excluded both sources.
    if pdf_forbidden and web_forbidden:

        search_pdf = False
        search_web = False

    # User explicitly excluded PDF.
    elif pdf_forbidden:

        search_pdf = False
        search_web = not web_forbidden

    # User explicitly excluded web.
    elif web_forbidden:

        search_pdf = pdf_available
        search_web = False

    # User requested both sources.
    elif explicit_pdf and explicit_web:

        search_pdf = pdf_available
        search_web = True

    # User requested web.
    elif explicit_web:

        search_pdf = False
        search_web = True

    # User requested PDF.
    elif explicit_pdf:

        if pdf_available:
            search_pdf = True
            search_web = False
        else:
            search_pdf = False
            search_web = True

    # No explicit source preference.
    elif pdf_available:

        search_pdf = True
        search_web = False

    else:

        search_pdf = False
        search_web = True

    logger.info(
        "SOURCE SELECTION | Question: %s | "
        "PDF available: %s | PDF: %s | Web: %s",
        question,
        pdf_available,
        search_pdf,
        search_web
    )

    return search_pdf, search_web


# ==========================================
# 7. NORMALIZE RESEARCH PLAN
# ==========================================

def normalize_research_plan(
    data,
    question,
    pdf_available=None
):

    search_pdf, search_web = decide_sources(
        question,
        pdf_available
    )

    raw_questions = data.get(
        "subquestions",
        []
    )

    if not isinstance(raw_questions, list):
        raw_questions = [raw_questions]

    unique_questions = []
    seen = set()

    for item in raw_questions:

        if isinstance(item, str):

            title = item
            description = ""

        elif isinstance(item, dict):

            title = (
                item.get("title")
                or item.get("question")
                or item.get("name")
                or ""
            )

            description = (
                item.get("description")
                or item.get("details")
                or ""
            )

        else:
            continue

        title = str(title).strip()
        description = str(description).strip()

        key = title.lower()

        if key and key not in seen:

            seen.add(key)

            unique_questions.append({

                "title": title,
                "description": description

            })

    if not unique_questions:

        unique_questions = [{

            "title": question,

            "description": (
                "Investigate the user's research question."
            )

        }]

    research_goal = str(
        data.get(
            "research_goal",
            "Research the user's question."
        )
    )

    return {

        "search_pdf": search_pdf,

        "search_web": search_web,

        "subquestions": unique_questions[:5],

        "research_goal": research_goal

    }


# ==========================================
# 8. DETERMINISTIC FALLBACK PLAN
# ==========================================

def create_fallback_plan(
    question: str,
    pdf_available=None
) -> ResearchPlan:

    search_pdf, search_web = decide_sources(
        question,
        pdf_available
    )

    fallback_plan = {

        "search_pdf": search_pdf,

        "search_web": search_web,

        "research_goal": question.strip(),

        "subquestions": [
            {
                "title": question.strip(),
                "description": (
                    "Answer the user's research question "
                    "using the selected evidence sources."
                )
            }
        ]

    }

    logger.warning(
        "Using deterministic research plan fallback. "
        "PDF: %s | Web: %s",
        search_pdf,
        search_web
    )

    return ResearchPlan.model_validate(
        fallback_plan
    )


# ==========================================
# 9. CREATE RESEARCH PLAN
# ==========================================

def create_research_plan(
    question: str
) -> ResearchPlan:

    if not question or not question.strip():
        raise ValueError(
            "Research question cannot be empty."
        )

    question = question.strip()

    logger.info(
        "Creating research plan: %s",
        question
    )

    pdf_available = is_pdf_available()

    try:

        # ----------------------------------
        # STEP 1: CALL LLM
        # ----------------------------------

        response = planner_chain.invoke({

            "question": question

        })

        content = response.content

        if isinstance(content, list):

            content = "".join(
                str(item.get("text", ""))
                if isinstance(item, dict)
                else str(item)
                for item in content
            )

        content = str(content).strip()

        if not content:
            raise ValueError(
                "LLM returned an empty response."
            )

        # ----------------------------------
        # STEP 2: REPAIR JSON
        # ----------------------------------

        data = repair_json(
            content,
            return_objects=True
        )

        if not isinstance(data, dict):
            raise ValueError(
                "LLM did not return a valid JSON object."
            )

        # ----------------------------------
        # STEP 3: NORMALIZE PLAN
        # ----------------------------------

        normalized_data = normalize_research_plan(
            data,
            question,
            pdf_available
        )

        # ----------------------------------
        # STEP 4: VALIDATE PLAN
        # ----------------------------------

        plan = ResearchPlan.model_validate(
            normalized_data
        )

        logger.info(
            "PLAN CREATED | PDF: %s | Web: %s | "
            "Subquestions: %d",
            plan.search_pdf,
            plan.search_web,
            len(plan.subquestions)
        )

        return plan

    except Exception as e:

        logger.warning(
            "LLM research planning failed: %s",
            e
        )

        logger.warning(
            "Switching to deterministic fallback plan."
        )

        # ----------------------------------
        # STEP 5: FALLBACK
        # ----------------------------------

        return create_fallback_plan(
            question,
            pdf_available
        )
