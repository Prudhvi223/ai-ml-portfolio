
import logging
import re

import chromadb

from json_repair import repair_json
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from config import (
    LLM_MODEL,
    CHROMA_DB_PATH,
    UPLOAD_COLLECTION_NAME
)


# ==========================================
# 1. LOGGING
# ==========================================

logger = logging.getLogger(__name__)


# ==========================================
# 2. SUBQUESTION MODEL
# ==========================================

class SubQuestion(BaseModel):

    title: str = Field(
        description="Short title of the research question"
    )

    description: str = Field(
        default="",
        description="Explanation of what should be researched"
    )


# ==========================================
# 3. RESEARCH PLAN MODEL
# ==========================================

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
# 4. INITIALIZE LOCAL LLM
# ==========================================

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)


# ==========================================
# 5. PLANNING PROMPT
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
# 6. CHECK PDF AVAILABILITY
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
# 7. SOURCE SELECTION POLICY
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

    # Check ChromaDB if availability was not supplied
    if pdf_available is None:

        pdf_available = is_pdf_available()

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
    # DETECT SOURCE INTENT
    # --------------------------------------

    explicit_pdf = any(
        re.search(pattern, q)
        for pattern in pdf_patterns
    )

    explicit_web = any(
        re.search(pattern, q)
        for pattern in web_patterns
    )

    both_requested = (
        explicit_pdf and explicit_web
    )

    # --------------------------------------
    # FINAL SOURCE DECISION
    # --------------------------------------

    if both_requested:

        search_pdf = pdf_available
        search_web = True

    elif explicit_web:

        search_pdf = (
            explicit_pdf and pdf_available
        )

        search_web = True

    elif explicit_pdf:

        if pdf_available:

            search_pdf = True
            search_web = False

        else:

            search_pdf = False
            search_web = True

    elif pdf_available:

        # PDF IS THE DEFAULT SOURCE
        # FOR NORMAL QUESTIONS

        search_pdf = True
        search_web = False

    else:

        # No uploaded PDF:
        # Use web search

        search_pdf = False
        search_web = True

    logger.info(
        "SOURCE SELECTION | Question: %s | PDF available: %s | PDF: %s | Web: %s",
        question,
        pdf_available,
        search_pdf,
        search_web
    )

    return search_pdf, search_web


# ==========================================
# 8. NORMALIZE LLM OUTPUT
# ==========================================

def normalize_research_plan(
    data,
    question,
    pdf_available=None
):

    # Source decisions are controlled by Python
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

    # --------------------------------------
    # PROCESS SUBQUESTIONS
    # --------------------------------------

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

    # --------------------------------------
    # FALLBACK SUBQUESTION
    # --------------------------------------

    if not unique_questions:

        unique_questions = [{

            "title": question,

            "description": (
                "Investigate the user's research question."
            )

        }]

    # --------------------------------------
    # RESEARCH GOAL
    # --------------------------------------

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
# 9. CREATE RESEARCH PLAN
# ==========================================

def create_research_plan(
    question: str
) -> ResearchPlan:

    if not question or not question.strip():

        raise ValueError(
            "Research question cannot be empty."
        )

    logger.info(
        "Creating research plan: %s",
        question
    )

    # --------------------------------------
    # STEP 1: CHECK PDF AVAILABILITY
    # --------------------------------------

    pdf_available = is_pdf_available()

    # --------------------------------------
    # STEP 2: GENERATE RESEARCH PLAN
    # --------------------------------------

    response = planner_chain.invoke({

        "question": question

    })

    content = response.content.strip()

    try:

        # ----------------------------------
        # STEP 3: REPAIR JSON
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
        # STEP 4: NORMALIZE OUTPUT
        # ----------------------------------

        normalized_data = normalize_research_plan(
            data,
            question,
            pdf_available
        )

        # ----------------------------------
        # STEP 5: PYDANTIC VALIDATION
        # ----------------------------------

        plan = ResearchPlan.model_validate(
            normalized_data
        )

        logger.info(
            "PLAN CREATED | PDF: %s | Web: %s | Subquestions: %d",
            plan.search_pdf,
            plan.search_web,
            len(plan.subquestions)
        )

        return plan

    except Exception as e:

        logger.exception(
            "Research planning failed"
        )

        raise ValueError(
            f"Could not create a valid research plan: {e}"
        ) from e
