
from typing import TypedDict


class ResearchState(TypedDict, total=False):

    question: str

    search_pdf: bool
    search_web: bool

    research_plan: dict

    rag_context: str
    web_context: str

    rag_sources: list
    web_sources: list

    rag_errors: list
    web_errors: list

    web_status: str

    sources: list
    errors: list

    report: str
