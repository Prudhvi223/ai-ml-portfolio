
import logging

from langgraph.graph import StateGraph, START, END

from state import ResearchState
from agents.planner import create_research_plan, decide_sources
from rag import RAGRetriever
from report_generator import generate_report
from tools import multi_query_web_search


# ==========================================
# 1. LOGGING
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research-assistant")


# ==========================================
# 2. INITIALIZE RETRIEVER
# ==========================================

retriever = RAGRetriever()


def refresh_retriever():
    """Reload uploaded PDF documents from ChromaDB."""

    global retriever

    retriever = RAGRetriever()

    logger.info("RAG retriever refreshed successfully")


# ==========================================
# NODE 1: RESEARCH PLANNER
# ==========================================

def planner_node(state: ResearchState):

    question = state["question"]

    logger.info("Creating research plan for: %s", question)

    # IMPORTANT:
    # Source selection is deterministic.
    # It does not depend on LLM output.

    search_pdf, search_web = decide_sources(question)

    research_plan = {}
    errors = []

    try:
        plan = create_research_plan(question)
        research_plan = plan.model_dump()

    except Exception as e:

        logger.exception("Research plan generation failed")

        errors.append(
            f"Research planning warning: {str(e)}"
        )

        # Keep the workflow working even if
        # the LLM cannot generate subquestions.
        research_plan = {
            "research_goal": question,
            "subquestions": []
        }

    logger.info(
        "Source selection: PDF=%s, Web=%s",
        search_pdf,
        search_web
    )

    return {
        "search_pdf": search_pdf,
        "search_web": search_web,
        "research_plan": research_plan,
        "errors": errors
    }


# ==========================================
# NODE 2: CONDITIONAL ROUTER
# ==========================================

def route_research(state: ResearchState):

    search_pdf = bool(state.get("search_pdf", False))
    search_web = bool(state.get("search_web", False))

    if search_pdf and search_web:

        logger.info("Routing to combined PDF and web research")
        return "both"

    if search_pdf:

        logger.info("Routing to PDF retrieval")
        return "rag"

    if search_web:

        logger.info("Routing to web research")
        return "web"

    logger.warning("No research source selected")
    return "synthesis"


# ==========================================
# NODE 3: PDF RETRIEVAL
# ==========================================

def rag_node(state: ResearchState):

    logger.info("Starting PDF retrieval")

    question = state["question"]

    try:

        results = retriever.retrieve(question)

        if not results:
            raise ValueError(
                "No relevant PDF chunks were retrieved."
            )

        context = []
        sources = []

        for item in results:

            source_name = item.get(
                "source",
                "Unknown source"
            )

            page = item.get(
                "page",
                "Unknown page"
            )

            evidence = (
                f"Source: {source_name}\n"
                f"Page: {page}\n"
                f"Evidence:\n{item['text']}"
            )

            context.append(evidence)

            source = f"{source_name}, Page {page}"

            if source not in sources:
                sources.append(source)

        logger.info(
            "PDF retrieval completed: %d chunks",
            len(results)
        )

        return {
            "rag_context": "\n\n".join(context),
            "rag_sources": sources,
            "rag_errors": []
        }

    except Exception as e:

        logger.exception("PDF retrieval failed")

        return {
            "rag_context": "",
            "rag_sources": [],
            "rag_errors": [
                f"PDF retrieval failed: {str(e)}"
            ]
        }


# ==========================================
# NODE 4: WEB RESEARCH
# ==========================================

def web_node(state: ResearchState):

    logger.info("Starting web research")

    question = state["question"]

    plan = state.get("research_plan", {})

    subquestions = plan.get("subquestions", [])

    try:

        results = multi_query_web_search(
            question=question,
            subquestions=subquestions,
            max_results_per_query=3
        )

        context = []
        sources = []

        for item in results:

            title = item.get("title", "")
            body = item.get("body", "")
            url = item.get("url", "")

            context.append(
                f"Title: {title}\n"
                f"Content: {body}\n"
                f"URL: {url}"
            )

            if url and url not in sources:
                sources.append(url)

        logger.info(
            "Web research completed: %d results",
            len(results)
        )

        return {
            "web_context": "\n\n".join(context),
            "web_sources": sources,
            "web_status": (
                "success" if results else "no_results"
            ),
            "web_errors": []
        }

    except Exception as e:

        logger.exception("Web research failed")

        return {
            "web_context": "",
            "web_sources": [],
            "web_status": "failed",
            "web_errors": [
                f"Web search failed: {str(e)}"
            ]
        }


# ==========================================
# NODE 5: COMBINED RETRIEVAL
# ==========================================

def both_node(state: ResearchState):

    logger.info("Starting combined PDF and web retrieval")

    pdf_result = rag_node(state)
    web_result = web_node(state)

    return {
        "rag_context": pdf_result.get("rag_context", ""),
        "rag_sources": pdf_result.get("rag_sources", []),
        "rag_errors": pdf_result.get("rag_errors", []),

        "web_context": web_result.get("web_context", ""),
        "web_sources": web_result.get("web_sources", []),
        "web_status": web_result.get("web_status", ""),
        "web_errors": web_result.get("web_errors", [])
    }


# ==========================================
# NODE 6: REPORT GENERATION
# ==========================================

def synthesis_node(state: ResearchState):

    logger.info("Starting report generation")

    rag_context = state.get("rag_context", "")
    web_context = state.get("web_context", "")

    rag_sources = state.get("rag_sources", [])
    web_sources = state.get("web_sources", [])

    sources = list(dict.fromkeys(
        rag_sources + web_sources
    ))

    errors = (
        state.get("errors", [])
        + state.get("rag_errors", [])
        + state.get("web_errors", [])
    )

    try:

        if not rag_context and not web_context:

            report = (
                "## Research could not be completed\n\n"
                "No usable evidence was retrieved from the "
                "selected research sources.\n\n"
                "Please check the uploaded PDF or web search "
                "connection and try again."
            )

        else:

            report = generate_report(
                question=state["question"],
                rag_context=rag_context,
                web_context=web_context,
                sources=sources
            )

        logger.info("Report generation completed")

        return {
            "report": report,
            "sources": sources,
            "errors": errors
        }

    except Exception as e:

        logger.exception("Report generation failed")

        return {
            "report": "Unable to generate the research report.",
            "sources": sources,
            "errors": errors + [
                f"Report generation failed: {str(e)}"
            ]
        }


# ==========================================
# 7. BUILD LANGGRAPH WORKFLOW
# ==========================================

builder = StateGraph(ResearchState)

builder.add_node("planner", planner_node)
builder.add_node("rag", rag_node)
builder.add_node("web", web_node)
builder.add_node("both", both_node)
builder.add_node("synthesis", synthesis_node)


# Entry point
builder.add_edge(START, "planner")


# Conditional routing
builder.add_conditional_edges(
    "planner",
    route_research,
    {
        "rag": "rag",
        "web": "web",
        "both": "both",
        "synthesis": "synthesis"
    }
)


# Retrieval to synthesis
builder.add_edge("rag", "synthesis")
builder.add_edge("web", "synthesis")
builder.add_edge("both", "synthesis")


# End
builder.add_edge("synthesis", END)


# ==========================================
# 8. COMPILE GRAPH
# ==========================================

research_graph = builder.compile()

logger.info(
    "Research Assistant LangGraph workflow compiled successfully"
)
