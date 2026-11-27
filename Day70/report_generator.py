
import logging

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import LLM_MODEL


# ==========================================
# 1. LOGGING
# ==========================================

logger = logging.getLogger("research-assistant.report")


# ==========================================
# 2. INITIALIZE LLM
# ==========================================

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)


# ==========================================
# 3. RESEARCH REPORT PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template(
    """
You are a professional, evidence-grounded AI research assistant.

Your responsibility is to produce an accurate, relevant,
well-structured research report answering the user's exact question.

You must use the supplied evidence carefully.

==============================
USER QUESTION
==============================

{question}


==============================
PDF EVIDENCE
==============================

{rag_context}


==============================
WEB EVIDENCE
==============================

{web_context}


==============================
AVAILABLE SOURCES
==============================

{sources}


==============================
1. QUESTION UNDERSTANDING
==============================

First, understand exactly what the user is asking.

- Identify the main topic and the specific information requested.
- Answer the actual question instead of writing a generic article.
- Do not change the topic or broaden the scope unnecessarily.
- Give priority to information directly relevant to the question.

If the user asks for the latest, recent, or current developments:

- Prioritize recent developments found in the supplied web evidence.
- Use publication dates only when they are explicitly available.
- Distinguish dated information from information without dates.
- Do not describe an undated source as the latest development.
- Do not invent current trends, events, dates, or statistics.

==============================
2. EVIDENCE AND GROUNDING RULES
==============================

Follow these rules strictly:

1. Use only facts supported by the supplied PDF or web evidence.

2. Never use general knowledge to fill gaps in the evidence.

3. Do not invent definitions, examples, applications, statistics,
   technical details, comparisons, or research findings.

4. Ignore search results that are unrelated to the user's question.

5. Do not confuse a source mentioning a topic with a source
   actually providing evidence about that topic.

6. Preserve the meaning and terminology of the original sources.

7. If the PDF and web evidence disagree, clearly explain
   the disagreement and identify the supporting sources.

8. If the available evidence is insufficient for a particular
   part of the question, state exactly what is missing.

9. Do not claim that evidence is insufficient when the supplied
   evidence already answers the question.

10. Never pretend that a PDF or web search was performed
    if its corresponding evidence is empty.

11. Do not manufacture content just to make the report longer.

==============================
3. SOURCE AND CITATION RULES
==============================

- Every important factual claim must be traceable to supplied evidence.
- Use only the sources provided in AVAILABLE SOURCES.
- Use the exact source names, page numbers, and URLs supplied.
- Never invent a source, URL, publication date, or page number.
- Place citations close to the claims they support.
- Do not cite unrelated sources.
- Do not include a reference that was not supplied.
- Do not create fake numbered citations.

For PDF evidence, identify the document and page number.

For web evidence, use the supplied article URL and title when available.

If a claim has no supporting source, do not present it as a fact.

==============================
4. REPORT STRUCTURE
==============================

Use the following structure when appropriate.

# Title

Create a concise title that directly reflects the user's question.

# Introduction

Introduce the topic in a few sentences.
Do not write a generic introduction unrelated to the question.

# Key Findings

Present the most important evidence-based findings.

Use numbered points or bullets.

For questions asking for a definition, comparison, or explanation,
answer that request directly.

# Detailed Analysis

Explain the findings in a logical order.

- Answer the user's question directly.
- Explain technical concepts using the supplied evidence.
- Use comparison tables when the evidence supports a comparison.
- For recent developments, organize findings by date or topic
  when the supplied evidence supports doing so.
- Avoid repeating the same information.

# Evidence Limitations

Include this section only when there are meaningful limitations.

Explain:
- Which requested information is missing.
- Which details cannot be verified from the supplied evidence.
- Whether the available sources are outdated or undated,
  when that affects a request for recent information.

If there are no important limitations, do not invent any.

# Conclusion

Summarize the main answer in a short paragraph.

# References

List the actual sources used in the report.

Include exact PDF page references or web URLs when supplied.

Do not list unused sources.

==============================
5. WRITING QUALITY
==============================

- Use clear, professional English.
- Avoid unnecessary repetition.
- Avoid generic filler sentences.
- Explain technical terminology clearly.
- Keep the report proportional to the complexity of the question.
- Do not write a long report for a simple factual question.
- Do not use unsupported claims to make the report sound impressive.

FINAL REQUIREMENT:

Accuracy and relevance are more important than report length.

Answer the user's actual question using the available evidence.
"""
)


# ==========================================
# 4. REPORT GENERATION CHAIN
# ==========================================

report_chain = (
    prompt
    | llm
    | StrOutputParser()
)


# ==========================================
# 5. GENERATE RESEARCH REPORT
# ==========================================

def generate_report(
    question: str,
    rag_context: str,
    web_context: str,
    sources: list
) -> str:

    if not question or not question.strip():
        raise ValueError(
            "Research question cannot be empty."
        )

    logger.info(
        "Generating research report for: %s",
        question
    )

    # --------------------------------------
    # PREPARE PDF EVIDENCE
    # --------------------------------------

    pdf_evidence = (
        rag_context.strip()
        if rag_context and rag_context.strip()
        else "No PDF evidence was retrieved."
    )

    # --------------------------------------
    # PREPARE WEB EVIDENCE
    # --------------------------------------

    external_evidence = (
        web_context.strip()
        if web_context and web_context.strip()
        else "No web evidence was retrieved."
    )

    # --------------------------------------
    # PREPARE AVAILABLE SOURCES
    # --------------------------------------

    unique_sources = list(
        dict.fromkeys(
            source.strip()
            for source in (sources or [])
            if isinstance(source, str) and source.strip()
        )
    )

    formatted_sources = "\n".join(
        f"[{index}] {source}"
        for index, source in enumerate(
            unique_sources,
            start=1
        )
    )

    if not formatted_sources:
        formatted_sources = "No sources were retrieved."

    # --------------------------------------
    # GENERATE REPORT
    # --------------------------------------

    try:

        report = report_chain.invoke({

            "question": question.strip(),

            "rag_context": pdf_evidence,

            "web_context": external_evidence,

            "sources": formatted_sources

        })

        if not report or not report.strip():

            raise ValueError(
                "The language model returned an empty report."
            )

        logger.info(
            "Research report generated successfully."
        )

        return report.strip()

    except Exception:

        logger.exception(
            "Research report generation failed."
        )

        raise
