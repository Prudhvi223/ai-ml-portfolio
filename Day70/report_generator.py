
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_factory import get_chat_model


# ==========================================
# 1. LOGGING
# ==========================================

logger = logging.getLogger("research-assistant.report")


# ==========================================
# 2. INITIALIZE LLM
# ==========================================

llm = get_chat_model()


# ==========================================
# 3. RESEARCH REPORT PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template(
    """
You are an evidence-grounded AI research assistant.

Your task is to answer the user's exact research question
using only the evidence supplied below.

Accuracy, relevance and source attribution are more important
than report length.

==================================================
USER QUESTION
==================================================

{question}


==================================================
PDF EVIDENCE
==================================================

{rag_context}


==================================================
WEB EVIDENCE
==================================================

{web_context}


==================================================
AVAILABLE SOURCES
==================================================

{sources}


==================================================
1. STRICT EVIDENCE RULES
==================================================

Follow these rules throughout the entire report:

1. Use only information supported by the supplied evidence.

2. Never invent facts, statistics, dates, examples, findings,
   technical concepts, research developments or comparisons.

3. Do not introduce a concept simply because it is related
   to the user's question.

4. Do not treat a source title or URL as proof of a claim.
   The actual evidence must support the claim.

5. Do not use general knowledge to fill gaps in the evidence.

6. If information is missing, explicitly say:
   "The supplied evidence does not establish this."

7. Do not repeat the same information in different sections.

8. Do not write generic filler or unnecessary introductions.

9. Do not make unsupported claims about improvements,
   performance, accuracy, efficiency or advantages.

10. Do not claim that a source supports something it does not
    actually discuss.

==================================================
2. PDF AND WEB SEPARATION
==================================================

Keep PDF and web information clearly separated.

PDF FINDINGS:

- Use only information from PDF EVIDENCE.
- Mention the exact PDF filename and page number when available.
- Do not include web information in this section.
- Do not invent page numbers.

WEB FINDINGS:

- Use only information from WEB EVIDENCE.
- Include the source title and URL when supplied.
- Use publication dates only when explicitly provided.
- Do not describe undated information as the latest development.
- Do not include PDF information in this section.

COMPARISON:

- Compare PDF findings with web findings only when both
  sources provide evidence for the same topic.
- Clearly explain similarities and differences.
- Do not manufacture comparisons.
- If the evidence does not support a comparison, say so.

==================================================
3. CITATION RULES
==================================================

Every important factual statement must have supporting evidence.

For PDF information:

Use this format:

(Source: filename, Page X)

For web information:

Use this format:

(Source: article title, URL)

Important:

- Use only references present in AVAILABLE SOURCES.
- Do not invent URLs, titles, dates or page numbers.
- Do not cite unrelated sources.
- Place citations immediately after the relevant statements.
- Do not create fake references.
- If the exact supporting reference is unavailable,
  state that the claim cannot be verified.

==================================================
4. REPORT STRUCTURE
==================================================

Use the following structure when appropriate.

# Research Report

## 1. Direct Answer

Answer the user's question directly in a short paragraph.

Do not provide a generic introduction.

## 2. PDF Findings

Explain the important findings from the uploaded document.

Use numbered points when appropriate.

Every point must be supported by PDF evidence
and include its relevant page citation.

If no PDF evidence exists, state that clearly.

## 3. Web Findings

Explain the relevant findings from web research.

Include only information supported by web evidence.

Cite the actual sources.

If no web evidence exists, state that clearly.

## 4. Comparison and Analysis

Include this section only if the user asks for a comparison
or if comparison is necessary to answer the question.

Use a table when it makes the comparison clearer.

Compare only points supported by both evidence sets.

Do not introduce unrelated concepts.

## 5. Evidence Limitations

Mention only meaningful limitations.

Explain what information is missing or cannot be verified.

Do not invent limitations.

## 6. Conclusion

Summarize the answer in a few sentences.

Do not introduce new information.

## 7. References

List only the sources actually used.

Use the exact source details supplied.

==================================================
5. SPECIAL INSTRUCTIONS FOR RECENT DEVELOPMENTS
==================================================

When the user asks about recent developments:

- Prioritize relevant web evidence.
- Use dates only when explicitly supplied.
- Distinguish confirmed developments from general explanations.
- Do not call an old or undated article a 2026 development.
- Do not invent future trends or predictions.
- Do not assume that a development is new simply because
  it appears in a search result.

==================================================
6. WRITING QUALITY
==================================================

- Use clear and simple professional English.
- Explain technical concepts without changing their meaning.
- Avoid unnecessary repetition.
- Keep the report proportional to the question.
- Prefer concise explanations over lengthy generic content.
- Use headings and bullet points for readability.
- Never add unsupported information to make the report longer.

FINAL CHECK BEFORE ANSWERING:

Check every factual claim against the supplied evidence.

Remove any claim that is not supported.

Make sure PDF and web findings are separated.

Make sure citations are accurate.

Now answer the user's exact question.
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
# 5. FORMAT SOURCES
# ==========================================

def format_sources(sources: List[str]) -> str:

    unique_sources = list(
        dict.fromkeys(
            source.strip()
            for source in (sources or [])
            if isinstance(source, str)
            and source.strip()
        )
    )

    if not unique_sources:
        return "No source references were retrieved."

    return "\n".join(
        f"{index}. {source}"
        for index, source in enumerate(
            unique_sources,
            start=1
        )
    )


# ==========================================
# 6. GENERATE RESEARCH REPORT
# ==========================================

def generate_report(
    question: str,
    rag_context: str,
    web_context: str,
    sources: list
) -> str:

    # --------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------

    if not question or not question.strip():

        raise ValueError(
            "Research question cannot be empty."
        )

    question = question.strip()

    logger.info(
        "Generating report for: %s",
        question
    )

    # --------------------------------------
    # PREPARE PDF EVIDENCE
    # --------------------------------------

    pdf_evidence = (
        rag_context.strip()
        if rag_context and rag_context.strip()
        else "No relevant PDF evidence was retrieved."
    )

    # --------------------------------------
    # PREPARE WEB EVIDENCE
    # --------------------------------------

    external_evidence = (
        web_context.strip()
        if web_context and web_context.strip()
        else "No relevant web evidence was retrieved."
    )

    # --------------------------------------
    # PREPARE SOURCE REFERENCES
    # --------------------------------------

    formatted_sources = format_sources(
        sources
    )

    # --------------------------------------
    # GENERATE REPORT
    # --------------------------------------

    try:

        logger.info(
            "Sending evidence to the language model."
        )

        report = report_chain.invoke({

            "question": question,

            "rag_context": pdf_evidence,

            "web_context": external_evidence,

            "sources": formatted_sources

        })

        if not report or not report.strip():

            raise ValueError(
                "The language model returned an empty report."
            )

        report = report.strip()

        logger.info(
            "Research report generated successfully."
        )

        return report

    except Exception:

        logger.exception(
            "Research report generation failed."
        )

        raise
