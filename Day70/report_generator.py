
import logging
from typing import List, Any

from langchain_core.prompts import ChatPromptTemplate

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

Answer the user's exact research question using the evidence
provided below.

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

1. Use only information supported by the supplied evidence.
2. Never invent facts, statistics, dates, examples or findings.
3. Do not introduce unrelated concepts.
4. A source title or URL alone is not proof of a claim.
5. Do not use general knowledge to fill evidence gaps.
6. If information is missing, say:
   "The supplied evidence does not establish this."
7. Avoid repeating the same information.
8. Do not write generic filler.
9. Do not make unsupported claims about performance,
   accuracy, efficiency or advantages.
10. Do not claim that a source supports something it does not
    actually discuss.

==================================================
2. PDF AND WEB SEPARATION
==================================================

PDF FINDINGS:

- Use only information from PDF EVIDENCE.
- Mention the exact PDF filename and page number when available.
- Do not include web information in this section.
- Never invent page numbers.

WEB FINDINGS:

- Use only information from WEB EVIDENCE.
- Present findings as numbered points, not a Markdown table.
- Each point must contain a specific supported finding.
- Include the source title and exact URL when supplied.
- Include publication dates only when explicitly supplied.
- Put citations immediately after the relevant finding.
- Do not invent titles, URLs, dates or authors.
- If no usable URL is supplied, say "URL not supplied".
- Do not combine unrelated search results.
- Do not include PDF information in this section.

COMPARISON:

- Compare PDF and web findings only when both provide
  evidence about the same topic.
- Explain similarities and differences only when supported.
- Do not manufacture comparisons.

==================================================
3. CITATION RULES
==================================================

Every important factual statement must have supporting evidence.

For PDF information, use:

(Source: filename, Page X)

For web information, use:

(Source: article title, URL)

Important:

- Use only references present in AVAILABLE SOURCES.
- Do not invent URLs, titles, dates or page numbers.
- Do not cite unrelated sources.
- Put citations immediately after relevant statements.
- If an exact supporting reference is unavailable,
  state that the claim cannot be verified.

==================================================
4. REPORT STRUCTURE
==================================================

# Research Report

## 1. Direct Answer

Answer the user's question directly in a short paragraph.
Avoid a generic introduction.

## 2. PDF Findings

Explain important findings from the uploaded document.

Use numbered points where appropriate.

Every point must be supported by PDF evidence and include
its relevant page citation.

If no PDF evidence exists, state that clearly.

## 3. Web Findings

Present relevant web research as a numbered list.

For each finding:
- State the supported finding clearly.
- Include the source title and exact URL when supplied.
- Include publication dates only when supplied.
- Put the citation immediately after the finding.

Do not use a table in this section.

If no web evidence exists, state that clearly.

## 4. Comparison and Analysis

Include this section only if the user asks for comparison
or comparison is necessary to answer the question.

Compare only points supported by both evidence sets.

## 5. Evidence Limitations

Mention meaningful limitations.
Explain what information is missing or cannot be verified.

## 6. Conclusion

Summarize the answer in a few sentences.
Do not introduce new information.

## 7. References

List only sources actually used.
Use the exact source details supplied.

==================================================
5. RECENT DEVELOPMENTS
==================================================

When the user asks about recent developments:

- Prioritize relevant web evidence.
- Use dates only when explicitly supplied.
- Distinguish confirmed developments from general explanations.
- Do not call an old or undated article a recent development.
- Do not invent future trends or predictions.

==================================================
6. WRITING QUALITY
==================================================

- Use clear and simple professional English.
- Explain technical concepts without changing their meaning.
- Avoid unnecessary repetition.
- Keep the report proportional to the question.
- Prefer concise explanations over generic content.
- Use headings and bullet points for readability.
- Never add unsupported information to make the report longer.

FINAL CHECK:

- Check every factual claim against supplied evidence.
- Keep PDF and web findings separate.
- Check that citations are accurate.
- Ensure every web finding is a complete numbered point.
- Copy supplied URLs exactly.
- Remove incomplete sentences and empty list items.

IMPORTANT OUTPUT REQUIREMENT:

You must return a non-empty research report.
If the evidence is insufficient, explain the limitation instead
of returning an empty response.

Now answer the user's exact question.
"""
)


# ==========================================
# 4. FORMAT SOURCES
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
# 5. EXTRACT TEXT FROM MODEL RESPONSE
# ==========================================

def extract_response_text(response: Any) -> str:
    """
    Extract text safely from a LangChain AIMessage.

    Supports:
    - Normal string content
    - List of text blocks
    - Dictionary-based content blocks
    """

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, str):

                text_parts.append(block)

            elif isinstance(block, dict):

                block_type = block.get("type")

                if block_type in ("text", None):

                    block_text = block.get("text")

                    if isinstance(block_text, str):
                        text_parts.append(block_text)

            else:

                block_text = getattr(block, "text", None)

                if isinstance(block_text, str):
                    text_parts.append(block_text)

        return "\n".join(text_parts).strip()

    if content is None:
        return ""

    return str(content).strip()


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
        if isinstance(rag_context, str) and rag_context.strip()
        else "No relevant PDF evidence was retrieved."
    )

    # --------------------------------------
    # PREPARE WEB EVIDENCE
    # --------------------------------------

    external_evidence = (
        web_context.strip()
        if isinstance(web_context, str) and web_context.strip()
        else "No relevant web evidence was retrieved."
    )

    # --------------------------------------
    # PREPARE SOURCE REFERENCES
    # --------------------------------------

    formatted_sources = format_sources(sources)

    # --------------------------------------
    # PREPARE PROMPT MESSAGES
    # --------------------------------------

    try:

        messages = prompt.format_messages(

            question=question,

            rag_context=pdf_evidence,

            web_context=external_evidence,

            sources=formatted_sources

        )

        logger.info(
            "Prompt prepared. PDF evidence length=%d, "
            "Web evidence length=%d, Sources=%d",
            len(pdf_evidence),
            len(external_evidence),
            len(sources or [])
        )

        # ----------------------------------
        # CALL MODEL DIRECTLY
        # ----------------------------------

        logger.info(
            "Sending report-generation request to LLM."
        )

        response = llm.invoke(messages)

        # ----------------------------------
        # LOG RAW RESPONSE DETAILS
        # ----------------------------------

        response_metadata = getattr(
            response,
            "response_metadata",
            {}
        )

        additional_kwargs = getattr(
            response,
            "additional_kwargs",
            {}
        )

        logger.info(
            "RAW RESPONSE TYPE: %s",
            type(response).__name__
        )

        logger.info(
            "RAW RESPONSE METADATA: %s",
            response_metadata
        )

        logger.info(
            "RAW RESPONSE ADDITIONAL KWARGS: %s",
            additional_kwargs
        )

        # ----------------------------------
        # EXTRACT RESPONSE TEXT
        # ----------------------------------

        report = extract_response_text(response)

        logger.info(
            "EXTRACTED REPORT LENGTH: %d",
            len(report)
        )

        if not report:

            logger.error(
                "LLM returned empty content. "
                "Response metadata=%s",
                response_metadata
            )

            raise ValueError(
                "The language model returned empty content. "
                "Check the model response metadata and finish reason."
            )

        # ----------------------------------
        # FINAL VALIDATION
        # ----------------------------------

        report = report.strip()

        if len(report) < 10:

            logger.warning(
                "LLM returned an unusually short report: %r",
                report
            )

        logger.info(
            "Research report generated successfully. "
            "Characters=%d",
            len(report)
        )

        return report

    except Exception:

        logger.exception(
            "Research report generation failed."
        )

        raise
