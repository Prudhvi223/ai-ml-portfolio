
import logging
import re

from ddgs import DDGS


# ==========================================
# 1. LOGGING
# ==========================================

logger = logging.getLogger("research-assistant.tools")


# ==========================================
# 2. SEARCH QUERY VARIATIONS
# ==========================================

def generate_query_variations(query: str) -> list[str]:

    query = query.strip()

    variations = [
        query,
        f"{query} explanation",
        f"{query} definition examples"
    ]

    # Remove duplicates while preserving order
    return list(dict.fromkeys(variations))


# ==========================================
# 3. SINGLE WEB SEARCH
# ==========================================

def web_search(
    query: str,
    max_results: int = 5
) -> list[dict]:

    if not query or not query.strip():
        return []

    max_results = max(1, min(max_results, 10))

    try:

        logger.info("Searching web: %s", query)

        with DDGS(timeout=15) as ddgs:

            results = ddgs.text(
                query,
                max_results=max_results,
                backend="auto"
            )

        formatted_results = []

        for result in results or []:

            title = result.get("title", "")
            body = result.get("body", "")
            url = result.get("href", "")

            if not title and not body:
                continue

            formatted_results.append({
                "title": title,
                "body": body,
                "url": url
            })

        logger.info(
            "Retrieved %d results for: %s",
            len(formatted_results),
            query
        )

        return formatted_results

    except Exception as e:

        logger.warning(
            "Search failed for '%s': %s",
            query,
            str(e)
        )

        return []


# ==========================================
# 4. SEARCH WITH FALLBACK QUERIES
# ==========================================

def search_with_fallback(
    query: str,
    max_results: int = 3
) -> list[dict]:

    variations = generate_query_variations(query)

    for variation in variations:

        logger.info(
            "Trying search variation: %s",
            variation
        )

        results = web_search(
            variation,
            max_results=max_results
        )

        if results:
            return results

    logger.warning(
        "All search variations failed for: %s",
        query
    )

    return []


# ==========================================
# 5. EXTRACT SUBQUESTION
# ==========================================

def extract_subquestion_title(item) -> str:

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):

        return str(
            item.get("title")
            or item.get("question")
            or ""
        ).strip()

    if hasattr(item, "title"):

        return str(item.title).strip()

    return ""


# ==========================================
# 6. MULTI-QUERY WEB SEARCH
# ==========================================

def multi_query_web_search(
    question: str,
    subquestions: list = None,
    max_results_per_query: int = 3
) -> list[dict]:

    if not question or not question.strip():
        return []

    if subquestions is None:
        subquestions = []

    queries = [question.strip()]

    # Add planner-generated subquestions
    for item in subquestions:

        title = extract_subquestion_title(item)

        if title and title not in queries:
            queries.append(title)

    all_results = []

    seen_urls = set()
    seen_content = set()

    logger.info(
        "Starting multi-query search with %d queries",
        len(queries)
    )

    for query in queries:

        results = search_with_fallback(
            query,
            max_results=max_results_per_query
        )

        for item in results:

            url = item.get("url", "").strip()

            content_key = (
                item.get("title", "").lower(),
                item.get("body", "").lower()
            )

            if url and url in seen_urls:
                continue

            if not url and content_key in seen_content:
                continue

            if url:
                seen_urls.add(url)

            seen_content.add(content_key)

            all_results.append(item)

    logger.info(
        "Web search completed: %d unique results",
        len(all_results)
    )

    return all_results
