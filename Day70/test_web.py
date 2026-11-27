
from tools import web_search, multi_query_web_search

print("\n========== TEST 1: SINGLE WEB SEARCH ==========")

results = web_search(
    "What is a GPU?",
    max_results=3
)

for result in results:

    print("\nTitle:", result["title"])
    print("Content:", result["body"])
    print("URL:", result["url"])


print("\n========== TEST 2: MULTI-QUERY SEARCH ==========")

results = multi_query_web_search(
    question="What are the latest AI developments?",
    subquestions=[],
    max_results_per_query=3
)

print("\nTotal results:", len(results))

for result in results:

    print("\nTitle:", result["title"])
    print("URL:", result["url"])


if results:
    print("\nWEB SEARCH TEST PASSED")
else:
    print("\nWEB SEARCH RETURNED NO RESULTS")
