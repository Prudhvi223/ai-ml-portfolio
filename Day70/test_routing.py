
from agents.planner import decide_sources


test_cases = [

    {
        "question": "Explain the concepts in my uploaded PDF.",
        "expected_pdf": True,
        "expected_web": False
    },

    {
        "question": "What are the latest developments in agentic AI?",
        "expected_pdf": False,
        "expected_web": True
    },

    {
        "question": "Compare RAG and Agentic RAG using my PDF and external research.",
        "expected_pdf": True,
        "expected_web": True
    },

    {
        "question": "Explain machine learning generally.",
        "expected_pdf": False,
        "expected_web": True
    }

]


print("=" * 60)
print("SOURCE SELECTION TESTS")
print("=" * 60)

passed = 0

for index, test in enumerate(test_cases, start=1):

    pdf, web = decide_sources(test["question"])

    success = (
        pdf == test["expected_pdf"]
        and web == test["expected_web"]
    )

    if success:
        passed += 1

    print(f"\nTest {index}")
    print("Question:", test["question"])
    print("PDF:", pdf)
    print("Web:", web)
    print("Result:", "PASS" if success else "FAIL")


print("\n" + "=" * 60)
print(f"RESULT: {passed}/{len(test_cases)} TESTS PASSED")
print("=" * 60)
