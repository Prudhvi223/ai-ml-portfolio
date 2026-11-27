
from graph import research_graph


questions = [

    "Explain the concepts in my uploaded PDF.",

    "What are the latest developments in agentic AI?",

    "Compare RAG and Agentic RAG using my PDF and external research."

]


for index, question in enumerate(questions, start=1):

    print("\n" + "=" * 60)
    print(f"TEST {index}")
    print("QUESTION:", question)

    initial_state = {
        "question": question,
        "rag_context": "",
        "web_context": "",
        "sources": [],
        "errors": []
    }

    try:

        result = research_graph.invoke(initial_state)

        print("\nSELECTED SOURCES:")
        print("PDF:", result.get("search_pdf"))
        print("Web:", result.get("search_web"))

        print("\nFINAL REPORT:")
        print(result.get("report", "No report generated"))

        print("\nSOURCES:")
        for source in result.get("sources", []):
            print("-", source)

        print("\nERRORS:")
        for error in result.get("errors", []):
            print("-", error)

    except Exception as e:

        print("\nGRAPH EXECUTION FAILED:")
        print(e)
