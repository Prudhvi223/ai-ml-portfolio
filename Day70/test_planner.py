
from agents.planner import decide_sources

questions = [
    "What are the three common learning approaches in Machine Learning?",
    "According to the uploaded PDF, what are the three common learning approaches in Machine Learning?",
    "Explain this document.",
    "What are the latest developments in AI?",
    "Compare my PDF with recent external research."
]

for question in questions:

    pdf, web = decide_sources(question)

    print("\nQuestion:", question)
    print("PDF Search:", pdf)
    print("Web Search:", web)
