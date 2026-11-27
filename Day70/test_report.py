
from report_generator import generate_report

question = "What is Artificial Intelligence?"

rag_context = """
Artificial Intelligence is a field of computing
focused on building systems that perform tasks
commonly associated with human intelligence.
"""

web_context = """
IBM describes AI as technology that enables
computers and machines to simulate human learning,
comprehension, problem-solving and decision-making.
"""

# Only actual sources used in this test
sources = [
    "Day61_RAG_Sample_AI_ML.pdf, Page 1",
    "https://www.ibm.com/think/topics/artificial-intelligence"
]

report = generate_report(
    question,
    rag_context,
    web_context,
    sources
)

print("\nFINAL RESEARCH REPORT\n")
print(report)
