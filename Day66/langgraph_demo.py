
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# 1. Define the state
class AgentState(TypedDict):
    question: str
    answer: str


# 2. Create the first node
def question_node(state: AgentState):

    print("Question received:", state["question"])

    return {}


# 3. Create the second node
def answer_node(state: AgentState):

    answer = (
        "Machine Learning is a branch of AI "
        "that learns patterns from data."
    )

    return {"answer": answer}


# 4. Create the graph
graph_builder = StateGraph(AgentState)


# 5. Add nodes
graph_builder.add_node("question", question_node)
graph_builder.add_node("answer", answer_node)


# 6. Connect nodes using edges
graph_builder.add_edge(START, "question")
graph_builder.add_edge("question", "answer")
graph_builder.add_edge("answer", END)


# 7. Add checkpointing
memory = InMemorySaver()


# 8. Compile the graph
app = graph_builder.compile(checkpointer=memory)


# 9. Run the graph
config = {
    "configurable": {
        "thread_id": "user_1"
    }
}

result = app.invoke(
    {
        "question": "What is Machine Learning?",
        "answer": ""
    },
    config=config
)


# 10. Display the result
print("\nFinal Answer:")
print(result["answer"])
