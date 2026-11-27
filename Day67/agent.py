
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from ddgs import DDGS


# =====================================
# TOOL 1: WEB SEARCH
# =====================================

@tool
def web_search(query: str) -> str:
    """Search the internet for current information about a topic."""

    try:
        results = DDGS().text(query, max_results=5)

        if not results:
            return "No search results found."

        output = []

        for item in results:
            output.append(
                f"Title: {item.get('title', '')}\n"
                f"Description: {item.get('body', '')}\n"
                f"URL: {item.get('href', '')}"
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Search error: {str(e)}"


# =====================================
# TOOL 2: CALCULATOR
# =====================================

@tool
def calculator(a: float, b: float, operation: str) -> str:
    """
    Perform mathematical calculations.

    For 500 divided by 25:
    a=500, b=25, operation='divide'.

    Always calculate a operation b.
    """

    a = float(a)
    b = float(b)

    operation = operation.lower().strip()

    # Normalize operation names
    if operation in ["addition", "plus"]:
        operation = "add"

    elif operation in ["subtraction", "minus"]:
        operation = "subtract"

    elif operation in ["multiplication", "times"]:
        operation = "multiply"

    elif operation in ["division", "divided by"]:
        operation = "divide"

    print(f"\nCalculator called: {a} {operation} {b}")

    if operation == "add":
        return str(a + b)

    elif operation == "subtract":
        return str(a - b)

    elif operation == "multiply":
        return str(a * b)

    elif operation == "divide":

        if b == 0:
            return "Cannot divide by zero."

        return str(a / b)

    return f"Unsupported operation: {operation}"


# =====================================
# LOAD LOCAL LLM
# =====================================

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)


# =====================================
# REGISTER TOOLS
# =====================================

tools = [
    web_search,
    calculator
]


# =====================================
# CREATE LANGGRAPH AGENT
# =====================================

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=(
        "You are a helpful AI assistant. "
        "Choose web_search for internet searches. "
        "Choose calculator for mathematical calculations. "
        "For division, use operation='divide'. "
        "After receiving a successful tool result, "
        "explain that result directly to the user. "
        "Do not call the same tool repeatedly for the same task."
    )
)


# =====================================
# RUN AGENT
# =====================================

question = input("\nAsk your agent: ")

try:

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config={
            "recursion_limit": 8
        }
    )

    # =================================
    # DISPLAY TOOL USAGE
    # =================================

    print("\nTools used:")

    tool_names = []

    for message in result["messages"]:

        for call in getattr(message, "tool_calls", []):
            tool_names.append(call["name"])

    if tool_names:

        for name in tool_names:
            print("-", name)

    else:
        print("No tool was called.")

    # =================================
    # DISPLAY FINAL ANSWER
    # =================================

    print("\nFinal Answer:")

    print(result["messages"][-1].content)

except Exception as e:

    print("\nAgent error:", str(e))
