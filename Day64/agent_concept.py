
# DAY 64 - Understanding AI Agents

# Tool: Calculator
def calculator(a, b):
    return a + b


# Simple agent
def simple_agent(question):

    print("User Goal:", question)

    # 1. Reasoning
    print("\n1. Reasoning:")
    print("I need to calculate the total.")

    # 2. Planning
    print("\n2. Planning:")
    print("I will use the calculator tool.")

    # 3. Tool use
    print("\n3. Tool Execution:")
    result = calculator(100, 200)

    # 4. Observation
    print("\n4. Observation:")
    print("Calculator returned:", result)

    # Final answer
    print("\nFinal Answer:")
    print("Total =", result)


simple_agent("Calculate 100 + 200")
