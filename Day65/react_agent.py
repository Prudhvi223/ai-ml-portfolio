
import ollama


# TOOL: Calculator function

def calculator(a, b, operation):

    # Convert string inputs into numbers
    a = float(a)
    b = float(b)

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":
        if b == 0:
            return "Cannot divide by zero"
        return a / b


# Describe the tool to the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operation": {
                        "type": "string",
                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide"
                        ]
                    }
                },
                "required": ["a", "b", "operation"]
            }
        }
    }
]


# User question
question = "What is 250 multiplied by 40?"

print("User:", question)


# REASON: Ask LLM to choose a tool
response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": question
        }
    ],
    tools=tools
)


# ACT: Check whether the LLM requested a tool
message = response["message"]

if message.get("tool_calls"):

    for tool_call in message["tool_calls"]:

        function_name = tool_call["function"]["name"]

        arguments = tool_call["function"]["arguments"]

        if function_name == "calculator":

            print("\nTool selected:", function_name)
            print("Arguments:", arguments)

            # Execute the Python function
            result = calculator(
                arguments["a"],
                arguments["b"],
                arguments["operation"]
            )

            print("\nTool result:", result)

            # OBSERVE: Send tool result back to LLM
            final_response = ollama.chat(
                model="llama3.2:1b",
                messages=[
                    {
                        "role": "user",
                        "content": question
                    },
                    message,
                    {
                        "role": "tool",
                        "content": str(result)
                    }
                ]
            )

            print("\nFinal Answer:")
            print(final_response["message"]["content"])

else:
    print("\nLLM did not call a tool.")
    print("LLM response:", message["content"])
