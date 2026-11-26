# Day 65 – Tool Calling and ReAct Agent

## Objective
Understand how an LLM selects and uses external tools to perform tasks.

## Concepts Learned
- Tool and function calling
- ReAct: Reason, Act, Observe
- LLM tool selection
- Passing arguments to Python functions
- Executing tools and returning results to the LLM

## Hands-on Project: Calculator Agent

Built a calculator agent using Python and Ollama with the `llama3.2:1b` model.

### Available Operations
- Addition
- Subtraction
- Multiplication
- Division

### Workflow
1. User submits a mathematical question.
2. LLM decides whether to use the calculator.
3. LLM selects the calculator and provides arguments.
4. Python executes the calculator function.
5. The result is returned to the LLM.
6. LLM generates the final response.

## Technologies Used
- Python
- Ollama
- Llama 3.2 (1B)

## Example Output
- Input: What is 250 multiplied by 40?
- Selected operation: Multiply
- Arguments: 250 and 40
- Calculator result: 10000

## Outcome
Successfully implemented a basic tool-calling agent and understood the ReAct workflow: Reason → Act → Observe.
