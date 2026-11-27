# Day 67 – Building a LangGraph Agent

## Objective
Build an AI agent using LangGraph that can decide which tool to use based on the user's question.

## Technologies Used
- Python
- LangGraph
- LangChain
- Ollama (Llama 3.2 1B)
- DuckDuckGo Search
- Python Calculator

## Tools Implemented

### 1. Web Search
Created a web search tool using DuckDuckGo to retrieve information from the internet.

### 2. Calculator
Created a calculator tool that supports:
- Addition
- Subtraction
- Multiplication
- Division

## Agent Workflow
1. Receive a question from the user.
2. The LLM analyses the question.
3. The agent chooses an appropriate tool.
4. The selected tool executes the task.
5. The tool result is returned to the LLM.
6. The LLM generates the final answer.

## Key Concepts Learned
- ReAct agents
- Tool calling
- Tool registration using `@tool`
- `create_react_agent()`
- Agent execution using `invoke()`
- Recursion limits

## Outcome
Successfully built a LangGraph ReAct agent with web search and calculator tools. The agent can select tools based on the user's request and generate a final response.

