# Day 64 – Agentic AI Concepts

## Overview

This day introduces the fundamentals of Agentic AI and explains how AI agents differ from traditional Large Language Model (LLM) applications.

An AI agent can understand a goal, plan steps, use tools, observe results, and produce an outcome.

## Concepts Learned

### 1. AI Agent
An AI agent is a system that uses an LLM to reason about a goal and perform actions to achieve it.

### 2. LLM vs AI Agent
- **LLM:** Primarily generates responses based on user input.
- **AI Agent:** Can perform multiple steps, use tools, and make decisions based on observations.

### 3. Four Components of an AI Agent

- **Reasoning:** Understands the user's goal and decides what needs to be done.
- **Planning:** Breaks a complex task into smaller steps.
- **Tool Use:** Uses external functions such as calculators, web search, and Python.
- **Memory:** Retains useful information for future steps or interactions.

## Hands-on Implementation

Created a simple Python agent simulation using a calculator function.

The program demonstrates:

1. Understanding the user's goal.
2. Reasoning about the task.
3. Planning to use a calculator.
4. Executing the calculator function.
5. Observing the result.
6. Producing the final answer.

## Project Structure

```text
Day64/
├── agent_concept.py
├── requirements.txt
└── README.md
```

## Run the Program

```bash
python Day64/agent_concept.py
```

## Learning Outcomes

- Understood the fundamentals of Agentic AI.
- Learned the difference between a single LLM call and an AI agent.
- Understood reasoning, planning, tool use, and memory.
- Implemented a basic agent workflow in Python.

## Conclusion

Successfully completed the introduction to Agentic AI and implemented a simple Python-based agent workflow. This provides the foundation for tool calling and ReAct agents in Day 65.