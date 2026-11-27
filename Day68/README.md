# Day 68 – CrewAI Multi-Agent System

## Overview
Built a multi-agent AI system using CrewAI and Ollama. Three AI agents collaborate to research, write, and review a report about Artificial Intelligence.

## Technologies Used
- Python
- CrewAI
- Ollama
- Llama 3.2:1b

## Project Structure
```text
Day68/
├── crewai_demo.py
├── requirements.txt
└── README.md
```

## Agents

### 1. Researcher
- Role: Researcher
- Finds information about the given topic.
- Produces a structured research summary.

### 2. Writer
- Role: Technical Writer
- Receives the researcher's output as context.
- Creates a clear report with an introduction, applications, and conclusion.

### 3. Reviewer
- Role: Reviewer
- Receives the writer's report as context.
- Checks the report and improves its clarity and accuracy.

## Workflow

The project uses sequential execution:

Researcher → Writer → Reviewer → Final Report

CrewAI manages the agents, tasks, and information flow.

## Key Concepts Learned
- Creating AI agents using `Agent`
- Assigning tasks using `Task`
- Passing task outputs through `context`
- Managing multiple agents using `Crew`
- Sequential execution using `Process.sequential`
- Connecting a local LLM through Ollama
- Executing workflows using `crew.kickoff()`

## How to Run

1. Install the dependencies:

```bash
pip install -r Day68/requirements.txt
```

2. Make sure Ollama is running and the model is available:

```bash
ollama pull llama3.2:1b
```

3. Run the project from the repository root:

```bash
python Day68/crewai_demo.py
```

## Outcome
Successfully created a CrewAI multi-agent system in which a Researcher, Writer, and Reviewer work together to generate a final AI report using a local Ollama model.