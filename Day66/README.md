# Day 66 – LangGraph Basics

## Objective
Understand the fundamentals of LangGraph and build a simple state-based workflow.

## Concepts Learned
- LangGraph
- State and TypedDict
- Nodes and edges
- START and END
- Graph compilation
- Checkpointing
- InMemorySaver

## Hands-on Project
Created a basic LangGraph workflow using Python.

### Workflow
START → Question Node → Answer Node → END

### Implementation
1. Defined the agent state using TypedDict.
2. Created a question node.
3. Created an answer node.
4. Connected nodes using add_edge().
5. Added checkpointing using InMemorySaver.
6. Compiled the graph.
7. Executed the workflow using invoke().

## Technologies Used
- Python
- LangGraph

## Outcome
Successfully created and executed a basic LangGraph workflow with state management and in-memory checkpointing.
