# Day 69 – Model Context Protocol (MCP)

## Overview
Built a simple MCP client-server application using Python and the Model Context Protocol (MCP) SDK. The server exposes mathematical tools, and the client discovers and executes those tools through an MCP connection.

## Technologies Used
- Python
- MCP Python SDK
- FastMCP
- Asyncio
- Logging

## Project Structure
```text
Day69/
├── server.py
├── client.py
├── requirements.txt
└── README.md
```

## MCP Server Tools

1. **add_numbers** – Adds two numbers.
2. **multiply_numbers** – Multiplies two numbers.

## Key Concepts Learned
- Model Context Protocol (MCP)
- MCP Client and Server
- FastMCP
- Tool registration using `@mcp.tool()`
- Tool discovery using `list_tools()`
- Tool execution using `call_tool()`
- Stdio communication
- Asynchronous programming using asyncio
- Basic input validation and logging
- Multi-agent design patterns
- Guardrails and observability

## How to Run

Activate the Python environment:

```powershell
.\.venv68\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r Day69/requirements.txt
```

Run the client:

```powershell
python Day69/client.py
```

The client automatically starts the MCP server, discovers its tools, and executes addition and multiplication operations.

## Expected Output

```text
Available Tools:
- add_numbers
- multiply_numbers

Addition Result:
30.0

Multiplication Result:
20.0
```

## Outcome
Successfully implemented an MCP client-server application that demonstrates standardized tool discovery, communication, and execution using the Model Context Protocol.