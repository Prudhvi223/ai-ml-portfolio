
from mcp.server.fastmcp import FastMCP
import logging
import math

# Logs go to stderr, keeping stdout available for MCP communication.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("math-server")

mcp = FastMCP("Math MCP Server")


@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    logger.info("Adding %s and %s", a, b)

    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Inputs must be finite numbers.")

    return a + b


@mcp.tool()
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    logger.info("Multiplying %s and %s", a, b)

    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Inputs must be finite numbers.")

    return a * b


if __name__ == "__main__":
    logger.info("Starting Math MCP Server...")
    mcp.run(transport="stdio")
