
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    # Tell the client how to start the MCP server.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["Day69/server.py"]
    )

    # Establish the connection.
    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize the MCP connection.
            await session.initialize()

            print("Connected to MCP Server!")

            # Discover available tools.
            tools = await session.list_tools()

            print("\nAvailable Tools:")

            for tool in tools.tools:
                print("-", tool.name)

            # Call the addition tool.
            result1 = await session.call_tool(
                "add_numbers",
                arguments={"a": 10, "b": 20}
            )

            print("\nAddition Result:")
            print(result1.content[0].text)

            # Call the multiplication tool.
            result2 = await session.call_tool(
                "multiply_numbers",
                arguments={"a": 5, "b": 4}
            )

            print("\nMultiplication Result:")
            print(result2.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
