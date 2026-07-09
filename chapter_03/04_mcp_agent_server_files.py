import asyncio
import os
import sys

from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from setup_openai import model


async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    async with MCPServerStdio(
        name="Filesymstem Server, via npx",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", current_dir],
        },
    ) as server:
        # Create an agent that usees the MCP server
        agent = Agent(
            model=model,
            name="Filesystem Agent",
            instructions="Use the filesystem tools to help the user with their tasks",
            mcp_servers=[server],
        )

        print("Running: Get the available files")
        result = await Runner.run(agent, "List all file names in the current directory")
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
