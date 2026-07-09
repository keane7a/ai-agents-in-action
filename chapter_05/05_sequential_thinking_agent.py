import asyncio
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from setup_openai import model


async def main():
    # Initi server
    thinking_srv = MCPServerStdio(
        name="sequential-thinking",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        },
    )

    instructions = """You are a helpful planning assistant"""

    agent = Agent(
        model=model,
        name="Assistant",
        instructions=instructions,
        mcp_servers=[thinking_srv],
    )

    async with thinking_srv:
        tools = await thinking_srv.list_tools()
        print("Available tools:", tools)

        goal = """Discover and output the tool and functions you have available"""

        print("Running...", goal)
        res = await Runner.run(agent, goal)
        print(res.final_output)


if __name__ == "__main__":
    asyncio.run(main())
