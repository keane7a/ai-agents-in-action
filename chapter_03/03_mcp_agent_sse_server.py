import asyncio
from pathlib import Path
from agents import Agent, Runner
from agents.mcp import MCPServerSse
from setup_openai import model

SCRIPT = Path(__file__).with_name("01_claude_mcp_server.py").resolve()


async def main():
    async with MCPServerSse(
        name="SSE Python server", params={"url": "http://127.0.0.1:8000/sse"}
    ) as research_server:
        agent = Agent(
            model=model,
            name="Assistant",
            instructions="Use the research tools to peform research",
            mcp_servers=[research_server],
        )

        print("Running: Get the available research sources")

        result = await Runner.run(agent, "Get the available research sources")

        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
