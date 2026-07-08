import asyncio
import os 
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerStdioParams
from setup_openai import model 

SANDBOX = os.path.dirname(os.path.abspath(__file__))
SCRIPT = Path(__file__).with_name("01_research_tools_mcp_server.py").resolve()

async def main():
    # Init servers 
    servers = [
        MCPServerStdio(
            name="Research Tools", 
            params=MCPServerStdioParams(
                command="mcp", 
                args=["run", str(SCRIPT)]
            )
        ), 
        MCPServerStdio(
            name="sequential-thinking", 
            params={
                "command": "npx", 
                "args": [
                    "-y", 
                    "@modelcontextprotocol/server-sequential-thinking"
                ]
            }
        ), 
        MCPServerStdio(
            name="filesystem", 
            params={
                "command": "npx", 
                "args": [
                    "-y", 
                    "@modelcontextprotocol/server-filesystem", SANDBOX
                ]
            }
        )
    ]

    instructions = """
    You are a research assistant who can use tools to perform and plan research.
    Given a research goal, use the research tools to find research sources.
    Then, use the sequential thinking tool to plan the research.
    Finally, use the filesystem tool to write the research plan as a text file.
    """
    
    async with (
        servers[0] as research_srv, 
        servers[1] as thinking_srv, 
        servers[2] as fs_srv, 
    ): 
        agent = Agent(
            model=model, 
            name="Assistant", 
            instructions=instructions, 
            mcp_servers=[research_srv, thinking_srv, fs_srv]
        )
        goal = """Produce a research plan to find the book 'The Hitchhiker's Guide to the Galaxy' """
        
        print("Running...", goal)
        result = await Runner.run(agent, goal)
        print(result.final_output)
        

if __name__ == "__main__":
    asyncio.run(main())

