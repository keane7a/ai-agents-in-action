from mcp.server.fastmcp import FastMCP

# Create MCP server 
mcp = FastMCP("Research Tools")

@mcp.tool()
def get_research_tools() -> list[str]: 
    """Provides a list of research resources"""
    search_sources = [
        "Wikipedia", 
        "Google", 
        "YouTube",
    ]
    
    return search_sources