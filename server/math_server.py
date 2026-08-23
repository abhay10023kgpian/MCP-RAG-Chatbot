"""
math_server.py — Calculator MCP Tool Server
=============================================
Exposes basic math operations as FastMCP tools.
Used as a demo/test MCP server to verify multi-server connectivity.

Based on: tempcodes/server/server.py
Run standalone:  python server/math_server.py
"""

from fastmcp import FastMCP

# ─── Initialize FastMCP Server ───
mcp = FastMCP("math_tools_server")


@mcp.tool()
def calculator_add(a: int, b: int) -> int:
    """
    Add two integers and return the sum.
    
    Use this tool when the user asks for basic arithmetic addition.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    return a + b


@mcp.tool()
def calculator_multiply(a: int, b: int) -> int:
    """
    Multiply two integers and return the product.
    
    Use this tool when the user asks for multiplication.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The product of a and b
    """
    return a * b


if __name__ == "__main__":
    print("🔧 Starting Math Tools Server...")
    mcp.run()
