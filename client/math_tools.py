"""
math_tools.py — Direct Calculator Tools (No MCP Subprocess)
==============================================================
Replaces the MCP-based math_server.py subprocess with direct
LangChain @tool functions.

These are simple pure functions — no external dependencies needed.
"""

from langchain_core.tools import tool


@tool
def calculator_add(a: int, b: int) -> int:
    """Add two integers and return the sum.

    Use this tool when the user asks for basic arithmetic addition.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b


@tool
def calculator_multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product.

    Use this tool when the user asks for multiplication.

    Args:
        a: First number
        b: Second number

    Returns:
        The product of a and b
    """
    return a * b
