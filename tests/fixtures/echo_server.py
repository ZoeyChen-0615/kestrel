"""A tiny stdio MCP server used by the MCP client tests."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("echo")


@server.tool()
def shout(text: str) -> str:
    """Upper-case the given text."""
    return text.upper()


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    server.run()
