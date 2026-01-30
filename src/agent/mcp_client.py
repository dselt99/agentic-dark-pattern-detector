"""MCP Client wrapper for agent tool calls.

This module provides a client interface for the agent to interact with
the MCP server tools without directly importing the server module.
"""

import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

# Import MCP server tools directly
# In a production setup, this would connect via stdio/HTTP to the MCP server
from ..mcp.server import (
    browser_navigate,
    get_accessibility_tree,
    take_screenshot,
    maps_topology,
)


class MCPClient:
    """Client wrapper for MCP server tools."""

    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call an MCP tool by name.

        Args:
            tool_name: Name of the tool to call.
            **kwargs: Arguments to pass to the tool.

        Returns:
            Tool response dictionary.
        """
        tool_map = {
            "browser_navigate": browser_navigate,
            "get_accessibility_tree": get_accessibility_tree,
            "take_screenshot": take_screenshot,
            "maps_topology": maps_topology,
        }

        if tool_name not in tool_map:
            return {
                "status": "error",
                "message": f"Unknown tool: {tool_name}",
            }

        try:
            tool_func = tool_map[tool_name]
            result = await tool_func(**kwargs)
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": f"Tool execution error: {str(e)}",
            }
