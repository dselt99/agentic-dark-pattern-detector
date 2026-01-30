"""Agent module for orchestrating dark pattern detection audits."""

from .core import DarkPatternAgent
from .mcp_client import MCPClient

__all__ = ["DarkPatternAgent", "MCPClient"]
