"""MCP Client wrapper for agent tool calls.

This module provides a client interface for the agent to interact with
the MCP server tools without directly importing the server module.
"""

import asyncio
import logging
import random
from typing import Dict, Any, Optional, Callable, TypeVar
from pathlib import Path
from functools import wraps

# Import MCP server tools and session management
from ..mcp.server import (
    browser_navigate,
    browser_click,
    browser_reload,
    get_accessibility_tree,
    deep_scan_element,
    get_page_url,
    take_screenshot,
    maps_topology,
    get_session_manager,
    cleanup_session,
)

logger = logging.getLogger(__name__)

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 30.0  # seconds
DEFAULT_EXPONENTIAL_BASE = 2

# Errors that are worth retrying
RETRYABLE_ERROR_PATTERNS = [
    "timeout",
    "timed out",
    "connection refused",
    "connection reset",
    "network",
    "temporarily unavailable",
    "rate limit",
    "too many requests",
    "503",
    "502",
    "504",
    "ECONNRESET",
    "ETIMEDOUT",
    "ENOTFOUND",
]


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is worth retrying.

    Args:
        error: The exception that occurred.

    Returns:
        True if the error is transient and worth retrying.
    """
    error_str = str(error).lower()
    return any(pattern.lower() in error_str for pattern in RETRYABLE_ERROR_PATTERNS)


async def retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    **kwargs,
) -> Any:
    """Execute a function with exponential backoff retry logic.

    Args:
        func: Async function to execute.
        *args: Positional arguments for the function.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff calculation.
        **kwargs: Keyword arguments for the function.

    Returns:
        The function's return value.

    Raises:
        The last exception if all retries fail.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            # Check if error is retryable
            if not is_retryable_error(e):
                logger.debug(f"Non-retryable error: {e}")
                raise

            # Check if we have retries left
            if attempt >= max_retries:
                logger.warning(
                    f"All {max_retries} retries exhausted. Last error: {e}"
                )
                raise

            # Calculate delay with exponential backoff and jitter
            delay = min(
                base_delay * (exponential_base ** attempt),
                max_delay
            )
            # Add jitter (0-25% of delay)
            jitter = delay * random.uniform(0, 0.25)
            actual_delay = delay + jitter

            logger.info(
                f"Retry {attempt + 1}/{max_retries} after {actual_delay:.2f}s. "
                f"Error: {e}"
            )
            await asyncio.sleep(actual_delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


class MCPClient:
    """Client wrapper for MCP server tools with session management and retry logic."""

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ):
        """Initialize the MCP client.

        Args:
            max_retries: Maximum retry attempts for transient failures.
            base_delay: Initial delay between retries in seconds.
            max_delay: Maximum delay between retries in seconds.
        """
        self._session_id: Optional[str] = None
        self._session_manager = get_session_manager()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    async def start_session(self) -> str:
        """Start a new isolated browser session.

        Returns:
            Session ID for this audit.
        """
        self._session_id = self._session_manager.create_session()
        session = self._session_manager.get_session(self._session_id)
        await session.initialize()
        logger.info(f"MCPClient started session: {self._session_id}")
        return self._session_id

    async def end_session(self) -> None:
        """End the current session and cleanup resources."""
        if self._session_id:
            await cleanup_session(self._session_id)
            logger.info(f"MCPClient ended session: {self._session_id}")
            self._session_id = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures cleanup."""
        await self.end_session()
        return False  # Don't suppress exceptions

    async def call_tool(
        self,
        tool_name: str,
        retry: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Call an MCP tool by name with automatic retry for transient failures.

        Args:
            tool_name: Name of the tool to call.
            retry: Whether to retry on transient failures (default True).
            **kwargs: Arguments to pass to the tool.

        Returns:
            Tool response dictionary.
        """
        tool_map = {
            "browser_navigate": browser_navigate,
            "browser_click": browser_click,
            "browser_reload": browser_reload,
            "get_accessibility_tree": get_accessibility_tree,
            "deep_scan_element": deep_scan_element,
            "get_page_url": get_page_url,
            "take_screenshot": take_screenshot,
            "maps_topology": maps_topology,
        }

        if tool_name not in tool_map:
            return {
                "status": "error",
                "message": f"Unknown tool: {tool_name}",
            }

        tool_func = tool_map[tool_name]

        async def _execute():
            return await tool_func(**kwargs)

        try:
            if retry:
                result = await retry_with_backoff(
                    _execute,
                    max_retries=self._max_retries,
                    base_delay=self._base_delay,
                    max_delay=self._max_delay,
                )
            else:
                result = await _execute()
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed after retries: {e}")
            return {
                "status": "error",
                "message": f"Tool execution error: {str(e)}",
                "retried": retry,
            }
