"""MCP Server for browser automation capabilities.

This module implements the "Dumb Tools" philosophy - deterministic, stateless
interactions with the browser via Playwright. The server acts as the "hands"
of the agent, executing commands without reasoning.
"""

import os
import re
import time
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import yaml
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright
from mcp.server.fastmcp import FastMCP


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Dark Pattern Hunter - Browser Automation")

# Tool definitions for Anthropic's tool use API
TOOL_DEFINITIONS = [
    {
        "name": "browser_reload",
        "description": "Reloads the current page. Use this to test for False Urgency patterns - if countdown timers reset to their original values after reload, this indicates fake urgency.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "browser_click",
        "description": "Clicks an element on the page. Use this to test interaction flows like signup vs. cancellation paths (Roach Motel), or to add items to cart (Sneak into Basket).",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the element to click",
                },
            },
            "required": ["selector"],
        },
    },
    {
        "name": "get_accessibility_tree",
        "description": "Returns a semantic YAML representation of the current page structure. Use this after navigation or clicks to see the updated page state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector to root the tree at. Defaults to 'body'.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth to traverse (default 15, max 50). Use higher values (20-30) when investigating nested UI like accordion menus, settings pages, or multi-step cancellation flows where Roach Motel patterns might be hidden.",
                    "minimum": 1,
                    "maximum": 50,
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "If true, include hidden elements (display:none). Useful for finding pre-checked options or hidden form fields (Sneak into Basket detection).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "deep_scan_element",
        "description": "Perform a deep scan of a specific element to find nested dark patterns. Use this when you suspect a Roach Motel pattern (easy signup, hard cancellation) might be hiding cancellation options deep in nested menus, accordions, or modal dialogs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the element to deeply scan (e.g., '#account-settings', '.subscription-panel').",
                },
            },
            "required": ["selector"],
        },
    },
    {
        "name": "take_screenshot",
        "description": "Captures visual evidence of the current page or a specific element.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for element to capture. If omitted, captures full viewport.",
                },
                "filename_prefix": {
                    "type": "string",
                    "description": "Prefix for saved file (e.g., 'false_urgency_evidence').",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_page_url",
        "description": "Returns the current page URL. Useful for tracking navigation state.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "submit_audit_result",
        "description": "Submit the final audit result. Call this when you have finished analyzing the page for all dark pattern types and are ready to report your findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "description": "List of detected dark patterns",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern_type": {
                                "type": "string",
                                "enum": ["roach_motel", "false_urgency", "confirmshaming", "sneak_into_basket", "forced_continuity"],
                            },
                            "confidence_score": {
                                "type": "number",
                                "minimum": 0.7,
                                "maximum": 1.0,
                            },
                            "element_selector": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["pattern_type", "confidence_score", "element_selector", "reasoning", "evidence"],
                    },
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary of the audit findings",
                },
            },
            "required": ["findings", "summary"],
        },
    },
]


class BrowserSession:
    """Isolated browser session for a single audit."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize browser resources."""
        if self._initialized:
            return

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
        )
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="DarkPatternHunter/1.0 (Research Bot; +https://github.com/dark-pattern-hunter)",
        )
        self.page = await self.context.new_page()
        self._initialized = True
        logger.info(f"Session {self.session_id}: Browser initialized")

    async def cleanup(self) -> None:
        """Cleanup browser resources."""
        errors = []
        if self.page:
            try:
                await self.page.close()
            except Exception as e:
                errors.append(f"page: {e}")
            self.page = None

        if self.context:
            try:
                await self.context.close()
            except Exception as e:
                errors.append(f"context: {e}")
            self.context = None

        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                errors.append(f"browser: {e}")
            self.browser = None

        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as e:
                errors.append(f"playwright: {e}")
            self.playwright = None

        self._initialized = False
        if errors:
            logger.warning(f"Session {self.session_id}: Cleanup errors: {errors}")
        else:
            logger.info(f"Session {self.session_id}: Cleaned up successfully")


class SessionManager:
    """Manages browser sessions with automatic cleanup."""

    def __init__(self):
        self._sessions: Dict[str, BrowserSession] = {}
        self._current_session_id: Optional[str] = None

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())[:8]
        self._sessions[session_id] = BrowserSession(session_id)
        self._current_session_id = session_id
        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: Optional[str] = None) -> Optional[BrowserSession]:
        """Get a session by ID, or the current session."""
        sid = session_id or self._current_session_id
        if sid is None:
            return None
        return self._sessions.get(sid)

    async def close_session(self, session_id: str) -> None:
        """Close and remove a session."""
        session = self._sessions.pop(session_id, None)
        if session:
            await session.cleanup()
            if self._current_session_id == session_id:
                self._current_session_id = None

    async def close_all(self) -> None:
        """Close all sessions."""
        for sid in list(self._sessions.keys()):
            await self.close_session(sid)

    @asynccontextmanager
    async def session_context(self):
        """Context manager for automatic session lifecycle."""
        session_id = self.create_session()
        session = self.get_session(session_id)
        try:
            await session.initialize()
            yield session
        finally:
            await self.close_session(session_id)


# Global session manager
_session_manager = SessionManager()

async def list_tools() -> list[dict[str, Any]]:
    """Return available tool schemas (MCP tools/list)."""
    return TOOL_DEFINITIONS

async def get_browser() -> tuple[Browser, BrowserContext, Page]:
    """Get browser from current session, creating one if needed."""
    session = _session_manager.get_session()

    # Auto-create session for backward compatibility
    if session is None:
        session_id = _session_manager.create_session()
        session = _session_manager.get_session(session_id)

    if not session._initialized:
        await session.initialize()

    return session.browser, session.context, session.page


def get_session_manager() -> SessionManager:
    """Get the global session manager."""
    return _session_manager


def check_robots_txt(url: str, fail_open: bool = False) -> tuple[bool, Optional[str]]:
    """Check if the URL is allowed by robots.txt.

    Args:
        url: The URL to check.
        fail_open: If True, allow access on parse errors (legacy behavior).
                   If False (default), deny access on parse errors (fail closed).

    Returns:
        Tuple of (is_allowed, error_message).
        If allowed, returns (True, None).
        If disallowed, returns (False, error_message).
    """
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        # Fetch robots.txt
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()

        # Check if our user agent is allowed
        user_agent = "DarkPatternHunter"
        path = parsed.path or "/"

        if not rp.can_fetch(user_agent, url):
            return False, f"robots.txt disallows {user_agent} from {path}"

        return True, None
    except FileNotFoundError:
        # No robots.txt means no restrictions
        logger.info(f"No robots.txt found for {parsed.netloc} - allowing access")
        return True, None
    except Exception as e:
        # Parse error or fetch error
        error_msg = f"robots.txt check failed for {url}: {e}"
        if fail_open:
            logger.warning(f"{error_msg} - allowing access (fail_open=True)")
            return True, None
        else:
            logger.warning(f"{error_msg} - denying access (fail_closed)")
            return False, f"robots.txt check failed: {e}. Use fail_open=True to override."


@mcp.tool()
async def browser_navigate(url: str) -> dict:
    """Navigates the active tab to a specified URL. Enforces robots.txt compliance.

    Args:
        url: The fully qualified URL to visit.

    Returns:
        Dictionary with status and message.
    """
    try:
        # Check robots.txt compliance (Responsible Auditor)
        is_allowed, error_msg = check_robots_txt(url)
        if not is_allowed:
            raise PermissionError(error_msg)

        # Get browser instance
        _, _, page = await get_browser()

        # Navigate with error handling
        timeout_ms = int(os.getenv("BROWSER_TIMEOUT_MS", "30000"))
        try:
            response = await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            if response is None:
                return {
                    "status": "error",
                    "message": "Navigation failed: No response received",
                }

            status_code = response.status
            if status_code >= 400:
                return {
                    "status": "error",
                    "message": f"HTTP {status_code}: {response.status_text}",
                }

            return {
                "status": "success",
                "message": f"Navigated to {url}",
                "status_code": status_code,
            }
        except Exception as nav_error:
            error_type = type(nav_error).__name__
            return {
                "status": "error",
                "message": f"Navigation failed ({error_type}): {str(nav_error)}",
            }

    except PermissionError as e:
        return {
            "status": "error",
            "message": f"Robots.txt violation: {str(e)}",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
        }


# Default and maximum depth limits
DEFAULT_TREE_DEPTH = 15
MAX_TREE_DEPTH = 50
DEEP_SCAN_DEPTH = 30  # For targeted deep scans


@mcp.tool()
async def get_accessibility_tree(
    selector: str = "body",
    max_depth: int = DEFAULT_TREE_DEPTH,
    include_hidden: bool = False,
) -> dict:
    """Returns a semantic YAML representation of the current page structure.

    Use this to understand the page content without visual noise.
    Extracts semantic information from the page including text, roles, and structure.

    Args:
        selector: CSS selector to root the tree at. Defaults to 'body'.
        max_depth: Maximum depth to traverse (default 15, max 50).
                   Use higher values (20-30) when investigating nested UI
                   like accordion menus, modals, or multi-step forms where
                   dark patterns might be hidden deep in the DOM.
        include_hidden: If True, include elements with display:none or
                        visibility:hidden. Useful for finding hidden form
                        fields or pre-checked options.

    Returns:
        Dictionary with the YAML representation of the page structure.
    """
    try:
        _, _, page = await get_browser()

        # Clamp depth to safe bounds
        effective_depth = max(1, min(max_depth, MAX_TREE_DEPTH))
        if max_depth != effective_depth:
            logger.warning(f"Depth {max_depth} clamped to {effective_depth}")

        # Extract semantic structure using JavaScript
        extract_script = """
        (args) => {
            const { selector, maxDepth, includeHidden } = args;
            let maxDepthReached = 0;
            let nodesProcessed = 0;
            let nodesTruncated = 0;

            function extractNode(element, depth = 0) {
                if (!element) return null;

                // Track max depth reached
                if (depth > maxDepthReached) maxDepthReached = depth;
                nodesProcessed++;

                // Depth limit check
                if (depth > maxDepth) {
                    nodesTruncated++;
                    return { tag: '...', text: '[depth limit reached]', _truncated: true };
                }

                const result = {};

                // Get tag and role
                result.tag = element.tagName?.toLowerCase() || 'text';
                const role = element.getAttribute?.('role');
                if (role) result.role = role;

                // Get text content (direct text, not children)
                const directText = Array.from(element.childNodes)
                    .filter(n => n.nodeType === Node.TEXT_NODE)
                    .map(n => n.textContent.trim())
                    .filter(t => t)
                    .join(' ');
                if (directText) result.text = directText.substring(0, 200);

                // Get important attributes
                const attrs = {};
                if (element.id) attrs.id = element.id;
                if (element.className && typeof element.className === 'string') {
                    attrs.class = element.className.split(' ')[0];
                }
                if (element.href) attrs.href = element.href.substring(0, 100);
                if (element.type) attrs.type = element.type;
                if (element.value) attrs.value = element.value.substring(0, 100);
                if (element.checked !== undefined) attrs.checked = element.checked;
                if (element.disabled) attrs.disabled = true;
                if (element.name) attrs.name = element.name;

                // Get aria attributes
                const ariaLabel = element.getAttribute?.('aria-label');
                if (ariaLabel) attrs['aria-label'] = ariaLabel;
                const ariaHidden = element.getAttribute?.('aria-hidden');
                if (ariaHidden === 'true') attrs['aria-hidden'] = true;
                const ariaExpanded = element.getAttribute?.('aria-expanded');
                if (ariaExpanded) attrs['aria-expanded'] = ariaExpanded;

                // Check for pre-checked inputs (sneak into basket detection)
                if (element.tagName?.toLowerCase() === 'input') {
                    const inputType = element.type?.toLowerCase();
                    if (inputType === 'checkbox' || inputType === 'radio') {
                        attrs.checked = element.checked;
                        if (element.defaultChecked) attrs.defaultChecked = true;
                    }
                }

                if (Object.keys(attrs).length > 0) result.attrs = attrs;

                // Get children (skip script, style, hidden elements unless requested)
                const skipTags = ['script', 'style', 'noscript', 'svg', 'path'];
                const children = Array.from(element.children || [])
                    .filter(child => {
                        if (skipTags.includes(child.tagName?.toLowerCase())) return false;
                        if (!includeHidden) {
                            try {
                                const style = window.getComputedStyle(child);
                                if (style.display === 'none' || style.visibility === 'hidden') {
                                    return false;
                                }
                            } catch (e) {
                                // getComputedStyle can fail on some elements
                            }
                        }
                        return true;
                    })
                    .map(child => extractNode(child, depth + 1))
                    .filter(Boolean);

                if (children.length > 0) result.children = children;

                return result;
            }

            const root = selector === 'body'
                ? document.body
                : document.querySelector(selector);

            if (!root) return { error: 'Selector not found: ' + selector };

            const tree = extractNode(root);

            return {
                tree: tree,
                stats: {
                    maxDepthReached: maxDepthReached,
                    nodesProcessed: nodesProcessed,
                    nodesTruncated: nodesTruncated,
                    requestedDepth: maxDepth
                }
            };
        }
        """

        result = await page.evaluate(
            extract_script,
            {
                "selector": selector,
                "maxDepth": effective_depth,
                "includeHidden": include_hidden,
            }
        )

        if result and result.get("error"):
            return {
                "status": "error",
                "message": result["error"],
            }

        structure = result.get("tree", {})
        stats = result.get("stats", {})

        # Convert to YAML string
        yaml_str = yaml.dump(structure, default_flow_style=False, allow_unicode=True)

        response = {
            "status": "success",
            "tree": yaml_str,
            "selector": selector,
            "depth_used": effective_depth,
            "max_depth_reached": stats.get("maxDepthReached", 0),
            "nodes_processed": stats.get("nodesProcessed", 0),
        }

        # Warn if depth limit was hit
        if stats.get("nodesTruncated", 0) > 0:
            response["warning"] = (
                f"Depth limit ({effective_depth}) reached at {stats['nodesTruncated']} nodes. "
                "Consider using a higher max_depth to see nested content."
            )
            logger.warning(
                f"Tree extraction hit depth limit: {stats['nodesTruncated']} nodes truncated"
            )

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting accessibility tree: {str(e)}",
        }


@mcp.tool()
async def deep_scan_element(selector: str) -> dict:
    """Perform a deep scan of a specific element to find nested content.

    This is specifically designed for detecting Roach Motel patterns where
    cancellation or unsubscribe options are buried deep in nested UI structures
    like accordion menus, modal dialogs, or multi-step settings pages.

    Uses a higher depth limit (30) and includes hidden elements.

    Args:
        selector: CSS selector for the element to deeply scan.

    Returns:
        Dictionary with deep tree structure and analysis hints.
    """
    try:
        _, _, page = await get_browser()

        # Check if element exists first
        element = await page.query_selector(selector)
        if not element:
            return {
                "status": "error",
                "message": f"Element not found: {selector}",
            }

        # Get the deep tree
        result = await get_accessibility_tree(
            selector=selector,
            max_depth=DEEP_SCAN_DEPTH,
            include_hidden=True,
        )

        if result.get("status") != "success":
            return result

        # Analyze for patterns that might indicate hidden options
        tree_text = result.get("tree", "").lower()

        hints = []

        # Look for cancellation-related terms
        cancel_terms = ["cancel", "unsubscribe", "deactivate", "close account", "delete account", "opt out", "remove"]
        found_cancel_terms = [term for term in cancel_terms if term in tree_text]
        if found_cancel_terms:
            hints.append(f"Found cancellation-related terms: {found_cancel_terms}")

        # Look for hidden elements that were revealed
        if "aria-hidden" in tree_text or "display: none" in tree_text:
            hints.append("Found hidden elements - may contain obscured options")

        # Look for collapsed/expandable sections
        if "aria-expanded" in tree_text:
            hints.append("Found expandable sections - check if important options are collapsed by default")

        # Look for multi-step indicators
        step_terms = ["step", "next", "continue", "proceed"]
        if any(term in tree_text for term in step_terms):
            hints.append("Found multi-step flow indicators - may be a Roach Motel pattern")

        result["deep_scan"] = True
        result["analysis_hints"] = hints
        result["selector_scanned"] = selector

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in deep scan: {str(e)}",
        }


@mcp.tool()
async def take_screenshot(
    selector: Optional[str] = None, filename_prefix: Optional[str] = None
) -> dict:
    """Captures visual evidence of the current page state or a specific element.

    Args:
        selector: CSS selector for the element to capture. If omitted, captures full viewport.
        filename_prefix: Prefix for the saved file (e.g., 'roach_motel_evidence').

    Returns:
        Dictionary with status and relative path to saved screenshot.
    """
    try:
        _, _, page = await get_browser()

        # Ensure artifacts directory exists
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = int(time.time())
        prefix = filename_prefix or "screenshot"
        safe_prefix = re.sub(r'[^\w\-_]', '_', prefix)
        filename = f"{safe_prefix}_{timestamp}.png"
        filepath = artifacts_dir / filename

        # Take screenshot
        if selector:
            try:
                element = await page.query_selector(selector)
                if not element:
                    return {
                        "status": "error",
                        "message": f"Selector '{selector}' not found",
                    }
                await element.screenshot(path=str(filepath))
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error capturing element '{selector}': {str(e)}",
                }
        else:
            await page.screenshot(path=str(filepath))

        # Return relative path
        relative_path = f"artifacts/{filename}"

        return {
            "status": "success",
            "path": relative_path,
            "message": f"Screenshot saved to {relative_path}",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error taking screenshot: {str(e)}",
        }


@mcp.tool()
async def browser_click(selector: str) -> dict:
    """Clicks an element on the page. Use this to test interaction flows.

    Useful for:
    - Testing Roach Motel: Click through signup vs. cancellation flows
    - Testing Sneak into Basket: Click "Add to Cart" and check what gets added
    - Navigating multi-step processes

    Args:
        selector: CSS selector for the element to click.

    Returns:
        Dictionary with status and any navigation/state changes detected.
    """
    try:
        _, _, page = await get_browser()

        # Find the element
        element = await page.query_selector(selector)
        if not element:
            return {
                "status": "error",
                "message": f"Element not found: {selector}",
            }

        # Get element info before click
        element_text = await element.text_content()
        element_tag = await element.evaluate("el => el.tagName.toLowerCase()")

        # Get current URL before click
        url_before = page.url

        # Click the element
        await element.click()

        # Wait for potential navigation or state change
        await page.wait_for_load_state("networkidle", timeout=5000)

        # Get URL after click
        url_after = page.url
        navigated = url_before != url_after

        return {
            "status": "success",
            "message": f"Clicked {element_tag}: '{element_text[:50] if element_text else 'no text'}'",
            "navigated": navigated,
            "url_before": url_before,
            "url_after": url_after,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error clicking element: {str(e)}",
        }


@mcp.tool()
async def get_page_url() -> dict:
    """Returns the current page URL. Useful for tracking navigation state."""
    try:
        _, _, page = await get_browser()
        return {
            "status": "success",
            "url": page.url,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting URL: {str(e)}",
        }


@mcp.tool()
async def browser_reload() -> dict:
    """Reloads the current page. Use this to test for False Urgency patterns.

    After reload, check if countdown timers reset to their original values,
    which indicates artificial/fake urgency rather than genuine time limits.

    Returns:
        Dictionary with status, message, and any timer values detected.
    """
    try:
        _, _, page = await get_browser()

        # Capture any visible timers/countdowns before reload
        timer_script = """
        () => {
            const timerPatterns = [
                /\\d{1,2}:\\d{2}(:\\d{2})?/,  // 05:00 or 05:00:00
                /\\d+\\s*(hours?|minutes?|seconds?|hrs?|mins?|secs?)/i,
            ];

            const timers = [];
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );

            let node;
            while (node = walker.nextNode()) {
                const text = node.textContent.trim();
                for (const pattern of timerPatterns) {
                    const match = text.match(pattern);
                    if (match) {
                        timers.push({
                            text: text.substring(0, 100),
                            match: match[0]
                        });
                        break;
                    }
                }
            }
            return timers.slice(0, 10);  // Limit to 10 timers
        }
        """

        timers_before = await page.evaluate(timer_script)

        # Reload the page
        await page.reload(wait_until="networkidle")

        # Capture timers after reload
        timers_after = await page.evaluate(timer_script)

        return {
            "status": "success",
            "message": "Page reloaded successfully",
            "timers_before": timers_before,
            "timers_after": timers_after,
            "hint": "If timers reset to original values, this indicates False Urgency",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reloading page: {str(e)}",
        }


@mcp.tool()
async def maps_topology() -> dict:
    """Maps the page's interactive surface area by scanning for interactive elements.

    Groups elements by region (Header, Main, Footer) to help identify
    navigation patterns and "Roach Motel" characteristics.

    Returns:
        Dictionary with topology grouped by region.
    """
    try:
        _, _, page = await get_browser()

        # JavaScript to scan for interactive elements by region
        topology_script = """
        () => {
            const result = {
                header: [],
                main: [],
                footer: []
            };

            // Helper to determine region
            function getRegion(element) {
                const rect = element.getBoundingClientRect();
                const viewportHeight = window.innerHeight;
                const top = rect.top;
                const bottom = rect.bottom;
                const middle = viewportHeight / 2;

                // If element is in top 20% of viewport, consider it header
                if (top < viewportHeight * 0.2) return 'header';
                // If element is in bottom 20% of viewport, consider it footer
                if (bottom > viewportHeight * 0.8) return 'footer';
                // Otherwise, main content
                return 'main';
            }

            // Find all interactive elements
            const selectors = [
                'a[href]',
                'button',
                'input[type="button"]',
                'input[type="submit"]',
                'input[type="checkbox"]',
                'input[type="radio"]',
                'select',
                '[role="button"]',
                '[role="link"]',
                '[tabindex]:not([tabindex="-1"])'
            ];

            const allElements = new Set();
            selectors.forEach(selector => {
                document.querySelectorAll(selector).forEach(el => {
                    if (el.offsetParent !== null) { // Only visible elements
                        allElements.add(el);
                    }
                });
            });

            // Group by region
            allElements.forEach(element => {
                const region = getRegion(element);
                const text = element.textContent?.trim() || element.value || element.alt || '';
                const tag = element.tagName.toLowerCase();
                const type = element.type || tag;
                const href = element.href || element.getAttribute('href') || '';

                result[region].push({
                    tag: tag,
                    type: type,
                    text: text.substring(0, 100), // Limit text length
                    href: href ? href.substring(0, 200) : '', // Limit URL length
                    selector: element.id ? `#${element.id}` : 
                             element.className ? `.${element.className.split(' ')[0]}` : 
                             tag
                });
            });

            return result;
        }
        """

        topology = await page.evaluate(topology_script)

        return {
            "status": "success",
            "topology": topology,
            "message": "Topology mapped successfully",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error mapping topology: {str(e)}",
        }


@mcp.tool()
async def browser_type(selector: str, text: str) -> dict:
    """Types text into an input field or textarea.

    Args:
        selector: CSS selector for the input element.
        text: Text to type.

    Returns:
        Dictionary with status and message.
    """
    try:
        _, _, page = await get_browser()

        element = await page.query_selector(selector)
        if not element:
            return {
                "status": "error",
                "message": f"Element not found: {selector}",
            }

        await element.fill(text)

        return {
            "status": "success",
            "message": f"Typed '{text[:50]}...' into {selector}",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error typing text: {str(e)}",
        }


@mcp.tool()
async def browser_scroll(direction: str = "down", amount: int = 500) -> dict:
    """Scrolls the viewport in a specified direction.

    Args:
        direction: Scroll direction ("up", "down", "left", "right").
        amount: Pixels to scroll (default 500).

    Returns:
        Dictionary with status and message.
    """
    try:
        _, _, page = await get_browser()

        if direction == "down":
            await page.evaluate(f"window.scrollBy(0, {amount})")
        elif direction == "up":
            await page.evaluate(f"window.scrollBy(0, -{amount})")
        elif direction == "left":
            await page.evaluate(f"window.scrollBy(-{amount}, 0)")
        elif direction == "right":
            await page.evaluate(f"window.scrollBy({amount}, 0)")
        else:
            return {
                "status": "error",
                "message": f"Invalid direction: {direction}. Use 'up', 'down', 'left', or 'right'",
            }

        await asyncio.sleep(0.5)  # Brief pause after scroll

        return {
            "status": "success",
            "message": f"Scrolled {direction} by {amount} pixels",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error scrolling: {str(e)}",
        }


@mcp.tool()
async def browser_wait_for_stability(
    network_idle_timeout: float = 5.0,
    visual_stability_threshold: float = 0.01,
) -> dict:
    """Wait for page stability (network idle and visual stability).

    Args:
        network_idle_timeout: Seconds to wait for network idle.
        visual_stability_threshold: Pixel difference threshold for visual stability.

    Returns:
        Dictionary with stability results.
    """
    try:
        _, _, page = await get_browser()

        # Wait for network idle
        network_idle = False
        try:
            await page.wait_for_load_state(
                "networkidle", timeout=int(network_idle_timeout * 1000)
            )
            network_idle = True
        except Exception:
            pass

        # Check visual stability (simplified - would use PIL/OpenCV in full implementation)
        visual_stable = False
        pixel_diff_ratio = None

        try:
            # Take two screenshots with small delay
            screenshot1 = await page.screenshot()
            await asyncio.sleep(0.5)
            screenshot2 = await page.screenshot()

            # Simple check: if screenshots are identical, visually stable
            # Full implementation would use PIL/OpenCV for pixel diff
            visual_stable = screenshot1 == screenshot2
            pixel_diff_ratio = 0.0 if visual_stable else 1.0
        except Exception:
            pass

        return {
            "status": "success",
            "network_idle": network_idle,
            "visual_stable": visual_stable,
            "pixel_diff_ratio": pixel_diff_ratio,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error waiting for stability: {str(e)}",
        }


@mcp.tool()
async def get_cart_state() -> dict:
    """Parse shopping cart/basket state from the current page.

    Returns:
        Dictionary with cart items and total.
    """
    try:
        _, _, page = await get_browser()

        cart_script = """
        () => {
            const items = [];
            
            // Common cart selectors
            const cartSelectors = [
                '.cart-item',
                '.basket-item',
                '[data-cart-item]',
                '.product-item',
                'tr[data-item]'
            ];
            
            let cartContainer = null;
            for (const selector of cartSelectors) {
                cartContainer = document.querySelector(selector)?.closest('.cart, .basket, [data-cart], table');
                if (cartContainer) break;
            }
            
            if (!cartContainer) {
                // Try to find any element with "cart" or "basket" in class/id
                cartContainer = document.querySelector('[class*="cart"], [class*="basket"], [id*="cart"], [id*="basket"]');
            }
            
            if (cartContainer) {
                const itemElements = cartContainer.querySelectorAll('[data-item], .item, tr, li');
                
                itemElements.forEach((el, index) => {
                    const nameEl = el.querySelector('.name, .product-name, [data-name], h3, h4, .title');
                    const priceEl = el.querySelector('.price, [data-price], .amount, .cost');
                    const qtyEl = el.querySelector('.quantity, [data-quantity], input[type="number"]');
                    
                    const name = nameEl?.textContent?.trim() || `Item ${index + 1}`;
                    const priceText = priceEl?.textContent?.trim() || '0';
                    const price = parseFloat(priceText.replace(/[^0-9.]/g, '')) || 0;
                    const quantity = parseInt(qtyEl?.value || qtyEl?.textContent?.trim() || '1') || 1;
                    
                    if (name && price > 0) {
                        items.push({
                            name: name.substring(0, 100),
                            price: price,
                            quantity: quantity,
                            selector: el.id ? `#${el.id}` : null
                        });
                    }
                });
            }
            
            // Get total
            const totalEl = document.querySelector('.total, .cart-total, [data-total], .grand-total, .subtotal');
            const totalText = totalEl?.textContent?.trim() || '0';
            const total = parseFloat(totalText.replace(/[^0-9.]/g, '')) || 0;
            
            return {
                items: items,
                total: total,
                item_count: items.length
            };
        }
        """

        cart_data = await page.evaluate(cart_script)

        return {
            "status": "success",
            "cart": cart_data,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting cart state: {str(e)}",
        }


@mcp.tool()
async def get_price_breakdown() -> dict:
    """Extract price breakdown from checkout/payment page.

    Returns:
        Dictionary with price components (subtotal, tax, shipping, fees, total).
    """
    try:
        _, _, page = await get_browser()

        price_script = """
        () => {
            const breakdown = {
                subtotal: 0,
                tax: 0,
                vat: 0,
                shipping: 0,
                service_fee: 0,
                processing_fee: 0,
                convenience_fee: 0,
                cleaning_fee: 0,
                total: 0
            };
            
            // Common price breakdown selectors
            const priceLabels = {
                'subtotal': ['.subtotal', '[data-subtotal]', '.item-total'],
                'tax': ['.tax', '[data-tax]', '.sales-tax', '.vat'],
                'vat': ['.vat', '[data-vat]'],
                'shipping': ['.shipping', '[data-shipping]', '.delivery'],
                'service_fee': ['.service-fee', '[data-service-fee]', '.fee'],
                'processing_fee': ['.processing-fee', '[data-processing-fee]'],
                'convenience_fee': ['.convenience-fee', '[data-convenience-fee]'],
                'cleaning_fee': ['.cleaning-fee', '[data-cleaning-fee]'],
                'total': ['.total', '.grand-total', '[data-total]', '.final-total']
            };
            
            for (const [key, selectors] of Object.entries(priceLabels)) {
                for (const selector of selectors) {
                    const el = document.querySelector(selector);
                    if (el) {
                        const text = el.textContent?.trim() || '';
                        const value = parseFloat(text.replace(/[^0-9.]/g, '')) || 0;
                        if (value > 0) {
                            breakdown[key] = value;
                            break;
                        }
                    }
                }
            }
            
            return breakdown;
        }
        """

        price_data = await page.evaluate(price_script)

        return {
            "status": "success",
            "price_breakdown": price_data,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting price breakdown: {str(e)}",
        }


@mcp.tool()
async def get_interactive_elements_marked() -> dict:
    """Get all interactive elements with numeric mark IDs (Set-of-Marks).

    Returns:
        Dictionary mapping mark IDs to element information.
    """
    try:
        _, _, page = await get_browser()

        marking_script = """
        () => {
            const markedElements = {};
            let markId = 1;
            
            // Find all interactive elements
            const selectors = [
                'a[href]',
                'button',
                'input[type="button"]',
                'input[type="submit"]',
                'input[type="text"]',
                'input[type="email"]',
                'input[type="checkbox"]',
                'input[type="radio"]',
                'select',
                'textarea',
                '[role="button"]',
                '[role="link"]',
                '[onclick]',
                '[tabindex]:not([tabindex="-1"])'
            ];
            
            const allElements = new Set();
            selectors.forEach(selector => {
                document.querySelectorAll(selector).forEach(el => {
                    if (el.offsetParent !== null) { // Only visible elements
                        allElements.add(el);
                    }
                });
            });
            
            allElements.forEach(element => {
                const rect = element.getBoundingClientRect();
                const text = element.textContent?.trim() || element.value || element.alt || '';
                const tag = element.tagName.toLowerCase();
                const type = element.type || tag;
                
                // Generate selector
                let selector = null;
                if (element.id) {
                    selector = `#${element.id}`;
                } else if (element.className && typeof element.className === 'string') {
                    const firstClass = element.className.split(' ')[0];
                    selector = `.${firstClass}`;
                } else {
                    selector = tag;
                }
                
                markedElements[markId.toString()] = {
                    mark_id: markId,
                    selector: selector,
                    tag: tag,
                    type: type,
                    text: text.substring(0, 100),
                    x: Math.round(rect.left + rect.width / 2),
                    y: Math.round(rect.top + rect.height / 2),
                    visible: true
                };
                
                markId++;
            });
            
            return {
                marked_elements: markedElements,
                total_marks: markId - 1
            };
        }
        """

        marked_data = await page.evaluate(marking_script)

        return {
            "status": "success",
            "marked_elements": marked_data.get("marked_elements", {}),
            "total_marks": marked_data.get("total_marks", 0),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error marking interactive elements: {str(e)}",
        }


async def cleanup():
    """Cleanup all browser sessions."""
    await _session_manager.close_all()


async def cleanup_session(session_id: str):
    """Cleanup a specific browser session."""
    await _session_manager.close_session(session_id)


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
