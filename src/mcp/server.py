"""MCP Server for browser automation capabilities.

This module implements the "Dumb Tools" philosophy - deterministic, stateless
interactions with the browser via Playwright. The server acts as the "hands"
of the agent, executing commands without reasoning.
"""

import os
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import yaml

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("Dark Pattern Hunter - Browser Automation")

# Global browser state (singleton pattern)
_browser: Optional[Browser] = None
_context: Optional[BrowserContext] = None
_page: Optional[Page] = None
_playwright = None


async def get_browser() -> tuple[Browser, BrowserContext, Page]:
    """Get or create browser instance (singleton pattern)."""
    global _browser, _context, _page, _playwright

    if _browser is None:
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(
            headless=os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
        )
        _context = await _browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="DarkPatternHunter/1.0 (Research Bot; +https://github.com/dark-pattern-hunter)",
        )
        _page = await _context.new_page()

    return _browser, _context, _page


def check_robots_txt(url: str) -> tuple[bool, Optional[str]]:
    """Check if the URL is allowed by robots.txt.

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
    except Exception as e:
        # If robots.txt doesn't exist or is unparseable, allow by default
        # (permissive approach - some sites don't have robots.txt)
        return True, None


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


@mcp.tool()
async def get_accessibility_tree(selector: str = "body") -> dict:
    """Returns a semantic YAML representation of the current page structure.

    Use this to understand the page content without visual noise.
    Extracts semantic information from the page including text, roles, and structure.

    Args:
        selector: Optional CSS selector to root the tree at. Defaults to 'body'.

    Returns:
        Dictionary with the YAML representation of the page structure.
    """
    try:
        _, _, page = await get_browser()

        # Extract semantic structure using JavaScript
        # This replaces the deprecated accessibility.snapshot() API
        extract_script = """
        (selector) => {
            function extractNode(element, depth = 0) {
                if (!element || depth > 10) return null;

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

                // Get aria attributes
                const ariaLabel = element.getAttribute?.('aria-label');
                if (ariaLabel) attrs['aria-label'] = ariaLabel;
                const ariaHidden = element.getAttribute?.('aria-hidden');
                if (ariaHidden === 'true') attrs['aria-hidden'] = true;

                if (Object.keys(attrs).length > 0) result.attrs = attrs;

                // Get children (skip script, style, hidden elements)
                const skipTags = ['script', 'style', 'noscript', 'svg', 'path'];
                const children = Array.from(element.children || [])
                    .filter(child => {
                        if (skipTags.includes(child.tagName?.toLowerCase())) return false;
                        const style = window.getComputedStyle(child);
                        if (style.display === 'none' || style.visibility === 'hidden') return false;
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

            return extractNode(root);
        }
        """

        structure = await page.evaluate(extract_script, selector)

        if structure and structure.get("error"):
            return {
                "status": "error",
                "message": structure["error"],
            }

        # Convert to YAML string
        yaml_str = yaml.dump(structure, default_flow_style=False, allow_unicode=True)

        return {
            "status": "success",
            "tree": yaml_str,
            "selector": selector,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting accessibility tree: {str(e)}",
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


async def cleanup():
    """Cleanup browser resources."""
    global _browser, _context, _page, _playwright

    try:
        if _page:
            try:
                await _page.close()
            except Exception:
                pass
            _page = None
        if _context:
            try:
                await _context.close()
            except Exception:
                pass
            _context = None
        if _browser:
            try:
                await _browser.close()
            except Exception:
                pass
            _browser = None
        if _playwright:
            try:
                await _playwright.stop()
            except Exception:
                pass
            _playwright = None
    except Exception:
        pass  # Ignore cleanup errors


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
