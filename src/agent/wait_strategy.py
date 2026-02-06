"""Wait Strategy for Phase 2 Architecture.

Multi-signal wait strategy combining DOM stability polling,
domcontentloaded, and optional networkidle for robust page-ready detection.
"""

import asyncio
from typing import Dict, Any

from playwright.async_api import Page


class WaitStrategy:
    """Multi-signal wait strategy for reliable page-ready detection."""

    def __init__(self, overall_timeout: float = 10.0, stability_interval: float = 0.5, stability_threshold: int = 50):
        """Initialize wait strategy.

        Args:
            overall_timeout: Maximum seconds to wait overall.
            stability_interval: Seconds between DOM stability snapshots.
            stability_threshold: Max character-count delta to consider DOM stable.
        """
        self.overall_timeout = overall_timeout
        self.stability_interval = stability_interval
        self.stability_threshold = stability_threshold

    async def wait_for_network_idle(self, page: Page) -> bool:
        """Wait until network requests subside (short timeout, non-blocking).

        Args:
            page: Playwright Page object.

        Returns:
            True if network became idle, False if timeout.
        """
        try:
            await page.wait_for_load_state("networkidle", timeout=3000)
            return True
        except Exception:
            return False

    async def _wait_for_dom_content_loaded(self, page: Page) -> bool:
        """Wait for DOMContentLoaded event (fast baseline).

        Args:
            page: Playwright Page object.

        Returns:
            True if loaded, False on timeout.
        """
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
            return True
        except Exception:
            return False

    async def _poll_dom_stability(self, page: Page) -> bool:
        """Poll DOM size until it stabilises between two snapshots.

        Takes two snapshots `stability_interval` apart and checks if
        the character-count delta is below `stability_threshold`.

        Args:
            page: Playwright Page object.

        Returns:
            True if DOM is stable, False on timeout.
        """
        try:
            snap1 = await page.evaluate("document.documentElement.innerHTML.length")
            await asyncio.sleep(self.stability_interval)
            snap2 = await page.evaluate("document.documentElement.innerHTML.length")
            return abs(snap2 - snap1) < self.stability_threshold
        except Exception:
            return False

    async def wait_for_stability(self, page: Page, **kwargs) -> Dict[str, Any]:
        """Wait for page stability using multiple signals.

        Strategy:
        1. domcontentloaded first (fast baseline)
        2. DOM size stability polling (two snapshots, check delta)
        3. Optional networkidle with short 3s timeout (non-blocking)
        4. Overall timeout of self.overall_timeout seconds

        Args:
            page: Playwright Page object.

        Returns:
            Dictionary with stability results.
        """
        dom_loaded = False
        dom_stable = False
        network_idle = False

        try:
            # Wrap everything in an overall timeout
            async with asyncio.timeout(self.overall_timeout):
                # Signal 1: DOMContentLoaded (fast)
                dom_loaded = await self._wait_for_dom_content_loaded(page)

                # Signal 2: DOM stability polling
                dom_stable = await self._poll_dom_stability(page)

                # Signal 3: networkidle (short, non-blocking)
                network_idle = await self.wait_for_network_idle(page)

        except (asyncio.TimeoutError, TimeoutError):
            # Overall timeout hit — return whatever we collected
            pass

        return {
            "network_idle": network_idle,
            "visual_stable": dom_loaded and dom_stable,
            "dom_content_loaded": dom_loaded,
            "dom_stable": dom_stable,
        }
