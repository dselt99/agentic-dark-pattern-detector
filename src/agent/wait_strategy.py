"""Dynamic Wait Strategy for Phase 2 Architecture.

This module implements polling-based wait strategies for network idle,
visual stability detection, and False Urgency timer handling.
"""

from typing import Optional, Tuple, Dict, Any
import asyncio
import io
from pathlib import Path
from PIL import Image
import numpy as np
from playwright.async_api import Page


class WaitStrategy:
    """Dynamic wait strategy for handling temporal patterns and page stability."""

    def __init__(
        self,
        network_idle_timeout: float = 5.0,
        visual_stability_threshold: float = 0.01,
        max_wait_time: float = 30.0,
    ):
        """Initialize wait strategy.

        Args:
            network_idle_timeout: Seconds to wait for network idle state.
            visual_stability_threshold: Pixel difference threshold (0-1) for stability.
            max_wait_time: Maximum time to wait before timeout.
        """
        self.network_idle_timeout = network_idle_timeout
        self.visual_stability_threshold = visual_stability_threshold
        self.max_wait_time = max_wait_time

    async def wait_for_network_idle(self, page: Page) -> bool:
        """Wait until network requests subside (indicating page load completion).

        Args:
            page: Playwright Page object.

        Returns:
            True if network became idle, False if timeout.
        """
        try:
            await page.wait_for_load_state("networkidle", timeout=int(self.network_idle_timeout * 1000))
            return True
        except Exception:
            # Timeout or error - return False but don't raise
            return False

    async def wait_for_visual_stability(
        self, page: Page, screenshot_dir: Optional[Path] = None
    ) -> Tuple[bool, Optional[float]]:
        """Wait until visual stability is achieved (pixel diff < threshold).

        Args:
            page: Playwright Page object.
            screenshot_dir: Optional directory to save comparison screenshots.

        Returns:
            Tuple of (is_stable, pixel_difference_ratio).
        """
        if screenshot_dir:
            screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Take initial screenshot
        screenshot1_path = None
        if screenshot_dir:
            screenshot1_path = screenshot_dir / "stability_check_1.png"
            await page.screenshot(path=str(screenshot1_path))
        else:
            screenshot1_bytes = await page.screenshot()
            screenshot1 = Image.open(io.BytesIO(screenshot1_bytes))

        # Wait a short interval
        await asyncio.sleep(0.5)

        # Take second screenshot
        screenshot2_path = None
        if screenshot_dir:
            screenshot2_path = screenshot_dir / "stability_check_2.png"
            await page.screenshot(path=str(screenshot2_path))
            screenshot1 = Image.open(screenshot1_path)
            screenshot2 = Image.open(screenshot2_path)
        else:
            screenshot2_bytes = await page.screenshot()
            screenshot2 = Image.open(io.BytesIO(screenshot2_bytes))

        # Calculate pixel difference
        diff_ratio = self._calculate_pixel_difference(screenshot1, screenshot2)

        is_stable = diff_ratio < self.visual_stability_threshold

        # Clean up temporary screenshots
        if screenshot1_path and screenshot1_path.exists():
            screenshot1_path.unlink()
        if screenshot2_path and screenshot2_path.exists():
            screenshot2_path.unlink()

        return is_stable, diff_ratio

    def _calculate_pixel_difference(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculate pixel difference ratio between two images.

        Args:
            img1: First image.
            img2: Second image.

        Returns:
            Pixel difference ratio (0-1).
        """
        # Resize to same dimensions if needed
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

        # Convert to numpy arrays
        arr1 = np.array(img1.convert("RGB"))
        arr2 = np.array(img2.convert("RGB"))

        # Calculate absolute difference
        diff = np.abs(arr1.astype(float) - arr2.astype(float))

        # Calculate ratio of changed pixels
        total_pixels = arr1.shape[0] * arr1.shape[1] * arr1.shape[2]
        changed_pixels = np.sum(diff > 0)
        diff_ratio = changed_pixels / total_pixels

        return diff_ratio

    async def wait_for_timer_expiration(
        self, page: Page, timer_selector: str, max_wait: float = 60.0
    ) -> Tuple[bool, Optional[str]]:
        """Wait for a False Urgency timer to expire and observe if it resets.

        Args:
            page: Playwright Page object.
            timer_selector: CSS selector for the timer element.
            max_wait: Maximum time to wait for expiration.

        Returns:
            Tuple of (timer_expired, final_timer_value).
        """
        try:
            # Get initial timer value
            initial_value = await page.text_content(timer_selector)
            if not initial_value:
                return False, None

            # Wait for timer to potentially expire
            await asyncio.sleep(min(max_wait, 10.0))  # Don't wait more than 10 seconds

            # Check final timer value
            final_value = await page.text_content(timer_selector)

            # Reload page to check if timer resets
            await page.reload(wait_until="networkidle")
            await asyncio.sleep(1.0)

            after_reload_value = await page.text_content(timer_selector)

            # If timer reset to initial value, it's False Urgency
            if after_reload_value == initial_value and initial_value != final_value:
                return True, after_reload_value

            return True, final_value

        except Exception:
            return False, None

    async def wait_for_stability(
        self, page: Page, screenshot_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Comprehensive wait for both network idle and visual stability.

        Args:
            page: Playwright Page object.
            screenshot_dir: Optional directory for screenshots.

        Returns:
            Dictionary with stability results.
        """
        results = {
            "network_idle": False,
            "visual_stable": False,
            "pixel_diff_ratio": None,
            "total_wait_time": 0.0,
        }

        import time
        start_time = time.time()

        # Wait for network idle
        results["network_idle"] = await self.wait_for_network_idle(page)

        # Wait for visual stability
        is_stable, diff_ratio = await self.wait_for_visual_stability(page, screenshot_dir)
        results["visual_stable"] = is_stable
        results["pixel_diff_ratio"] = diff_ratio

        results["total_wait_time"] = time.time() - start_time

        return results
