"""Privacy Zuckering Detector: Consent Maze Navigation.

Detects confusing or arduous consent management interfaces.
"""

from typing import List, Optional, Dict, Any
from ..ledger import JourneyLedger
from ...schemas import AuditFlag, PatternType


class PrivacyZuckeringDetector:
    """Detector for Privacy Zuckering patterns via consent maze navigation."""

    def __init__(self, ledger: JourneyLedger):
        """Initialize Privacy Zuckering detector.

        Args:
            ledger: JourneyLedger instance for state tracking.
        """
        self.ledger = ledger

    async def detect(
        self,
        navigation_path: List[str],
        zuckering_score: Optional[int] = None,
        cookies_after_reject: Optional[List[str]] = None,
    ) -> List[AuditFlag]:
        """Detect Privacy Zuckering pattern via friction scoring and verification.

        Args:
            navigation_path: Path taken to reject cookies (e.g., ["Layer 1", "Settings", "Toggle"]).
            zuckering_score: Pre-calculated Zuckering score (0-10+).
            cookies_after_reject: List of cookie names found after rejection.

        Returns:
            List of AuditFlags for Privacy Zuckering violations.
        """
        flags = []

        # Calculate Zuckering score if not provided
        if zuckering_score is None:
            zuckering_score = self._calculate_zuckering_score(navigation_path)

        # High score indicates dark pattern
        if zuckering_score >= 5:
            flag = AuditFlag(
                pattern_type=PatternType.PRIVACY_ZUCKERING,
                confidence=min(0.9, 0.7 + (zuckering_score - 5) * 0.05),
                step_id=len(self.ledger.snapshots) - 1,
                evidence=(
                    f"Zuckering score: {zuckering_score}. "
                    f"Navigation path: {' → '.join(navigation_path)}"
                ),
                priority="high" if zuckering_score >= 7 else "normal",
            )
            flags.append(flag)

        # Check for false consent (cookies set despite rejection)
        if cookies_after_reject:
            tracking_cookies = [
                cookie
                for cookie in cookies_after_reject
                if any(
                    tracker in cookie.lower()
                    for tracker in ["_ga", "uuid", "_fbp", "_gid", "analytics", "tracking"]
                )
            ]

            if tracking_cookies:
                flag = AuditFlag(
                    pattern_type=PatternType.PRIVACY_ZUCKERING,
                    confidence=0.95,
                    step_id=len(self.ledger.snapshots) - 1,
                    evidence=(
                        f"Tracking cookies found after rejection: {', '.join(tracking_cookies)}. "
                        "This indicates False Consent / Deceptive Implementation."
                    ),
                    priority="high",
                )
                flags.append(flag)

        return flags

    def _calculate_zuckering_score(self, navigation_path: List[str]) -> int:
        """Calculate Zuckering score based on navigation path.

        Args:
            navigation_path: Path taken to reject cookies.

        Returns:
            Zuckering score (0-10+).
        """
        score = 0

        # Score = 0: "Reject All" on Layer 1 with equal prominence
        if len(navigation_path) == 1 and "reject" in navigation_path[0].lower():
            return 0

        # Score +1: "Reject" is text-only (not a button)
        if len(navigation_path) == 1:
            score += 1

        # Score +2: Requires entering "Settings" (Layer 2)
        if len(navigation_path) >= 2:
            score += 2

        # Score +5: Requires toggling multiple individual vendors
        if "toggle" in str(navigation_path).lower() or "vendor" in str(navigation_path).lower():
            score += 5

        # Score +10: "Legitimate Interest" is pre-enabled and requires separate toggling
        if "legitimate interest" in str(navigation_path).lower():
            score += 10

        return score

    async def verify_consent(
        self, page, expected_status: str = "rejected"
    ) -> List[str]:
        """Verify actual consent status by checking cookies and localStorage.

        Args:
            page: Playwright Page object.
            expected_status: Expected consent status ("accepted" or "rejected").

        Returns:
            List of cookie names found.
        """
        # Get cookies
        cookies = await page.context.cookies()
        cookie_names = [cookie["name"] for cookie in cookies]

        # Get localStorage (requires JavaScript evaluation)
        try:
            local_storage = await page.evaluate("() => Object.keys(localStorage)")
            cookie_names.extend(local_storage)
        except Exception:
            pass

        return cookie_names
