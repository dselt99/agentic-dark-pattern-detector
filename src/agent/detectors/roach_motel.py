"""Roach Motel Detector: Friction Comparison.

Detects patterns where entry (Sign Up) is easy but exit (Cancel) is difficult.
"""

from typing import Dict, Any, List, Optional
from ..ledger import JourneyLedger
from ...schemas import AuditFlag, PatternType


class RoachMotelDetector:
    """Detector for Roach Motel patterns by comparing entry vs exit friction."""

    def __init__(self, ledger: JourneyLedger):
        """Initialize Roach Motel detector.

        Args:
            ledger: JourneyLedger instance for state tracking.
        """
        self.ledger = ledger
        self.friction_threshold = 2.0  # Exit clicks > 2x entry clicks triggers alert

    async def detect(
        self,
        entry_metrics: Optional[Dict[str, Any]] = None,
        exit_metrics: Optional[Dict[str, Any]] = None,
    ) -> List[AuditFlag]:
        """Detect Roach Motel pattern by comparing click counts.

        Args:
            entry_metrics: Metrics for entry (signup) - must include "clicks".
            exit_metrics: Metrics for exit (cancel) - must include "clicks".

        Returns:
            List of AuditFlags for Roach Motel violations.
        """
        flags = []

        if not entry_metrics or not exit_metrics:
            return flags

        clicks_in = entry_metrics.get("clicks", 0)
        clicks_out = exit_metrics.get("clicks", 0)

        # Avoid division by zero
        if clicks_in == 0:
            return flags

        # Simple ratio: how many more clicks to exit vs enter?
        friction_ratio = clicks_out / clicks_in

        if friction_ratio > self.friction_threshold:
            flag = AuditFlag(
                pattern_type=PatternType.ROACH_MOTEL,
                confidence=min(0.9, 0.6 + friction_ratio * 0.1),
                step_id=len(self.ledger.snapshots) - 1,
                evidence=(
                    f"Exit requires {clicks_out} clicks vs {clicks_in} clicks to enter "
                    f"(ratio: {friction_ratio:.1f}x). "
                    "Cancellation is significantly harder than signup."
                ),
                priority="high",
            )
            flags.append(flag)

        return flags

    async def analyze_confirmshaming(self, cancellation_text: str) -> float:
        """Analyze cancellation text for guilt-tripping language.

        Args:
            cancellation_text: Text from cancellation buttons/links.

        Returns:
            Guilt score (0-1).
        """
        guilt_phrases = ["prefer paying", "don't care", "miss out", "lose benefits"]
        text_lower = cancellation_text.lower()

        matches = sum(1 for phrase in guilt_phrases if phrase in text_lower)
        return min(1.0, matches * 0.25)
