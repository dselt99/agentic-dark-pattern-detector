"""Roach Motel Detector: Friction Quantification Algorithm.

Detects patterns where entry (Sign Up) is easy but exit (Cancel) is difficult.
"""

from typing import Dict, Any, List, Optional
from ..ledger import JourneyLedger
from ...schemas import AuditFlag, PatternType


class RoachMotelDetector:
    """Detector for Roach Motel patterns via comparative A/B traversal."""

    def __init__(self, ledger: JourneyLedger):
        """Initialize Roach Motel detector.

        Args:
            ledger: JourneyLedger instance for state tracking.
        """
        self.ledger = ledger
        self.friction_threshold = 2.5  # R_f > 2.5 triggers alert

    async def detect(
        self,
        entry_metrics: Optional[Dict[str, Any]] = None,
        exit_metrics: Optional[Dict[str, Any]] = None,
    ) -> List[AuditFlag]:
        """Detect Roach Motel pattern via friction ratio calculation.

        Args:
            entry_metrics: Metrics for entry traversal (T_in, C_in, S_in).
            exit_metrics: Metrics for exit traversal (T_out, C_out, S_out).

        Returns:
            List of AuditFlags for Roach Motel violations.
        """
        flags = []

        if not entry_metrics or not exit_metrics:
            return flags

        # Extract metrics
        clicks_in = entry_metrics.get("clicks", 0)
        screens_in = entry_metrics.get("screens", 0)
        time_in = entry_metrics.get("time", 0)

        clicks_out = exit_metrics.get("clicks", 0)
        screens_out = exit_metrics.get("screens", 0)
        time_out = exit_metrics.get("time", 0)

        # Calculate friction ratio
        # R_f = (C_out * W_c + S_out * W_s) / (C_in * W_c + S_in * W_s)
        weight_clicks = 1.0
        weight_screens = 0.5

        numerator = clicks_out * weight_clicks + screens_out * weight_screens
        denominator = clicks_in * weight_clicks + screens_in * weight_screens

        if denominator == 0:
            return flags  # Cannot calculate if entry had zero friction

        friction_ratio = numerator / denominator

        # Check threshold
        if friction_ratio > self.friction_threshold:
            flag = AuditFlag(
                pattern_type=PatternType.ROACH_MOTEL,
                confidence=min(0.9, 0.7 + (friction_ratio - self.friction_threshold) * 0.1),
                step_id=len(self.ledger.snapshots) - 1,
                evidence=(
                    f"Friction ratio {friction_ratio:.2f} exceeds threshold {self.friction_threshold}. "
                    f"Entry: {clicks_in} clicks, {screens_in} screens. "
                    f"Exit: {clicks_out} clicks, {screens_out} screens."
                ),
                priority="high",
            )
            flags.append(flag)

        return flags

    async def analyze_confirmshaming(self, cancellation_text: str) -> float:
        """Analyze cancellation flow text for Confirmshaming (emotional manipulation).

        Args:
            cancellation_text: Text from cancellation buttons/links.

        Returns:
            Guilt score (0-1) indicating level of emotional manipulation.
        """
        guilt_keywords = [
            "hate",
            "prefer paying",
            "don't care",
            "miss out",
            "lose benefits",
            "waste money",
        ]

        text_lower = cancellation_text.lower()
        guilt_score = 0.0

        for keyword in guilt_keywords:
            if keyword in text_lower:
                guilt_score += 0.2

        return min(1.0, guilt_score)
