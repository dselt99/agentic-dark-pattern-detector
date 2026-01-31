"""False Urgency Detector: Temporal Inconsistency Analysis.

Detects countdown timers and scarcity claims that reset on page reload.
"""

from typing import List, Optional, Dict, Any
from ..ledger import JourneyLedger
from ...schemas import AuditFlag, PatternType


class FalseUrgencyDetector:
    """Detector for False Urgency patterns via timer reset detection."""

    def __init__(self, ledger: JourneyLedger):
        """Initialize False Urgency detector.

        Args:
            ledger: JourneyLedger instance for state tracking.
        """
        self.ledger = ledger
        self.timer_history: List[Dict[str, Any]] = []

    async def detect(
        self,
        timer_value: Optional[str] = None,
        timer_selector: Optional[str] = None,
        after_reload: bool = False,
    ) -> List[AuditFlag]:
        """Detect False Urgency pattern via timer reset analysis.

        Args:
            timer_value: Current timer value (e.g., "05:00").
            timer_selector: CSS selector for the timer element.
            after_reload: Whether this observation is after a page reload.

        Returns:
            List of AuditFlags for False Urgency violations.
        """
        flags = []

        if not timer_value or not timer_selector:
            return flags

        # Record timer observation
        self.timer_history.append({
            "value": timer_value,
            "selector": timer_selector,
            "after_reload": after_reload,
        })

        # Check for reset pattern
        if len(self.timer_history) >= 2:
            first_timer = self.timer_history[0]
            last_timer = self.timer_history[-1]

            # If timer reset to original value after reload, it's False Urgency
            if (
                last_timer["after_reload"]
                and first_timer["value"] == last_timer["value"]
                and first_timer["value"] not in ["00:00", "0:00"]  # Not already expired
            ):
                flag = AuditFlag(
                    pattern_type=PatternType.FALSE_URGENCY,
                    confidence=0.9,
                    step_id=len(self.ledger.snapshots) - 1,
                    evidence=(
                        f"Timer reset to original value '{timer_value}' after page reload. "
                        "This indicates artificial urgency rather than a real deadline."
                    ),
                    element_selector=timer_selector,
                    priority="high",
                )
                flags.append(flag)

        return flags

    async def check_vague_quantification(self, text: str) -> bool:
        """Check for vague quantification claims without verifiable data.

        Args:
            text: Text content to analyze.

        Returns:
            True if vague quantification detected.
        """
        vague_phrases = [
            "selling fast",
            "high demand",
            "limited stock",
            "only a few left",
            "almost gone",
            "going fast",
        ]

        text_lower = text.lower()
        for phrase in vague_phrases:
            if phrase in text_lower:
                return True

        return False
