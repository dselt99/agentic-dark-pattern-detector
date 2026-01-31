"""Auditor: Observation layer for Phase 2 architecture.

The Auditor runs in parallel to the Actor, observing state changes
and detecting dark patterns without interfering in navigation.
"""

from typing import List, Dict, Any, Optional
from ..schemas import AuditFlag, PatternType, InteractionSnapshot, CartItem
from .ledger import JourneyLedger
from .detectors import (
    SneakIntoBasketDetector,
    DripPricingDetector,
    RoachMotelDetector,
    ForcedContinuityDetector,
    PrivacyZuckeringDetector,
    FalseUrgencyDetector,
)


class Auditor:
    """Observation layer for detecting dark patterns during navigation."""

    def __init__(self, ledger: JourneyLedger):
        """Initialize the Auditor.

        Args:
            ledger: JourneyLedger instance for state tracking.
        """
        self.ledger = ledger
        self.flags: List[AuditFlag] = []
        
        # Initialize detectors
        self.sneak_detector = SneakIntoBasketDetector(ledger)
        self.drip_detector = DripPricingDetector(ledger)
        self.roach_detector = RoachMotelDetector(ledger)
        self.forced_continuity_detector = ForcedContinuityDetector(ledger)
        self.privacy_detector = PrivacyZuckeringDetector(ledger)
        self.false_urgency_detector = FalseUrgencyDetector(ledger)

    async def observe_state(
        self,
        snapshot: InteractionSnapshot,
        dom_tree: Optional[str] = None,
        price_breakdown: Optional[Dict[str, float]] = None,
    ) -> List[AuditFlag]:
        """Observe current state and detect potential dark patterns.

        Args:
            snapshot: Current interaction snapshot.
            dom_tree: Current accessibility tree.
            price_breakdown: Price breakdown from checkout page.

        Returns:
            List of AuditFlags raised during observation.
        """
        new_flags: List[AuditFlag] = []

        # Use detector modules for pattern detection
        try:
            # Sneak into Basket detection
            sneak_flags = await self.sneak_detector.detect(snapshot)
            new_flags.extend(sneak_flags)

            # Pre-selection check
            if dom_tree:
                pre_selection_flags = await self.sneak_detector.check_pre_selection(dom_tree)
                new_flags.extend(pre_selection_flags)

            # Drip Pricing detection
            drip_flags = await self.drip_detector.detect(snapshot, price_breakdown)
            new_flags.extend(drip_flags)

            # False Urgency detection (if timer data available)
            # This would be triggered after a reload action
            if dom_tree:
                # Look for timer patterns in DOM
                timer_patterns = ["timer", "countdown", "expires"]
                if any(pattern in dom_tree.lower() for pattern in timer_patterns):
                    # Extract timer value and selector (simplified)
                    # Full implementation would parse DOM properly
                    pass

        except Exception as e:
            # Log error but continue
            pass

        # Store flags
        self.flags.extend(new_flags)

        return new_flags


    def get_high_priority_flags(self) -> List[AuditFlag]:
        """Get flags with high priority that should interrupt the Planner.

        Returns:
            List of high-priority AuditFlags.
        """
        return [flag for flag in self.flags if flag.priority == "high"]

    def get_all_flags(self) -> List[AuditFlag]:
        """Get all flags raised during the audit.

        Returns:
            List of all AuditFlags.
        """
        return self.flags
