"""Sneak into Basket Detector: Negative Option Billing Analysis.

Detects items added to cart without explicit user consent.
"""

from typing import List, Optional
from ..ledger import JourneyLedger
from ...schemas import AuditFlag, PatternType, InteractionSnapshot


class SneakIntoBasketDetector:
    """Detector for Sneak into Basket patterns via set integrity verification."""

    def __init__(self, ledger: JourneyLedger):
        """Initialize Sneak into Basket detector.

        Args:
            ledger: JourneyLedger instance for state tracking.
        """
        self.ledger = ledger

    async def detect(
        self, snapshot: Optional[InteractionSnapshot] = None
    ) -> List[AuditFlag]:
        """Detect Sneak into Basket pattern via cart reconciliation.

        Args:
            snapshot: Current interaction snapshot.

        Returns:
            List of AuditFlags for Sneak into Basket violations.
        """
        flags = []

        if not snapshot:
            snapshot = self.ledger.get_latest_snapshot()

        if not snapshot:
            return flags

        # Reconcile intent vs reality
        reconciliation = self.ledger.reconcile_intent_vs_reality(snapshot)

        if reconciliation["has_discrepancy"]:
            extra_items = reconciliation["extra_items"]

            for item in extra_items:
                # Check if item was explicitly added
                if not item.added_explicitly:
                    flag = AuditFlag(
                        pattern_type=PatternType.SNEAK_INTO_BASKET,
                        confidence=0.85,
                        step_id=snapshot.sequence_id,
                        evidence=(
                            f"Item '{item.name}' (${item.price:.2f}) found in cart "
                            "but not explicitly added by user"
                        ),
                        element_selector=item.selector,
                        priority="high",
                    )
                    flags.append(flag)

        return flags

    async def check_pre_selection(self, dom_tree: str) -> List[AuditFlag]:
        """Check for pre-selected checkboxes with financial keywords.

        Args:
            dom_tree: Accessibility tree (YAML format).

        Returns:
            List of AuditFlags for pre-selection violations.
        """
        flags = []

        # Financial keywords that indicate dark pattern
        financial_keywords = [
            "insurance",
            "warranty",
            "protection",
            "donation",
            "priority",
            "express",
            "extended",
        ]

        # Look for checked checkboxes in DOM tree
        # This is simplified - full implementation would parse YAML properly
        dom_lower = dom_tree.lower()

        for keyword in financial_keywords:
            if keyword in dom_lower and "checked" in dom_lower:
                # Check if it's a checkbox input
                if "input" in dom_lower and "checkbox" in dom_lower:
                    flag = AuditFlag(
                        pattern_type=PatternType.SNEAK_INTO_BASKET,
                        confidence=0.8,
                        step_id=len(self.ledger.snapshots) - 1,
                        evidence=f"Pre-selected checkbox found with financial keyword: {keyword}",
                        priority="normal",
                    )
                    flags.append(flag)
                    break  # Only flag once per DOM tree

        return flags
