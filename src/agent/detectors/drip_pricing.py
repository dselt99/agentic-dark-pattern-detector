"""Drip Pricing Detector: Hidden Fee Forensics.

Detects sequential disclosure of mandatory fees.
"""

from typing import List, Optional, Dict, Any
from ..ledger import JourneyLedger
from ...schemas import AuditFlag, PatternType, InteractionSnapshot


class DripPricingDetector:
    """Detector for Drip Pricing patterns via waterfall price tracking."""

    def __init__(self, ledger: JourneyLedger):
        """Initialize Drip Pricing detector.

        Args:
            ledger: JourneyLedger instance for state tracking.
        """
        self.ledger = ledger

    async def detect(
        self,
        snapshot: Optional[InteractionSnapshot] = None,
        price_breakdown: Optional[Dict[str, float]] = None,
    ) -> List[AuditFlag]:
        """Detect Drip Pricing pattern via price delta analysis.

        Args:
            snapshot: Current interaction snapshot.
            price_breakdown: Price breakdown from checkout page.

        Returns:
            List of AuditFlags for Drip Pricing violations.
        """
        flags = []

        if not snapshot:
            snapshot = self.ledger.get_latest_snapshot()

        if not snapshot:
            return flags

        # Get anchor and terminal prices
        anchor_price = self.ledger.get_anchor_price()
        terminal_price = self.ledger.get_terminal_price()

        if not anchor_price or not terminal_price:
            return flags

        # Calculate delta
        price_delta = terminal_price - anchor_price

        if price_delta <= 0:
            return flags  # No price increase

        # Categorize fees
        if price_breakdown:
            legitimate_fees = (
                price_breakdown.get("tax", 0)
                + price_breakdown.get("vat", 0)
                + price_breakdown.get("shipping", 0)
            )

            drip_fees = (
                price_breakdown.get("service_fee", 0)
                + price_breakdown.get("processing_fee", 0)
                + price_breakdown.get("convenience_fee", 0)
                + price_breakdown.get("cleaning_fee", 0)
            )

            unexplained_delta = price_delta - legitimate_fees

            if drip_fees > 0 or unexplained_delta > 0:
                # Note: Using SNEAK_INTO_BASKET as closest match until DRIP_PRICING is added to enum
                flag = AuditFlag(
                    pattern_type=PatternType.SNEAK_INTO_BASKET,  # Drip Pricing pattern
                    confidence=0.85,
                    step_id=snapshot.sequence_id,
                    evidence=(
                        f"Drip Pricing: Price increased from ${anchor_price:.2f} to ${terminal_price:.2f}. "
                        f"Legitimate fees: ${legitimate_fees:.2f}, "
                        f"Drip fees: ${drip_fees:.2f}, "
                        f"Unexplained: ${unexplained_delta:.2f}"
                    ),
                    priority="high",
                )
                flags.append(flag)
        elif price_delta > 5.0:  # Threshold for suspicious increase without breakdown
            flag = AuditFlag(
                pattern_type=PatternType.SNEAK_INTO_BASKET,  # Drip Pricing pattern
                confidence=0.7,
                step_id=snapshot.sequence_id,
                evidence=(
                    f"Drip Pricing: Price increased by ${price_delta:.2f} without clear explanation. "
                    f"Anchor: ${anchor_price:.2f}, Terminal: ${terminal_price:.2f}"
                ),
                priority="normal",
            )
            flags.append(flag)

        return flags

    async def check_visibility(
        self, plp_text: str, fee_keywords: List[str] = None
    ) -> bool:
        """Check if fees were mentioned on Product Listing Page (PLP).

        Args:
            plp_text: Text content from PLP.
            fee_keywords: Keywords to search for (default: common fee terms).

        Returns:
            True if fees were mentioned, False otherwise.
        """
        if fee_keywords is None:
            fee_keywords = ["plus fees", "additional fees", "service fee", "processing fee"]

        plp_lower = plp_text.lower()

        for keyword in fee_keywords:
            if keyword in plp_lower:
                return True

        return False
