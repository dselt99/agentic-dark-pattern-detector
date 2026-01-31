"""Forced Continuity Detector: Terms & Conditions Extraction.

Detects situations where users unknowingly enter recurring billing agreements.
"""

from typing import List, Optional, Dict, Any
from ..ledger import JourneyLedger
from ...schemas import AuditFlag, PatternType


class ForcedContinuityDetector:
    """Detector for Forced Continuity patterns via proximity zone analysis."""

    def __init__(self, ledger: JourneyLedger):
        """Initialize Forced Continuity detector.

        Args:
            ledger: JourneyLedger instance for state tracking.
        """
        self.ledger = ledger
        self.proximity_zone_pixels = 150

    async def detect(
        self,
        cta_selector: str,
        cta_text: str,
        proximity_text: str,
        font_sizes: Optional[Dict[str, float]] = None,
        contrast_ratios: Optional[Dict[str, float]] = None,
    ) -> List[AuditFlag]:
        """Detect Forced Continuity pattern via proximity and obfuscation analysis.

        Args:
            cta_selector: CSS selector for the Call-to-Action button.
            cta_text: Text of the CTA button.
            proximity_text: Text within 150px of CTA.
            font_sizes: Dictionary with 'cta' and 'terms' font sizes.
            contrast_ratios: Dictionary with contrast ratios.

        Returns:
            List of AuditFlags for Forced Continuity violations.
        """
        flags = []

        # Check for keywords
        keywords = ["auto-renew", "bill after", "cancel anytime", "monthly charge", "recurring"]
        found_keywords = [kw for kw in keywords if kw.lower() in proximity_text.lower()]

        if not found_keywords:
            return flags  # No relevant terms found

        # Check obfuscation metrics
        obfuscation_score = 0.0

        # Font size delta
        if font_sizes:
            cta_size = font_sizes.get("cta", 16)
            terms_size = font_sizes.get("terms", 16)
            if cta_size > 0:
                size_ratio = cta_size / terms_size
                if size_ratio > 3.0:  # CTA is 3x larger than terms
                    obfuscation_score += 0.4

        # Contrast ratio
        if contrast_ratios:
            terms_contrast = contrast_ratios.get("terms", 1.0)
            if terms_contrast < 3.0:  # Low contrast (WCAG AA threshold is 4.5)
                obfuscation_score += 0.3

        # Check if terms are in proximity zone
        if len(proximity_text) < 50:  # Very short text suggests hidden
            obfuscation_score += 0.3

        confidence = 0.7 + (obfuscation_score * 0.2)  # 0.7 to 0.9

        if confidence >= 0.7:
            flag = AuditFlag(
                pattern_type=PatternType.FORCED_CONTINUITY,
                confidence=min(0.9, confidence),
                step_id=len(self.ledger.snapshots) - 1,
                evidence=(
                    f"Found terms near CTA: {', '.join(found_keywords)}. "
                    f"Obfuscation score: {obfuscation_score:.2f}"
                ),
                element_selector=cta_selector,
                priority="normal",
            )
            flags.append(flag)

        return flags

    async def check_modal_trapping(self, terms_link_clicked: bool, scrollable_box: bool) -> bool:
        """Check if terms are trapped in a modal or scrollable box.

        Args:
            terms_link_clicked: Whether clicking "Terms" opens a modal.
            scrollable_box: Whether terms are in a scrollable container.

        Returns:
            True if modal trapping detected.
        """
        return terms_link_clicked and scrollable_box
