"""Auditor: Observation layer for Phase 2 architecture.

The Auditor runs in parallel to the Actor, observing state changes
and detecting dark patterns without interfering in navigation.
"""

import re
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

            # DOM-based pattern detection
            if dom_tree:
                dom_lower = dom_tree.lower()

                # False Urgency - check for urgency language
                urgency_flags = self._detect_urgency_language(dom_lower)
                new_flags.extend(urgency_flags)

                # Forced Continuity - check for subscription/auto-renew language
                continuity_flags = self._detect_forced_continuity_language(dom_lower)
                new_flags.extend(continuity_flags)

                # Privacy Zuckering - check for confusing consent patterns
                privacy_flags = self._detect_privacy_zuckering_patterns(dom_lower)
                new_flags.extend(privacy_flags)

                # Roach Motel - check for hidden cancel/unsubscribe
                roach_flags = self._detect_roach_motel_patterns(dom_lower)
                new_flags.extend(roach_flags)

        except Exception as e:
            # Log error but continue
            pass

        # Store flags
        self.flags.extend(new_flags)

        return new_flags

    def _detect_urgency_language(self, dom_lower: str) -> List[AuditFlag]:
        """Detect False Urgency via urgency language in DOM."""
        flags = []

        urgency_phrases = [
            ("only .* left", "scarcity claim"),
            ("selling fast", "urgency claim"),
            ("limited time", "time pressure"),
            ("offer expires", "expiration pressure"),
            ("hurry", "urgency language"),
            ("don't miss out", "FOMO language"),
            ("last chance", "scarcity pressure"),
            ("act now", "urgency language"),
            ("today only", "time pressure"),
            (".* people (are )?viewing", "social proof pressure"),
            (".* people (are )?looking", "social proof pressure"),
            ("in high demand", "demand pressure"),
            ("almost sold out", "scarcity claim"),
            ("book now", "urgency CTA"),
        ]

        import re
        for pattern, description in urgency_phrases:
            if re.search(pattern, dom_lower):
                flags.append(AuditFlag(
                    pattern_type=PatternType.FALSE_URGENCY,
                    confidence=0.7,
                    step_id=len(self.ledger.snapshots) - 1,
                    evidence=f"Urgency language detected: {description} (pattern: '{pattern}')",
                    priority="normal",
                ))
                break  # Only flag once per observation

        return flags

    def _detect_forced_continuity_language(self, dom_lower: str) -> List[AuditFlag]:
        """Detect Forced Continuity via subscription language in DOM."""
        flags = []

        # Keywords that indicate potential forced continuity
        continuity_keywords = [
            "auto-renew",
            "automatically renew",
            "recurring charge",
            "monthly charge",
            "annual charge",
            "subscription will continue",
            "billed automatically",
            "cancel anytime",  # Often used to hide auto-renewal
            "free trial",  # Often leads to auto-charge
        ]

        found_keywords = [kw for kw in continuity_keywords if kw in dom_lower]

        if found_keywords:
            # Check if near a CTA button
            cta_keywords = ["subscribe", "start", "continue", "begin", "join", "sign up", "try"]
            has_cta = any(cta in dom_lower for cta in cta_keywords)

            if has_cta:
                flags.append(AuditFlag(
                    pattern_type=PatternType.FORCED_CONTINUITY,
                    confidence=0.75,
                    step_id=len(self.ledger.snapshots) - 1,
                    evidence=f"Subscription terms detected: {', '.join(found_keywords[:3])}",
                    priority="normal",
                ))

        return flags

    def _detect_privacy_zuckering_patterns(self, dom_lower: str) -> List[AuditFlag]:
        """Detect Privacy Zuckering via confusing consent patterns in DOM."""
        flags = []

        # Check for cookie consent with asymmetric options
        if "cookie" in dom_lower or "consent" in dom_lower or "privacy" in dom_lower:
            # Check for "accept all" prominence vs hidden reject
            has_accept = "accept" in dom_lower or "agree" in dom_lower
            has_reject = "reject" in dom_lower or "decline" in dom_lower or "refuse" in dom_lower

            # If accept exists but reject doesn't, it's suspicious
            if has_accept and not has_reject:
                flags.append(AuditFlag(
                    pattern_type=PatternType.PRIVACY_ZUCKERING,
                    confidence=0.7,
                    step_id=len(self.ledger.snapshots) - 1,
                    evidence="Cookie consent with 'Accept' option but no clear 'Reject' option",
                    priority="normal",
                ))

            # Check for "manage preferences" as hidden reject
            if "manage" in dom_lower and "preference" in dom_lower and not has_reject:
                flags.append(AuditFlag(
                    pattern_type=PatternType.PRIVACY_ZUCKERING,
                    confidence=0.65,
                    step_id=len(self.ledger.snapshots) - 1,
                    evidence="Cookie reject hidden behind 'Manage Preferences' instead of clear reject button",
                    priority="normal",
                ))

        return flags

    def _detect_roach_motel_patterns(self, dom_lower: str) -> List[AuditFlag]:
        """Detect Roach Motel via hidden unsubscribe/cancel patterns in DOM."""
        flags = []

        # Check for account/subscription pages
        if "account" in dom_lower or "subscription" in dom_lower or "membership" in dom_lower:
            # Look for cancel/unsubscribe
            has_cancel = "cancel" in dom_lower or "unsubscribe" in dom_lower

            # Look for confirmshaming language
            confirmshaming_phrases = [
                "i don't want",
                "no thanks",
                "i prefer not",
                "lose my benefits",
                "miss out",
                "give up",
            ]

            found_shaming = [phrase for phrase in confirmshaming_phrases if phrase in dom_lower]

            if found_shaming:
                flags.append(AuditFlag(
                    pattern_type=PatternType.ROACH_MOTEL,
                    confidence=0.8,
                    step_id=len(self.ledger.snapshots) - 1,
                    evidence=f"Confirmshaming language detected: {', '.join(found_shaming[:2])}",
                    priority="high",
                ))

        return flags


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
