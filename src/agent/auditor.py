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

                # Drip Pricing - check for hidden fees on checkout/order pages
                drip_flags = self._detect_drip_pricing_language(dom_lower)
                new_flags.extend(drip_flags)

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

        # Try to find a consent banner region (usually a few hundred chars
        # around "cookie" or "consent" keywords).  This avoids false negatives
        # caused by "reject"/"decline" appearing in unrelated article text on
        # large news sites like BBC.
        consent_regions = self._extract_consent_regions(dom_lower)

        if consent_regions:
            for region in consent_regions:
                has_accept = "accept" in region or "agree" in region or "ok" in region.split()
                has_reject = "reject" in region or "decline" in region or "refuse" in region

                if has_accept and not has_reject:
                    flags.append(AuditFlag(
                        pattern_type=PatternType.PRIVACY_ZUCKERING,
                        confidence=0.7,
                        step_id=len(self.ledger.snapshots) - 1,
                        evidence="Cookie consent with 'Accept' option but no clear 'Reject' option",
                        priority="normal",
                    ))
                    break

                if "manage" in region and "preference" in region and not has_reject:
                    flags.append(AuditFlag(
                        pattern_type=PatternType.PRIVACY_ZUCKERING,
                        confidence=0.65,
                        step_id=len(self.ledger.snapshots) - 1,
                        evidence="Cookie reject hidden behind 'Manage Preferences' instead of clear reject button",
                        priority="normal",
                    ))
                    break
        else:
            # Fallback: whole-page scan for small DOMs (simulations, <10K chars)
            if len(dom_lower) < 10000:
                if "cookie" in dom_lower or "consent" in dom_lower or "privacy" in dom_lower:
                    has_accept = "accept" in dom_lower or "agree" in dom_lower
                    has_reject = "reject" in dom_lower or "decline" in dom_lower or "refuse" in dom_lower

                    if has_accept and not has_reject:
                        flags.append(AuditFlag(
                            pattern_type=PatternType.PRIVACY_ZUCKERING,
                            confidence=0.7,
                            step_id=len(self.ledger.snapshots) - 1,
                            evidence="Cookie consent with 'Accept' option but no clear 'Reject' option",
                            priority="normal",
                        ))

                    if "manage" in dom_lower and "preference" in dom_lower and not has_reject:
                        flags.append(AuditFlag(
                            pattern_type=PatternType.PRIVACY_ZUCKERING,
                            confidence=0.65,
                            step_id=len(self.ledger.snapshots) - 1,
                            evidence="Cookie reject hidden behind 'Manage Preferences' instead of clear reject button",
                            priority="normal",
                        ))

        return flags

    def _extract_consent_regions(self, dom_lower: str) -> List[str]:
        """Extract text regions around cookie/consent keywords.

        Returns chunks of ~500 chars centred on each consent-related keyword
        occurrence.  Deduplicates overlapping regions.
        """
        regions = []
        consent_anchors = ["cookie", "consent", "gdpr", "privacy banner", "cookie banner"]
        radius = 500
        seen_starts = set()

        for anchor in consent_anchors:
            start = 0
            while True:
                idx = dom_lower.find(anchor, start)
                if idx == -1:
                    break
                region_start = max(0, idx - radius)
                # Deduplicate overlapping windows
                bucket = region_start // radius
                if bucket not in seen_starts:
                    seen_starts.add(bucket)
                    region_end = min(len(dom_lower), idx + len(anchor) + radius)
                    regions.append(dom_lower[region_start:region_end])
                start = idx + len(anchor)

        return regions

    def _detect_roach_motel_patterns(self, dom_lower: str) -> List[AuditFlag]:
        """Detect Roach Motel via hidden unsubscribe/cancel patterns in DOM."""
        flags = []

        # Check for account/subscription pages
        is_subscription_page = "subscription" in dom_lower or "membership" in dom_lower or "account" in dom_lower

        if not is_subscription_page:
            return flags

        # Path 1: Confirmshaming language
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

        # Path 2: Navigation friction — prominent subscribe/sign-up but cancel is
        # hidden (accordion, small text, buried in help section, etc.)
        has_prominent_cta = bool(re.search(
            r'subscribe|sign.?up|join now|get started|start.*(trial|plan)',
            dom_lower,
        ))
        has_cancel = "cancel" in dom_lower or "unsubscribe" in dom_lower

        if has_prominent_cta:
            if has_cancel:
                # Cancel exists but may be hidden behind friction
                friction_indicators = [
                    "help", "faq", "frequently asked", "support",
                    "accordion", "footer", "contact",
                ]
                has_friction = any(ind in dom_lower for ind in friction_indicators)

                if has_friction:
                    flags.append(AuditFlag(
                        pattern_type=PatternType.ROACH_MOTEL,
                        confidence=0.75,
                        step_id=len(self.ledger.snapshots) - 1,
                        evidence="Cancel/unsubscribe option buried in help/FAQ section while subscribe is prominent",
                        priority="high",
                    ))
            else:
                # Subscribe is prominent but cancel is NOT visible at all.
                # Only flag if the page has management/help context (indicating
                # this is a subscription management page, not just a signup form).
                management_indicators = [
                    "help", "faq", "frequently asked", "support",
                    "manage", "settings", "billing history",
                ]
                has_management = any(ind in dom_lower for ind in management_indicators)

                if has_management:
                    flags.append(AuditFlag(
                        pattern_type=PatternType.ROACH_MOTEL,
                        confidence=0.7,
                        step_id=len(self.ledger.snapshots) - 1,
                        evidence="Subscription page has prominent subscribe action but no visible cancel/unsubscribe option",
                        priority="high",
                    ))

        return flags


    def _detect_drip_pricing_language(self, dom_lower: str) -> List[AuditFlag]:
        """Detect Drip Pricing via hidden fees on checkout/order summary pages."""
        flags = []

        # Look for checkout/order context
        checkout_keywords = ["order summary", "checkout", "payment", "total", "booking"]
        is_checkout = any(kw in dom_lower for kw in checkout_keywords)

        if not is_checkout:
            return flags

        # Look for hidden fee indicators
        fee_keywords = [
            "service fee", "processing fee", "convenience fee",
            "booking fee", "booking service fee", "handling fee",
            "payment processing", "surcharge",
        ]
        found_fees = [kw for kw in fee_keywords if kw in dom_lower]

        if found_fees:
            # Extract dollar amounts to check for price inflation
            import re
            prices = re.findall(r'\$(\d+\.?\d*)', dom_lower)
            prices_float = [float(p) for p in prices if float(p) > 0]

            if len(prices_float) >= 2:
                min_price = min(prices_float)
                max_price = max(prices_float)
                # If the highest price is >2x the lowest, it's likely drip pricing
                if max_price > min_price * 2 and max_price > 10:
                    flags.append(AuditFlag(
                        pattern_type=PatternType.SNEAK_INTO_BASKET,
                        confidence=0.8,
                        step_id=len(self.ledger.snapshots) - 1,
                        evidence=(
                            f"Drip pricing detected: hidden fees ({', '.join(found_fees[:3])}) "
                            f"inflate price from ${min_price:.2f} to ${max_price:.2f}"
                        ),
                        priority="high",
                    ))
                    return flags

            # Even without price extraction, multiple hidden fees are suspicious
            if len(found_fees) >= 2:
                flags.append(AuditFlag(
                    pattern_type=PatternType.SNEAK_INTO_BASKET,
                    confidence=0.7,
                    step_id=len(self.ledger.snapshots) - 1,
                    evidence=f"Multiple hidden fees on checkout page: {', '.join(found_fees[:3])}",
                    priority="normal",
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
