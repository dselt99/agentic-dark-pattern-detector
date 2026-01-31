"""Detector modules for Phase 2 dynamic pattern detection."""

from .roach_motel import RoachMotelDetector
from .sneak_into_basket import SneakIntoBasketDetector
from .drip_pricing import DripPricingDetector
from .forced_continuity import ForcedContinuityDetector
from .privacy_zuckering import PrivacyZuckeringDetector
from .false_urgency import FalseUrgencyDetector

__all__ = [
    "RoachMotelDetector",
    "SneakIntoBasketDetector",
    "DripPricingDetector",
    "ForcedContinuityDetector",
    "PrivacyZuckeringDetector",
    "FalseUrgencyDetector",
]
