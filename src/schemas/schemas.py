"""Pydantic schemas for structuring the agent's output.

This module defines the strict data models that enforce JSON output protocols,
ensuring deterministic, structured data from the probabilistic LLM.
"""

from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime


class PatternType(str, Enum):
    """Enumeration of dark pattern types based on SKILL.md definitions."""

    ROACH_MOTEL = "roach_motel"
    FALSE_URGENCY = "false_urgency"
    CONFIRMSHAMING = "confirmshaming"
    SNEAK_INTO_BASKET = "sneak_into_basket"
    FORCED_CONTINUITY = "forced_continuity"
    PRIVACY_ZUCKERING = "privacy_zuckering"


class DetectedPattern(BaseModel):
    """Represents a single instance of a suspected dark pattern.

    This model captures the what, where, and why of a detected pattern,
    ensuring the agent provides structured evidence rather than free-form text.
    """

    pattern_type: PatternType = Field(
        ...,
        description="The classification of the dark pattern based on definitions in SKILL.md.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A normalized score (0-1) representing the agent's certainty.",
    )
    element_selector: str = Field(
        ...,
        description="The precise CSS or XPath selector of the DOM element involved.",
    )
    reasoning: str = Field(
        ...,
        description=(
            "A concise explanation of WHY this fits the definition. "
            "Must reference specific heuristics (e.g., 'Timer reset on reload')."
        ),
    )
    evidence: Optional[str] = Field(
        None,
        description=(
            "The text content or attribute value that triggered the detection "
            "(e.g., 'No, I hate saving money')."
        ),
    )

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if finding meets confidence threshold.

        Args:
            threshold: Minimum confidence score (default 0.7).

        Returns:
            True if confidence meets or exceeds threshold.
        """
        return self.confidence_score >= threshold


class AuditResult(BaseModel):
    """Encapsulates the entire audit session, aggregating all findings and metadata.

    This is the primary output structure that the agent must produce,
    ensuring all audit results are consistently formatted and validated.
    """

    target_url: str = Field(..., description="The URL that was audited.")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the audit was performed.",
    )
    findings: List[DetectedPattern] = Field(
        default_factory=list,
        description="List of detected dark patterns. Empty if none found.",
    )
    screenshot_paths: List[str] = Field(
        default_factory=list,
        description="Relative paths to screenshot evidence files in artifacts/ directory.",
    )
    summary: str = Field(
        ...,
        description="An executive summary of the audit for a human reader.",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "target_url": "https://dark-shop.com",
                "findings": [],
            }
        }

    def get_high_confidence_findings(self, threshold: float = 0.7) -> List[DetectedPattern]:
        """Filter findings to only include those meeting confidence threshold.

        Args:
            threshold: Minimum confidence score (default 0.7).

        Returns:
            List of findings meeting the threshold.
        """
        return [f for f in self.findings if f.is_high_confidence(threshold)]

    @property
    def high_confidence_count(self) -> int:
        """Count of findings meeting default confidence threshold."""
        return len(self.get_high_confidence_findings())


class CartItem(BaseModel):
    """Represents an item in a shopping cart/basket."""

    name: str = Field(..., description="Name of the item")
    price: float = Field(..., description="Price of the item")
    quantity: int = Field(default=1, description="Quantity of the item")
    added_explicitly: bool = Field(
        default=False,
        description="Whether the user explicitly added this item (vs. pre-selected or auto-added)",
    )
    selector: Optional[str] = Field(
        None, description="CSS selector for the cart item element"
    )


class ConsentStatus(str, Enum):
    """Enumeration of consent status states."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class InteractionSnapshot(BaseModel):
    """Represents a single snapshot of the interaction state at a point in time.

    This is the core data structure for the Journey Ledger, recording
    not just technical logs but semantic interpretations of user intent
    and system response.
    """

    sequence_id: int = Field(..., description="Monotonic counter for steps in the journey")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Exact time of observation (crucial for timer analysis)",
    )
    url: str = Field(..., description="Current page location")
    user_intent: str = Field(
        ..., description="The Planner's current goal (e.g., 'Add Item X to Cart')"
    )
    action_taken: Optional[Dict[str, Any]] = Field(
        None, description="The specific action executed (e.g., CLICK(selector='#btn-add'))"
    )
    perceived_value: Optional[float] = Field(
        None, description="The price displayed prominently to the user at this step"
    )
    cart_state: List[CartItem] = Field(
        default_factory=list,
        description="A parsed list of items currently detected in the shopping cart/basket",
    )
    consent_status: ConsentStatus = Field(
        default=ConsentStatus.UNKNOWN,
        description="The agent's belief about tracking status",
    )
    dom_hash: Optional[str] = Field(
        None, description="A hash of the accessibility tree to detect content changes"
    )
    screenshot_ref: Optional[str] = Field(
        None, description="Link to the visual evidence for this step"
    )


class JourneyLedger(BaseModel):
    """Container for a time-series sequence of InteractionSnapshots.

    This serves as the agent's episodic memory, enabling temporal analysis
    and intent reconciliation across the entire user journey.
    """

    snapshots: List[InteractionSnapshot] = Field(
        default_factory=list, description="Time-series sequence of interaction states"
    )
    target_url: str = Field(..., description="The URL being audited")
    session_id: Optional[str] = Field(None, description="Session identifier for this journey")

    def record_snapshot(self, snapshot: InteractionSnapshot) -> None:
        """Append a new interaction snapshot to the ledger."""
        self.snapshots.append(snapshot)

    def get_intended_cart(self) -> List[CartItem]:
        """Extract the intended cart ($C_{intent}$) from user actions.

        Returns items that were explicitly added via user actions.
        """
        intended = []
        for snapshot in self.snapshots:
            if snapshot.action_taken:
                action_type = snapshot.action_taken.get("type", "")
                if action_type in ["add_to_cart", "click_add"]:
                    # Extract cart items from this snapshot
                    intended.extend(snapshot.cart_state)
        return intended

    def get_actual_cart(self, latest_snapshot: Optional[InteractionSnapshot] = None) -> List[CartItem]:
        """Get the actual cart state ($C_{actual}$) from the most recent snapshot.

        Args:
            latest_snapshot: Optional snapshot to use. If None, uses the latest in ledger.

        Returns:
            List of items currently in the cart.
        """
        if latest_snapshot:
            return latest_snapshot.cart_state
        if self.snapshots:
            return self.snapshots[-1].cart_state
        return []

    def calculate_price_delta(
        self, anchor_price: Optional[float] = None, terminal_price: Optional[float] = None
    ) -> Optional[float]:
        """Calculate price delta from anchor ($P_0$) to terminal ($P_n$).

        Args:
            anchor_price: The price at first "Add to Cart" intent. If None, auto-detects.
            terminal_price: The final price at checkout. If None, uses latest snapshot.

        Returns:
            Price delta, or None if insufficient data.
        """
        # Auto-detect anchor price if not provided
        if anchor_price is None:
            for snapshot in self.snapshots:
                if snapshot.action_taken and snapshot.action_taken.get("type") == "add_to_cart":
                    anchor_price = snapshot.perceived_value
                    break

        # Auto-detect terminal price if not provided
        if terminal_price is None and self.snapshots:
            terminal_price = self.snapshots[-1].perceived_value

        if anchor_price is not None and terminal_price is not None:
            return terminal_price - anchor_price

        return None

    def reconcile_intent_vs_reality(
        self, latest_snapshot: Optional[InteractionSnapshot] = None
    ) -> Dict[str, Any]:
        """Reconcile intended cart vs actual cart for Sneak into Basket detection.

        Returns:
            Dictionary with reconciliation results including:
            - intended_items: List of intended items
            - actual_items: List of actual items
            - extra_items: Items in actual but not in intended
            - missing_items: Items in intended but not in actual
        """
        intended = self.get_intended_cart()
        actual = self.get_actual_cart(latest_snapshot)

        # Create sets for comparison (using name as identifier)
        intended_names = {item.name for item in intended}
        actual_names = {item.name for item in actual}

        extra_items = [item for item in actual if item.name not in intended_names]
        missing_items = [item for item in intended if item.name not in actual_names]

        return {
            "intended_items": intended,
            "actual_items": actual,
            "extra_items": extra_items,
            "missing_items": missing_items,
            "has_discrepancy": len(extra_items) > 0 or len(missing_items) > 0,
        }


class AuditFlag(BaseModel):
    """Represents a potential dark pattern violation detected by the Auditor.

    These flags are raised during state observation and can interrupt
    the Planner to force investigation.
    """

    pattern_type: PatternType = Field(..., description="Type of dark pattern detected")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for this flag"
    )
    step_id: int = Field(..., description="Sequence ID of the snapshot where flag was raised")
    evidence: str = Field(..., description="Evidence or reasoning for the flag")
    element_selector: Optional[str] = Field(
        None, description="CSS selector of the element involved"
    )
    priority: str = Field(
        default="normal",
        description="Priority level: 'high' (interrupt), 'normal' (log), 'low' (note)",
    )
