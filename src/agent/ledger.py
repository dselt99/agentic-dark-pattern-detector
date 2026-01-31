"""Journey Ledger: Episodic Memory for Dynamic Pattern Detection.

This module implements the Journey Ledger system that tracks interaction
states across time, enabling temporal analysis and intent reconciliation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib

from ..schemas.schemas import (
    JourneyLedger as JourneyLedgerSchema,
    InteractionSnapshot,
    CartItem,
    ConsentStatus,
)


class JourneyLedger:
    """Journey Ledger implementation for tracking interaction states.

    This class wraps the JourneyLedgerSchema and provides additional
    methods for state management, cart reconciliation, and price tracking.
    """

    def __init__(self, target_url: str, session_id: Optional[str] = None):
        """Initialize a new Journey Ledger.

        Args:
            target_url: The URL being audited.
            session_id: Optional session identifier.
        """
        self.schema = JourneyLedgerSchema(
            target_url=target_url, session_id=session_id, snapshots=[]
        )

    def record_snapshot(
        self,
        url: str,
        user_intent: str,
        action_taken: Optional[Dict[str, Any]] = None,
        perceived_value: Optional[float] = None,
        cart_state: Optional[List[CartItem]] = None,
        consent_status: ConsentStatus = ConsentStatus.UNKNOWN,
        screenshot_ref: Optional[str] = None,
    ) -> InteractionSnapshot:
        """Record a new interaction snapshot.

        Args:
            url: Current page location.
            user_intent: The Planner's current goal.
            action_taken: The specific action executed.
            perceived_value: Price displayed to user.
            cart_state: Current cart items.
            consent_status: Tracking consent status.
            screenshot_ref: Path to screenshot evidence.

        Returns:
            The created InteractionSnapshot.
        """
        sequence_id = len(self.schema.snapshots)
        timestamp = datetime.now()

        # Calculate DOM hash if we have accessibility tree data
        dom_hash = None
        if action_taken and "dom_tree" in action_taken:
            dom_hash = self._hash_dom(action_taken["dom_tree"])

        snapshot = InteractionSnapshot(
            sequence_id=sequence_id,
            timestamp=timestamp,
            url=url,
            user_intent=user_intent,
            action_taken=action_taken,
            perceived_value=perceived_value,
            cart_state=cart_state or [],
            consent_status=consent_status,
            dom_hash=dom_hash,
            screenshot_ref=screenshot_ref,
        )

        self.schema.record_snapshot(snapshot)
        return snapshot

    def get_intended_cart(self) -> List[CartItem]:
        """Extract the intended cart ($C_{intent}$) from user actions.

        Returns items that were explicitly added via user actions.
        """
        return self.schema.get_intended_cart()

    def get_actual_cart(
        self, latest_snapshot: Optional[InteractionSnapshot] = None
    ) -> List[CartItem]:
        """Get the actual cart state ($C_{actual}$) from the most recent snapshot.

        Args:
            latest_snapshot: Optional snapshot to use. If None, uses the latest in ledger.

        Returns:
            List of items currently in the cart.
        """
        return self.schema.get_actual_cart(latest_snapshot)

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
        return self.schema.calculate_price_delta(anchor_price, terminal_price)

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
            - has_discrepancy: Boolean indicating if discrepancy exists
        """
        return self.schema.reconcile_intent_vs_reality(latest_snapshot)

    def get_anchor_price(self) -> Optional[float]:
        """Get the anchor price ($P_0$) - price at first 'Add to Cart' intent.

        Returns:
            Anchor price, or None if not found.
        """
        for snapshot in self.schema.snapshots:
            if snapshot.action_taken:
                action_type = snapshot.action_taken.get("type", "")
                if action_type in ["add_to_cart", "click_add"]:
                    return snapshot.perceived_value
        return None

    def get_terminal_price(self) -> Optional[float]:
        """Get the terminal price ($P_n$) - final price at checkout.

        Returns:
            Terminal price from latest snapshot, or None if no snapshots.
        """
        if self.schema.snapshots:
            return self.schema.snapshots[-1].perceived_value
        return None

    def get_latest_snapshot(self) -> Optional[InteractionSnapshot]:
        """Get the most recent interaction snapshot.

        Returns:
            Latest snapshot, or None if ledger is empty.
        """
        if self.schema.snapshots:
            return self.schema.snapshots[-1]
        return None

    def get_snapshots_by_intent(self, intent_pattern: str) -> List[InteractionSnapshot]:
        """Get all snapshots matching a user intent pattern.

        Args:
            intent_pattern: Pattern to match in user_intent field.

        Returns:
            List of matching snapshots.
        """
        return [
            snapshot
            for snapshot in self.schema.snapshots
            if intent_pattern.lower() in snapshot.user_intent.lower()
        ]

    def get_snapshots_since(self, sequence_id: int) -> List[InteractionSnapshot]:
        """Get all snapshots since a given sequence ID.

        Args:
            sequence_id: Starting sequence ID.

        Returns:
            List of snapshots with sequence_id >= given ID.
        """
        return [
            snapshot
            for snapshot in self.schema.snapshots
            if snapshot.sequence_id >= sequence_id
        ]

    def get_short_term_context(self, num_snapshots: int = 3) -> List[InteractionSnapshot]:
        """Get the last N snapshots for short-term memory (Actor context).

        Args:
            num_snapshots: Number of recent snapshots to return.

        Returns:
            List of last N snapshots.
        """
        return self.schema.snapshots[-num_snapshots:] if self.schema.snapshots else []

    def _hash_dom(self, dom_tree: str) -> str:
        """Calculate hash of DOM tree for change detection.

        Args:
            dom_tree: DOM tree string (YAML or HTML).

        Returns:
            SHA256 hash of the DOM tree.
        """
        return hashlib.sha256(dom_tree.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert ledger to dictionary representation.

        Returns:
            Dictionary representation of the ledger.
        """
        return self.schema.model_dump()

    @property
    def target_url(self) -> str:
        """Get the target URL being audited."""
        return self.schema.target_url

    @property
    def session_id(self) -> Optional[str]:
        """Get the session ID."""
        return self.schema.session_id

    @property
    def snapshots(self) -> List[InteractionSnapshot]:
        """Get all snapshots in the ledger."""
        return self.schema.snapshots
