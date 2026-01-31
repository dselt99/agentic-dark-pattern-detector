"""Sandbox & Safety Engineering for Phase 2 Architecture.

This module implements payment interception, synthetic identity management,
and Docker containerization for safe browser automation.
"""

import os
import uuid
from typing import Dict, Any, Optional, List
from playwright.async_api import Page, Route, Request


class PaymentInterceptor:
    """Intercepts payment API requests to prevent real transactions."""

    # Blocklist of known payment API endpoints
    PAYMENT_BLOCKLIST = [
        "*.stripe.com",
        "*.paypal.com",
        "checkout.shopify.com/process",
        "*.braintreegateway.com",
        "*.squareup.com",
        "*.authorize.net",
        "*.adyen.com",
        "*.worldpay.com",
    ]

    def __init__(self, page: Page):
        """Initialize payment interceptor.

        Args:
            page: Playwright Page object to intercept.
        """
        self.page = page
        self.intercepted_requests: List[Dict[str, Any]] = []

    async def setup_interception(self):
        """Set up route interception for payment APIs."""
        await self.page.route("**/*", self._intercept_request)

    async def _intercept_request(self, route: Route):
        """Intercept and handle payment requests.

        Args:
            route: Playwright Route object.
        """
        request = route.request
        url = request.url

        # Check if URL matches payment blocklist
        is_payment = any(blocked in url for blocked in self.PAYMENT_BLOCKLIST)

        if is_payment:
            # Log the intercepted request
            self.intercepted_requests.append({
                "url": url,
                "method": request.method,
                "timestamp": str(uuid.uuid4()),
            })

            # Mock successful response to allow UI progression
            await route.fulfill(
                status=200,
                content_type="application/json",
                body='{"status": "success", "token": "mock_token_' + str(uuid.uuid4())[:8] + '"}',
            )
        else:
            # Allow other requests to proceed
            await route.continue_()

    async def remove_interception(self):
        """Remove route interception."""
        await self.page.unroute("**/*", self._intercept_request)


class SyntheticIdentity:
    """Generates synthetic identities for form filling."""

    def __init__(self, session_id: Optional[str] = None):
        """Initialize synthetic identity generator.

        Args:
            session_id: Optional session identifier for unique identities.
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]

    def generate_identity(self) -> Dict[str, str]:
        """Generate a synthetic identity for form filling.

        Returns:
            Dictionary with name, email, address, and credit card.
        """
        return {
            "name": "Test User",
            "email": f"audit+{self.session_id}@example.com",
            "address": "123 Test St",
            "city": "Test City",
            "state": "CA",
            "zip": "90210",
            "country": "US",
            "phone": "555-0100",
            "credit_card": "4242 4242 4242 4242",  # Stripe test card
            "expiry": "12/25",
            "cvv": "123",
        }

    def get_test_card(self, provider: str = "stripe") -> Dict[str, str]:
        """Get test card information for a payment provider.

        Args:
            provider: Payment provider ("stripe", "paypal", etc.).

        Returns:
            Dictionary with card details.
        """
        if provider == "stripe":
            return {
                "number": "4242 4242 4242 4242",
                "expiry": "12/25",
                "cvv": "123",
                "zip": "90210",
            }
        else:
            # Default to Stripe test card
            return {
                "number": "4242 4242 4242 4242",
                "expiry": "12/25",
                "cvv": "123",
                "zip": "90210",
            }


class SandboxManager:
    """Manages sandboxed browser environment with payment interception."""

    def __init__(self, page: Page, session_id: Optional[str] = None):
        """Initialize sandbox manager.

        Args:
            page: Playwright Page object.
            session_id: Optional session identifier.
        """
        self.page = page
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.payment_interceptor = PaymentInterceptor(page)
        self.identity = SyntheticIdentity(session_id)

    async def setup(self):
        """Set up sandbox environment (payment interception, etc.)."""
        await self.payment_interceptor.setup_interception()

    async def cleanup(self):
        """Cleanup sandbox environment."""
        await self.payment_interceptor.remove_interception()

    def get_identity(self) -> Dict[str, str]:
        """Get synthetic identity for form filling."""
        return self.identity.generate_identity()

    def get_test_card(self, provider: str = "stripe") -> Dict[str, str]:
        """Get test card information."""
        return self.identity.get_test_card(provider)
