"""Planner: Strategic goal decomposition for Phase 2 architecture.

The Planner is responsible for decomposing abstract user queries into
a sequence of executable sub-goals using a "Plan-and-Solve" strategy.
"""

import asyncio
import json
import os
import re
from typing import List, Dict, Any, Optional

from anthropic import AsyncAnthropic

# Rate limit configuration
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0
MAX_DOM_SNIPPET_CHARS = 6000  # Reduced from 8000 to be safer


class Planner:
    """Strategic layer for goal decomposition and task planning."""

    def __init__(self, model: str = "claude-3-5-sonnet-20240620", provider: str = "anthropic"):
        """Initialize the Planner.

        Args:
            model: Model name to use (defaults to Sonnet 3.5 for best reasoning).
            provider: LLM provider.
        """
        self.model = model
        self.provider = provider

        if provider != "anthropic":
            raise ValueError(f"Planner currently only supports 'anthropic' provider, got: {provider}")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = AsyncAnthropic(api_key=api_key)

    async def decompose_goal(self, user_query: str, target_url: str) -> List[Dict[str, Any]]:
        """Decompose a high-level user query into a task queue.

        Args:
            user_query: The user's high-level goal.
            target_url: The target URL to audit.

        Returns:
            List of task dictionaries.
        """
        system_prompt = """You are a strategic planner for a web automation agent that detects dark patterns on websites.
Your task is to decompose high-level user goals into a sequence of executable sub-tasks.

Each task should be:
- Specific and actionable
- Sequenced logically
- Testable (has clear success criteria)

Return a JSON array of tasks, each with:
- id: Sequential number (0, 1, 2, ...)
- type: One of ["navigate", "observe", "interact", "analyze", "verify", "reload", "dismiss"]
- goal: Clear description of what to accomplish
- target: (Optional) Specific element description or text to interact with
- dependencies: List of task IDs that must complete first (optional)

IMPORTANT RULES:

1. Keep plans SHORT and action-focused. Maximum 10-12 tasks. Prefer interact/click tasks over observe tasks.
   Do NOT generate multiple observe tasks in a row — one observe is enough to assess page state.

2. Dismiss tasks: Include at most ONE dismiss task early in the plan. If no popup is found, the agent will
   move on automatically. Do NOT plan elaborate multi-step popup detection sequences.

3. False Urgency detection: If the page might have countdown timers or urgency messaging, include ONE
   "reload" task followed by ONE "observe" to compare timer states.

4. PURCHASE / CHECKOUT FLOW: When the user goal involves buying, purchasing, shopping, booking, or
   subscribing, you MUST plan the FULL end-to-end journey through checkout:
   a) Find the product/service (search, browse, or navigate to it)
   b) Click on the item to view its details
   c) Select required options (size, color, dates, quantity, plan tier) AND add to cart / buy — do
      this in ONE task so it happens on the product page, not as separate tasks
   d) Proceed through each checkout step (cart → shipping → payment)
   e) Analyze each checkout step for dark patterns (hidden fees, pre-checked extras, drip pricing,
      sneak-into-basket items, forced continuity opt-ins, confusing opt-out flows)
   f) Fill payment info if a form appears (use test card: 4242 4242 4242 4242, exp 12/29, CVC 123)
   g) Verify the final total matches the originally advertised price

   The checkout flow is where the most important dark patterns hide. Getting to checkout is CRITICAL.
   Do not stop at the product page — always push through to the final price screen.

5. Each task type implies a specific action:
   - "navigate": Go to a URL
   - "interact": Click, type, or submit — requires a concrete action on a specific element
   - "observe": Look at page state (use sparingly — only when you need to assess before acting)
   - "analyze": Deep analysis of current page for dark patterns
   - "verify": Confirm expected state (e.g., price matches, item in cart)
   - "reload": Refresh page (for false urgency detection)
   - "dismiss": Close popups/banners

Example for "Buy the cheapest item and look for dark patterns":
[
  {"id": 0, "type": "navigate", "goal": "Navigate to site homepage"},
  {"id": 1, "type": "dismiss", "goal": "Dismiss any cookie banners or pop-ups"},
  {"id": 2, "type": "interact", "goal": "Search for the product using the site's search bar"},
  {"id": 3, "type": "observe", "goal": "Scan search results for urgency claims, scarcity messaging, or misleading pricing"},
  {"id": 4, "type": "interact", "goal": "Click on a product listing to view its details"},
  {"id": 5, "type": "analyze", "goal": "Analyze product page for dark patterns: fake urgency, misleading pricing, pre-checked add-ons"},
  {"id": 6, "type": "interact", "goal": "Select any required product options (size, color, quantity, dates, plan tier, etc.) then click Add to Cart, Buy Now, Book, or equivalent purchase button. If no options are required, click the purchase button directly. If no cart exists, proceed to checkout."},
  {"id": 7, "type": "interact", "goal": "Proceed to checkout or next step in the purchase flow"},
  {"id": 8, "type": "analyze", "goal": "Analyze checkout for hidden fees, drip pricing, sneak-into-basket items, forced continuity opt-ins"},
  {"id": 9, "type": "interact", "goal": "Fill payment form with test card (4242424242424242, exp 12/29, CVC 123) if prompted"},
  {"id": 10, "type": "verify", "goal": "Verify final total matches advertised price, document any price discrepancies or unexpected charges"}
]"""

        user_prompt = f"""User Query: {user_query}
Target URL: {target_url}

Generate a task queue to accomplish this goal. Be specific."""

        tasks = await self._call_model(system_prompt, user_prompt)
        return self._normalize_tasks(tasks)

    async def replan_goal(
        self,
        original_query: str,
        failed_step: str,
        error_context: str,
        current_dom: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate a new plan when a task fails, using visible context.

        Args:
            original_query: The original high-level goal.
            failed_step: The specific goal of the step that failed.
            error_context: The error message (e.g., "Timeout", "Element not found").
            current_dom: (Optional) Snippet of the current DOM/Accessibility tree.

        Returns:
            New task queue starting from the corrective action.
        """
        system_prompt = """You are a Tactical Recovery Expert for a web agent.
Your goal is to analyze a failure and generate a *corrective* sequence of actions.

### DIAGNOSIS
1. Analyze the `error_context` and `current_dom`.
2. CHECK: Is a popup/modal blocking the view? -> Plan a `dismiss` action.
3. CHECK: Did the selector fail? -> Plan an `observe` action to find the new selector, then `interact`.
4. CHECK: Are we in a loop? -> Plan a `Maps` to a known safe URL.

### OUTPUT
Return a JSON array of tasks to fix the error and resume progress.
"""
        
        # Truncate DOM to avoid token limits, keeping the most relevant structure
        dom_snippet = (current_dom[:MAX_DOM_SNIPPET_CHARS] + "...") if current_dom else "No DOM provided"

        user_prompt = f"""CONTEXT:
Original Goal: {original_query}
Failed Step: {failed_step}
Error Message: {error_context}

CURRENT PAGE STATE (DOM SNIPPET):
{dom_snippet}

ACTION:
Generate a recovery plan. If you see a popup in the DOM, dismiss it first."""

        tasks = await self._call_model(system_prompt, user_prompt)
        return self._normalize_tasks(tasks)

    async def _call_model(self, system: str, user: str) -> List[Dict[str, Any]]:
        """Helper to call Anthropic and parse JSON with retry logic."""
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                content = response.content[0].text

                # Extract JSON array - try multiple strategies
                # Strategy 1: Find JSON code block
                code_block_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', content)
                if code_block_match:
                    try:
                        return json.loads(code_block_match.group(1))
                    except json.JSONDecodeError:
                        pass

                # Strategy 2: Find first complete JSON array using bracket matching
                start_idx = content.find('[')
                if start_idx != -1:
                    bracket_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(content[start_idx:], start_idx):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i + 1
                                break

                    if end_idx > start_idx:
                        try:
                            return json.loads(content[start_idx:end_idx])
                        except json.JSONDecodeError:
                            pass

                # Strategy 3: Try parsing full content
                return json.loads(content)

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Check if it's a rate limit error
                if "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < MAX_RETRIES - 1:
                        delay = BASE_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                        print(f"[Planner] Rate limited, waiting {delay}s before retry {attempt + 2}/{MAX_RETRIES}")
                        await asyncio.sleep(delay)
                        continue

                # For other errors, log and break
                print(f"Planner Error: {e}")
                break

        # All retries exhausted - return fallback
        print(f"[Planner] Failed after {MAX_RETRIES} attempts: {last_error}")
        return [
            {"id": 0, "type": "observe", "goal": "Re-assess page state after error"},
            {"id": 1, "type": "verify", "goal": "Check for blocking elements"}
        ]

    def _normalize_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize task structure to ensure consistent format.

        The LLM may return different field names. This ensures we have
        the expected 'type' and 'goal' fields.

        Args:
            tasks: Raw task list from LLM.

        Returns:
            Normalized task list.
        """
        normalized = []
        for i, task in enumerate(tasks):
            norm_task = {"id": task.get("id", i)}

            # Normalize 'type' field (may be 'action' or 'type')
            task_type = task.get("type") or task.get("action", "unknown")
            # Map action values to expected types
            action_map = {
                "observe": "observe",
                "interact": "interact",
                "click": "interact",
                "type": "interact",
                "navigate": "navigate",
                "dismiss": "dismiss",
                "reload": "reload",
                "wait": "observe",
                "analyze": "analyze",
                "verify": "verify",
            }
            norm_task["type"] = action_map.get(task_type.lower(), task_type)

            # Normalize 'goal' field (may be 'goal', 'description', or 'reason')
            norm_task["goal"] = (
                task.get("goal")
                or task.get("description")
                or task.get("reason")
                or "Unknown task"
            )

            # Preserve target if specified
            if task.get("target"):
                norm_task["target"] = task.get("target")
            elif task.get("selector"):
                norm_task["target"] = task.get("selector")

            # Preserve any other useful fields
            if task.get("value"):
                norm_task["value"] = task.get("value")

            normalized.append(norm_task)

        return normalized