"""Actor: Execution layer for Phase 2 architecture.

The Actor is responsible for translating Planner sub-goals into
Playwright actions using Set-of-Marks for precise element selection.
"""

import asyncio
import json
import os
import re
from typing import Dict, Any, Optional, List

from anthropic import AsyncAnthropic

# Rate limit configuration
MAX_DOM_CHARS = 40000  # Truncate DOM to avoid token explosion
MAX_MARKED_ELEMENTS = 150  # Limit marked elements sent to LLM
MAX_RETRIES = 3  # Retries for rate limit errors
BASE_RETRY_DELAY = 2.0  # Base delay for exponential backoff


class Actor:
    """Execution layer for browser automation with Set-of-Marks support."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", provider: str = "anthropic", mcp_client=None):
        """Initialize the Actor.

        Args:
            model: Model name to use.
            provider: LLM provider (currently only "anthropic" is supported).
            mcp_client: MCP client for browser automation.
        """
        self.model = model
        self.provider = provider
        self.mcp_client = mcp_client

        # Currently only Anthropic is supported for the Actor
        if provider != "anthropic":
            raise ValueError(f"Actor currently only supports 'anthropic' provider, got: {provider}")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = AsyncAnthropic(api_key=api_key)

    async def execute_task(
        self,
        task: Dict[str, Any],
        dom_tree: Optional[str] = None,
        marked_elements: Optional[Dict[str, Any]] = None,
        short_term_context: Optional[List[Dict[str, Any]]] = None,
        failed_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute a task by generating and performing browser actions.

        Args:
            task: Task dictionary with type and goal.
            dom_tree: Current accessibility tree (YAML).
            marked_elements: Dictionary mapping mark IDs to element selectors.
            short_term_context: Last 3 interaction snapshots for context.

        Returns:
            Dictionary with action result and next state.
        """
        # Prune DOM tree to avoid rate limits
        pruned_dom = self._prune_dom_tree(dom_tree, task)

        # Filter marked elements to most relevant ones
        filtered_marks = self._filter_marked_elements(marked_elements, task)

        task_type = task.get('type', 'unknown')

        system_prompt = """You are a web automation actor that mimics a "naive user" behavior.
Your goal is to interact with web pages naturally, experiencing friction and dark patterns
as a real user would.

IMPORTANT: You must NOT use developer tools or bypass UI elements. You must interact
through the normal UI, even if it's slow or frustrating. This is intentional - we need
to experience the dark patterns to detect them.

When you see marked elements (numbered overlays), use the mark ID for precise actions:
- CLICK(42) means click element with mark ID 42
- TYPE(42, "text") means type into element with mark ID 42
- SCROLL(direction) means scroll the viewport

If no marked elements are available, use CSS selectors from the DOM tree.

CRITICAL RULES BY TASK TYPE:
- For "interact" tasks: You MUST return "click" or "type" action. DO NOT return "observe".
  If you cannot find the target element, look for similar elements or scroll to find it.
- For "type" tasks: You MUST return a "type" action with the text to enter.
- For "dismiss" tasks: You MUST return a "click" action on a close/dismiss button.
- For "observe" tasks: Return "observe" to analyze the page state.
- For "navigate" tasks: Return "navigate" with the URL.

If the task says "Enter destination" or "Enter [something] in search/input":
1. Look for input fields, search boxes, or comboboxes in the marked elements
2. Use "type" action with the value to enter
3. Common selectors: input[type="text"], [role="combobox"], [role="searchbox"]

Return a JSON object with:
{
  "action_type": "click" | "type" | "scroll" | "wait" | "navigate" | "reload" | "observe",
  "target": mark_id (integer) or CSS selector (string),
  "value": text to type (required for "type" actions) or URL (for "navigate"),
  "reasoning": why you chose this action
}

Use "reload" when testing for False Urgency patterns - reload the page to check if countdown
timers reset to their original values (indicating fake urgency rather than real deadlines).

Never repeat an action that just failed. If an action failed, try a different selector, scroll to find the element, or take an alternative path."""

        context_summary = ""
        if short_term_context:
            context_summary = "\n\nRecent actions:\n"
            for ctx in short_term_context[-3:]:
                if ctx.get("action_taken"):
                    context_summary += f"- {ctx['action_taken']}\n"

        failed_actions_summary = ""
        if failed_actions:
            failed_actions_summary = "\n\nFailed actions (DO NOT repeat these):\n"
            for fa in failed_actions[-5:]:
                action_type = fa.get("type", "unknown")
                target = fa.get("target", "unknown")
                error = fa.get("error", "unknown")
                failed_actions_summary += f"- {action_type} {target} -> {error}\n"

        user_prompt = f"""Task: {task.get('goal', 'Unknown')}
Task Type: {task.get('type', 'unknown')}

Current DOM Tree:
```yaml
{pruned_dom or 'Not available'}
```

Marked Elements:
{self._format_marked_elements(filtered_marks) if filtered_marks else 'Not available'}
{context_summary}{failed_actions_summary}

What action should I take to accomplish this task? Be specific and use mark IDs when available."""

        # Call LLM with retry logic for rate limits
        action = await self._call_llm_with_retry(system_prompt, user_prompt)

        # Execute the action via MCP client
        if self.mcp_client and action.get("action_type") != "observe":
            result = await self._execute_action(action, marked_elements)
            action["execution_result"] = result

        # Determine task completion based on task type and action taken
        # "observe" only completes "observe" tasks, NOT interact/type/dismiss tasks
        action_type = action.get("action_type")
        task_completed = False

        if task_type == "observe":
            # Observe tasks complete when we observe
            task_completed = action_type == "observe"
        elif task_type in ["interact", "type", "click"]:
            # Interact tasks only complete when we actually interact
            task_completed = action_type in ["click", "type"]
        elif task_type == "dismiss":
            # Dismiss tasks complete when we click (to dismiss)
            task_completed = action_type == "click"
        elif task_type == "navigate":
            task_completed = action_type == "navigate"
        elif task_type == "reload":
            task_completed = action_type == "reload"
        elif task_type == "scroll":
            task_completed = action_type == "scroll"
        else:
            # Default: any action completes unknown task types
            task_completed = action_type in ["observe", "wait"]

        return {
            "action": action,
            "task_completed": task_completed,
        }

    async def _execute_action(
        self, action: Dict[str, Any], marked_elements: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute an action via MCP client.

        Args:
            action: Action dictionary from LLM.
            marked_elements: Mapping of mark IDs to selectors.

        Returns:
            Execution result dictionary.
        """
        action_type = action.get("action_type")
        target = action.get("target")

        # Resolve mark ID to selector if needed
        selector = target
        if isinstance(target, int) and marked_elements:
            selector = marked_elements.get(str(target), {}).get("selector", target)

        if action_type == "click" and selector:
            return await self.mcp_client.call_tool("browser_click", selector=selector)
        elif action_type == "type" and selector:
            text = action.get("value", "")
            return await self.mcp_client.call_tool("browser_type", selector=selector, text=text)
        elif action_type == "scroll":
            direction = action.get("value", "down")
            return await self.mcp_client.call_tool("browser_scroll", direction=direction)
        elif action_type == "navigate":
            url = action.get("value", "")
            return await self.mcp_client.call_tool("browser_navigate", url=url)
        elif action_type == "reload":
            return await self.mcp_client.call_tool("browser_reload")
        elif action_type == "wait":
            return await self.mcp_client.call_tool("browser_wait_for_stability")
        else:
            return {"status": "skipped", "message": f"Action type {action_type} not executable"}

    def _format_marked_elements(self, marked_elements: Dict[str, Any]) -> str:
        """Format marked elements for prompt."""
        if not marked_elements:
            return "No marked elements available"

        formatted = "Mark ID -> Element Info:\n"
        for mark_id, element_info in marked_elements.items():
            selector = element_info.get("selector", "unknown")
            tag = element_info.get("tag", "unknown")
            text = element_info.get("text", "")[:50]
            formatted += f"  {mark_id}: {tag} ({selector}) - {text}\n"

        return formatted

    def _prune_dom_tree(self, dom_tree: Optional[str], task: Dict[str, Any]) -> Optional[str]:
        """Prune DOM tree to reduce token usage while preserving relevant content.

        Args:
            dom_tree: Full DOM tree in YAML format.
            task: Current task for context-aware pruning.

        Returns:
            Pruned DOM tree string.
        """
        if not dom_tree:
            return None

        # If DOM is within limits, return as-is
        if len(dom_tree) <= MAX_DOM_CHARS:
            return dom_tree

        # Extract task keywords for relevance scoring
        task_goal = task.get("goal", "").lower()
        task_type = task.get("type", "").lower()

        # Priority keywords based on task type
        priority_keywords = []
        if task_type in ["interact", "click", "type"]:
            priority_keywords = ["button", "input", "link", "a ", "form", "submit", "search"]
        elif task_type == "dismiss":
            priority_keywords = ["close", "dismiss", "accept", "cookie", "banner", "modal", "popup", "dialog"]
        elif task_type == "navigate":
            priority_keywords = ["nav", "menu", "link", "href", "a "]
        elif task_type in ["observe", "analyze"]:
            priority_keywords = ["price", "timer", "countdown", "cart", "total", "fee", "shipping"]

        # Add keywords from task goal
        for word in task_goal.split():
            if len(word) > 3:
                priority_keywords.append(word)

        # Split DOM into lines and score by relevance
        lines = dom_tree.split('\n')
        scored_lines = []

        for i, line in enumerate(lines):
            line_lower = line.lower()
            score = 0

            # Score based on keyword matches
            for keyword in priority_keywords:
                if keyword in line_lower:
                    score += 2

            # Boost interactive elements
            if any(tag in line_lower for tag in ['button', 'input', 'select', 'a ', 'link']):
                score += 1

            # Keep structural lines (indentation markers)
            if line.strip().startswith('-') or line.strip().startswith('tag:'):
                score += 0.5

            scored_lines.append((i, score, line))

        # Sort by score (descending) but maintain some structure
        # Keep first 20 lines for context (header/structure)
        header_lines = lines[:20]
        remaining = scored_lines[20:]

        # Sort remaining by score
        remaining.sort(key=lambda x: (-x[1], x[0]))

        # Calculate how many chars we can use after header
        header_chars = sum(len(line) for line in header_lines)
        remaining_budget = MAX_DOM_CHARS - header_chars - 100  # Reserve for truncation message

        # Select highest-scoring lines that fit budget
        selected_lines = []
        current_chars = 0
        for idx, score, line in remaining:
            if current_chars + len(line) > remaining_budget:
                break
            selected_lines.append((idx, line))
            current_chars += len(line)

        # Sort selected lines by original index to maintain some structure
        selected_lines.sort(key=lambda x: x[0])

        # Combine header and selected lines
        result_lines = header_lines + [line for _, line in selected_lines]
        result = '\n'.join(result_lines)

        return result + f"\n... [DOM truncated: {len(dom_tree)} -> {len(result)} chars]"

    def _filter_marked_elements(
        self, marked_elements: Optional[Dict[str, Any]], task: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Filter marked elements to most relevant ones for the task.

        Args:
            marked_elements: Full dictionary of marked elements.
            task: Current task for relevance filtering.

        Returns:
            Filtered dictionary of marked elements.
        """
        if not marked_elements:
            return None

        # If within limits, return as-is
        if len(marked_elements) <= MAX_MARKED_ELEMENTS:
            return marked_elements

        task_goal = task.get("goal", "").lower()
        task_type = task.get("type", "").lower()

        # Score elements by relevance
        scored_elements = []
        priority_elements = []  # Always include these

        for mark_id, element_info in marked_elements.items():
            score = 0
            tag = element_info.get("tag", "").lower()
            elem_type = element_info.get("type", "").lower()
            text = element_info.get("text", "").lower()
            selector = element_info.get("selector", "").lower()

            # ALWAYS include form input elements (text inputs, search boxes, textareas)
            # These are critical for interaction tasks
            is_text_input = (
                tag == "input" and elem_type in ["text", "search", "email", "tel", "url", ""]
            ) or tag == "textarea"

            # Check for combobox/searchbox roles in selector
            is_searchbox = any(kw in selector for kw in ["search", "combobox", "autocomplete", "destination"])

            if is_text_input or is_searchbox:
                priority_elements.append((mark_id, element_info, 100))  # High priority
                continue

            # Task-type based scoring
            if task_type == "dismiss":
                if any(kw in text for kw in ["close", "dismiss", "accept", "reject", "x", "got it", "no thanks"]):
                    score += 5
                if any(kw in selector for kw in ["close", "dismiss", "modal", "dialog", "banner"]):
                    score += 3
            elif task_type in ["interact", "click"]:
                if tag in ["button", "a", "input"]:
                    score += 2
            elif task_type == "type":
                if tag in ["input", "textarea"]:
                    score += 5

            # Match task goal keywords
            for word in task_goal.split():
                if len(word) > 2 and word in text:
                    score += 2
                if len(word) > 2 and word in selector:
                    score += 1

            # Boost common interactive elements
            if tag in ["button", "a", "select"]:
                score += 1

            # Boost elements that look like primary actions
            if any(kw in text for kw in ["search", "submit", "continue", "next", "book", "reserve"]):
                score += 3

            # Boost elements closer to current viewport (lower y = more visible)
            y_coord = element_info.get("y", 9999)
            if y_coord < 600:
                score += 3  # Above fold
            elif y_coord < 1200:
                score += 1  # Near viewport

            scored_elements.append((mark_id, element_info, score))

        # Sort by score and take top N
        scored_elements.sort(key=lambda x: -x[2])

        # Combine priority elements with top scored elements
        # Ensure priority elements are always included
        remaining_slots = MAX_MARKED_ELEMENTS - len(priority_elements)
        filtered = {
            mark_id: info
            for mark_id, info, score in priority_elements
        }
        filtered.update({
            mark_id: info
            for mark_id, info, score in scored_elements[:remaining_slots]
        })

        return filtered

    async def _call_llm_with_retry(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call LLM with exponential backoff retry for rate limits.

        Args:
            system_prompt: System prompt for the LLM.
            user_prompt: User prompt with task and context.

        Returns:
            Parsed action dictionary.
        """
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                content = response.content[0].text

                # Strategy 1: Find JSON code block
                code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
                if code_block_match:
                    try:
                        return json.loads(code_block_match.group(1))
                    except json.JSONDecodeError:
                        pass

                # Strategy 2: Bracket-matching extraction (handles nested JSON correctly)
                start_idx = content.find('{')
                if start_idx != -1:
                    bracket_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(content[start_idx:], start_idx):
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        try:
                            return json.loads(content[start_idx:end_idx])
                        except json.JSONDecodeError:
                            pass

                # Strategy 3: Try full content
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"action_type": "observe", "target": None, "reasoning": "Failed to parse action"}

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Check if it's a rate limit error
                if "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < MAX_RETRIES - 1:
                        delay = BASE_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                        print(f"[Actor] Rate limited, waiting {delay}s before retry {attempt + 2}/{MAX_RETRIES}")
                        await asyncio.sleep(delay)
                        continue

                # For other errors, don't retry
                break

        # All retries exhausted or non-retryable error
        return {
            "action_type": "observe",
            "target": None,
            "reasoning": f"Error generating action after {MAX_RETRIES} attempts: {str(last_error)}",
        }
