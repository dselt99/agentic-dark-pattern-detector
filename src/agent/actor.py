"""Actor: Execution layer for Phase 2 architecture.

The Actor is responsible for translating Planner sub-goals into
Playwright actions using Set-of-Marks for precise element selection.
"""

from typing import Dict, Any, Optional, List
import os
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


class Actor:
    """Execution layer for browser automation with Set-of-Marks support."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet",
        provider: str = "anthropic",
        mcp_client=None,
    ):
        """Initialize the Actor.

        Args:
            model: LLM model name to use.
            provider: LLM provider ("openai" or "anthropic").
            mcp_client: MCP client for browser automation.
        """
        self.model = model
        self.provider = provider
        self.mcp_client = mcp_client
        self.llm_client = self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client based on provider."""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return AsyncOpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return AsyncAnthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def execute_task(
        self,
        task: Dict[str, Any],
        dom_tree: Optional[str] = None,
        marked_elements: Optional[Dict[str, Any]] = None,
        short_term_context: Optional[List[Dict[str, Any]]] = None,
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

Return a JSON object with:
{
  "action_type": "click" | "type" | "scroll" | "wait" | "navigate" | "observe",
  "target": mark_id or selector,
  "value": optional value (for type actions),
  "reasoning": why you chose this action
}"""

        context_summary = ""
        if short_term_context:
            context_summary = "\n\nRecent actions:\n"
            for ctx in short_term_context[-3:]:
                if ctx.get("action_taken"):
                    context_summary += f"- {ctx['action_taken']}\n"

        user_prompt = f"""Task: {task.get('goal', 'Unknown')}
Task Type: {task.get('type', 'unknown')}

Current DOM Tree:
```yaml
{dom_tree or 'Not available'}
```

Marked Elements:
{self._format_marked_elements(marked_elements) if marked_elements else 'Not available'}
{context_summary}

What action should I take to accomplish this task? Be specific and use mark IDs when available."""

        try:
            if self.provider == "openai":
                response = await self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                content = response.choices[0].message.content
                import json
                action = json.loads(content)
            else:  # anthropic
                response = await self.llm_client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                content = response.content[0].text
                import json
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    action = json.loads(json_match.group())
                else:
                    # Fallback action
                    action = {"action_type": "observe", "target": None, "reasoning": "Failed to parse action"}
        except Exception as e:
            # Fallback to observation
            action = {
                "action_type": "observe",
                "target": None,
                "reasoning": f"Error generating action: {str(e)}",
            }

        # Execute the action via MCP client
        if self.mcp_client and action.get("action_type") != "observe":
            result = await self._execute_action(action, marked_elements)
            action["execution_result"] = result

        return {
            "action": action,
            "task_completed": action.get("action_type") in ["observe", "wait"],
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
