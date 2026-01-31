"""Planner: Strategic goal decomposition for Phase 2 architecture.

The Planner is responsible for decomposing abstract user queries into
a sequence of executable sub-goals using a "Plan-and-Solve" strategy.
"""

from typing import List, Dict, Any, Optional
import os
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


class Planner:
    """Strategic layer for goal decomposition and task planning."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet",
        provider: str = "anthropic",
    ):
        """Initialize the Planner.

        Args:
            model: LLM model name to use.
            provider: LLM provider ("openai" or "anthropic").
        """
        self.model = model
        self.provider = provider
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

    async def decompose_goal(self, user_query: str, target_url: str) -> List[Dict[str, Any]]:
        """Decompose a high-level user query into a task queue.

        Args:
            user_query: The user's high-level goal (e.g., "Audit cancellation flow").
            target_url: The target URL to audit.

        Returns:
            List of task dictionaries with id, type, and goal fields.
        """
        system_prompt = """You are a strategic planner for a web automation agent.
Your task is to decompose high-level user goals into a sequence of executable sub-tasks.

Each task should be:
- Specific and actionable
- Sequenced logically
- Testable (has clear success criteria)

Return a JSON array of tasks, each with:
- id: Sequential number (0, 1, 2, ...)
- type: One of ["navigate", "observe", "interact", "analyze", "verify"]
- goal: Clear description of what to accomplish
- dependencies: List of task IDs that must complete first (optional)

Example for "Buy the cheapest item":
[
  {"id": 0, "type": "navigate", "goal": "Navigate to product listing page"},
  {"id": 1, "type": "observe", "goal": "Get accessibility tree of product list"},
  {"id": 2, "type": "interact", "goal": "Sort products by price (ascending)"},
  {"id": 3, "type": "interact", "goal": "Click on first (cheapest) product"},
  {"id": 4, "type": "interact", "goal": "Add product to cart"},
  {"id": 5, "type": "observe", "goal": "Verify cart contains only the selected item"},
  {"id": 6, "type": "interact", "goal": "Proceed to checkout"},
  {"id": 7, "type": "analyze", "goal": "Compare final price to initial price"}
]"""

        user_prompt = f"""User Query: {user_query}
Target URL: {target_url}

Generate a task queue to accomplish this goal. Be specific and break down complex actions into smaller steps."""

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
                # Parse JSON response
                import json
                result = json.loads(content)
                tasks = result.get("tasks", [])
            else:  # anthropic
                response = await self.llm_client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                content = response.content[0].text
                # Extract JSON from response
                import json
                import re
                # Try to find JSON array in response
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    tasks = json.loads(json_match.group())
                else:
                    # Fallback: try parsing entire response as JSON
                    result = json.loads(content)
                    tasks = result.get("tasks", [])
        except Exception as e:
            # Fallback to simple task queue on error
            tasks = [
                {"id": 0, "type": "navigate", "goal": f"Navigate to {target_url}"},
                {"id": 1, "type": "observe", "goal": "Get accessibility tree"},
                {"id": 2, "type": "analyze", "goal": f"Analyze for: {user_query}"},
            ]

        return tasks

    async def re_plan(
        self,
        failed_task: Dict[str, Any],
        error_message: str,
        completed_tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate a new plan when a task fails.

        Args:
            failed_task: The task that failed.
            error_message: Description of why it failed.
            completed_tasks: List of tasks that were successfully completed.

        Returns:
            New task queue starting from the failed task.
        """
        system_prompt = """You are a strategic planner that adapts when tasks fail.
Generate an alternative approach to accomplish the same goal when the original plan fails."""

        user_prompt = f"""Original task that failed:
{failed_task}

Error: {error_message}

Completed tasks so far:
{completed_tasks}

Generate a new task queue that:
1. Builds on completed tasks
2. Uses an alternative approach for the failed task
3. Continues toward the original goal"""

        try:
            if self.provider == "openai":
                response = await self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,  # Slightly higher for creativity
                )
                content = response.choices[0].message.content
                import json
                result = json.loads(content)
                new_tasks = result.get("tasks", [])
            else:  # anthropic
                response = await self.llm_client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                content = response.content[0].text
                import json
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    new_tasks = json.loads(json_match.group())
                else:
                    result = json.loads(content)
                    new_tasks = result.get("tasks", [])
        except Exception:
            # Fallback: return original failed task with error handling
            new_tasks = [
                {
                    "id": failed_task["id"],
                    "type": "observe",
                    "goal": f"Investigate error: {error_message}",
                }
            ]

        return new_tasks
