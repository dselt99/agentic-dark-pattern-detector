"""Dark Pattern Agent - The Orchestrator.

This module implements Pillar 4: The Agent, using the ReAct pattern
(Reason + Act) to orchestrate dark pattern detection audits.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from ..schemas import AuditResult, DetectedPattern
from ..schemas.utils import (
    get_audit_result_schema,
    validate_and_parse,
    create_self_correction_prompt,
    inject_schema_into_system_prompt,
)
from .mcp_client import MCPClient


class DarkPatternAgent:
    """Orchestrates dark pattern detection audits using the ReAct pattern.

    The agent manages the lifecycle of an audit session, coordinating
    between the MCP server (capabilities), skill definitions (reasoning),
    and LLM (decision-making).
    """

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        model: str = "claude-3-5-sonnet",
        provider: str = "anthropic",
        max_steps: int = 50,
    ):
        """Initialize the Dark Pattern Agent.

        Args:
            mcp_client: MCP client for browser automation. If None, creates default.
            model: LLM model name to use.
            provider: LLM provider ("openai" or "anthropic").
            max_steps: Maximum number of ReAct loop iterations.
        """
        self.client = mcp_client or MCPClient()
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.skills = self._load_skill("skills/detect-manipulation.md")
        self.schema = get_audit_result_schema()

        # State management
        self.visited_urls: Set[str] = set()
        self.observed_elements: Set[str] = set()
        self.memory: List[Dict[str, Any]] = []
        self.screenshot_paths: List[str] = []

        # Rate limiting
        self.rate_limit_delay = float(
            os.getenv("RATE_LIMIT_DELAY_SECONDS", "2")
        )

        # Initialize LLM client
        self.llm_client = self._init_llm_client()

    def _load_skill(self, skill_path: str) -> str:
        """Load skill definition from markdown file.

        Args:
            skill_path: Path to the skill file.

        Returns:
            Skill content as string.
        """
        # Try relative path first
        skill_file = Path(skill_path)
        if not skill_file.exists():
            # Try relative to project root (find it by looking for pyproject.toml)
            current = Path(__file__).parent
            while current != current.parent:
                project_root = current
                if (project_root / "pyproject.toml").exists():
                    skill_file = project_root / skill_path
                    break
                current = current.parent

        if not skill_file.exists():
            raise FileNotFoundError(
                f"Skill file not found: {skill_path}. "
                "Please ensure skills/detect-manipulation.md exists."
            )
        return skill_file.read_text(encoding="utf-8")

    def _init_llm_client(self):
        """Initialize LLM client based on provider.

        Returns:
            Initialized LLM client.
        """
        if self.provider == "openai":
            try:
                from openai import AsyncOpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                return AsyncOpenAI(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )

        elif self.provider == "anthropic":
            try:
                from anthropic import AsyncAnthropic

                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError(
                        "ANTHROPIC_API_KEY environment variable not set"
                    )
                return AsyncAnthropic(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _check_robots_compliance(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt.

        Args:
            url: URL to check.

        Returns:
            True if allowed, False otherwise.
        """
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()

            user_agent = "DarkPatternHunter"
            return rp.can_fetch(user_agent, url)
        except Exception:
            # If robots.txt doesn't exist or is unparseable, allow by default
            return True

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        max_retries: int = 3,
    ) -> AuditResult:
        """Call LLM with schema enforcement and self-correction.

        Args:
            prompt: User prompt with current context.
            system_prompt: System prompt with skills and schema.
            max_retries: Maximum number of retry attempts.

        Returns:
            Validated AuditResult instance.
        """
        for attempt in range(1, max_retries + 1):
            try:
                if self.provider == "openai":
                    response = await self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1,  # Low temperature for deterministic output
                    )
                    json_str = response.choices[0].message.content

                elif self.provider == "anthropic":
                    response = await self.llm_client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    json_str = response.content[0].text

                # Validate and parse
                result, error = validate_and_parse(json_str, AuditResult)
                if result is not None:
                    return result

                # Validation failed - create correction prompt
                if attempt < max_retries:
                    prompt = create_self_correction_prompt(
                        prompt, error, attempt, max_retries
                    )
                else:
                    raise ValueError(
                        f"Failed to generate valid AuditResult after {max_retries} attempts. "
                        f"Last error: {error}"
                    )

            except Exception as e:
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"LLM call failed after {max_retries} attempts: {str(e)}"
                    )
                await asyncio.sleep(1)  # Brief delay before retry

        raise RuntimeError("Unexpected error in LLM call loop")

    async def run_audit(self, url: str) -> AuditResult:
        """Run a complete dark pattern audit on the target URL.

        This implements the ReAct pattern:
        - Observe: Get accessibility tree
        - Reason: LLM analyzes against skills and schema
        - Act: Execute tools or return final result

        Args:
            url: Target URL to audit.

        Returns:
            AuditResult with findings and metadata.
        """
        # Reset state for new audit
        self.visited_urls.clear()
        self.observed_elements.clear()
        self.memory.clear()
        self.screenshot_paths.clear()

        # 1. Respect Robots.txt (Responsible Auditor)
        if not self._check_robots_compliance(url):
            return AuditResult(
                target_url=url,
                findings=[],
                screenshot_paths=[],
                summary="Audit aborted: robots.txt disallows access to this URL.",
            )

        # 2. Navigate to target URL
        nav_result = await self.client.call_tool("browser_navigate", url=url)
        if nav_result.get("status") != "success":
            return AuditResult(
                target_url=url,
                findings=[],
                screenshot_paths=[],
                summary=f"Audit failed: Could not navigate to URL. {nav_result.get('message', 'Unknown error')}",
            )

        self.visited_urls.add(url)

        # 3. Main ReAct Loop
        for step in range(self.max_steps):
            # Observe: Get accessibility tree
            tree_result = await self.client.call_tool("get_accessibility_tree")
            if tree_result.get("status") != "success":
                continue  # Skip this iteration if tree fetch failed

            tree_yaml = tree_result.get("tree", "")

            # Build prompt with current context
            prompt = self._build_reasoning_prompt(url, tree_yaml, step)

            # Inject schema into system prompt
            system_prompt = inject_schema_into_system_prompt(
                self.skills, self.schema, self.provider
            )

            # Reason: LLM call with schema enforcement
            try:
                decision = await self._call_llm(prompt, system_prompt)

                # Check if audit is complete
                if self._is_audit_complete(decision, step):
                    # Add screenshot paths from memory
                    decision.screenshot_paths = self.screenshot_paths
                    return decision

                # Act: Check if LLM wants to use tools
                # (In a more sophisticated implementation, the LLM would explicitly
                # request tool calls. For now, we check if findings are complete.)
                if decision.findings:
                    # Take screenshots for evidence
                    for finding in decision.findings:
                        screenshot_result = await self.client.call_tool(
                            "take_screenshot",
                            selector=finding.element_selector,
                            filename_prefix=f"{finding.pattern_type.value}_evidence",
                        )
                        if screenshot_result.get("status") == "success":
                            self.screenshot_paths.append(
                                screenshot_result.get("path", "")
                            )

                # Store decision in memory
                self.memory.append(decision.model_dump())

            except Exception as e:
                # If LLM call fails, continue with next iteration
                # or return partial results
                if step >= self.max_steps - 1:
                    return AuditResult(
                        target_url=url,
                        findings=[],
                        screenshot_paths=self.screenshot_paths,
                        summary=f"Audit incomplete: Error during reasoning step. {str(e)}",
                    )

            # Rate Limiting (Responsible Auditor)
            await asyncio.sleep(self.rate_limit_delay)

        # Max steps reached - return final result
        final_result = self._synthesize_final_result(url)
        return final_result

    def _build_reasoning_prompt(
        self, url: str, tree_yaml: str, step: int
    ) -> str:
        """Build the reasoning prompt for the LLM.

        Args:
            url: Current URL being audited.
            tree_yaml: Accessibility tree in YAML format.
            step: Current step number.

        Returns:
            Formatted prompt string.
        """
        memory_summary = ""
        if self.memory:
            memory_summary = f"\n\nPrevious observations ({len(self.memory)} steps):\n"
            for i, mem in enumerate(self.memory[-3:], 1):  # Last 3 steps
                memory_summary += f"Step {i}: {mem.get('summary', 'N/A')}\n"

        return f"""You are auditing the website at {url} for dark patterns.

Current accessibility tree (YAML):
```yaml
{tree_yaml}
```

Your task:
1. Analyze the accessibility tree for potential dark patterns
2. Reference the skill definitions to identify specific patterns
3. If you detect a pattern, provide:
   - pattern_type (from PatternType enum)
   - confidence_score (≥ 0.7, only report if confident)
   - element_selector (precise CSS/XPath selector)
   - reasoning (reference specific heuristics)
   - evidence (text content that triggered detection)

4. If you need more information, you can request tool calls (though the current implementation focuses on static analysis)

Current step: {step + 1}/{self.max_steps}
{memory_summary}

Provide your audit result as a JSON object matching the AuditResult schema."""

    def _is_audit_complete(self, decision: AuditResult, step: int) -> bool:
        """Determine if the audit is complete.

        Args:
            decision: Current LLM decision.
            step: Current step number.

        Returns:
            True if audit should terminate.
        """
        # Complete if we have findings and confidence is high
        if decision.findings and len(decision.findings) > 0:
            return True

        # Complete if we've reached max steps
        if step >= self.max_steps - 1:
            return True

        # Complete if summary indicates completion
        if "complete" in decision.summary.lower() or "finished" in decision.summary.lower():
            return True

        return False

    def _synthesize_final_result(self, url: str) -> AuditResult:
        """Synthesize final audit result from memory.

        Args:
            url: Target URL.

        Returns:
            Combined AuditResult from all steps.
        """
        # Aggregate findings from all steps
        all_findings: List[DetectedPattern] = []
        seen_selectors: Set[str] = set()

        for mem in self.memory:
            if "findings" in mem:
                for finding_data in mem["findings"]:
                    selector = finding_data.get("element_selector", "")
                    if selector and selector not in seen_selectors:
                        try:
                            finding = DetectedPattern.model_validate(finding_data)
                            all_findings.append(finding)
                            seen_selectors.add(selector)
                        except Exception:
                            # Skip invalid findings
                            pass

        # Create summary
        if all_findings:
            summary = f"Audit completed. Found {len(all_findings)} dark pattern(s): "
            summary += ", ".join(
                f.pattern_type.value for f in all_findings
            )
        else:
            summary = "Audit completed. No dark patterns detected."

        return AuditResult(
            target_url=url,
            findings=all_findings,
            screenshot_paths=self.screenshot_paths,
            summary=summary,
        )
