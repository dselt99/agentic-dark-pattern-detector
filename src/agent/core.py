"""Dark Pattern Agent - The Orchestrator.

This module implements Pillar 4: The Agent, using the ReAct pattern
(Reason + Act) to orchestrate dark pattern detection audits.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import uuid
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
from .ledger import JourneyLedger
from .planner import Planner
from .actor import Actor
from .auditor import Auditor
from .graph import create_state_graph, AgentState
from .sandbox import SandboxManager
from .wait_strategy import WaitStrategy


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

        # Phase 2 components (initialized on demand)
        self._planner: Optional[Planner] = None
        self._actor: Optional[Actor] = None
        self._auditor: Optional[Auditor] = None
        self._graph = None
        self._sandbox: Optional[SandboxManager] = None
        self._wait_strategy: Optional[WaitStrategy] = None

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
        skill_file = Path(skill_path)
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

                backoff_delay = min(2**attempt, 60)
                await asyncio.sleep(backoff_delay)  # Brief delay before retry

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

    async def run_dynamic_audit(
        self, url: str, user_query: str = "Audit this website for dark patterns"
    ) -> AuditResult:
        """Run a dynamic Phase 2 audit using LangGraph and Planner-Actor-Auditor.

        This is the new Phase 2 architecture that supports multi-step journeys,
        state tracking, and dynamic pattern detection.

        Args:
            url: Target URL to audit.
            user_query: High-level user goal (e.g., "Audit cancellation flow").

        Returns:
            AuditResult with findings and metadata.
        """
        # Check robots.txt
        if not self._check_robots_compliance(url):
            return AuditResult(
                target_url=url,
                findings=[],
                screenshot_paths=[],
                summary="Audit aborted: robots.txt disallows access to this URL.",
            )

        # Initialize Phase 2 components
        session_id = str(uuid.uuid4())[:8]
        ledger = JourneyLedger(target_url=url, session_id=session_id)

        if not self._planner:
            self._planner = Planner(model=self.model, provider=self.provider)
        if not self._actor:
            self._actor = Actor(
                model=self.model, provider=self.provider, mcp_client=self.client
            )
        if not self._auditor:
            self._auditor = Auditor(ledger=ledger)
        if not self._graph:
            self._graph = create_state_graph()
        if not self._wait_strategy:
            self._wait_strategy = WaitStrategy()

        # Initialize browser session and sandbox
        await self.client.start_session()
        try:
            from ..mcp.server import get_browser

            _, _, page = await get_browser()
            sandbox = SandboxManager(page, session_id)
            await sandbox.setup()

            # Create initial state
            initial_state: AgentState = {
                "ledger": ledger,
                "planner_state": {
                    "user_query": user_query,
                    "task_queue": [],
                    "current_task": None,
                    "re_planning_needed": False,
                    "plan_dag": None,
                },
                "browser_state": {
                    "url": url,
                    "dom_tree": None,
                    "screenshot_path": None,
                    "marked_elements": None,
                    "network_idle": False,
                    "visual_stable": False,
                    "last_reload_timers": None,
                },
                "audit_log": {
                    "flags": [],
                    "price_history": [],
                    "cart_history": [],
                    "consent_history": [],
                    "violations_detected": False,
                },
                "control_signal": {
                    "action": "continue",
                    "next_node": None,
                    "error_message": None,
                    "wait_reason": None,
                },
                "security_clearance": {
                    "allowed": True,
                    "restricted_actions": [],
                    "reason": None,
                },
                "planner": self._planner,
                "actor": self._actor,
                "auditor": self._auditor,
                "mcp_client": self.client,
                "wait_strategy": self._wait_strategy,
                "session_id": session_id,
                "target_url": url,
                "max_steps": self.max_steps,
                "current_step": 0,
            }

            # Reset debug state
            from .debug import reset_debug_state, log_performance_summary, log_state_summary
            reset_debug_state()
            
            # Execute graph
            final_state = await self._graph.ainvoke(initial_state)
            
            # Log debug summaries
            log_performance_summary()
            log_state_summary(final_state)

            # Extract findings from audit log
            flags = final_state["audit_log"]["flags"]
            findings = []
            for flag in flags:
                # Convert AuditFlag to DetectedPattern
                finding = DetectedPattern(
                    pattern_type=flag.pattern_type,
                    confidence_score=flag.confidence,
                    element_selector=flag.element_selector or "",
                    reasoning=flag.evidence,
                    evidence=flag.evidence,
                )
                findings.append(finding)

            # Collect screenshot paths
            screenshot_paths = []
            for snapshot in ledger.snapshots:
                if snapshot.screenshot_ref:
                    screenshot_paths.append(snapshot.screenshot_ref)

            # Generate comprehensive analysis summary
            summary = self._generate_analysis_summary(
                final_state, ledger, findings, user_query
            )

            return AuditResult(
                target_url=url,
                findings=findings,
                screenshot_paths=screenshot_paths,
                summary=summary,
            )

        finally:
            await sandbox.cleanup()
            await self.client.end_session()

    def _generate_analysis_summary(
        self,
        final_state: AgentState,
        ledger: JourneyLedger,
        findings: List[DetectedPattern],
        user_query: str,
    ) -> str:
        """Generate a comprehensive analysis summary from audit execution.
        
        Args:
            final_state: Final state from graph execution.
            ledger: Journey ledger with interaction snapshots.
            findings: List of detected patterns.
            user_query: Original user query.
            
        Returns:
            Detailed analysis summary.
        """
        planner_state = final_state.get("planner_state", {})
        task_queue = planner_state.get("task_queue", [])
        current_step = final_state.get("current_step", 0)
        max_steps = final_state.get("max_steps", 50)
        audit_log = final_state.get("audit_log", {})
        
        # Check if audit completed properly
        completed_tasks = [t for t in task_queue if t.get("completed", False)]
        all_tasks_completed = len(completed_tasks) == len(task_queue) and len(task_queue) > 0
        hit_max_steps = current_step >= max_steps
        
        # Build summary
        summary_parts = []
        
        # Header
        if findings:
            summary_parts.append(
                f"Phase 2 audit completed successfully. Found {len(findings)} dark pattern(s): "
            )
            pattern_names = [f.pattern_type.value.replace("_", " ").title() for f in findings]
            summary_parts.append(", ".join(pattern_names) + ".")
        else:
            summary_parts.append("Phase 2 audit completed. No dark patterns detected.")
        
        # Audit execution details
        summary_parts.append(f"\nExecution Summary:")
        summary_parts.append(f"- Steps executed: {current_step}/{max_steps}")
        summary_parts.append(f"- Tasks completed: {len(completed_tasks)}/{len(task_queue)}")
        summary_parts.append(f"- Interaction snapshots: {len(ledger.snapshots)}")
        
        # Completion status
        if hit_max_steps and not all_tasks_completed:
            summary_parts.append(
                f"\n⚠️  Audit reached maximum step limit ({max_steps}) before completing all tasks. "
                f"Some analysis may be incomplete."
            )
        elif all_tasks_completed:
            summary_parts.append("\n✓ All planned tasks completed successfully.")
        else:
            summary_parts.append(
                f"\n⚠️  Audit completed but {len(task_queue) - len(completed_tasks)} task(s) "
                f"were not completed."
            )
        
        # What was checked
        if task_queue:
            summary_parts.append(f"\nAudit Scope:")
            task_types = {}
            for task in completed_tasks:
                task_type = task.get("type", "unknown")
                task_types[task_type] = task_types.get(task_type, 0) + 1
            
            for task_type, count in sorted(task_types.items()):
                summary_parts.append(f"- {task_type.title()} tasks: {count}")
        
        # Patterns checked for
        summary_parts.append(f"\nPatterns Analyzed:")
        pattern_types_checked = [
            "False Urgency", "Roach Motel", "Sneak into Basket", 
            "Drip Pricing", "Forced Continuity", "Privacy Zuckering"
        ]
        for pattern in pattern_types_checked:
            summary_parts.append(f"- {pattern}")
        
        # Key interactions
        unique_urls = set()
        if ledger.snapshots:
            summary_parts.append(f"\nKey Interactions:")
            action_types = {}
            for snapshot in ledger.snapshots:
                if snapshot.url:
                    unique_urls.add(snapshot.url)
                if snapshot.action_taken:
                    action_type = snapshot.action_taken.get("type", "unknown")
                    action_types[action_type] = action_types.get(action_type, 0) + 1
            
            summary_parts.append(f"- Pages visited: {len(unique_urls)}")
            for action_type, count in sorted(action_types.items()):
                if action_type != "unknown":
                    summary_parts.append(f"- {action_type.title()} actions: {count}")
        
        # Price tracking (if applicable)
        price_history = audit_log.get("price_history", [])
        if price_history:
            summary_parts.append(f"\nPrice Tracking:")
            summary_parts.append(f"- Price points recorded: {len(price_history)}")
            if len(price_history) > 1:
                price_delta = price_history[-1] - price_history[0]
                summary_parts.append(
                    f"- Price change: ${price_history[0]:.2f} → ${price_history[-1]:.2f} "
                    f"({price_delta:+.2f})"
                )
        
        # Cart tracking (if applicable)
        cart_history = audit_log.get("cart_history", [])
        if cart_history:
            summary_parts.append(f"\nCart Analysis:")
            summary_parts.append(f"- Cart states recorded: {len(cart_history)}")
        
        # Findings details
        if findings:
            summary_parts.append(f"\nDetected Patterns:")
            for i, finding in enumerate(findings, 1):
                summary_parts.append(
                    f"{i}. {finding.pattern_type.value.replace('_', ' ').title()} "
                    f"(confidence: {finding.confidence_score:.2f})"
                )
                if finding.element_selector:
                    summary_parts.append(f"   Element: {finding.element_selector}")
                if finding.reasoning:
                    summary_parts.append(f"   Reasoning: {finding.reasoning[:200]}...")
        else:
            summary_parts.append(
                f"\nAnalysis Result: After examining {len(ledger.snapshots)} interaction points "
                f"across {len(unique_urls) if ledger.snapshots else 0} page(s), no dark patterns "
                f"were identified. The site appears to follow ethical design practices for the "
                f"tested user journey."
            )
        
        return "\n".join(summary_parts)
