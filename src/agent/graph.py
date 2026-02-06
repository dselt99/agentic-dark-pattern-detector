"""LangGraph State Machine for Phase 2 Architecture.

This module implements the Planner-Actor-Auditor orchestration using
LangGraph's StateGraph for managing complex, multi-step journeys.

Supports checkpointing for resuming from specific steps.
"""

from typing import TypedDict, List, Optional, Dict, Any, Literal
from langgraph.graph import StateGraph, END
import asyncio

from .ledger import JourneyLedger
from .debug import (
    debug_node,
    log_edge_transition,
    log_task_execution,
    log_detector_result,
    log_performance_summary,
    log_state_summary,
    reset_debug_state,
    DEBUG_ENABLED,
)
from .security import sanitize_untrusted_content, log_security_event
from .checkpoints import save_checkpoint
from ..schemas import AuditFlag, InteractionSnapshot, CartItem, ConsentStatus, PatternType


async def _refresh_browser_state(mcp_client, browser_state: Dict[str, Any]) -> None:
    """Fetch fresh page state from the browser after an action.

    Updates browser_state in-place with current URL, DOM tree, and marked elements.
    This is the single most important function for preventing stale-state loops.
    """
    try:
        # 1. Fresh URL
        url_result = await mcp_client.call_tool("get_page_url")
        if url_result.get("status") == "success":
            browser_state["url"] = url_result.get("url", browser_state.get("url", ""))

        # 2. Fresh DOM tree
        tree_result = await mcp_client.call_tool("get_accessibility_tree")
        if tree_result.get("status") == "success":
            raw_tree = tree_result.get("tree", "")
            sanitization = sanitize_untrusted_content(raw_tree)
            browser_state["dom_tree"] = sanitization.sanitized_content
            if sanitization.injection_detected:
                log_security_event(
                    "INJECTION_ATTEMPT",
                    {"url": browser_state.get("url"), "warnings": sanitization.warnings},
                    severity="WARNING",
                )

        # 3. Fresh marked elements
        marked_result = await mcp_client.call_tool("get_interactive_elements_marked")
        if marked_result.get("status") == "success":
            browser_state["marked_elements"] = marked_result.get("marked_elements", {})

    except Exception as e:
        if DEBUG_ENABLED:
            from .debug import debug_logger
            debug_logger.warning(f"  Re-observation failed: {e}")


class PlannerState(TypedDict):
    """State for the Planner node - high-level goal decomposition."""

    user_query: str
    task_queue: List[Dict[str, Any]]
    current_task: Optional[Dict[str, Any]]
    re_planning_needed: bool
    plan_dag: Optional[Dict[str, Any]]


class BrowserState(TypedDict):
    """State for browser interaction - current page state."""

    url: str
    dom_tree: Optional[str]
    screenshot_path: Optional[str]
    marked_elements: Optional[Dict[str, Any]]
    network_idle: bool
    visual_stable: bool
    last_reload_timers_before: Optional[List[Dict[str, Any]]]
    last_reload_timers_after: Optional[List[Dict[str, Any]]]


class AuditLog(TypedDict):
    """State for the Auditor - observation and flagging."""

    flags: List[AuditFlag]
    price_history: List[float]
    cart_history: List[List[Any]]
    consent_history: List[str]
    violations_detected: bool


class ControlSignal(TypedDict):
    """Control flow signals between nodes."""

    action: Literal["continue", "retry", "replan", "complete", "error", "wait"]
    next_node: Optional[str]
    error_message: Optional[str]
    wait_reason: Optional[str]


class SecurityClearance(TypedDict):
    """Security checks for restricted actions."""

    allowed: bool
    restricted_actions: List[str]
    reason: Optional[str]


class AgentState(TypedDict):
    """Complete agent state passed through the graph."""

    # Core components
    ledger: JourneyLedger
    planner_state: PlannerState
    browser_state: BrowserState
    audit_log: AuditLog
    control_signal: ControlSignal
    security_clearance: SecurityClearance

    # Component instances (stored as Any to avoid TypedDict issues)
    planner: Any  # Planner instance
    actor: Any  # Actor instance
    auditor: Any  # Auditor instance
    mcp_client: Any  # MCPClient instance
    wait_strategy: Any  # WaitStrategy instance

    # Metadata
    session_id: str
    target_url: str
    max_steps: int
    current_step: int


def create_state_graph():
    """Create and configure the LangGraph StateGraph for Phase 2 architecture.

    Returns:
        Compiled StateGraph instance.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("PLAN_GENESIS", plan_genesis_node)
    graph.add_node("NAV_ACTOR", nav_actor_node)
    graph.add_node("DOM_AUDITOR", dom_auditor_node)
    graph.add_node("STATE_EVAL", state_eval_node)
    graph.add_node("SAFE_GUARD", safe_guard_node)
    graph.add_node("WAIT_AND_RETRY", wait_and_retry_node)
    graph.add_node("HUMAN_INTERVENTION", human_intervention_node)

    # Define edges
    graph.set_entry_point("PLAN_GENESIS")

    # From PLAN_GENESIS
    graph.add_edge("PLAN_GENESIS", "SAFE_GUARD")

    # From SAFE_GUARD
    graph.add_conditional_edges(
        "SAFE_GUARD",
        check_security_clearance,
        {
            "allowed": "NAV_ACTOR",
            "blocked": "HUMAN_INTERVENTION",
        },
    )

    # From NAV_ACTOR
    graph.add_edge("NAV_ACTOR", "DOM_AUDITOR")

    # From DOM_AUDITOR
    graph.add_edge("DOM_AUDITOR", "STATE_EVAL")

    # From STATE_EVAL
    graph.add_conditional_edges(
        "STATE_EVAL",
        check_completion,
        {
            "continue": "SAFE_GUARD",  
            "complete": END,
            "retry": "WAIT_AND_RETRY",
            "replan": "PLAN_GENESIS",
            "error": "HUMAN_INTERVENTION",
        },
    )

    # From WAIT_AND_RETRY
    graph.add_edge("WAIT_AND_RETRY", "NAV_ACTOR")

    # From HUMAN_INTERVENTION
    graph.add_edge("HUMAN_INTERVENTION", END)

    return graph.compile()


@debug_node("PLAN_GENESIS")
async def plan_genesis_node(state: AgentState) -> AgentState:
    """PLAN_GENESIS node: Decompose goal OR Re-plan based on failure."""
    
    # 1. Unpack State
    user_query = state["planner_state"]["user_query"]
    target_url = state["target_url"]
    planner = state.get("planner")
    planner_state = state["planner_state"]
    
    # Check if we are here because of a failure (Re-planning Loop)
    is_replanning = planner_state.get("re_planning_needed", False)
    
    if DEBUG_ENABLED:
        from .debug import debug_logger
        mode = "RE-PLANNING" if is_replanning else "INITIAL PLANNING"
        debug_logger.debug(f"  Genesis Mode: {mode}")

    task_queue = []

    if planner:
        try:
            if is_replanning:
                # === PATH A: RECOVERY MODE ===
                # We failed. Give the planner context so it can fix it.
                failed_task = planner_state.get("current_task", {})
                error_msg = state["control_signal"].get("error_message", "Unknown error")
                current_dom = state["browser_state"].get("dom_tree", "")
                
                if DEBUG_ENABLED:
                    debug_logger.warning(f"  Re-planning context: Failed '{failed_task.get('goal')}' due to '{error_msg}'")

                task_queue = await planner.replan_goal(
                    original_query=user_query,
                    failed_step=failed_task.get("goal", "Unknown Step"),
                    error_context=error_msg,
                    current_dom=current_dom  # Critical: Let planner see obstructions!
                )
            else:
                # === PATH B: FRESH START ===
                # Standard decomposition for a new run
                task_queue = await planner.decompose_goal(user_query, target_url)

            # Safety: Ensure ID numbering is correct
            for i, task in enumerate(task_queue):
                task["id"] = i
                
        except Exception as e:
            # Fallback if LLM fails completely
            if DEBUG_ENABLED:
                debug_logger.error(f"  Planning failed: {e}")
            task_queue = [
                {"id": 0, "type": "navigate", "goal": f"Navigate to {target_url}"},
                {"id": 1, "type": "observe", "goal": "Get accessibility tree"},
                {"id": 2, "type": "analyze", "goal": "Fallback analysis"},
            ]
    
    # 2. Update State
    state["planner_state"]["task_queue"] = task_queue
    state["planner_state"]["current_task"] = task_queue[0] if task_queue else None
    
    # CRITICAL: Reset the flag so we don't get stuck in re-planning mode forever
    state["planner_state"]["re_planning_needed"] = False 

    if DEBUG_ENABLED:
        debug_logger.debug(f"  Generated {len(task_queue)} task(s)")

    return state

@debug_node("NAV_ACTOR")
async def nav_actor_node(state: AgentState) -> AgentState:
    """NAV_ACTOR node: Execute browser actions to advance the task.

    This node receives a task from the Planner and translates it into
    Playwright actions using Set-of-Marks for precise element selection.
    """
    current_task = state["planner_state"]["current_task"]
    if not current_task:
        if DEBUG_ENABLED:
            from .debug import debug_logger
            debug_logger.warning("  No current task to execute")
        return state

    actor = state.get("actor")
    mcp_client = state.get("mcp_client")
    wait_strategy = state.get("wait_strategy")
    ledger = state["ledger"]
    browser_state = state["browser_state"]

    try:
        # Get current DOM tree and marked elements
        dom_tree = browser_state.get("dom_tree")
        marked_elements = browser_state.get("marked_elements", {}) # Ensure dict if None

        # Get short-term context for Actor (include action results for deduplication)
        short_term_context = ledger.get_short_term_context(5)
        context_list = [
            {
                "action_taken": s.action_taken,
                "user_intent": s.user_intent,
                "url": s.url,
            }
            for s in short_term_context
        ]

        # Build failed-action list for deduplication
        failed_actions = []
        for s in short_term_context:
            if s.action_taken and s.action_taken.get("result") == "failed":
                failed_actions.append({
                    "type": s.action_taken.get("type", "unknown"),
                    "target": str(s.action_taken.get("target", "unknown"))[:80],
                    "error": s.action_taken.get("error", "unknown"),
                })

        # Execute task via Actor
        if actor and mcp_client:
            if DEBUG_ENABLED:
                log_task_execution(current_task, {"status": "starting"})

            action_result = await actor.execute_task(
                task=current_task,
                dom_tree=dom_tree,
                marked_elements=marked_elements,
                short_term_context=context_list,
                failed_actions=failed_actions if failed_actions else None,
            )

            action = action_result.get("action", {})
            action_type = action.get("action_type")
            
            # --- 🔧 FIX STARTS HERE: Define raw_target and translate ---
            raw_target = action.get("target")
            final_selector = raw_target

            # Check if target is a Mark ID (int or string-digit)
            if raw_target is not None:
                is_mark_id = (isinstance(raw_target, int)) or \
                             (isinstance(raw_target, str) and raw_target.isdigit())
                
                if is_mark_id and marked_elements:
                    mark_key = str(raw_target)
                    if mark_key in marked_elements:
                        # Translate ID -> CSS Selector
                        final_selector = marked_elements[mark_key].get("selector")
                        if DEBUG_ENABLED:
                            from .debug import debug_logger
                            debug_logger.debug(f" 🔄 Translated Mark {raw_target} -> {final_selector}")
                    else:
                        if DEBUG_ENABLED:
                            from .debug import debug_logger
                            debug_logger.warning(f" ⚠️ Mark ID {raw_target} not found in current element map")
            
            # Update the Action Object with the real selector
            if final_selector:
                action["target"] = final_selector
            # --- FIX ENDS HERE ---

            if DEBUG_ENABLED:
                from .debug import debug_logger
                debug_logger.debug(f"  Action type: {action_type}")
                if final_selector:
                    debug_logger.debug(f"  Target: {str(final_selector)[:50]}")

            # Capture URL before action for change detection
            url_before_action = browser_state.get("url", state["target_url"])

            # Record action in ledger (will be updated with result after execution)
            current_url = browser_state.get("url", state["target_url"])
            action_record = {
                "type": action_type,
                "target": final_selector,
                "value": action.get("value"),
                "reasoning": action.get("reasoning"),
            }
            snapshot = ledger.record_snapshot(
                url=current_url,
                user_intent=current_task.get("goal", ""),
                action_taken=action_record,
            )

            # Track whether this is a state-changing action that needs re-observation
            needs_reobserve = False

            # Handle different action types
            if action_type == "navigate":
                url = action.get("value") or state["target_url"]
                if DEBUG_ENABLED:
                    from .debug import debug_logger
                    debug_logger.debug(f"  Navigating to: {url}")

                # Invalidate stale DOM before navigation
                browser_state["dom_tree"] = None
                browser_state["marked_elements"] = None

                nav_result = await mcp_client.call_tool("browser_navigate", url=url)
                if nav_result.get("status") == "success":
                    browser_state["url"] = url
                    if wait_strategy:
                        from ..mcp.server import get_browser
                        _, _, page = await get_browser()
                        await wait_strategy.wait_for_stability(page)
                    current_task["completed"] = True
                    needs_reobserve = True
                    action_record["result"] = "success"

                    try:
                        screenshot_result = await mcp_client.call_tool(
                            "take_screenshot", filename_prefix="navigation",
                        )
                        if screenshot_result.get("status") == "success":
                            browser_state["screenshot_path"] = screenshot_result.get("path", "")
                    except Exception:
                        pass

                    if DEBUG_ENABLED:
                        from .debug import debug_logger
                        debug_logger.debug("  Navigation successful")
                else:
                    current_task["failed"] = True
                    current_task["error"] = nav_result.get("message", "Navigation failed")
                    action_record["result"] = "failed"
                    action_record["error"] = current_task["error"]
                    if DEBUG_ENABLED:
                        from .debug import debug_logger
                        debug_logger.warning(f"  Navigation failed: {current_task['error']}")

            elif action_type == "observe":
                # Use the shared re-observation helper
                await _refresh_browser_state(mcp_client, browser_state)
                current_task["completed"] = True
                action_record["result"] = "success"

            elif action_type == "click":
                if final_selector and isinstance(final_selector, str):
                    # Invalidate stale DOM before click (may navigate)
                    browser_state["dom_tree"] = None
                    browser_state["marked_elements"] = None

                    click_result = await mcp_client.call_tool("browser_click", selector=final_selector)
                    if click_result.get("status") == "success":
                        if wait_strategy:
                            from ..mcp.server import get_browser
                            _, _, page = await get_browser()
                            await wait_strategy.wait_for_stability(page)
                        current_task["completed"] = True
                        needs_reobserve = True
                        action_record["result"] = "success"
                    else:
                        current_task["failed"] = True
                        current_task["error"] = click_result.get("message", "Click failed")
                        action_record["result"] = "failed"
                        action_record["error"] = current_task["error"]
                else:
                    current_task["failed"] = True
                    current_task["error"] = f"Invalid selector: {final_selector} (Raw: {raw_target})"
                    action_record["result"] = "failed"
                    action_record["error"] = current_task["error"]

            elif action_type == "type":
                text = action.get("value", "")
                if final_selector and isinstance(final_selector, str):
                    type_result = await mcp_client.call_tool("browser_type", selector=final_selector, text=text)
                    if type_result.get("status") == "success":
                        current_task["completed"] = True
                        needs_reobserve = True
                        action_record["result"] = "success"
                    else:
                        current_task["failed"] = True
                        current_task["error"] = type_result.get("message", "Type failed")
                        action_record["result"] = "failed"
                        action_record["error"] = current_task["error"]
                else:
                    current_task["failed"] = True
                    current_task["error"] = f"Invalid selector: {final_selector}"
                    action_record["result"] = "failed"
                    action_record["error"] = current_task["error"]

            elif action_type == "scroll":
                direction = action.get("value", "down")
                scroll_result = await mcp_client.call_tool("browser_scroll", direction=direction)
                if scroll_result.get("status") == "success":
                    current_task["completed"] = True
                    needs_reobserve = True
                    action_record["result"] = "success"
                else:
                    current_task["failed"] = True
                    current_task["error"] = scroll_result.get("message", "Scroll failed")
                    action_record["result"] = "failed"
                    action_record["error"] = current_task["error"]

            elif action_type == "wait":
                current_task["completed"] = True
                action_record["result"] = "success"

            elif action_type == "reload":
                browser_state["dom_tree"] = None
                browser_state["marked_elements"] = None

                reload_result = await mcp_client.call_tool("browser_reload")
                if reload_result.get("status") == "success":
                    browser_state["last_reload_timers_before"] = reload_result.get("timers_before", [])
                    browser_state["last_reload_timers_after"] = reload_result.get("timers_after", [])
                    current_task["completed"] = True
                    needs_reobserve = True
                    action_record["result"] = "success"
                else:
                    current_task["failed"] = True
                    current_task["error"] = reload_result.get("message", "Reload failed")
                    action_record["result"] = "failed"
                    action_record["error"] = current_task["error"]

            elif action_type == "dismiss":
                from ..mcp.server import dismiss_popup
                dismiss_result = await mcp_client.call_tool("dismiss_popup")
                if dismiss_result.get("status") == "success":
                    current_task["completed"] = True
                    needs_reobserve = True
                    action_record["result"] = "success"
                else:
                    await mcp_client.call_tool("browser_key", key="Escape")
                    current_task["completed"] = True
                    needs_reobserve = True
                    action_record["result"] = "success"

            else:
                if DEBUG_ENABLED:
                    from .debug import debug_logger
                    debug_logger.warning(f"  Unknown action type: {action_type} - marking task complete")
                current_task["completed"] = True
                action_record["result"] = "success"

            # === POST-ACTION RE-OBSERVATION ===
            # This is the critical fix: fetch fresh page state after every
            # state-changing action so the next step doesn't use stale DOM.
            if needs_reobserve and mcp_client:
                if DEBUG_ENABLED:
                    from .debug import debug_logger
                    debug_logger.debug("  Re-observing browser state after action...")
                await _refresh_browser_state(mcp_client, browser_state)

                # Detect URL change — if URL changed, the action worked even if
                # the task didn't explicitly mark it complete
                url_after = browser_state.get("url", "")
                if url_after and url_after != url_before_action:
                    if DEBUG_ENABLED:
                        from .debug import debug_logger
                        debug_logger.debug(f"  URL changed: {url_before_action[:60]} -> {url_after[:60]}")

            # Update browser state
            browser_state["network_idle"] = True
            browser_state["visual_stable"] = True

        else:
            current_task["completed"] = True

    except Exception as e:
        current_task["completed"] = False
        state["control_signal"]["error_message"] = str(e)
        state["control_signal"]["action"] = "error"
        if DEBUG_ENABLED:
            from .debug import debug_logger
            debug_logger.error(f"  Actor error: {e}")

    return state


@debug_node("DOM_AUDITOR")
async def dom_auditor_node(state: AgentState) -> AgentState:
    """DOM_AUDITOR node: Analyze state for differential changes and log flags.

    This node runs in parallel to the Actor, observing state changes
    without interfering in navigation.
    """
    # Import debug_logger at the top for consistent logging
    from .debug import debug_logger

    auditor = state.get("auditor")
    mcp_client = state.get("mcp_client")
    ledger = state["ledger"]
    browser_state = state["browser_state"]

    # Initialize audit log if needed
    if "flags" not in state["audit_log"]:
        state["audit_log"]["flags"] = []
        state["audit_log"]["price_history"] = []
        state["audit_log"]["cart_history"] = []
        state["audit_log"]["consent_history"] = []
        state["audit_log"]["violations_detected"] = False

    if not auditor:
        debug_logger.warning("  No auditor instance - skipping pattern detection")
        return state

    try:
        # Get latest snapshot
        latest_snapshot = ledger.get_latest_snapshot()
        if not latest_snapshot:
            debug_logger.warning("  No snapshot available - skipping pattern detection")
            return state

        current_url = browser_state.get("url", state["target_url"])
        debug_logger.debug(f"  Auditor starting for URL: {current_url}")

        # Use cached DOM tree if available and recent (from Actor's observe action)
        # Only fetch fresh DOM if cache is empty - this reduces API calls significantly
        dom_tree = browser_state.get("dom_tree")

        if dom_tree:
            debug_logger.debug(f"  Using cached DOM tree: {len(dom_tree)} chars")
        elif mcp_client:
            # Only fetch if we don't have a cached tree
            debug_logger.debug("  No cached DOM, fetching accessibility tree for analysis...")

            tree_result = await mcp_client.call_tool("get_accessibility_tree")
            if tree_result.get("status") == "success":
                raw_tree = tree_result.get("tree", "")
                # Sanitize untrusted DOM content
                sanitization = sanitize_untrusted_content(raw_tree)
                dom_tree = sanitization.sanitized_content
                browser_state["dom_tree"] = dom_tree

                if sanitization.injection_detected:
                    log_security_event(
                        "INJECTION_ATTEMPT",
                        {
                            "url": current_url,
                            "warnings": sanitization.warnings,
                        },
                        severity="WARNING",
                    )

                debug_logger.debug(f"  DOM tree fetched: {len(dom_tree)} chars")
            else:
                debug_logger.warning(f"  Failed to fetch DOM tree: {tree_result.get('message', 'unknown')}")
        else:
            debug_logger.warning("  No MCP client and no cached DOM tree")

        # Get cart state and price breakdown if available
        cart_state = None
        price_breakdown = None

        if mcp_client:
            # Try to get cart state
            cart_result = await mcp_client.call_tool("get_cart_state")
            if cart_result.get("status") == "success":
                cart_data = cart_result.get("cart", {})
                cart_items = []
                for item in cart_data.get("items", []):
                    cart_items.append(
                        CartItem(
                            name=item.get("name", ""),
                            price=item.get("price", 0.0),
                            quantity=item.get("quantity", 1),
                            added_explicitly=False,  # Would need to track this
                            selector=item.get("selector"),
                        )
                    )
                cart_state = cart_items

            # Try to get price breakdown
            price_result = await mcp_client.call_tool("get_price_breakdown")
            if price_result.get("status") == "success":
                price_breakdown = price_result.get("price_breakdown", {})

        # Update snapshot with current state
        if cart_state:
            latest_snapshot.cart_state = cart_state
        if price_breakdown:
            total = price_breakdown.get("total", 0)
            if total > 0:
                latest_snapshot.perceived_value = total

        # Observe state with Auditor first
        debug_logger.debug("  Running pattern detectors...")

        new_flags = await auditor.observe_state(
            snapshot=latest_snapshot,
            dom_tree=dom_tree,
            price_breakdown=price_breakdown,
        )

        if new_flags:
            debug_logger.info(f"  Detected {len(new_flags)} new pattern flag(s)")
            for flag in new_flags[:3]:  # Log first 3
                pattern = flag.pattern_type.value if hasattr(flag, "pattern_type") else "unknown"
                confidence = flag.confidence if hasattr(flag, "confidence") else 0.0
                debug_logger.debug(f"    - {pattern} (confidence: {confidence:.2f})")
        else:
            debug_logger.debug("  No patterns detected in this step")

        # Take screenshots: always for evidence when patterns detected, every step only in debug mode
        current_step = state.get("current_step", 0)
        current_task = state["planner_state"].get("current_task", {})
        task_type = current_task.get("type", "unknown") if current_task else "unknown"

        # Check DEBUG_ENABLED dynamically from the module (not the imported copy)
        from .debug import DEBUG_ENABLED as debug_enabled_now
        should_screenshot = bool(new_flags) or debug_enabled_now  # Evidence screenshots always, debug screenshots when enabled
        debug_logger.debug(f"  should_screenshot={should_screenshot} (new_flags={bool(new_flags)}, debug={debug_enabled_now})")

        if mcp_client and should_screenshot:
            try:
                # Take screenshot for this step
                prefix = f"step_{current_step:03d}_{task_type}"
                if new_flags:
                    # Include pattern name in filename if patterns detected
                    pattern_names = "_".join([f.pattern_type.value for f in new_flags[:2]])
                    prefix = f"step_{current_step:03d}_evidence_{pattern_names}"

                debug_logger.debug(f"  Taking screenshot: {prefix}")
                screenshot_result = await mcp_client.call_tool(
                    "take_screenshot",
                    filename_prefix=prefix,
                )
                if screenshot_result.get("status") == "success":
                    screenshot_path = screenshot_result.get("path", "")
                    latest_snapshot.screenshot_ref = screenshot_path
                    browser_state["screenshot_path"] = screenshot_path
                    debug_logger.debug(f"  Screenshot captured: {screenshot_path}")
                else:
                    debug_logger.warning(f"  Screenshot failed: {screenshot_result.get('message', 'unknown')}")
            except Exception as e:
                debug_logger.warning(f"  Failed to capture screenshot: {e}")

        # Check for False Urgency if reload happened
        timers_before = browser_state.get("last_reload_timers_before", [])
        timers_after = browser_state.get("last_reload_timers_after", [])

        if timers_before and timers_after:
            # Compare timer values before and after reload
            before_values = {t.get("match", "") for t in timers_before if t.get("match")}
            after_values = {t.get("match", "") for t in timers_after if t.get("match")}

            # Find timers that reset to original values (not 00:00)
            reset_timers = before_values & after_values - {"00:00", "0:00", ""}

            if reset_timers:
                # Timers reset to original values - this is False Urgency
                for timer_value in reset_timers:
                    flag = AuditFlag(
                        pattern_type=PatternType.FALSE_URGENCY,
                        confidence=0.9,
                        step_id=len(ledger.snapshots) - 1,
                        evidence=(
                            f"Timer '{timer_value}' reset to original value after page reload. "
                            "This indicates artificial urgency rather than a real deadline."
                        ),
                        element_selector="detected_timer",
                        priority="high",
                    )
                    new_flags.append(flag)

                if DEBUG_ENABLED:
                    from .debug import debug_logger
                    debug_logger.info(f"  False Urgency detected: {len(reset_timers)} timer(s) reset on reload")

                # Take screenshot as evidence for False Urgency
                if mcp_client:
                    try:
                        screenshot_result = await mcp_client.call_tool(
                            "take_screenshot",
                            filename_prefix="evidence_false_urgency_timer_reset",
                        )
                        if screenshot_result.get("status") == "success":
                            screenshot_path = screenshot_result.get("path", "")
                            latest_snapshot.screenshot_ref = screenshot_path
                            browser_state["screenshot_path"] = screenshot_path
                            if DEBUG_ENABLED:
                                debug_logger.debug(f"  False Urgency screenshot: {screenshot_path}")
                    except Exception as e:
                        if DEBUG_ENABLED:
                            debug_logger.warning(f"  Failed to capture False Urgency screenshot: {e}")

            # Clear reload timers after processing
            browser_state["last_reload_timers_before"] = None
            browser_state["last_reload_timers_after"] = None

        # Add new flags to audit log
        state["audit_log"]["flags"].extend(new_flags)
        if new_flags:
            state["audit_log"]["violations_detected"] = True

        # Update price history
        if latest_snapshot.perceived_value:
            state["audit_log"]["price_history"].append(latest_snapshot.perceived_value)

        # Update cart history
        if latest_snapshot.cart_state:
            state["audit_log"]["cart_history"].append(
                [{"name": item.name, "price": item.price} for item in latest_snapshot.cart_state]
            )

    except Exception as e:
        # Log error but don't fail the audit
        state["control_signal"]["error_message"] = f"Auditor error: {str(e)}"
        if DEBUG_ENABLED:
            from .debug import debug_logger
            debug_logger.error(f"  Auditor error: {e}")

    return state


@debug_node("STATE_EVAL")
async def state_eval_node(state: AgentState) -> AgentState:
    """STATE_EVAL node: Determine if sub-task is complete, failed, or needs re-planning."""
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", 50)
    
    if DEBUG_ENABLED:
        from .debug import debug_logger
        debug_logger.debug(f"  Step {current_step}/{max_steps}")

    # Check if we've exceeded max steps
    if current_step >= max_steps:
        state["control_signal"]["action"] = "complete"
        state["control_signal"]["next_node"] = None
        if DEBUG_ENABLED:
            from .debug import debug_logger
            debug_logger.warning("  Max steps reached - completing audit")
        return state

    # Check if current task is complete
    current_task = state["planner_state"]["current_task"]
    task_queue = state["planner_state"]["task_queue"]

    if not current_task:
        # No current task - check if we have tasks in queue
        if task_queue:
            state["planner_state"]["current_task"] = task_queue[0]
            state["control_signal"]["action"] = "continue"
        else:
            # No tasks left
            state["control_signal"]["action"] = "complete"
    elif current_task.get("completed", False):
        # Current task completed - move to next task
        task_index = next(
            (i for i, t in enumerate(task_queue) if t.get("id") == current_task.get("id")), -1
        )
        if task_index >= 0 and task_index < len(task_queue) - 1:
            # Move to next task
            next_task = task_queue[task_index + 1]
            state["planner_state"]["current_task"] = next_task
            state["control_signal"]["action"] = "continue"
            if DEBUG_ENABLED:
                from .debug import debug_logger
                debug_logger.debug(f"  Task {current_task.get('id')} completed, moving to task {next_task.get('id')}")
        else:
            # All tasks complete
            state["control_signal"]["action"] = "complete"
            if DEBUG_ENABLED:
                from .debug import debug_logger
                debug_logger.info("  All tasks completed")
    elif current_task.get("failed", False):
        # Task failed - check if we should retry or replan
        retry_count = current_task.get("retry_count", 0)
        if retry_count < 2:  # Allow 2 retries
            current_task["retry_count"] = retry_count + 1
            current_task["failed"] = False
            state["control_signal"]["action"] = "retry"
            if DEBUG_ENABLED:
                from .debug import debug_logger
                debug_logger.warning(f"  Task failed, retrying ({retry_count + 1}/2)")
        else:
            # Too many retries - try re-planning
            state["planner_state"]["re_planning_needed"] = True
            state["control_signal"]["action"] = "replan"
            if DEBUG_ENABLED:
                from .debug import debug_logger
                debug_logger.warning("  Task failed after retries, re-planning needed")
    else:
        # Task neither completed nor failed - track attempts to detect stuck loops
        attempt_count = current_task.get("attempt_count", 0) + 1
        current_task["attempt_count"] = attempt_count

        if attempt_count >= 8:
            # Ultimate fallback: force-complete as FAILURE after 8 attempts
            if DEBUG_ENABLED:
                from .debug import debug_logger
                debug_logger.warning(f"  Task {current_task.get('id')} stuck after {attempt_count} attempts - force-completing as FAILURE")
            current_task["completed"] = True
            current_task["forced"] = True
            current_task["failed_forced"] = True  # Mark as forced failure for reporting
            state["control_signal"]["action"] = "continue"
        elif attempt_count >= 5:
            # Escalation 2: trigger re-plan — current plan is not working
            if DEBUG_ENABLED:
                from .debug import debug_logger
                debug_logger.warning(f"  Task {current_task.get('id')} stuck after {attempt_count} attempts - triggering re-plan")
            state["planner_state"]["re_planning_needed"] = True
            state["control_signal"]["action"] = "replan"
            state["control_signal"]["error_message"] = (
                f"Task '{current_task.get('goal', '')}' stuck after {attempt_count} attempts. "
                f"Current URL: {state['browser_state'].get('url', 'unknown')}"
            )
        elif attempt_count >= 3:
            # Escalation 1: force re-observe to get fresh DOM
            if DEBUG_ENABLED:
                from .debug import debug_logger
                debug_logger.warning(f"  Task {current_task.get('id')} stuck after {attempt_count} attempts - forcing re-observe")
            # Invalidate DOM to force fresh fetch on next NAV_ACTOR cycle
            state["browser_state"]["dom_tree"] = None
            state["browser_state"]["marked_elements"] = None
            state["control_signal"]["action"] = "continue"
        else:
            # Continue with current task
            state["control_signal"]["action"] = "continue"

    state["current_step"] = current_step + 1

    # Save checkpoint after each step
    thread_id = state.get("session_id", "unknown")
    save_checkpoint(
        thread_id=thread_id,
        step=current_step + 1,
        state=state,
        node="STATE_EVAL"
    )

    return state


@debug_node("SAFE_GUARD")
async def safe_guard_node(state: AgentState) -> AgentState:
    """SAFE_GUARD node: Check for restricted actions (buying real items, downloading binaries)."""
    current_task = state["planner_state"]["current_task"]
    if not current_task:
        state["security_clearance"]["allowed"] = True
        return state

    # Check for restricted action types
    restricted_keywords = ["buy", "purchase", "download", "submit_payment", "confirm_order"]
    task_goal = current_task.get("goal", "").lower()

    restricted_actions = [
        keyword for keyword in restricted_keywords if keyword in task_goal
    ]

    if restricted_actions:
        # In Phase 2, these should be intercepted by sandbox, but log for safety
        state["security_clearance"]["allowed"] = True  # Sandbox will handle interception
        state["security_clearance"]["restricted_actions"] = restricted_actions
        state["security_clearance"]["reason"] = "Action will be intercepted by sandbox"
        if DEBUG_ENABLED:
            from .debug import debug_logger
            debug_logger.debug(f"  Restricted actions detected: {restricted_actions} (sandbox will intercept)")
    else:
        state["security_clearance"]["allowed"] = True
        state["security_clearance"]["restricted_actions"] = []

    return state


@debug_node("WAIT_AND_RETRY")
async def wait_and_retry_node(state: AgentState) -> AgentState:
    """WAIT_AND_RETRY node: Handle retry logic for transient failures."""
    wait_strategy = state.get("wait_strategy")
    mcp_client = state.get("mcp_client")
    
    if DEBUG_ENABLED:
        from .debug import debug_logger
        debug_logger.debug("  Waiting for stability before retry...")

    if wait_strategy and mcp_client:
        try:
            from ..mcp.server import get_browser
            _, _, page = await get_browser()
            stability_result = await wait_strategy.wait_for_stability(page)
            state["browser_state"]["network_idle"] = stability_result.get("network_idle", False)
            state["browser_state"]["visual_stable"] = stability_result.get("visual_stable", False)
        except Exception:
            pass

    # Refresh browser state so the retry has fresh DOM
    if mcp_client:
        await _refresh_browser_state(mcp_client, state["browser_state"])

    state["control_signal"]["action"] = "continue"
    state["control_signal"]["wait_reason"] = "Retrying after wait"
    return state


@debug_node("HUMAN_INTERVENTION")
async def human_intervention_node(state: AgentState) -> AgentState:
    """HUMAN_INTERVENTION node: Handle cases requiring human attention."""
    if DEBUG_ENABLED:
        from .debug import debug_logger
        error_msg = state["control_signal"].get("error_message", "Unknown error")
        debug_logger.warning(f"  Human intervention required: {error_msg}")
    state["control_signal"]["action"] = "complete"
    state["control_signal"]["error_message"] = "Human intervention required"
    return state


def check_security_clearance(state: AgentState) -> Literal["allowed", "blocked"]:
    """Conditional edge function for security clearance check."""
    result = "allowed" if state["security_clearance"]["allowed"] else "blocked"
    if DEBUG_ENABLED:
        log_edge_transition("SAFE_GUARD", "NAV_ACTOR" if result == "allowed" else "HUMAN_INTERVENTION", 
                          f"security_{result}", state)
    return result


def check_completion(state: AgentState) -> Literal["continue", "complete", "retry", "replan", "error"]:
    """Conditional edge function for state evaluation."""
    action = state["control_signal"]["action"]
    
    # Map action to next node
    node_map = {
        "complete": "END",
        "retry": "WAIT_AND_RETRY",
        "replan": "PLAN_GENESIS",
        "error": "HUMAN_INTERVENTION",
        "continue": "SAFE_GUARD",
    }
    
    result = "complete" if action == "complete" else \
             "retry" if action == "retry" else \
             "replan" if action == "replan" else \
             "error" if action == "error" else \
             "continue"
    
    if DEBUG_ENABLED:
        next_node = node_map.get(action, "UNKNOWN")
        log_edge_transition("STATE_EVAL", next_node, action, state)
    
    return result
