"""LangGraph State Machine for Phase 2 Architecture.

This module implements the Planner-Actor-Auditor orchestration using
LangGraph's StateGraph for managing complex, multi-step journeys.
"""

from typing import TypedDict, List, Optional, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
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
from ..schemas import AuditFlag, InteractionSnapshot, CartItem, ConsentStatus


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
    last_reload_timers: Optional[List[Dict[str, Any]]]


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


def create_state_graph() -> StateGraph:
    """Create and configure the LangGraph StateGraph for Phase 2 architecture.

    Returns:
        Configured StateGraph instance.
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
    """PLAN_GENESIS node: Decompose high-level goal into task queue.

    This node takes the user query and generates a DAG of sub-goals.
    """
    user_query = state["planner_state"]["user_query"]
    target_url = state["target_url"]
    planner = state.get("planner")
    
    if DEBUG_ENABLED:
        from .debug import debug_logger
        debug_logger.debug(f"  Decomposing goal: {user_query[:60]}...")

    if planner:
        # Use Planner to decompose goal
        try:
            task_queue = await planner.decompose_goal(user_query, target_url)
            # Ensure first task is navigation if not already
            if task_queue and task_queue[0].get("type") != "navigate":
                task_queue.insert(0, {"id": 0, "type": "navigate", "goal": f"Navigate to {target_url}"})
                # Renumber IDs
                for i, task in enumerate(task_queue):
                    task["id"] = i
        except Exception as e:
            # Fallback to simple task queue on error
            task_queue = [
                {"id": 0, "type": "navigate", "goal": f"Navigate to {target_url}"},
                {"id": 1, "type": "observe", "goal": "Get accessibility tree"},
                {"id": 2, "type": "analyze", "goal": "Analyze for dark patterns"},
            ]
    else:
        # Fallback to simple task queue
        task_queue = [
            {"id": 0, "type": "navigate", "goal": f"Navigate to {target_url}"},
            {"id": 1, "type": "observe", "goal": "Get accessibility tree"},
            {"id": 2, "type": "analyze", "goal": "Analyze for dark patterns"},
        ]

    state["planner_state"]["task_queue"] = task_queue
    state["planner_state"]["current_task"] = task_queue[0] if task_queue else None
    state["planner_state"]["re_planning_needed"] = False

    if DEBUG_ENABLED:
        from .debug import debug_logger
        debug_logger.debug(f"  Generated {len(task_queue)} task(s)")

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
        marked_elements = browser_state.get("marked_elements")

        # Get short-term context for Actor
        short_term_context = ledger.get_short_term_context(3)
        context_list = [
            {
                "action_taken": s.action_taken,
                "user_intent": s.user_intent,
                "url": s.url,
            }
            for s in short_term_context
        ]

        # Execute task via Actor
        if actor and mcp_client:
            if DEBUG_ENABLED:
                log_task_execution(current_task, {"status": "starting"})
            
            action_result = await actor.execute_task(
                task=current_task,
                dom_tree=dom_tree,
                marked_elements=marked_elements,
                short_term_context=context_list,
            )

            action = action_result.get("action", {})
            action_type = action.get("action_type")
            
            if DEBUG_ENABLED:
                from .debug import debug_logger
                debug_logger.debug(f"  Action type: {action_type}")
                if action.get("target"):
                    debug_logger.debug(f"  Target: {action.get('target')[:50]}")

            # Record action in ledger
            current_url = browser_state.get("url", state["target_url"])
            snapshot = ledger.record_snapshot(
                url=current_url,
                user_intent=current_task.get("goal", ""),
                action_taken={
                    "type": action_type,
                    "target": action.get("target"),
                    "value": action.get("value"),
                    "reasoning": action.get("reasoning"),
                },
            )

            # Handle different action types
            if action_type == "navigate":
                url = action.get("value") or state["target_url"]
                if DEBUG_ENABLED:
                    from .debug import debug_logger
                    debug_logger.debug(f"  Navigating to: {url}")
                nav_result = await mcp_client.call_tool("browser_navigate", url=url)
                if nav_result.get("status") == "success":
                    browser_state["url"] = url
                    # Wait for page to load
                    if wait_strategy:
                        from ..mcp.server import get_browser
                        _, _, page = await get_browser()
                        await wait_strategy.wait_for_network_idle(page)
                    current_task["completed"] = True
                    if DEBUG_ENABLED:
                        from .debug import debug_logger
                        debug_logger.debug("  Navigation successful")
                else:
                    current_task["failed"] = True
                    current_task["error"] = nav_result.get("message", "Navigation failed")
                    if DEBUG_ENABLED:
                        from .debug import debug_logger
                        debug_logger.warning(f"  Navigation failed: {current_task['error']}")

            elif action_type == "observe":
                # Get accessibility tree
                tree_result = await mcp_client.call_tool("get_accessibility_tree")
                if tree_result.get("status") == "success":
                    browser_state["dom_tree"] = tree_result.get("tree", "")
                    current_task["completed"] = True
                else:
                    current_task["failed"] = True

                # Get marked elements for Set-of-Marks
                marked_result = await mcp_client.call_tool("get_interactive_elements_marked")
                if marked_result.get("status") == "success":
                    browser_state["marked_elements"] = marked_result.get("marked_elements", {})

            elif action_type == "click":
                selector = action.get("target")
                if selector:
                    click_result = await mcp_client.call_tool("browser_click", selector=selector)
                    if click_result.get("status") == "success":
                        # Wait for stability after click
                        if wait_strategy:
                            from ..mcp.server import get_browser
                            _, _, page = await get_browser()
                            await wait_strategy.wait_for_stability(page)
                        current_task["completed"] = True

            elif action_type == "type":
                selector = action.get("target")
                text = action.get("value", "")
                if selector:
                    type_result = await mcp_client.call_tool("browser_type", selector=selector, text=text)
                    if type_result.get("status") == "success":
                        current_task["completed"] = True

            elif action_type == "scroll":
                direction = action.get("value", "down")
                scroll_result = await mcp_client.call_tool("browser_scroll", direction=direction)
                if scroll_result.get("status") == "success":
                    current_task["completed"] = True

            elif action_type == "wait":
                # Wait action - just mark as completed
                current_task["completed"] = True

            elif action_type == "reload":
                # Reload page (for False Urgency testing)
                reload_result = await mcp_client.call_tool("browser_reload")
                if reload_result.get("status") == "success":
                    # Store reload info for False Urgency detection
                    browser_state["last_reload_timers"] = reload_result.get("timers_after", [])
                    current_task["completed"] = True

            # Update browser state
            browser_state["network_idle"] = True
            browser_state["visual_stable"] = True

        else:
            # Fallback: mark task as completed if no actor
            current_task["completed"] = True

    except Exception as e:
        # On error, mark task as failed and set error message
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
        return state

    try:
        # Get latest snapshot
        latest_snapshot = ledger.get_latest_snapshot()
        if not latest_snapshot:
            return state

        # Get current page state
        dom_tree = browser_state.get("dom_tree")
        current_url = browser_state.get("url", state["target_url"])

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
        if DEBUG_ENABLED:
            from .debug import debug_logger
            debug_logger.debug("  Running pattern detectors...")
        
        new_flags = await auditor.observe_state(
            snapshot=latest_snapshot,
            dom_tree=dom_tree,
            price_breakdown=price_breakdown,
        )
        
        if DEBUG_ENABLED:
            if new_flags:
                from .debug import debug_logger
                debug_logger.info(f"  Detected {len(new_flags)} new pattern flag(s)")
                for flag in new_flags[:3]:  # Log first 3
                    pattern = flag.pattern_type.value if hasattr(flag, "pattern_type") else "unknown"
                    confidence = flag.confidence if hasattr(flag, "confidence") else 0.0
                    debug_logger.debug(f"    - {pattern} (confidence: {confidence:.2f})")

        # Check for False Urgency if reload happened
        reload_timers = browser_state.get("last_reload_timers")
        if reload_timers:
            # Check if timers reset (False Urgency pattern)
            for timer_info in reload_timers:
                timer_value = timer_info.get("match", "")
                if timer_value:
                    false_urgency_flags = await auditor.false_urgency_detector.detect(
                        timer_value=timer_value,
                        timer_selector="#timer",  # Default selector
                        after_reload=True,
                    )
                    new_flags.extend(false_urgency_flags)

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
        # Continue with current task
        state["control_signal"]["action"] = "continue"

    state["current_step"] = current_step + 1
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
