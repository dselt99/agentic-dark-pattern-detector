"""Enhanced debugging utilities for Phase 2 graph execution.

This module provides comprehensive logging and debugging capabilities for
the LangGraph state machine, including node execution tracking, state
snapshots, performance metrics, and transition logging.
"""

import os
import time
import logging
import json
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime

# Configure debug logger
debug_logger = logging.getLogger("dark_pattern_agent.debug")
debug_logger.setLevel(logging.DEBUG)

# Check if debug mode is enabled
DEBUG_ENABLED = os.getenv("DEBUG_GRAPH", "false").lower() in ("true", "1", "yes")
DEBUG_VERBOSE = os.getenv("DEBUG_GRAPH_VERBOSE", "false").lower() in ("true", "1", "yes")

# Performance tracking
_node_timings: Dict[str, list] = {}
_state_history: list = []


def reset_debug_state():
    """Reset debug state between audits."""
    global _node_timings, _state_history
    _node_timings = {}
    _state_history = []


def log_node_entry(node_name: str, state: Dict[str, Any]):
    """Log when a node is entered."""
    if not DEBUG_ENABLED:
        return
    
    step = state.get("current_step", 0)
    debug_logger.info(f"[STEP {step}] → ENTER {node_name}")
    
    # Log current task if available
    current_task = state.get("planner_state", {}).get("current_task")
    if current_task:
        task_type = current_task.get("type", "unknown")
        task_goal = current_task.get("goal", "")[:50]
        debug_logger.debug(f"  Task: {task_type} - {task_goal}")
    
    # Log control signal
    control = state.get("control_signal", {})
    action = control.get("action", "unknown")
    if action != "continue":
        debug_logger.debug(f"  Control: {action}")
    
    # Log URL if changed
    url = state.get("browser_state", {}).get("url")
    if url:
        debug_logger.debug(f"  URL: {url}")
    
    # Log flags count
    flags_count = len(state.get("audit_log", {}).get("flags", []))
    if flags_count > 0:
        debug_logger.debug(f"  Flags: {flags_count} pattern(s) detected")


def log_node_exit(node_name: str, state: Dict[str, Any], duration: float, error: Optional[Exception] = None):
    """Log when a node exits."""
    if not DEBUG_ENABLED:
        return
    
    step = state.get("current_step", 0)
    if error:
        debug_logger.error(f"[STEP {step}] ← EXIT {node_name} (ERROR: {error}) [{duration:.3f}s]")
    else:
        debug_logger.info(f"[STEP {step}] ← EXIT {node_name} [{duration:.3f}s]")
    
    # Track timing
    if node_name not in _node_timings:
        _node_timings[node_name] = []
    _node_timings[node_name].append(duration)
    
    # Log state changes if verbose
    if DEBUG_VERBOSE:
        log_state_snapshot(node_name, state)


def log_state_snapshot(node_name: str, state: Dict[str, Any]):
    """Log a snapshot of the current state."""
    if not DEBUG_VERBOSE:
        return
    
    snapshot = {
        "node": node_name,
        "step": state.get("current_step", 0),
        "timestamp": datetime.now().isoformat(),
        "planner": {
            "task_queue_length": len(state.get("planner_state", {}).get("task_queue", [])),
            "current_task_id": state.get("planner_state", {}).get("current_task", {}).get("id"),
            "re_planning_needed": state.get("planner_state", {}).get("re_planning_needed", False),
        },
        "browser": {
            "url": state.get("browser_state", {}).get("url"),
            "network_idle": state.get("browser_state", {}).get("network_idle", False),
            "visual_stable": state.get("browser_state", {}).get("visual_stable", False),
        },
        "audit": {
            "flags_count": len(state.get("audit_log", {}).get("flags", [])),
            "violations_detected": state.get("audit_log", {}).get("violations_detected", False),
            "price_history_length": len(state.get("audit_log", {}).get("price_history", [])),
        },
        "control": state.get("control_signal", {}),
        "ledger_snapshots": len(state.get("ledger", {}).snapshots if hasattr(state.get("ledger"), "snapshots") else []),
    }
    
    _state_history.append(snapshot)
    debug_logger.debug(f"  State snapshot: {json.dumps(snapshot, indent=2, default=str)}")


def log_edge_transition(from_node: str, to_node: str, reason: str, state: Dict[str, Any]):
    """Log a graph edge transition."""
    if not DEBUG_ENABLED:
        return
    
    step = state.get("current_step", 0)
    debug_logger.info(f"[STEP {step}] → TRANSITION: {from_node} → {to_node} ({reason})")


def log_task_execution(task: Dict[str, Any], result: Dict[str, Any]):
    """Log task execution details."""
    if not DEBUG_ENABLED:
        return
    
    task_type = task.get("type", "unknown")
    task_goal = task.get("goal", "")[:60]
    status = result.get("status", "unknown")
    
    debug_logger.debug(f"  Executing task: {task_type}")
    debug_logger.debug(f"    Goal: {task_goal}")
    debug_logger.debug(f"    Status: {status}")
    
    if result.get("error"):
        debug_logger.warning(f"    Error: {result.get('error')}")


def log_detector_result(detector_name: str, flags: list):
    """Log detector execution results."""
    if not DEBUG_ENABLED:
        return
    
    if flags:
        debug_logger.info(f"  {detector_name}: {len(flags)} flag(s) raised")
        for flag in flags[:3]:  # Log first 3 flags
            pattern = flag.pattern_type.value if hasattr(flag, "pattern_type") else "unknown"
            confidence = flag.confidence if hasattr(flag, "confidence") else 0.0
            debug_logger.debug(f"    - {pattern} (confidence: {confidence:.2f})")
    else:
        debug_logger.debug(f"  {detector_name}: No flags")


def log_performance_summary():
    """Log performance summary at end of execution."""
    if not DEBUG_ENABLED or not _node_timings:
        return
    
    debug_logger.info("=" * 70)
    debug_logger.info("PERFORMANCE SUMMARY")
    debug_logger.info("=" * 70)
    
    total_time = sum(sum(times) for times in _node_timings.values())
    debug_logger.info(f"Total execution time: {total_time:.3f}s")
    debug_logger.info("")
    debug_logger.info("Node timings:")
    
    for node_name, times in sorted(_node_timings.items(), key=lambda x: sum(x[1]), reverse=True):
        count = len(times)
        total = sum(times)
        avg = total / count if count > 0 else 0
        max_time = max(times) if times else 0
        debug_logger.info(f"  {node_name:20s}: {count:3d} calls, {total:7.3f}s total, {avg:6.3f}s avg, {max_time:6.3f}s max")
    
    debug_logger.info("=" * 70)


def log_state_summary(final_state: Dict[str, Any]):
    """Log final state summary."""
    if not DEBUG_ENABLED:
        return
    
    debug_logger.info("=" * 70)
    debug_logger.info("FINAL STATE SUMMARY")
    debug_logger.info("=" * 70)
    
    step = final_state.get("current_step", 0)
    debug_logger.info(f"Total steps: {step}")
    
    # Planner state
    planner_state = final_state.get("planner_state", {})
    task_queue = planner_state.get("task_queue", [])
    completed_tasks = [t for t in task_queue if t.get("completed", False)]
    debug_logger.info(f"Tasks: {len(completed_tasks)}/{len(task_queue)} completed")
    
    # Browser state
    browser_state = final_state.get("browser_state", {})
    debug_logger.info(f"Final URL: {browser_state.get('url', 'N/A')}")
    
    # Audit results
    audit_log = final_state.get("audit_log", {})
    flags = audit_log.get("flags", [])
    debug_logger.info(f"Patterns detected: {len(flags)}")
    
    if flags:
        pattern_counts = {}
        for flag in flags:
            pattern = flag.pattern_type.value if hasattr(flag, "pattern_type") else "unknown"
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        for pattern, count in sorted(pattern_counts.items()):
            debug_logger.info(f"  - {pattern}: {count}")
    
    # Ledger
    ledger = final_state.get("ledger")
    if ledger and hasattr(ledger, "snapshots"):
        debug_logger.info(f"Interaction snapshots: {len(ledger.snapshots)}")
    
    debug_logger.info("=" * 70)


def debug_node(node_name: str):
    """Decorator to add debugging to a graph node function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            error = None
            
            try:
                log_node_entry(node_name, state)
                result = await func(state)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                duration = time.time() - start_time
                log_node_exit(node_name, state, duration, error)
        
        return wrapper
    return decorator


def enable_debug(enabled: bool = True, verbose: bool = False):
    """Enable or disable debug logging."""
    global DEBUG_ENABLED, DEBUG_VERBOSE
    
    DEBUG_ENABLED = enabled
    DEBUG_VERBOSE = verbose
    
    if enabled:
        # Set up console handler if not already present
        if not debug_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            debug_logger.addHandler(handler)
        
        debug_logger.info("=" * 70)
        debug_logger.info("GRAPH DEBUG MODE ENABLED")
        debug_logger.info(f"Verbose: {verbose}")
        debug_logger.info("=" * 70)
    else:
        debug_logger.setLevel(logging.WARNING)


# Initialize debug state
if DEBUG_ENABLED:
    enable_debug(True, DEBUG_VERBOSE)
