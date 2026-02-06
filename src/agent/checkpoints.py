"""Simple checkpoint management using JSON files.

Saves agent state to JSON files so you can resume from any step.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


CHECKPOINT_DIR = Path("checkpoints")


def save_checkpoint(thread_id: str, step: int, state: Dict[str, Any], node: str = "") -> Path:
    """Save current state to a JSON file.

    Args:
        thread_id: Unique ID for this run.
        step: Current step number.
        state: State dict to save (will be serialized to JSON).
        node: Name of the current node.

    Returns:
        Path to the saved checkpoint file.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Create a serializable version of state (skip non-serializable objects)
    serializable_state = _make_serializable(state, strip_large=True)
    serializable_state["_meta"] = {
        "thread_id": thread_id,
        "step": step,
        "node": node,
        "timestamp": datetime.now().isoformat(),
    }

    filename = f"{thread_id}_step_{step:03d}.json"
    filepath = CHECKPOINT_DIR / filename

    with open(filepath, "w") as f:
        json.dump(serializable_state, f, indent=2, default=str)

    return filepath


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """Load state from a checkpoint file.

    Args:
        filepath: Path to the checkpoint JSON file.

    Returns:
        State dict from the checkpoint.
    """
    with open(filepath, "r") as f:
        return json.load(f)


def list_checkpoints(thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available checkpoints.

    Args:
        thread_id: Optional filter by thread ID.

    Returns:
        List of checkpoint info dicts.
    """
    if not CHECKPOINT_DIR.exists():
        return []

    checkpoints = []
    for filepath in sorted(CHECKPOINT_DIR.glob("*.json")):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                meta = data.get("_meta", {})

                if thread_id and meta.get("thread_id") != thread_id:
                    continue

                checkpoints.append({
                    "file": str(filepath),
                    "thread_id": meta.get("thread_id", "unknown"),
                    "step": meta.get("step", 0),
                    "node": meta.get("node", ""),
                    "timestamp": meta.get("timestamp", ""),
                })
        except (json.JSONDecodeError, IOError):
            continue

    return checkpoints


def get_latest_checkpoint(thread_id: str) -> Optional[str]:
    """Get the most recent checkpoint for a thread.

    Args:
        thread_id: Thread ID to look for.

    Returns:
        Path to the latest checkpoint file, or None.
    """
    checkpoints = list_checkpoints(thread_id)
    if not checkpoints:
        return None
    return checkpoints[-1]["file"]


def new_thread_id(prefix: str = "audit") -> str:
    """Generate a new thread ID.

    Args:
        prefix: Prefix for the ID.

    Returns:
        Unique thread ID.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


# Fields that bloat checkpoint files — truncate or strip
_LARGE_FIELDS = {"dom_tree", "dom"}
_MAX_LARGE_FIELD_CHARS = 1000


def _make_serializable(obj: Any, strip_large: bool = False) -> Any:
    """Convert object to JSON-serializable form.

    Args:
        obj: Object to serialize.
        strip_large: If True, truncate large fields like dom_tree.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k.startswith("_") or callable(v):
                continue
            if strip_large and k in _LARGE_FIELDS and isinstance(v, str) and len(v) > _MAX_LARGE_FIELD_CHARS:
                result[k] = v[:_MAX_LARGE_FIELD_CHARS] + f"... [truncated from {len(v)} chars]"
            else:
                result[k] = _make_serializable(v, strip_large=strip_large)
        return result
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item, strip_large=strip_large) for item in obj]
    elif hasattr(obj, "__dict__"):
        return _make_serializable(vars(obj), strip_large=strip_large)
    else:
        return str(obj)
