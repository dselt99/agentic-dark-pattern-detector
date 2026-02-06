"""Scoring utilities for evaluation framework.

Handles TP/FP/FN calculation, aggregate metrics, and baseline comparison.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Set, Optional, Dict, Any


EVALS_DIR = Path(__file__).parent
BASELINE_PATH = EVALS_DIR / "baseline.json"
RESULTS_DIR = EVALS_DIR / "results"
LATEST_PATH = RESULTS_DIR / "latest.json"


@dataclass
class TestCaseResult:
    """Result of a single test case."""

    test_id: str
    tier: str
    description: str
    expected_patterns: Set[str]
    found_patterns: Set[str]
    status: str = ""  # PASS, FAIL, ERROR, SKIP, WARN
    error_message: str = ""
    duration_seconds: float = 0.0
    # Tier 1b/2 only
    tasks_completed: int = 0
    known_flaky: bool = False

    def __post_init__(self):
        if not self.status:
            self.status = self._compute_status()

    def _compute_status(self) -> str:
        if self.error_message:
            return "ERROR"
        tp = self.expected_patterns & self.found_patterns
        fp = self.found_patterns - self.expected_patterns
        fn = self.expected_patterns - self.found_patterns
        if len(fn) == 0 and len(fp) == 0:
            return "PASS"
        return "FAIL"

    @property
    def tp(self) -> Set[str]:
        return self.expected_patterns & self.found_patterns

    @property
    def fp(self) -> Set[str]:
        return self.found_patterns - self.expected_patterns

    @property
    def fn(self) -> Set[str]:
        return self.expected_patterns - self.found_patterns

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "tier": self.tier,
            "description": self.description,
            "expected_patterns": sorted(self.expected_patterns),
            "found_patterns": sorted(self.found_patterns),
            "status": self.status,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
            "tasks_completed": self.tasks_completed,
            "known_flaky": self.known_flaky,
        }


@dataclass
class TierSummary:
    """Aggregate metrics for a tier."""

    tier: str
    results: List[TestCaseResult] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == "ERROR")

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == "WARN")

    @property
    def sum_tp(self) -> int:
        return sum(len(r.tp) for r in self.results)

    @property
    def sum_fp(self) -> int:
        return sum(len(r.fp) for r in self.results)

    @property
    def sum_fn(self) -> int:
        return sum(len(r.fn) for r in self.results)

    @property
    def precision(self) -> float:
        denom = self.sum_tp + self.sum_fp
        return self.sum_tp / denom if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.sum_tp + self.sum_fn
        return self.sum_tp / denom if denom > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "warnings": self.warnings,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
            "pass_rate": round(self.pass_rate, 3),
            "total_duration": round(self.total_duration, 2),
            "results": [r.to_dict() for r in self.results],
        }


def _get_git_sha() -> str:
    """Get short git SHA of current commit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_results(summaries: List[TierSummary], path: Optional[Path] = None) -> Path:
    """Save evaluation results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or LATEST_PATH

    data = {
        "timestamp": datetime.now().isoformat(),
        "git_sha": _get_git_sha(),
        "tiers": {s.tier: s.to_dict() for s in summaries},
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def save_baseline(summaries: List[TierSummary]) -> Path:
    """Save current results as the baseline."""
    return save_results(summaries, BASELINE_PATH)


def load_baseline() -> Optional[Dict[str, Any]]:
    """Load baseline results if they exist."""
    if not BASELINE_PATH.exists():
        return None
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_to_baseline(current: List[TierSummary]) -> Optional[str]:
    """Compare current results to saved baseline. Returns formatted diff string."""
    baseline = load_baseline()
    if baseline is None:
        return None

    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append(f"BASELINE COMPARISON (vs {baseline['timestamp'][:10]} @ {baseline['git_sha']})")
    lines.append("=" * 64)
    lines.append(f"{'Metric':<16}{'Baseline':<12}{'Current':<12}{'Delta':<12}")
    lines.append("-" * 64)

    regressions = []
    improvements = []

    for summary in current:
        tier = summary.tier
        if tier not in baseline.get("tiers", {}):
            continue

        bl = baseline["tiers"][tier]

        for metric_name, cur_val, bl_val in [
            ("F1", summary.f1, bl["f1"]),
            ("Precision", summary.precision, bl["precision"]),
            ("Recall", summary.recall, bl["recall"]),
        ]:
            delta = cur_val - bl_val
            delta_str = f"{delta:+.3f}" if delta != 0 else "  0.000"
            lines.append(
                f"{metric_name:<16}{bl_val:<12.3f}{cur_val:<12.3f}{delta_str}"
            )

        # Check per-test regressions/improvements
        bl_results = {r["test_id"]: r for r in bl.get("results", [])}
        for result in summary.results:
            bl_result = bl_results.get(result.test_id)
            if bl_result is None:
                continue
            if bl_result["status"] == "PASS" and result.status in ("FAIL", "ERROR"):
                regressions.append(result.test_id)
            elif bl_result["status"] in ("FAIL", "ERROR") and result.status == "PASS":
                improvements.append(result.test_id)

    lines.append("-" * 64)
    lines.append(
        f"Regressions: {len(regressions)} | "
        f"Improvements: {len(improvements)} | "
        f"New failures: 0"
    )

    if regressions:
        lines.append(f"  REGRESSED: {', '.join(regressions)}")
    if improvements:
        lines.append(f"  IMPROVED:  {', '.join(improvements)}")

    return "\n".join(lines)


def format_results_table(summary: TierSummary) -> str:
    """Format a tier's results as a human-readable table."""
    git_sha = _get_git_sha()
    tier_labels = {
        "1a": "Tier 1a (Auditor-Only)",
        "1b": "Tier 1b (Simulation)",
        "2": "Tier 2 (Real-Site)",
    }
    label = tier_labels.get(summary.tier, f"Tier {summary.tier}")

    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"EVAL RESULTS — {label:<36}git:{git_sha}")
    lines.append("=" * 72)
    lines.append(
        f"{'Test':<32}{'Expected':<16}{'Found':<16}{'Status'}"
    )
    lines.append("-" * 72)

    for r in summary.results:
        expected = _short_patterns(r.expected_patterns)
        found = _short_patterns(r.found_patterns)
        status = r.status
        if status == "WARN":
            status = "WARN (flaky)"
        lines.append(f"{r.test_id:<32}{expected:<16}{found:<16}{status}")
        if r.error_message:
            lines.append(f"  ERROR: {r.error_message[:70]}")

    lines.append("-" * 72)
    lines.append(
        f"{summary.passed}/{summary.total} passed | "
        f"P: {summary.precision:.3f} | "
        f"R: {summary.recall:.3f} | "
        f"F1: {summary.f1:.3f} | "
        f"{summary.total_duration:.1f}s"
    )

    return "\n".join(lines)


def _short_patterns(patterns: Set[str]) -> str:
    """Abbreviate pattern names for table display."""
    if not patterns:
        return "[]"
    abbrevs = {
        "false_urgency": "urgency",
        "roach_motel": "roach",
        "sneak_into_basket": "sneak",
        "forced_continuity": "forced",
        "privacy_zuckering": "privacy",
        "confirmshaming": "confirm",
    }
    short = [abbrevs.get(p, p) for p in sorted(patterns)]
    return "[" + ",".join(short) + "]"
