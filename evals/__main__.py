"""CLI entry point for evaluation framework.

Usage:
    python -m evals --tier 1a                     # Fast detector tests
    python -m evals --tier 1b                     # Simulation tests
    python -m evals --tier 2                      # Real site tests
    python -m evals --tier all                    # Everything
    python -m evals --tier 1a --baseline          # Save as baseline
    python -m evals --tier 1a --compare           # Compare vs baseline
    python -m evals --tier 1b --id t1b_roach_motel  # Single test
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Dark Pattern Detection Evaluation Framework",
        prog="python -m evals",
    )
    parser.add_argument(
        "--tier",
        required=True,
        choices=["1a", "1b", "2", "all"],
        help="Which tier(s) to run",
    )
    parser.add_argument(
        "--id",
        dest="test_id",
        default=None,
        help="Run a single test by ID",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Save results as baseline for future comparison",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results against saved baseline",
    )

    args = parser.parse_args()

    if args.tier == "all":
        tiers = ["1a", "1b", "2"]
    else:
        tiers = [args.tier]

    from evals.runner import run_evaluation

    summaries = asyncio.run(
        run_evaluation(
            tiers=tiers,
            test_id_filter=args.test_id,
            save_as_baseline=args.baseline,
            compare=args.compare,
        )
    )

    # Exit with non-zero if any failures
    has_failures = any(
        r.status in ("FAIL", "ERROR")
        for s in summaries
        for r in s.results
    )
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
