"""Evaluation runner: loads manifest, dispatches to tier harnesses, aggregates results."""

import asyncio
import time
from pathlib import Path
from typing import List, Optional

import yaml

from evals.scorer import TestCaseResult, TierSummary, save_results, save_baseline, compare_to_baseline, format_results_table


MANIFEST_PATH = Path(__file__).parent / "manifest.yaml"


def load_manifest() -> dict:
    """Load the test case manifest from YAML."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def run_tier(
    tier: str,
    manifest: dict,
    test_id_filter: Optional[str] = None,
) -> TierSummary:
    """Run a single tier of tests.

    Args:
        tier: "1a", "1b", or "2"
        manifest: Loaded manifest dict.
        test_id_filter: Optional single test ID to run.

    Returns:
        TierSummary with results.
    """
    tier_key = f"tier{tier}"
    test_cases = manifest.get(tier_key, [])

    if not test_cases:
        print(f"No test cases found for tier {tier}")
        return TierSummary(tier=tier)

    print(f"\nRunning Tier {tier} ({len(test_cases)} tests)...")
    start = time.monotonic()

    if tier == "1a":
        from evals.tier1a.harness import run_tier1a
        results = await run_tier1a(test_cases, test_id_filter)
    elif tier == "1b":
        from evals.tier1b.harness import run_tier1b
        results = await run_tier1b(test_cases, test_id_filter)
    elif tier == "2":
        from evals.tier2.harness import run_tier2
        results = await run_tier2(test_cases, test_id_filter)
    else:
        print(f"Unknown tier: {tier}")
        return TierSummary(tier=tier)

    duration = time.monotonic() - start

    summary = TierSummary(
        tier=tier,
        results=results,
        total_duration=duration,
    )

    return summary


async def run_evaluation(
    tiers: List[str],
    test_id_filter: Optional[str] = None,
    save_as_baseline: bool = False,
    compare: bool = False,
) -> List[TierSummary]:
    """Run the full evaluation pipeline.

    Args:
        tiers: List of tier strings to run ("1a", "1b", "2").
        test_id_filter: Optional single test ID to run.
        save_as_baseline: If True, save results as baseline.
        compare: If True, compare results to saved baseline.

    Returns:
        List of TierSummary for all tiers run.
    """
    manifest = load_manifest()
    summaries = []

    for tier in tiers:
        summary = await run_tier(tier, manifest, test_id_filter)
        summaries.append(summary)
        print(format_results_table(summary))

    # Save results
    save_results(summaries)

    if save_as_baseline:
        path = save_baseline(summaries)
        print(f"\nBaseline saved to {path}")

    if compare:
        diff = compare_to_baseline(summaries)
        if diff:
            print(diff)
        else:
            print("\nNo baseline found. Run with --baseline first.")

    return summaries
