"""Tier 2 harness: Run full Phase 2 agent against real websites.

Failures on known_flaky tests are reported as WARN instead of FAIL.
"""

import asyncio
import os
import time
from typing import List, Dict, Any

from evals.scorer import TestCaseResult


async def _run_single_test(test_case: Dict[str, Any]) -> TestCaseResult:
    """Run a single Tier 2 test case."""
    test_id = test_case["id"]
    description = test_case.get("description", test_id)
    url = test_case["url"]
    query = test_case["query"]
    max_steps = test_case.get("max_steps", 20)
    expected = set(test_case.get("expected_patterns", []))
    min_tasks = test_case.get("min_tasks_completed", 0)
    known_flaky = test_case.get("known_flaky", False)

    start = time.monotonic()
    try:
        from src.agent import DarkPatternAgent

        model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
        provider = os.getenv("LLM_PROVIDER", "anthropic")

        agent = DarkPatternAgent(
            model=model,
            provider=provider,
            max_steps=max_steps,
        )

        result = await agent.run_dynamic_audit(url=url, user_query=query)

        found = set()
        for finding in result.findings:
            found.add(finding.pattern_type.value)

        duration = time.monotonic() - start

        test_result = TestCaseResult(
            test_id=test_id,
            tier="2",
            description=description,
            expected_patterns=expected,
            found_patterns=found,
            duration_seconds=duration,
            known_flaky=known_flaky,
        )

        # Downgrade FAIL to WARN for known-flaky tests
        if test_result.status == "FAIL" and known_flaky:
            test_result.status = "WARN"

        return test_result

    except Exception as e:
        duration = time.monotonic() - start
        status = "WARN" if known_flaky else "ERROR"
        return TestCaseResult(
            test_id=test_id,
            tier="2",
            description=description,
            expected_patterns=expected,
            found_patterns=set(),
            status=status,
            error_message=str(e)[:200],
            duration_seconds=duration,
            known_flaky=known_flaky,
        )


async def run_tier2(
    test_cases: List[Dict[str, Any]], test_id_filter: str = None
) -> List[TestCaseResult]:
    """Run all Tier 2 test cases.

    Args:
        test_cases: List of test case dicts from manifest.
        test_id_filter: Optional single test ID to run.

    Returns:
        List of TestCaseResult.
    """
    if test_id_filter:
        test_cases = [tc for tc in test_cases if tc["id"] == test_id_filter]

    results = []
    for tc in test_cases:
        print(f"  Running {tc['id']} ({tc['url']})...")
        result = await _run_single_test(tc)
        results.append(result)
        status_str = result.status
        if result.known_flaky:
            status_str += " (flaky)"
        print(f"    -> {status_str} ({result.duration_seconds:.1f}s)")

    return results
