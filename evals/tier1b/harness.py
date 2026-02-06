"""Tier 1b harness: Run full Phase 2 agent against localhost simulations.

Uses SimulationServer to host HTML files and run_dynamic_audit() for detection.
"""

import asyncio
import os
import time
from typing import List, Dict, Any

from evals.scorer import TestCaseResult


# Import SimulationServer from existing evals/run.py
from evals.run import SimulationServer


async def _run_single_test(
    test_case: Dict[str, Any], port: int
) -> TestCaseResult:
    """Run a single Tier 1b test case."""
    test_id = test_case["id"]
    description = test_case.get("description", test_id)
    simulation = test_case["simulation"]
    query = test_case["query"]
    max_steps = test_case.get("max_steps", 15)
    expected = set(test_case.get("expected_patterns", []))
    forbidden = set(test_case.get("forbidden_patterns", []))

    url = f"http://localhost:{port}/{simulation}"

    start = time.monotonic()
    try:
        # Import here to avoid loading heavy deps for tier1a-only runs
        from src.agent import DarkPatternAgent

        model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
        provider = os.getenv("LLM_PROVIDER", "anthropic")

        agent = DarkPatternAgent(
            model=model,
            provider=provider,
            max_steps=max_steps,
        )

        result = await agent.run_dynamic_audit(url=url, user_query=query)

        # Extract pattern types from findings
        found = set()
        for finding in result.findings:
            found.add(finding.pattern_type.value)

        duration = time.monotonic() - start

        # Check for forbidden patterns (count as FP)
        forbidden_found = found & forbidden
        status = ""
        if forbidden_found:
            status = "FAIL"

        return TestCaseResult(
            test_id=test_id,
            tier="1b",
            description=description,
            expected_patterns=expected,
            found_patterns=found,
            status=status if status else "",
            duration_seconds=duration,
        )

    except Exception as e:
        duration = time.monotonic() - start
        return TestCaseResult(
            test_id=test_id,
            tier="1b",
            description=description,
            expected_patterns=expected,
            found_patterns=set(),
            status="ERROR",
            error_message=str(e)[:200],
            duration_seconds=duration,
        )


async def run_tier1b(
    test_cases: List[Dict[str, Any]], test_id_filter: str = None
) -> List[TestCaseResult]:
    """Run all Tier 1b test cases.

    Args:
        test_cases: List of test case dicts from manifest.
        test_id_filter: Optional single test ID to run.

    Returns:
        List of TestCaseResult.
    """
    if test_id_filter:
        test_cases = [tc for tc in test_cases if tc["id"] == test_id_filter]

    if not test_cases:
        return []

    port = int(os.getenv("EVAL_SIM_PORT", "8765"))
    server = SimulationServer(port=port)

    try:
        server.start()
        # Give server a moment to start
        await asyncio.sleep(0.5)

        results = []
        for tc in test_cases:
            print(f"  Running {tc['id']}...")
            result = await _run_single_test(tc, port)
            results.append(result)
            print(f"    -> {result.status} ({result.duration_seconds:.1f}s)")

        return results

    finally:
        server.stop()
