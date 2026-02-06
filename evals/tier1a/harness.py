"""Tier 1a harness: Feed synthetic DOM to Auditor.observe_state() and check flags.

No browser, no API tokens — pure heuristic detector testing.
"""

import asyncio
import time
from typing import List, Dict, Any, Set

from evals.scorer import TestCaseResult
from src.agent.auditor import Auditor
from src.agent.ledger import JourneyLedger
from src.schemas.schemas import InteractionSnapshot


def _create_minimal_snapshot(url: str = "http://test.local") -> InteractionSnapshot:
    """Create a minimal InteractionSnapshot for auditor testing."""
    return InteractionSnapshot(
        sequence_id=0,
        url=url,
        user_intent="Audit page for dark patterns",
    )


async def _run_single_test(test_case: Dict[str, Any]) -> TestCaseResult:
    """Run a single Tier 1a test case."""
    test_id = test_case["id"]
    description = test_case.get("description", "")
    dom_fixture = test_case["dom_fixture"]
    expected = set(test_case.get("expected_patterns", []))

    start = time.monotonic()
    try:
        # Create fresh auditor with minimal ledger
        ledger = JourneyLedger(target_url="http://test.local")
        auditor = Auditor(ledger)

        # Record a snapshot so the ledger has at least one entry
        snapshot = ledger.record_snapshot(
            url="http://test.local",
            user_intent="Audit page for dark patterns",
        )

        # Run detection with DOM fixture
        flags = await auditor.observe_state(
            snapshot=snapshot,
            dom_tree=dom_fixture,
        )

        # Extract pattern types from flags
        found = set()
        for flag in flags:
            found.add(flag.pattern_type.value)

        duration = time.monotonic() - start

        return TestCaseResult(
            test_id=test_id,
            tier="1a",
            description=description,
            expected_patterns=expected,
            found_patterns=found,
            duration_seconds=duration,
        )

    except Exception as e:
        duration = time.monotonic() - start
        return TestCaseResult(
            test_id=test_id,
            tier="1a",
            description=description,
            expected_patterns=expected,
            found_patterns=set(),
            status="ERROR",
            error_message=str(e)[:200],
            duration_seconds=duration,
        )


async def run_tier1a(test_cases: List[Dict[str, Any]], test_id_filter: str = None) -> List[TestCaseResult]:
    """Run all Tier 1a test cases.

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
        result = await _run_single_test(tc)
        results.append(result)

    return results
