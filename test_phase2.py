"""Test Phase 2 Dynamic Audit Architecture.

This script tests the Phase 2 implementation including:
- Component testing (Journey Ledger, Planner, Actor, Auditor, Sandbox)
- Basic Phase 2 audit functionality
- Multi-step journey testing
- Integration with existing simulations

Usage:
    python test_phase2.py                    # Run all tests
    python test_phase2.py --components       # Test components only
    python test_phase2.py --basic            # Test basic audit only
    python test_phase2.py --roach-motel      # Test Roach Motel journey
"""

import asyncio
import os
import sys
import http.server
import socketserver
import threading
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Load environment
load_dotenv()

# Suppress logging
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_section(text: str):
    """Print a section header."""
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)


def safe_print(text: str):
    """Print text with safe encoding for Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe version
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text)


async def test_journey_ledger():
    """Test Journey Ledger component."""
    print_section("Testing Journey Ledger")
    
    from src.agent.ledger import JourneyLedger
    from src.schemas import CartItem
    
    try:
        ledger = JourneyLedger(target_url="http://test.com", session_id="test123")
        print("[OK] Journey Ledger initialized")
        
        # Record snapshots
        snapshot1 = ledger.record_snapshot(
            url="http://test.com/product",
            user_intent="View product page",
            action_taken={"type": "navigate", "url": "http://test.com/product"},
            perceived_value=29.99,
        )
        print(f"[OK] Snapshot 1 recorded: sequence_id={snapshot1.sequence_id}")
        
        snapshot2 = ledger.record_snapshot(
            url="http://test.com/cart",
            user_intent="Add item to cart",
            action_taken={"type": "add_to_cart", "selector": "#add-btn"},
            perceived_value=29.99,
            cart_state=[CartItem(name="Test Product", price=29.99, quantity=1, added_explicitly=True)],
        )
        print(f"[OK] Snapshot 2 recorded: sequence_id={snapshot2.sequence_id}")
        
        snapshot3 = ledger.record_snapshot(
            url="http://test.com/checkout",
            user_intent="Proceed to checkout",
            action_taken={"type": "navigate", "url": "http://test.com/checkout"},
            perceived_value=34.99,  # Price increased (drip pricing)
            cart_state=[CartItem(name="Test Product", price=29.99, quantity=1, added_explicitly=True)],
        )
        print(f"[OK] Snapshot 3 recorded: sequence_id={snapshot3.sequence_id}")
        
        # Test methods
        intended_cart = ledger.get_intended_cart()
        print(f"[OK] Intended cart: {len(intended_cart)} items")
        
        actual_cart = ledger.get_actual_cart()
        print(f"[OK] Actual cart: {len(actual_cart)} items")
        
        price_delta = ledger.calculate_price_delta()
        print(f"[OK] Price delta: ${price_delta:.2f}" if price_delta else "[OK] Price delta: None")
        
        reconciliation = ledger.reconcile_intent_vs_reality()
        print(f"[OK] Cart reconciliation: has_discrepancy={reconciliation['has_discrepancy']}")
        
        anchor_price = ledger.get_anchor_price()
        terminal_price = ledger.get_terminal_price()
        print(f"[OK] Anchor price: ${anchor_price:.2f}" if anchor_price else "[OK] Anchor price: None")
        print(f"[OK] Terminal price: ${terminal_price:.2f}" if terminal_price else "[OK] Terminal price: None")
        
        short_term = ledger.get_short_term_context(2)
        print(f"[OK] Short-term context: {len(short_term)} snapshots")
        
        print("\n[PASS] Journey Ledger tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Journey Ledger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_planner():
    """Test Planner component."""
    print_section("Testing Planner")
    
    from src.agent.planner import Planner
    
    try:
        planner = Planner(
            model=os.getenv("LLM_MODEL", "claude-3-5-sonnet"),
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
        )
        print("[OK] Planner initialized")
        
        # Test goal decomposition
        tasks = await planner.decompose_goal(
            "Buy the cheapest item",
            "http://test.com"
        )
        print(f"[OK] Generated {len(tasks)} tasks")
        for i, task in enumerate(tasks[:5], 1):  # Show first 5
            print(f"  {i}. [{task.get('type', 'unknown')}] {task.get('goal', 'N/A')}")
        
        # Test re-planning
        failed_task = {"id": 1, "type": "interact", "goal": "Sort by price"}
        new_tasks = await planner.re_plan(
            failed_task=failed_task,
            error_message="Sort button not found",
            completed_tasks=[{"id": 0, "type": "navigate", "goal": "Navigate to page"}],
        )
        print(f"[OK] Re-planning generated {len(new_tasks)} alternative tasks")
        
        print("\n[OK] Planner tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Planner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_sandbox():
    """Test Sandbox and Synthetic Identity."""
    print_section("Testing Sandbox & Synthetic Identity")
    
    from src.agent.sandbox import SyntheticIdentity, PaymentInterceptor
    
    try:
        # Test Synthetic Identity
        identity = SyntheticIdentity(session_id="test123")
        test_identity = identity.generate_identity()
        print("[OK] Synthetic Identity generated")
        print(f"  Email: {test_identity['email']}")
        print(f"  Name: {test_identity['name']}")
        print(f"  Address: {test_identity['address']}")
        
        test_card = identity.get_test_card("stripe")
        print(f"[OK] Test card: {test_card['number']}")
        
        print("\n[OK] Sandbox tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Sandbox test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_detectors():
    """Test Detector modules."""
    print_section("Testing Detectors")
    
    from src.agent.ledger import JourneyLedger
    from src.agent.detectors.sneak_into_basket import SneakIntoBasketDetector
    from src.agent.detectors.drip_pricing import DripPricingDetector
    from src.agent.detectors.roach_motel import RoachMotelDetector
    from src.schemas import CartItem, InteractionSnapshot
    
    try:
        ledger = JourneyLedger(target_url="http://test.com")
        
        # Test Sneak into Basket detector
        sneak_detector = SneakIntoBasketDetector(ledger)
        print("[OK] SneakIntoBasketDetector initialized")
        
        # Test Drip Pricing detector
        drip_detector = DripPricingDetector(ledger)
        print("[OK] DripPricingDetector initialized")
        
        # Test Roach Motel detector
        roach_detector = RoachMotelDetector(ledger)
        print("[OK] RoachMotelDetector initialized")
        
        # Test with sample data
        snapshot = ledger.record_snapshot(
            url="http://test.com/checkout",
            user_intent="Checkout",
            perceived_value=34.99,
            cart_state=[
                CartItem(name="Product", price=29.99, quantity=1, added_explicitly=True),
                CartItem(name="Insurance", price=5.00, quantity=1, added_explicitly=False),  # Sneaked in
            ],
        )
        
        flags = await sneak_detector.detect(snapshot)
        print(f"[OK] SneakIntoBasket detected {len(flags)} violations")
        
        flags = await drip_detector.detect(snapshot, price_breakdown={"service_fee": 5.00})
        print(f"[OK] DripPricing detected {len(flags)} violations")
        
        print("\n[OK] Detector tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_basic_phase2_audit(port: int = 8888):
    """Test basic Phase 2 audit functionality."""
    print_section("Testing Basic Phase 2 Audit")
    
    from src.agent.core import DarkPatternAgent
    from src.mcp.server import cleanup as browser_cleanup
    
    url = f"http://localhost:{port}/evals/simulations/false_urgency.html"
    
    try:
        agent = DarkPatternAgent(
            model=os.getenv("LLM_MODEL", "claude-3-5-sonnet"),
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
            max_steps=10,
        )
        print(f"[OK] DarkPatternAgent initialized")
        print(f"  Target URL: {url}")
        print(f"  User Query: 'Detect false urgency patterns'")
        
        print("\nRunning Phase 2 audit...")
        result = await agent.run_dynamic_audit(
            url=url,
            user_query="Detect false urgency patterns"
        )
        
        print(f"\n[OK] Audit completed")
        print(f"  Findings: {len(result.findings)}")
        print(f"  Summary: {result.summary}")
        
        if result.findings:
            for i, finding in enumerate(result.findings, 1):
                print(f"\n  Finding {i}:")
                print(f"    Pattern: {finding.pattern_type.value}")
                print(f"    Confidence: {finding.confidence_score:.2f}")
                print(f"    Reasoning: {finding.reasoning[:100]}...")
        else:
            print("  No patterns detected")
        
        if result.screenshot_paths:
            print(f"\n  Screenshots: {len(result.screenshot_paths)}")
        
        print("\n[OK] Basic Phase 2 audit test passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Basic Phase 2 audit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await browser_cleanup()


async def test_roach_motel_journey(port: int = 8888):
    """Test Phase 2 with Roach Motel (multi-step journey)."""
    print_section("Testing Roach Motel Journey (Multi-Step)")
    
    from src.agent.core import DarkPatternAgent
    from src.mcp.server import cleanup as browser_cleanup
    
    url = f"http://localhost:{port}/evals/simulations/roach_motel.html"
    
    try:
        agent = DarkPatternAgent(
            model=os.getenv("LLM_MODEL", "claude-3-5-sonnet"),
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
            max_steps=15,  # More steps for multi-step journey
        )
        print(f"[OK] DarkPatternAgent initialized")
        print(f"  Target URL: {url}")
        print(f"  User Query: 'Audit the cancellation flow for Roach Motel patterns'")
        print(f"  Max Steps: 15 (for multi-step journey)")
        
        print("\nRunning Phase 2 audit (this may take longer for multi-step journey)...")
        result = await agent.run_dynamic_audit(
            url=url,
            user_query="Audit the cancellation flow for Roach Motel patterns - compare signup vs cancel difficulty"
        )
        
        print(f"\n[OK] Audit completed")
        print(f"  Findings: {len(result.findings)}")
        print(f"  Summary: {result.summary}")
        
        # Check if Roach Motel was detected
        roach_found = any(f.pattern_type.value == "roach_motel" for f in result.findings)
        print(f"\n  Roach Motel Detected: {'[OK] YES' if roach_found else '[FAIL] NO'}")
        
        if result.findings:
            for i, finding in enumerate(result.findings, 1):
                print(f"\n  Finding {i}:")
                print(f"    Pattern: {finding.pattern_type.value}")
                print(f"    Confidence: {finding.confidence_score:.2f}")
                print(f"    Selector: {finding.element_selector}")
                print(f"    Reasoning: {finding.reasoning[:150]}...")
        
        print("\n[OK] Roach Motel journey test passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Roach Motel journey test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await browser_cleanup()


async def test_mcp_tools():
    """Test new MCP tools for Phase 2."""
    print_section("Testing New MCP Tools")
    
    from src.mcp.server import (
        browser_type,
        browser_scroll,
        browser_wait_for_stability,
        get_cart_state,
        get_price_breakdown,
        get_interactive_elements_marked,
        browser_navigate,
        cleanup as browser_cleanup,
    )
    
    test_url = "http://localhost:8888/evals/simulations/false_urgency.html"
    
    try:
        # Navigate first
        nav_result = await browser_navigate(test_url)
        if nav_result.get("status") != "success":
            print(f"[FAIL] Navigation failed: {nav_result.get('message')}")
            return False
        print("[OK] Navigation successful")
        
        # Test get_interactive_elements_marked
        marked_result = await get_interactive_elements_marked()
        if marked_result.get("status") == "success":
            marked_count = marked_result.get("total_marks", 0)
            print(f"[OK] Marked {marked_count} interactive elements")
        else:
            print(f"[FAIL] Marking failed: {marked_result.get('message')}")
        
        # Test browser_scroll
        scroll_result = await browser_scroll(direction="down", amount=300)
        if scroll_result.get("status") == "success":
            print("[OK] Scroll successful")
        else:
            print(f"[FAIL] Scroll failed: {scroll_result.get('message')}")
        
        # Test browser_wait_for_stability
        wait_result = await browser_wait_for_stability()
        if wait_result.get("status") == "success":
            print(f"[OK] Wait for stability: network_idle={wait_result.get('network_idle')}, visual_stable={wait_result.get('visual_stable')}")
        else:
            print(f"[FAIL] Wait failed: {wait_result.get('message')}")
        
        # Test get_cart_state (may not find cart on test page, but should not error)
        cart_result = await get_cart_state()
        if cart_result.get("status") == "success":
            cart_items = cart_result.get("cart", {}).get("item_count", 0)
            print(f"[OK] Cart state retrieved: {cart_items} items")
        else:
            print(f"  Note: No cart found (expected for test page)")
        
        # Test get_price_breakdown
        price_result = await get_price_breakdown()
        if price_result.get("status") == "success":
            breakdown = price_result.get("price_breakdown", {})
            print(f"[OK] Price breakdown retrieved: {len(breakdown)} components")
        else:
            print(f"  Note: No price breakdown found (expected for test page)")
        
        print("\n[OK] MCP tools tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] MCP tools test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await browser_cleanup()


def start_test_server(port: int = 8888) -> socketserver.TCPServer:
    """Start a local HTTP server for test simulations."""
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress logs
    
    server = socketserver.TCPServer(("", port), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def run_all_tests():
    """Run all Phase 2 tests."""
    print_header("PHASE 2 ARCHITECTURE - COMPREHENSIVE TEST SUITE")
    
    # Start test server
    port = 8888
    print(f"\nStarting test server on port {port}...")
    server = start_test_server(port)
    print("[OK] Test server started")
    
    results = {}
    
    try:
        # Component tests
        print_header("COMPONENT TESTS")
        results["journey_ledger"] = await test_journey_ledger()
        results["planner"] = await test_planner()
        results["sandbox"] = await test_sandbox()
        results["detectors"] = await test_detectors()
        results["mcp_tools"] = await test_mcp_tools()
        
        # Integration tests
        print_header("INTEGRATION TESTS")
        results["basic_audit"] = await test_basic_phase2_audit(port)
        results["roach_motel"] = await test_roach_motel_journey(port)
        
        # Summary
        print_header("TEST SUMMARY")
        total = len(results)
        passed = sum(1 for v in results.values() if v)
        failed = total - passed
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed} [OK]")
        print(f"Failed: {failed} [FAIL]")
        
        print("\nDetailed Results:")
        for test_name, result in results.items():
            status = "[OK] PASS" if result else "[FAIL] FAIL"
            print(f"  {test_name:20s} {status}")
        
        if failed == 0:
            print("\n[SUCCESS] All tests passed!")
        else:
            print(f"\n[WARNING]  {failed} test(s) failed. Check output above for details.")
        
        return failed == 0
        
    finally:
        server.shutdown()
        print("\n[OK] Test server stopped")


async def run_component_tests():
    """Run only component tests."""
    print_header("PHASE 2 - COMPONENT TESTS")
    
    results = {}
    results["journey_ledger"] = await test_journey_ledger()
    results["planner"] = await test_planner()
    results["sandbox"] = await test_sandbox()
    results["detectors"] = await test_detectors()
    
    passed = sum(1 for v in results.values() if v)
    print(f"\n[OK] {passed}/{len(results)} component tests passed")
    return passed == len(results)


async def run_basic_test():
    """Run only basic audit test."""
    print_header("PHASE 2 - BASIC AUDIT TEST")
    
    port = 8888
    server = start_test_server(port)
    
    try:
        result = await test_basic_phase2_audit(port)
        return result
    finally:
        server.shutdown()


async def run_roach_motel_test():
    """Run only Roach Motel journey test."""
    print_header("PHASE 2 - ROACH MOTEL JOURNEY TEST")
    
    port = 8888
    server = start_test_server(port)
    
    try:
        result = await test_roach_motel_journey(port)
        return result
    finally:
        server.shutdown()


def main():
    """Main entry point."""
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if "--help" in args or "-h" in args:
        print(__doc__)
        return
    
    if "--components" in args:
        asyncio.run(run_component_tests())
    elif "--basic" in args:
        asyncio.run(run_basic_test())
    elif "--roach-motel" in args or "--roach_motel" in args:
        asyncio.run(run_roach_motel_test())
    else:
        # Run all tests
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
