"""Demo script for the Dark Pattern Hunter.

This script starts a local HTTP server to serve test HTML files,
then runs the agent against them to demonstrate detection capabilities.

Usage:
    python demo.py                    # Run all demos
    python demo.py false_urgency      # Run specific demo
    python demo.py --list             # List available demos
    python demo.py --dynamic URL      # Run Phase 2 dynamic audit on URL
"""

import asyncio
import http.server
import socketserver
import threading
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Suppress httpx logging
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from src.agent.core import DarkPatternAgent
from src.mcp.server import cleanup as browser_cleanup


# Available test simulations
SIMULATIONS = {
    "false_urgency": {
        "file": "evals/simulations/false_urgency.html",
        "description": "Countdown timer that resets on page reload",
        "expected_pattern": "false_urgency",
    },
    "roach_motel": {
        "file": "evals/simulations/roach_motel.html",
        "description": "Easy subscribe, hard to cancel (hidden in accordion)",
        "expected_pattern": "roach_motel",
    },
    "clean_stock": {
        "file": "evals/simulations/clean_stock.html",
        "description": "Legitimate stock counter (negative control - should find nothing)",
        "expected_pattern": None,
    },
}


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that suppresses logging."""

    def log_message(self, format, *args):
        pass  # Suppress HTTP request logs


def start_server(port: int = 8888) -> socketserver.TCPServer:
    """Start a simple HTTP server in a background thread."""
    os.chdir(project_root)
    handler = QuietHTTPHandler
    server = socketserver.TCPServer(("", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_result(result):
    """Print audit result in a readable format."""
    print(f"\nURL: {result.target_url}")
    print(f"Timestamp: {result.timestamp}")
    print(f"Findings: {len(result.findings)}")
    print(f"\nSummary:\n{result.summary}")

    if result.findings:
        print("\nDetected Patterns:")
        for i, finding in enumerate(result.findings, 1):
            print(f"\n  {i}. {finding.pattern_type.value.upper()}")
            print(f"     Confidence: {finding.confidence_score:.2f}")
            print(f"     Selector: {finding.element_selector}")
            print(f"     Reasoning: {finding.reasoning}")
            if finding.evidence:
                print(f"     Evidence: {finding.evidence}")

    if result.screenshot_paths:
        print(f"\nScreenshots captured: {len(result.screenshot_paths)}")
        for path in result.screenshot_paths:
            print(f"  - {path}")


async def run_demo(simulation_name: str, port: int = 8888, verbose: bool = False):
    """Run a single demo against a simulation."""
    if simulation_name not in SIMULATIONS:
        print(f"Unknown simulation: {simulation_name}")
        print(f"Available: {', '.join(SIMULATIONS.keys())}")
        return

    sim = SIMULATIONS[simulation_name]
    url = f"http://localhost:{port}/{sim['file']}"

    print_header(f"Demo: {simulation_name.upper()}")
    print(f"Description: {sim['description']}")
    print(f"Expected: {sim['expected_pattern'] or 'No patterns (clean)'}")
    print(f"URL: {url}")
    print("-" * 60)

    # Initialize agent
    model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    provider = os.getenv("LLM_PROVIDER", "anthropic")

    print(f"Using model: {model} ({provider})")
    print("Starting audit...")

    # If verbose, show accessibility tree
    if verbose:
        from src.mcp.server import browser_navigate, get_accessibility_tree
        print("\n[VERBOSE] Fetching accessibility tree...")
        nav_result = await browser_navigate(url)
        print(f"[VERBOSE] Navigation: {nav_result.get('status')}")
        tree_result = await get_accessibility_tree()
        if tree_result.get('status') == 'success':
            print(f"[VERBOSE] Accessibility tree:\n{tree_result.get('tree', '')[:2000]}...")
        print("-" * 60)

    # Use more steps for patterns that need exploration
    # roach_motel needs clicking and exploration
    # false_urgency needs reload testing
    # clean pages need full analysis to confirm no patterns
    step_config = {
        "roach_motel": 5,
        "false_urgency": 4,
        "clean_stock": 3,
    }
    max_steps = step_config.get(simulation_name, 3)

    agent = DarkPatternAgent(
        model=model,
        provider=provider,
        max_steps=max_steps,
    )

    try:
        result = await agent.run_audit(url)
        print_result(result)

        # Check if result matches expectation
        expected = sim['expected_pattern']
        if expected:
            found = any(f.pattern_type.value == expected for f in result.findings)
            if found:
                print(f"\n[PASS] Correctly detected {expected}")
            else:
                print(f"\n[MISS] Expected {expected} but did not detect it")
        else:
            if not result.findings:
                print("\n[PASS] Correctly identified as clean (no false positives)")
            else:
                print(f"\n[FAIL] False positive: Found {len(result.findings)} pattern(s) on clean page")

    except Exception as e:
        print(f"\nError during audit: {e}")
        import traceback
        traceback.print_exc()


async def run_all_demos(port: int = 8888, verbose: bool = False):
    """Run all available demos."""
    print_header("DARK PATTERN HUNTER - DEMO")
    print("Running all simulations...")

    for name in SIMULATIONS:
        await run_demo(name, port, verbose)
        print("\n")


async def run_url_audit(url: str, verbose: bool = False):
    """Run full agentic audit against a custom URL."""
    print_header("DARK PATTERN HUNTER - LIVE SITE AUDIT")
    print(f"Target URL: {url}")
    print("-" * 60)

    # Initialize agent
    model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    provider = os.getenv("LLM_PROVIDER", "anthropic")

    print(f"Using model: {model} ({provider})")

    # If verbose, show accessibility tree before audit
    if verbose:
        from src.mcp.server import browser_navigate, get_accessibility_tree
        print("\n[VERBOSE] Fetching accessibility tree...")
        nav_result = await browser_navigate(url)
        print(f"[VERBOSE] Navigation: {nav_result.get('status')}")
        tree_result = await get_accessibility_tree()
        if tree_result.get('status') == 'success':
            print(f"[VERBOSE] Accessibility tree:\n{tree_result.get('tree', '')[:2000]}...")
        print("-" * 60)

    print("Starting full agentic audit...")

    agent = DarkPatternAgent(
        model=model,
        provider=provider,
        max_steps=10,
    )

    try:
        result = await agent.run_audit(url)
        print_result(result)
    except Exception as e:
        print(f"\nError during audit: {e}")
        import traceback
        traceback.print_exc()


async def run_dynamic_audit(url: str, user_query: str = None, verbose: bool = False):
    """Run Phase 2 dynamic audit using LangGraph and Planner-Actor-Auditor.

    This mode supports multi-step journeys, state tracking, and temporal
    pattern detection (like drip pricing and sneak-into-basket).
    """
    print_header("DARK PATTERN HUNTER - PHASE 2 DYNAMIC AUDIT")
    print(f"Target URL: {url}")
    if user_query:
        print(f"User Query: {user_query}")
    print("-" * 60)

    # Initialize agent
    model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    provider = os.getenv("LLM_PROVIDER", "anthropic")

    print(f"Using model: {model} ({provider})")
    print("Mode: Phase 2 (LangGraph + Planner-Actor-Auditor)")
    print("-" * 60)

    # Enable debug logging if verbose
    if verbose:
        os.environ["DEBUG_ENABLED"] = "true"
        print("[VERBOSE] Debug logging enabled")

    print("Starting dynamic audit...")

    agent = DarkPatternAgent(
        model=model,
        provider=provider,
        max_steps=25,  # More steps for dynamic exploration
    )

    try:
        query = user_query or "Audit this website for dark patterns"
        result = await agent.run_dynamic_audit(url, user_query=query)
        print_result(result)
    except Exception as e:
        print(f"\nError during audit: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    # Parse arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    dynamic = "--dynamic" in sys.argv or "-d" in sys.argv

    # Parse --query argument
    user_query = None
    for i, arg in enumerate(sys.argv):
        if arg in ("--query", "-q") and i + 1 < len(sys.argv):
            user_query = sys.argv[i + 1]
            break

    # Filter out flags and query value
    args = []
    skip_next = False
    for i, a in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if a in ("--query", "-q"):
            skip_next = True
            continue
        if not a.startswith("-"):
            args.append(a)

    if "--list" in sys.argv:
        print("Available simulations:")
        for name, sim in SIMULATIONS.items():
            expected = sim['expected_pattern'] or 'none (clean)'
            print(f"  {name}: {sim['description']} [expects: {expected}]")
        print("\nOr provide any URL: python demo.py https://example.com")
        print("\nPhase 2 Dynamic Audit:")
        print("  python demo.py --dynamic https://example.com")
        return
    elif "--help" in sys.argv:
        print(__doc__)
        print("\nOptions:")
        print("  --verbose, -v  Show accessibility tree and debug info")
        print("  --dynamic, -d  Use Phase 2 dynamic audit (LangGraph)")
        print("  --query, -q    User query for Phase 2 audit (use with --dynamic)")
        print("  --list         List available simulations")
        print("\nURL support:")
        print("  python demo.py https://example.com              # Phase 1 static audit")
        print("  python demo.py --dynamic https://example.com    # Phase 2 dynamic audit")
        print("  python demo.py --dynamic https://ryanair.com --query \"Book a flight to Dublin\"")
        return

    arg = args[0] if args else None

    # Check if argument is a URL
    if arg and (arg.startswith("http://") or arg.startswith("https://")):
        server = None
        try:
            # Start local server if it's a localhost URL
            if "localhost:8888" in arg or "127.0.0.1:8888" in arg:
                port = 8888
                print(f"Starting local server on port {port}...")
                server = start_server(port)

            if dynamic:
                asyncio.run(run_dynamic_audit(arg, user_query=user_query, verbose=verbose))
            else:
                asyncio.run(run_url_audit(arg, verbose))
        finally:
            asyncio.run(browser_cleanup())
            if server:
                server.shutdown()
                print("\nServer stopped.")
        return

    # Otherwise treat as simulation name
    simulation = arg

    # Start local server for simulations
    port = 8888
    print(f"Starting local server on port {port}...")
    server = start_server(port)
    try:
        if simulation:
            asyncio.run(run_demo(simulation, port, verbose))
        else:
            asyncio.run(run_all_demos(port, verbose))
    finally:
        # Cleanup browser
        asyncio.run(browser_cleanup())
        server.shutdown()
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
