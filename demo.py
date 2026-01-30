"""Demo script for the Dark Pattern Hunter.

This script starts a local HTTP server to serve test HTML files,
then runs the agent against them to demonstrate detection capabilities.

Usage:
    python demo.py                    # Run all demos
    python demo.py false_urgency      # Run specific demo
    python demo.py --list             # List available demos
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

    agent = DarkPatternAgent(
        model=model,
        provider=provider,
        max_steps=1,  # Single pass for demo - faster
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


def main():
    """Main entry point."""
    # Parse arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if "--list" in sys.argv:
        print("Available simulations:")
        for name, sim in SIMULATIONS.items():
            expected = sim['expected_pattern'] or 'none (clean)'
            print(f"  {name}: {sim['description']} [expects: {expected}]")
        return
    elif "--help" in sys.argv:
        print(__doc__)
        print("\nOptions:")
        print("  --verbose, -v  Show accessibility tree and debug info")
        print("  --list         List available simulations")
        return

    simulation = args[0] if args else None

    # Start local server
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
