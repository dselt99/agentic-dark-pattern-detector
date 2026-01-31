"""Test Phase 2 with real websites.

This script provides an easy way to test the Phase 2 dynamic audit system
against real websites. It uses the Planner-Actor-Auditor architecture to
perform multi-step journey simulations and detect dark patterns.

Usage:
    python test_real_site.py <url> [--query "custom query"] [--steps 20]
    
Examples:
    # Basic audit
    python test_real_site.py https://example.com
    
    # Custom query for specific pattern
    python test_real_site.py https://example.com --query "Audit checkout flow for Drip Pricing"
    
    # More steps for complex sites
    python test_real_site.py https://example.com --steps 30
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Setup
load_dotenv()
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agent.core import DarkPatternAgent
from src.mcp.server import cleanup as browser_cleanup


async def test_real_site(
    url: str,
    user_query: str = None,
    max_steps: int = 20,
    verbose: bool = False,
):
    """Test Phase 2 against a real website.
    
    Args:
        url: Target URL to audit
        user_query: Custom audit query (default: general dark pattern audit)
        max_steps: Maximum number of steps for the audit
        verbose: Print detailed progress information
    """
    
    if not user_query:
        user_query = "Audit this website for dark patterns"
    
    print("=" * 70)
    print("PHASE 2 - REAL SITE TEST")
    print("=" * 70)
    print(f"URL: {url}")
    print(f"Query: {user_query}")
    print(f"Max Steps: {max_steps}")
    print("-" * 70)
    
    # Initialize agent
    model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    
    if verbose:
        print(f"Model: {model} ({provider})")
        print("-" * 70)
    
    agent = DarkPatternAgent(
        model=model,
        provider=provider,
        max_steps=max_steps,
    )
    
    try:
        if verbose:
            print("\n[INFO] Starting Phase 2 dynamic audit...")
            print("[INFO] This may take several minutes for complex sites...")
        
        result = await agent.run_dynamic_audit(
            url=url,
            user_query=user_query
        )
        
        print("\n" + "=" * 70)
        print("AUDIT RESULTS")
        print("=" * 70)
        print(f"URL: {result.target_url}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Findings: {len(result.findings)}")
        print(f"\nSummary:\n{result.summary}")
        
        if result.findings:
            print("\n" + "-" * 70)
            print("DETECTED PATTERNS")
            print("-" * 70)
            for i, finding in enumerate(result.findings, 1):
                print(f"\n{i}. {finding.pattern_type.value.upper().replace('_', ' ')}")
                print(f"   Confidence: {finding.confidence_score:.2f}")
                if finding.element_selector:
                    print(f"   Element: {finding.element_selector}")
                print(f"   Reasoning: {finding.reasoning}")
                if finding.evidence:
                    print(f"   Evidence: {finding.evidence[:200]}...")
        else:
            print("\n" + "-" * 70)
            print("NO DARK PATTERNS DETECTED")
            print("-" * 70)
            print("The audit completed but did not identify any dark patterns.")
            print("This could mean:")
            print("  - The site is clean")
            print("  - Patterns require more steps to detect")
            print("  - Patterns require specific user interactions")
        
        if result.screenshot_paths:
            print("\n" + "-" * 70)
            print("SCREENSHOTS")
            print("-" * 70)
            print(f"Captured {len(result.screenshot_paths)} screenshot(s):")
            for path in result.screenshot_paths:
                print(f"  - {path}")
        
        print("\n" + "=" * 70)
        
        # Output JSON for programmatic use
        if verbose:
            import json
            print("\nJSON Output:")
            print(json.dumps(result.model_dump(), indent=2, default=str))
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Audit interrupted by user")
        return None
    except Exception as e:
        print(f"\n[ERROR] Audit failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None
    finally:
        if verbose:
            print("\n[INFO] Cleaning up browser session...")
        await browser_cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Phase 2 dynamic audit against real websites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic audit
  python test_real_site.py https://example.com
  
  # Custom query for specific pattern
  python test_real_site.py https://example.com --query "Audit checkout for Drip Pricing"
  
  # More steps for complex sites
  python test_real_site.py https://example.com --steps 30
  
  # Verbose output
  python test_real_site.py https://example.com --verbose
        """
    )
    
    parser.add_argument(
        "url",
        help="Target URL to audit"
    )
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Custom audit query (default: general dark pattern audit)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=20,
        help="Maximum number of audit steps (default: 20)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable graph execution debugging (shows node transitions, timing, state)"
    )
    parser.add_argument(
        "--debug-verbose",
        action="store_true",
        help="Enable verbose graph debugging (includes state snapshots)"
    )
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith(("http://", "https://")):
        print(f"[ERROR] Invalid URL: {args.url}")
        print("URL must start with http:// or https://")
        sys.exit(1)
    
    # Enable debugging if requested
    if args.debug or args.debug_verbose:
        from src.agent.debug import enable_debug
        enable_debug(enabled=True, verbose=args.debug_verbose)
        if args.verbose:
            print("[INFO] Graph debugging enabled")
            if args.debug_verbose:
                print("[INFO] Verbose debugging enabled (state snapshots)")
    
    asyncio.run(test_real_site(
        url=args.url,
        user_query=args.query,
        max_steps=args.steps,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
