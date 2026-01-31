"""Command-line entry point for the Dark Pattern Agent."""

import asyncio
import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .core import DarkPatternAgent
from ..mcp.server import cleanup as browser_cleanup


async def main():
    """Main entry point for running audits."""
    parser = argparse.ArgumentParser(
        description="Dark Pattern Agent - Audit websites for dark patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 2 audit (default)
  python -m src.agent https://example.com
  
  # Phase 2 with custom query
  python -m src.agent https://example.com --query "Audit checkout flow"
  
  # Phase 1 audit (legacy)
  python -m src.agent https://example.com --phase1
  
  # Phase 2 with more steps
  python -m src.agent https://example.com --steps 30
        """
    )
    
    parser.add_argument(
        "url",
        help="Target URL to audit"
    )
    parser.add_argument(
        "--query", "-q",
        default="Audit this website for dark patterns",
        help="Custom audit query for Phase 2 (default: general audit)"
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Use Phase 1 audit (legacy single-step mode)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=None,
        help="Maximum number of audit steps (default: 50 for Phase 1, 20 for Phase 2)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable graph execution debugging (Phase 2 only)"
    )
    parser.add_argument(
        "--debug-verbose",
        action="store_true",
        help="Enable verbose graph debugging with state snapshots (Phase 2 only)"
    )
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith(("http://", "https://")):
        print(f"Error: Invalid URL: {args.url}")
        print("URL must start with http:// or https://")
        sys.exit(1)

    # Initialize agent
    model = os.getenv("LLM_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"))
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    
    # Set default steps based on phase
    default_steps = args.steps or (50 if args.phase1 else 20)
    
    # Enable debugging if requested (Phase 2 only)
    if not args.phase1 and (args.debug or args.debug_verbose):
        from .debug import enable_debug
        enable_debug(enabled=True, verbose=args.debug_verbose)
        print("[INFO] Graph debugging enabled")
        if args.debug_verbose:
            print("[INFO] Verbose debugging enabled (state snapshots)")
    
    agent = DarkPatternAgent(
        model=model,
        provider=provider,
        max_steps=default_steps,
    )

    phase_name = "Phase 1" if args.phase1 else "Phase 2"
    print(f"Starting {phase_name} audit of {args.url}...")
    print("-" * 60)

    # Run audit
    try:
        if args.phase1:
            result = await agent.run_audit(args.url)
        else:
            result = await agent.run_dynamic_audit(
                url=args.url,
                user_query=args.query
            )

        # Print results
        print("\n" + "=" * 60)
        print("AUDIT RESULTS")
        print("=" * 60)
        print(f"URL: {result.target_url}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Findings: {len(result.findings)}")
        print(f"\nSummary:\n{result.summary}")

        if result.findings:
            print("\nDetected Patterns:")
            for i, finding in enumerate(result.findings, 1):
                print(f"\n{i}. {finding.pattern_type.value.upper()}")
                print(f"   Confidence: {finding.confidence_score:.2f}")
                print(f"   Selector: {finding.element_selector}")
                print(f"   Reasoning: {finding.reasoning}")
                if finding.evidence:
                    print(f"   Evidence: {finding.evidence}")

        if result.screenshot_paths:
            print(f"\nScreenshots: {len(result.screenshot_paths)} captured")
            for path in result.screenshot_paths:
                print(f"  - {path}")

        print("\n" + "=" * 60)

        # Output JSON for programmatic use
        import json
        print("\nJSON Output:")
        print(json.dumps(result.model_dump(), indent=2, default=str))

    except KeyboardInterrupt:
        print("\n\nAudit interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError during audit: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup browser session
        try:
            await browser_cleanup()
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    import os
    asyncio.run(main())
