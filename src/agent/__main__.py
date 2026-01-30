"""Command-line entry point for the Dark Pattern Agent."""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .core import DarkPatternAgent


async def main():
    """Main entry point for running audits."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.agent <url>")
        print("Example: python -m src.agent https://example.com")
        sys.exit(1)

    url = sys.argv[1]

    # Initialize agent
    model = os.getenv("LLM_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"))
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    agent = DarkPatternAgent(
        model=model,
        provider=provider,
        max_steps=int(os.getenv("MAX_AUDIT_STEPS", "50")),
    )

    print(f"Starting audit of {url}...")
    print("-" * 60)

    # Run audit
    try:
        result = await agent.run_audit(url)

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

    except Exception as e:
        print(f"\nError during audit: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import os
    asyncio.run(main())
