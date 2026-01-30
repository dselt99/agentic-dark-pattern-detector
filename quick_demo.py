"""Quick Demo - Simple single-pass dark pattern detection.

Usage:
    python quick_demo.py                          # Run against false_urgency.html
    python quick_demo.py roach_motel              # Run against roach_motel.html
    python quick_demo.py https://example.com      # Run against any URL
"""

import asyncio
import http.server
import socketserver
import threading
import sys
import os
import json
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Suppress logging
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)


SIMULATIONS = {
    "false_urgency": "evals/simulations/false_urgency.html",
    "roach_motel": "evals/simulations/roach_motel.html",
    "clean_stock": "evals/simulations/clean_stock.html",
}


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args): pass


async def run_quick_demo(simulation: str = "false_urgency"):
    """Run a quick single-pass demo."""
    from src.mcp.server import browser_navigate, get_accessibility_tree, cleanup
    from anthropic import AsyncAnthropic

    # Start local server
    port = 8899
    server = socketserver.TCPServer(("", port), QuietHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        html_file = SIMULATIONS.get(simulation, SIMULATIONS["false_urgency"])
        url = f"http://localhost:{port}/{html_file}"

        print("=" * 60)
        print("DARK PATTERN HUNTER - QUICK DEMO")
        print("=" * 60)
        print(f"Target: {simulation}")
        print(f"URL: {url}")
        print("-" * 60)

        # Navigate and get tree
        print("Navigating...")
        await browser_navigate(url)

        print("Extracting page structure...")
        # Use include_hidden=True to catch roach motel patterns (cancel links in collapsed menus)
        tree_result = await get_accessibility_tree(max_depth=20, include_hidden=True)
        if tree_result.get("status") != "success":
            print(f"Error: {tree_result.get('message')}")
            return

        tree_yaml = tree_result.get("tree", "")
        print(f"Page structure extracted ({len(tree_yaml)} chars)")

        # Load skill
        skill_path = project_root / "skills" / "detect-manipulation.md"
        skill_content = skill_path.read_text(encoding="utf-8")

        # Call LLM
        print("Analyzing with Claude...")
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        system_prompt = f"""{skill_content}

You are analyzing a web page for dark patterns. Respond with a JSON object:
{{
    "findings": [
        {{
            "pattern_type": "false_urgency|roach_motel|confirmshaming|sneak_into_basket|forced_continuity",
            "confidence_score": 0.7-1.0,
            "element_selector": "CSS selector",
            "reasoning": "Why this is a dark pattern",
            "evidence": "The actual text or behavior"
        }}
    ],
    "summary": "Brief summary of findings"
}}

If no dark patterns found, return {{"findings": [], "summary": "No dark patterns detected."}}
Only return JSON, no other text."""

        user_prompt = f"""Analyze this page structure for dark patterns:

```yaml
{tree_yaml}
```

Look for:
- False urgency (countdown timers, "limited time" claims)
- Roach motel (easy signup, hard cancellation) - Look for ASYMMETRY: prominent signup buttons vs hidden cancel links in accordions/FAQs/footers. The tree includes hidden elements.
- Confirmshaming (guilt-inducing decline options)
- Sneak into basket (pre-checked add-ons)
- Forced continuity (hidden auto-renewal)

Respond with JSON only."""

        response = await client.messages.create(
            model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        result_text = response.content[0].text
        print("-" * 60)
        print("RESULTS")
        print("-" * 60)

        try:
            # Clean up response - extract JSON if wrapped in markdown
            clean_text = result_text.strip()
            if clean_text.startswith("```"):
                # Remove markdown code blocks
                lines = clean_text.split("\n")
                clean_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            result = json.loads(clean_text)
            findings = result.get("findings", [])
            summary = result.get("summary", "")

            # Remove emojis for Windows console compatibility
            summary = summary.encode('ascii', 'ignore').decode('ascii')
            print(f"Summary: {summary}")
            print(f"Findings: {len(findings)}")

            if findings:
                for i, f in enumerate(findings, 1):
                    pattern = f.get('pattern_type', 'unknown').upper()
                    print(f"\n{i}. {pattern}")
                    print(f"   Confidence: {f.get('confidence_score', 0):.2f}")
                    print(f"   Selector: {f.get('element_selector', 'N/A')}")
                    reasoning = f.get('reasoning', 'N/A').encode('ascii', 'ignore').decode('ascii')
                    print(f"   Reasoning: {reasoning}")
                    if f.get("evidence"):
                        evidence = f.get('evidence').encode('ascii', 'ignore').decode('ascii')
                        print(f"   Evidence: {evidence}")
            else:
                print("\nNo dark patterns detected.")

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            # Write raw response to file for debugging
            with open("debug_response.txt", "w", encoding="utf-8") as f:
                f.write(result_text)
            print("Raw response saved to debug_response.txt")

        print("=" * 60)

    finally:
        await cleanup()
        server.shutdown()


async def run_real_url_demo(url: str):
    """Run demo against a real URL."""
    from src.mcp.server import browser_navigate, get_accessibility_tree, cleanup
    from anthropic import AsyncAnthropic

    try:
        print("=" * 60)
        print("DARK PATTERN HUNTER - LIVE SITE ANALYSIS")
        print("=" * 60)
        print(f"Target: {url}")
        print("-" * 60)

        # Navigate and get tree
        print("Navigating (this may take a moment)...")
        nav_result = await browser_navigate(url)
        if nav_result.get("status") != "success":
            print(f"Navigation failed: {nav_result.get('message')}")
            return

        print("Extracting page structure...")
        tree_result = await get_accessibility_tree()
        if tree_result.get("status") != "success":
            print(f"Error: {tree_result.get('message')}")
            return

        tree_yaml = tree_result.get("tree", "")
        print(f"Page structure extracted ({len(tree_yaml)} chars)")

        # Load skill
        skill_path = project_root / "skills" / "detect-manipulation.md"
        skill_content = skill_path.read_text(encoding="utf-8")

        # Call LLM
        print("Analyzing with Claude...")
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        system_prompt = f"""{skill_content}

You are analyzing a web page for dark patterns. Respond with a JSON object:
{{
    "findings": [
        {{
            "pattern_type": "false_urgency|roach_motel|confirmshaming|sneak_into_basket|forced_continuity",
            "confidence_score": 0.7-1.0,
            "element_selector": "CSS selector",
            "reasoning": "Why this is a dark pattern",
            "evidence": "The actual text or behavior"
        }}
    ],
    "summary": "Brief summary of findings"
}}

If no dark patterns found, return {{"findings": [], "summary": "No dark patterns detected."}}
Only return JSON, no other text."""

        # Truncate tree if too long
        max_tree_len = 8000
        if len(tree_yaml) > max_tree_len:
            tree_yaml = tree_yaml[:max_tree_len] + "\n... (truncated)"

        user_prompt = f"""Analyze this page structure for dark patterns:

```yaml
{tree_yaml}
```

Look for:
- False urgency (countdown timers, "limited time" claims)
- Roach motel (easy signup, hard cancellation)
- Confirmshaming (guilt-inducing decline options)
- Sneak into basket (pre-checked add-ons)
- Forced continuity (hidden auto-renewal)

Respond with JSON only."""

        response = await client.messages.create(
            model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        result_text = response.content[0].text
        print("-" * 60)
        print("RESULTS")
        print("-" * 60)

        try:
            clean_text = result_text.strip()
            if clean_text.startswith("```"):
                lines = clean_text.split("\n")
                clean_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            result = json.loads(clean_text)
            findings = result.get("findings", [])
            summary = result.get("summary", "")

            summary = summary.encode('ascii', 'ignore').decode('ascii')
            print(f"Summary: {summary}")
            print(f"Findings: {len(findings)}")

            if findings:
                for i, f in enumerate(findings, 1):
                    pattern = f.get('pattern_type', 'unknown').upper()
                    print(f"\n{i}. {pattern}")
                    print(f"   Confidence: {f.get('confidence_score', 0):.2f}")
                    print(f"   Selector: {f.get('element_selector', 'N/A')}")
                    reasoning = f.get('reasoning', 'N/A').encode('ascii', 'ignore').decode('ascii')
                    print(f"   Reasoning: {reasoning}")
                    if f.get("evidence"):
                        evidence = str(f.get('evidence')).encode('ascii', 'ignore').decode('ascii')
                        print(f"   Evidence: {evidence[:200]}")
            else:
                print("\nNo dark patterns detected.")

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            with open("debug_response.txt", "w", encoding="utf-8") as f:
                f.write(result_text)
            print("Raw response saved to debug_response.txt")

        print("=" * 60)

    finally:
        await cleanup()


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "false_urgency"

    if arg == "--list":
        print("Available simulations:", ", ".join(SIMULATIONS.keys()))
        print("\nOr provide any URL: python quick_demo.py https://example.com")
        return

    # Check if it's a URL
    if arg.startswith("http://") or arg.startswith("https://"):
        asyncio.run(run_real_url_demo(arg))
    else:
        asyncio.run(run_quick_demo(arg))


if __name__ == "__main__":
    main()
