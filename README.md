# Dark Pattern Hunter

An agentic AI system that autonomously navigates websites, simulates real user journeys (search, add-to-cart, checkout), and detects deceptive design patterns (dark patterns) in web interfaces.

## Overview

Dark Pattern Hunter uses a **Planner-Actor-Auditor** architecture powered by LangGraph to navigate multi-step user flows end-to-end. It doesn't just scan static pages — it searches for products, clicks through listings, adds items to cart, proceeds through checkout, and audits every step for manipulative UI patterns.

### What It Detects

| Pattern | Description | How It Detects |
|---------|-------------|----------------|
| **False Urgency** | Countdown timers that reset, fake scarcity claims | Reloads page and compares timer values |
| **Drip Pricing** | Hidden fees revealed late in checkout | Tracks price from product page through checkout |
| **Sneak into Basket** | Pre-checked add-ons, unauthorized cart items | Monitors cart state for unexpected additions |
| **Roach Motel** | Easy signup, difficult cancellation | Compares signup vs cancellation flow complexity |
| **Forced Continuity** | Hidden auto-renewal terms | Identifies buried subscription terms |
| **Privacy Zuckering** | Manipulative consent UIs | Analyzes cookie/privacy consent dialogs |
| **Confirmshaming** | Guilt-inducing decline options | Detects emotional language on opt-out buttons |

## Quick Start

### Installation

```bash
git clone https://github.com/dselt99/agentic-dark-pattern-detector.git
cd agentic-dark-pattern-detector

pip install -r requirements.txt
playwright install chromium
```

### Configuration

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your-api-key-here
LLM_MODEL=claude-haiku-4-5-20251001
BROWSER_HEADLESS=true
```

### Running an Audit

```bash
# Audit a site with a purchase flow
python test_real_site.py https://www.saucedemo.com \
  --query "Log in, buy the cheapest item, go through checkout, look for dark patterns" \
  --steps 25 --debug

# Audit any e-commerce site
python test_real_site.py https://example-shop.com \
  --query "Buy a product and look out for dark patterns" \
  --steps 20

# Via the CLI entry point
python -m src.agent https://example.com --query "Audit checkout flow for hidden fees" --steps 30 --debug
```

### Example Output

```
======================================================================
AUDIT RESULTS
======================================================================
URL: https://www.saucedemo.com
Findings: 0

Execution Summary:
- Steps executed: 18/25
- Tasks completed: 16/16
- Interaction snapshots: 18

[OK] All planned tasks completed successfully.

Key Interactions:
- Pages visited: 7
- Click actions: 8
- Type actions: 5

Analysis Result: After examining 18 interaction points across 7 page(s),
no dark patterns were identified.
======================================================================
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph State Machine                   │
│  PLAN → ACT → AUDIT → EVAL → (loop) → COMPLETE             │
└─────────────────────────────────────────────────────────────┘
        │           │           │
        ▼           ▼           ▼
  ┌──────────┐ ┌─────────┐ ┌──────────┐
  │ Planner  │ │  Actor  │ │ Auditor  │
  │ LLM goal │ │ LLM     │ │ Pattern  │
  │ decomp.  │ │ actions │ │ detectors│
  └──────────┘ └─────────┘ └──────────┘
        │           │           │
        └───────────┴───────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  ┌───────────┐          ┌────────────┐
  │ Journey   │          │ MCP Server │
  │ Ledger    │          │ (Playwright│
  │ (Memory)  │          │  Browser)  │
  └───────────┘          └────────────┘
```

### How It Works

1. **Planner** decomposes the user goal into 10-12 actionable tasks (navigate, search, click, add-to-cart, checkout, verify)
2. **Actor** uses Claude to decide which browser action to take based on the current DOM and marked interactive elements
3. **Auditor** runs pattern detectors after each action, checking for dark patterns in the current page state
4. **State Evaluator** decides whether to advance to the next task, re-observe, replan, or escalate
5. **Journey Ledger** tracks the full interaction history for stateful detection (e.g., price changes across pages)

### Key Components

| File | Purpose |
|------|---------|
| `src/agent/graph.py` | LangGraph state machine with 7 nodes (PLAN_GENESIS, NAV_ACTOR, DOM_AUDITOR, STATE_EVAL, SAFE_GUARD, WAIT_AND_RETRY, HUMAN_INTERVENTION) |
| `src/agent/planner.py` | LLM-powered goal decomposition with checkout-flow awareness |
| `src/agent/actor.py` | LLM-powered action selection from DOM + marked elements |
| `src/agent/auditor.py` | Modular pattern detectors (false urgency, drip pricing, etc.) |
| `src/mcp/server.py` | Playwright browser tools (click, type, navigate, screenshot, accessibility tree) |
| `src/agent/ledger.py` | Episodic memory tracking interaction history |
| `src/agent/checkpoints.py` | JSON checkpoint save/load for debugging |

### Browser Automation Features

- **Set-of-Marks**: Interactive elements are stamped with unique `data-mark-id` attributes for reliable targeting
- **New tab detection**: Handles `target="_blank"` links by detecting new tabs and switching context
- **Link click fallback**: If a JS click handler prevents navigation on `<a>` elements, falls back to direct `page.goto(href)`
- **Stuck detection**: Tiered escalation — 3 attempts re-observe, 5 replan, 8 force-complete
- **DOM refresh**: Automatic DOM invalidation and re-fetch after every state-changing action

### Planner Checkout Flow

When the user goal involves buying, shopping, booking, or subscribing, the planner generates the full end-to-end journey:

1. Navigate to site
2. Dismiss popups
3. Search for product
4. Click product listing
5. Analyze product page for dark patterns
6. Select options (size/color/etc.) + add to cart *(combined in one task)*
7. Proceed to checkout
8. Analyze checkout for hidden fees, drip pricing, sneak-into-basket
9. Fill payment form (test card: `4242 4242 4242 4242`)
10. Verify final total matches advertised price

## Testing

### Live Site Audit (`test_real_site.py`)

The primary CLI for running full audits against real websites:

```bash
# Audit a site with a purchase flow
python test_real_site.py https://www.saucedemo.com \
  --query "Log in, buy the cheapest item, go through checkout, look for dark patterns" \
  --steps 25 --debug

# Audit any e-commerce site
python test_real_site.py https://example-shop.com \
  --query "Buy a product and look out for dark patterns" \
  --steps 20
```

### Local Simulations (`demo.py`)

Runs against locally-served HTML pages with known dark patterns — useful for testing detection without hitting real sites:

```bash
python demo.py                    # All Phase 1 simulations (false urgency, roach motel, etc.)
python demo.py false_urgency      # Single scenario
python demo.py roach_motel        # Single scenario
python demo.py --dynamic          # Phase 2 multi-step audit against local simulation
```

### Component Tests (`test_phase2.py`)

Developer test suite for verifying individual Phase 2 components (planner, actor, auditor, graph wiring):

```bash
python test_phase2.py                 # Full test suite
python test_phase2.py --components    # Component unit tests only
python test_phase2.py --basic         # Basic integration test
```

## Debugging

```bash
# Basic debugging (node transitions, timing, task execution)
python test_real_site.py https://example.com \
  --query "Look for dark patterns" --debug

# Verbose debugging (includes full state snapshots)
python test_real_site.py https://example.com \
  --query "Look for dark patterns" --debug --verbose
```

Debug output shows node entry/exit timing, task execution details, pattern detection results, and a performance summary.

## Limitations

- **Bot protection**: Some sites (e.g., eBay) block add-to-cart and checkout from unauthenticated automated sessions
- **Visual patterns**: Color/size manipulation and visual misdirection require additional CV work
- **Complex JS sites**: Heavy SPAs may need more steps to fully explore
- **Login-gated flows**: Sites requiring authentication need credentials passed in the query

## License

MIT
