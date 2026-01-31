# Dark Pattern Hunter

An agentic AI system for detecting deceptive design patterns (dark patterns) in web interfaces.

## Overview

Dark Pattern Hunter uses a five-pillar architecture to autonomously audit websites for manipulative UI patterns:

| Pillar | Purpose |
|--------|---------|
| **MCP Server** | Playwright-based browser automation tools |
| **Skills** | Markdown-defined detection rules and heuristics |
| **Schemas** | Pydantic models for structured LLM output |
| **Agent** | ReAct-pattern orchestrator for audits |
| **Evals** | Test simulations with ground truth |

## Detected Patterns

- **False Urgency** - Countdown timers that reset, fake scarcity claims
- **Roach Motel** - Easy signup, difficult cancellation
- **Confirmshaming** - Guilt-inducing decline options
- **Sneak into Basket** - Pre-checked add-ons, hidden charges
- **Forced Continuity** - Hidden auto-renewal terms

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dselt99/agentic-dark-pattern-detector.git
cd agentic-dark-pattern-detector

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium
```

### Configuration

Create a `.env` file:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-api-key-here
LLM_MODEL=claude-haiku-4-5-20251001
BROWSER_HEADLESS=true
```

### Running the Demo

### Phase 1 (Legacy - Single Step)

```bash
# Test against local simulation (false urgency pattern)
python quick_demo.py

# Test against other simulations
python quick_demo.py roach_motel
python quick_demo.py clean_stock

# Test against a live website (Phase 1)
python -m src.agent https://example.com --phase1
```

### Phase 2 (Dynamic Multi-Step - Recommended)

Phase 2 uses a Planner-Actor-Auditor architecture with LangGraph for multi-step journey simulation and dynamic pattern detection.

```bash
# Basic Phase 2 audit (default)
python -m src.agent https://example.com

# Phase 2 with custom query
python -m src.agent https://example.com --query "Audit checkout flow for Drip Pricing"

# Phase 2 with more steps for complex sites
python -m src.agent https://example.com --steps 30

# Phase 2 with graph debugging enabled
python -m src.agent https://example.com --debug
python -m src.agent https://example.com --debug-verbose  # Includes state snapshots

# Or use the dedicated test script
python test_real_site.py https://example.com
python test_real_site.py https://example.com --query "Test cancellation flow" --steps 30 --verbose
python test_real_site.py https://example.com --debug  # Enable graph debugging
```

### Testing Phase 2 Components

```bash
# Run all Phase 2 tests
python test_phase2.py

# Run specific test suites
python test_phase2.py --components    # Component tests only
python test_phase2.py --basic         # Basic integration test
python test_phase2.py --roach-motel   # Roach Motel journey test
python test_phase2.py --mcp-tools     # MCP tools test
```

## Example Output

```
============================================================
DARK PATTERN HUNTER - QUICK DEMO
============================================================
Target: false_urgency
URL: http://localhost:8899/evals/simulations/false_urgency.html
------------------------------------------------------------
Analyzing with Claude...
------------------------------------------------------------
RESULTS
------------------------------------------------------------
Summary: One potential dark pattern detected: False Urgency
Findings: 1

1. FALSE_URGENCY
   Confidence: 0.75
   Selector: div.timer#timer
   Reasoning: A countdown timer displaying '05:00' is present with
   accompanying urgency language ('Limited Time Offer - Act Now!').
   Evidence: Timer element with text '05:00', urgency-text class
============================================================
```

## Architecture

### Phase 1 (Legacy)
```
┌─────────────────────────────────────────────────────────┐
│                      Agent (ReAct Loop)                 │
│  Observe → Reason → Act → Observe → ... → Report        │
└─────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────┐          ┌──────────────────────────┐
│   MCP Server     │          │   Skills (Markdown)      │
│  - navigate      │          │  - Pattern definitions   │
│  - get_tree      │          │  - Detection heuristics  │
│  - screenshot    │          │  - Negative constraints  │
└──────────────────┘          └──────────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────┐          ┌──────────────────────────┐
│   Playwright     │          │   Pydantic Schemas       │
│   (Browser)      │          │   (Structured Output)    │
└──────────────────┘          └──────────────────────────┘
```

### Phase 2 (Dynamic Multi-Step)
```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                │
│  PLAN → ACT → AUDIT → EVAL → PLAN → ... → COMPLETE      │
└─────────────────────────────────────────────────────────┘
          │         │         │
          ▼         ▼         ▼
    ┌─────────┐ ┌──────┐ ┌─────────┐
    │ Planner │ │Actor │ │ Auditor │
    │(Strategy)│ │(Exec)│ │(Observe)│
    └─────────┘ └──────┘ └─────────┘
          │         │         │
          └─────────┴─────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    ┌──────────┐      ┌──────────────┐
    │Journey   │      │  Detectors   │
    │Ledger    │      │  (Modular)   │
    │(Memory)  │      └──────────────┘
    └──────────┘
```

**Phase 2 Features:**
- **Planner**: Decomposes goals into actionable tasks
- **Actor**: Executes browser interactions (click, type, scroll, navigate)
- **Auditor**: Observes state changes and detects patterns in real-time
- **Journey Ledger**: Tracks interaction history for stateful pattern detection
- **Modular Detectors**: Specialized modules for each dark pattern type
- **Sandboxing**: Payment interception and synthetic identity for safe testing

## Project Structure

```
├── src/
│   ├── agent/
│   │   ├── core.py          # ReAct loop orchestrator
│   │   ├── mcp_client.py    # Tool calling wrapper
│   │   └── __main__.py      # CLI entry point
│   ├── mcp/
│   │   └── server.py        # Browser automation tools
│   └── schemas/
│       ├── schemas.py       # Pydantic models
│       └── utils.py         # Schema injection utilities
├── skills/
│   └── detect-manipulation.md   # Detection rules
├── evals/
│   ├── simulations/         # Test HTML files
│   └── run.py               # Evaluation framework
├── quick_demo.py            # Simple demo script
├── demo.py                  # Full demo with all features
├── test_phase2.py           # Phase 2 component and integration tests
└── test_real_site.py        # Real website testing script (Phase 2)
```

## Phase 2 Capabilities

Phase 2 extends Phase 1 with:

- ✅ **Multi-step journey simulation** - Navigate through complex user flows
- ✅ **Dynamic pattern detection** - Detects patterns that emerge over time
- ✅ **State tracking** - Remembers previous interactions for context-aware detection
- ✅ **False Urgency detection** - Verifies timer reset behavior on reload
- ✅ **Drip Pricing detection** - Tracks price changes through checkout
- ✅ **Sneak into Basket detection** - Monitors cart state for unauthorized additions
- ✅ **Roach Motel detection** - Compares signup vs cancellation difficulty
- ✅ **Forced Continuity detection** - Identifies hidden auto-renewal terms
- ✅ **Privacy Zuckering detection** - Analyzes consent UI for manipulation

## Debugging

Phase 2 includes comprehensive debugging capabilities for graph execution:

### Enable Debugging

**Via command line:**
```bash
# Basic debugging (node transitions, timing, task execution)
python -m src.agent https://example.com --debug

# Verbose debugging (includes state snapshots)
python -m src.agent https://example.com --debug-verbose
```

**Via environment variable:**
```bash
# Enable basic debugging
export DEBUG_GRAPH=true
python -m src.agent https://example.com

# Enable verbose debugging
export DEBUG_GRAPH=true
export DEBUG_GRAPH_VERBOSE=true
python -m src.agent https://example.com
```

### Debug Output Includes

- **Node Entry/Exit**: Logs when each graph node is entered and exited
- **Timing Metrics**: Execution time for each node
- **State Transitions**: Graph edge transitions with reasons
- **Task Execution**: Details of task execution and results
- **Pattern Detection**: Detector execution and flags raised
- **Performance Summary**: Aggregate timing statistics
- **State Summary**: Final state overview with task completion, patterns detected

### Example Debug Output

```
[STEP 0] → ENTER PLAN_GENESIS
  Decomposing goal: Audit this website for dark patterns...
  Generated 5 task(s)
[STEP 0] ← EXIT PLAN_GENESIS [0.234s]

[STEP 1] → ENTER NAV_ACTOR
  Task: navigate - Navigate to https://example.com
  Executing task: navigate
    Goal: Navigate to https://example.com
    Status: starting
  Action type: navigate
  Navigating to: https://example.com
  Navigation successful
[STEP 1] ← EXIT NAV_ACTOR [1.456s]

[STEP 2] → ENTER DOM_AUDITOR
  Running pattern detectors...
  Detected 2 new pattern flag(s)
    - DRIP_PRICING (confidence: 0.85)
    - SNEAK_INTO_BASKET (confidence: 0.72)
[STEP 2] ← EXIT DOM_AUDITOR [0.789s]

======================================================================
PERFORMANCE SUMMARY
======================================================================
Total execution time: 12.345s

Node timings:
  NAV_ACTOR          :  8 calls,   9.234s total,  1.154s avg,  2.345s max
  DOM_AUDITOR        :  8 calls,   2.456s total,  0.307s avg,  0.567s max
  STATE_EVAL         :  8 calls,   0.123s total,  0.015s avg,  0.034s max
  PLAN_GENESIS       :  1 calls,   0.234s total,  0.234s avg,  0.234s max
======================================================================
```

## Limitations

- Some sites have bot protection that may cause timeouts
- Visual-only patterns (color, size manipulation) require additional CV work
- Complex JavaScript-heavy sites may need more steps to fully explore

## License

MIT
