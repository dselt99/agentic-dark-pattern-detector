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

```bash
# Test against local simulation (false urgency pattern)
python quick_demo.py

# Test against other simulations
python quick_demo.py roach_motel
python quick_demo.py clean_stock

# Test against a live website
python quick_demo.py https://www.booking.com
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
└── demo.py                  # Full demo with all features
```

## Limitations

- Static analysis only - cannot detect patterns hidden behind interactions
- Does not verify dynamic behavior (e.g., timer reset on reload)
- Some sites have bot protection that may cause timeouts
- Visual-only patterns (color, size manipulation) not yet detected

## License

MIT
