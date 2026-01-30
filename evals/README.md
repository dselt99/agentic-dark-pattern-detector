# Evaluation Framework

This directory contains the evaluation framework for the Dark Pattern Hunter (Pillar 5).

## Structure

- `simulations/` - HTML test files with known dark patterns
- `run.py` - Main evaluation script
- `results/` - Output directory for evaluation results

## Test Simulations

### roach_motel.html
- **Pattern**: Roach Motel
- **Features**: 
  - Easy entry: Giant green "Subscribe" button
  - Hard exit: Cancel link hidden in accordion, styled as footer text
- **Expected**: Agent should detect ROACH_MOTEL due to Navigation Friction and Visual Interference

### false_urgency.html
- **Pattern**: False Urgency
- **Features**: 
  - Countdown timer that resets to 5:00 on every page reload
- **Expected**: Agent should detect FALSE_URGENCY due to Temporal Inconsistency

### clean_stock.html
- **Pattern**: None (Negative Control)
- **Features**: 
  - Legitimate stock counter that doesn't reset
- **Expected**: Agent should NOT flag this (false positive test)

## Running Evaluations

```bash
# Set environment variables
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here  # For judge LLM

# Run evaluation
python evals/run.py
```

The script will:
1. Start a local HTTP server on port 8000
2. Run the agent against each simulation
3. Use LLM-as-a-Judge to grade results
4. Calculate metrics (Precision, Recall, F1, Reasoning Score)
5. Save results to `evals/results/results.json`

## Metrics

- **Precision**: (True Positives) / (True Positives + False Positives)
  - Crucial for avoiding false accusations
- **Recall**: (True Positives) / (True Positives + False Negatives)
  - Crucial for safety (not missing real patterns)
- **F1 Score**: Harmonic mean of Precision and Recall
- **Reasoning Score**: 1-5 scale rating of explanation quality
