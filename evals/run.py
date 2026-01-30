"""Evaluation script for Dark Pattern Hunter.

This script implements Pillar 5: The Evaluation Framework, using
LLM-as-a-Judge to grade agent performance against ground truth test cases.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import DarkPatternAgent
from src.schemas import AuditResult


# Ground truth definitions for each simulation
GROUND_TRUTH = {
    "roach_motel.html": {
        "url": "http://localhost:8000/roach_motel.html",
        "patterns": ["roach_motel"],
        "description": (
            "A subscription page with a giant green 'Subscribe' button. "
            "The 'Cancel' button is hidden as a text link, styled to look like footer text, "
            "nested inside a 'Help' accordion. This is a Roach Motel pattern due to "
            "Navigation Friction (cancel requires opening accordion) and Visual Interference "
            "(cancel link styled as low-contrast footer text)."
        ),
    },
    "false_urgency.html": {
        "url": "http://localhost:8000/false_urgency.html",
        "patterns": ["false_urgency"],
        "description": (
            "An e-commerce product page with a countdown timer that resets to 5:00 "
            "every time the page is reloaded. This is a False Urgency pattern due to "
            "Temporal Inconsistency - the timer resets on page reload, indicating "
            "artificial urgency rather than a real deadline."
        ),
    },
    "clean_stock.html": {
        "url": "http://localhost:8000/clean_stock.html",
        "patterns": [],  # Negative control - no dark patterns
        "description": (
            "A product page with 'Only 3 items left' stock counter. "
            "The count is static and does not reset on reload, indicating it's tied to "
            "a legitimate inventory system. This is NOT a dark pattern - it's a negative control. "
            "If the agent flags this, it's a false positive."
        ),
    },
}


class SimulationServer:
    """Simple HTTP server to host simulation HTML files."""

    def __init__(self, port: int = 8000, directory: str = "evals/simulations"):
        self.port = port
        self.directory = Path(directory).absolute()
        self.server = None
        self.thread = None

    def start(self):
        """Start the HTTP server in a background thread."""
        handler = type(
            "Handler",
            (SimpleHTTPRequestHandler,),
            {
                "__init__": lambda self, *args, **kwargs: SimpleHTTPRequestHandler.__init__(
                    self, *args, directory=str(self.directory), **kwargs
                )
            },
        )

        self.server = HTTPServer(("localhost", self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        print(f"Started simulation server on http://localhost:{self.port}")

    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("Stopped simulation server")


class Judge:
    """LLM-as-a-Judge for evaluating agent performance."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider
        self.model = model
        self.client = self._init_client()

    def _init_client(self):
        """Initialize LLM client."""
        if self.provider == "openai":
            from openai import AsyncOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return AsyncOpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return AsyncAnthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def grade_audit(
        self, ground_truth: Dict, audit_result: AuditResult
    ) -> Dict:
        """Grade an audit result against ground truth.

        Args:
            ground_truth: Ground truth dictionary with patterns and description.
            audit_result: Agent's audit result.

        Returns:
            Dictionary with grading metrics.
        """
        expected_patterns = set(ground_truth["patterns"])
        found_patterns = set(
            f.pattern_type.value for f in audit_result.findings
        )

        # Build judge prompt
        prompt = self._build_judge_prompt(ground_truth, audit_result)

        # Call judge LLM
        try:
            if self.provider == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_judge_system_prompt(),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                json_str = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=self._get_judge_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                )
                json_str = response.content[0].text

            # Parse judge response
            try:
                grade = json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback to manual calculation if JSON parsing fails
                grade = self._calculate_metrics_manual(
                    expected_patterns, found_patterns, audit_result
                )

            return grade

        except Exception as e:
            print(f"Warning: Judge LLM call failed: {e}")
            # Fallback to manual calculation
            return self._calculate_metrics_manual(
                expected_patterns, found_patterns, audit_result
            )

    def _get_judge_system_prompt(self) -> str:
        """Get the system prompt for the judge."""
        return """You are an expert QA Auditor specializing in dark pattern detection. 
Your task is to evaluate an agent's audit report against known ground truth.

You will receive:
1. Ground Truth: What dark patterns actually exist on the page
2. Agent's Audit Report: What the agent found

Your task:
- Compare the findings
- Count True Positives (agent found pattern that exists)
- Count False Positives (agent found pattern that doesn't exist)
- Count False Negatives (agent missed a pattern that exists)
- Rate the agent's reasoning quality (1-5 scale)

Output strictly valid JSON matching this schema:
{
    "true_positives": int,
    "false_positives": int,
    "false_negatives": int,
    "reasoning_score": int (1-5),
    "critique": "string explanation"
}"""

    def _build_judge_prompt(
        self, ground_truth: Dict, audit_result: AuditResult
    ) -> str:
        """Build the prompt for the judge."""
        expected_str = ", ".join(ground_truth["patterns"]) or "None"
        found_str = (
            ", ".join(f.pattern_type.value for f in audit_result.findings)
            or "None"
        )

        findings_details = ""
        if audit_result.findings:
            for i, finding in enumerate(audit_result.findings, 1):
                findings_details += f"\n{i}. {finding.pattern_type.value}\n"
                findings_details += f"   Confidence: {finding.confidence_score}\n"
                findings_details += f"   Reasoning: {finding.reasoning}\n"
                if finding.evidence:
                    findings_details += f"   Evidence: {finding.evidence}\n"

        return f"""GROUND TRUTH:
Expected Patterns: {expected_str}
Description: {ground_truth['description']}

AGENT'S AUDIT REPORT:
URL: {audit_result.target_url}
Findings: {found_str}
Summary: {audit_result.summary}
{findings_details}

Please evaluate the agent's performance and output the JSON grading."""

    def _calculate_metrics_manual(
        self,
        expected_patterns: set,
        found_patterns: set,
        audit_result: AuditResult,
    ) -> Dict:
        """Manually calculate metrics as fallback."""
        true_positives = len(expected_patterns & found_patterns)
        false_positives = len(found_patterns - expected_patterns)
        false_negatives = len(expected_patterns - found_patterns)

        # Simple reasoning score based on whether reasoning exists
        reasoning_score = 3  # Default
        if audit_result.findings:
            avg_confidence = sum(
                f.confidence_score for f in audit_result.findings
            ) / len(audit_result.findings)
            reasoning_score = min(5, max(1, int(avg_confidence * 5)))

        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "reasoning_score": reasoning_score,
            "critique": "Manual calculation (Judge LLM unavailable)",
        }


async def run_evaluation():
    """Run the complete evaluation suite."""
    print("=" * 70)
    print("DARK PATTERN HUNTER - EVALUATION SUITE")
    print("=" * 70)
    print()

    # Start simulation server
    server = SimulationServer(port=8000, directory="evals/simulations")
    server.start()
    time.sleep(2)  # Give server time to start

    try:
        # Initialize agent
        print("Initializing agent...")
        model = os.getenv("LLM_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"))
        provider = os.getenv("LLM_PROVIDER", "anthropic")
        agent = DarkPatternAgent(
            model=model,
            provider=provider,
            max_steps=30,  # Reduced for faster evaluation
        )

        # Initialize judge
        print("Initializing judge...")
        judge_model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
        judge = Judge(provider="anthropic", model=judge_model)

        # Run evaluations
        results = []
        for sim_file, ground_truth in GROUND_TRUTH.items():
            print(f"\n{'=' * 70}")
            print(f"Evaluating: {sim_file}")
            print(f"Expected: {ground_truth['patterns'] or 'None'}")
            print(f"{'=' * 70}")

            url = ground_truth["url"]
            print(f"Running audit on {url}...")

            try:
                # Run agent audit
                audit_result = await agent.run_audit(url)

                print(f"Agent found: {len(audit_result.findings)} pattern(s)")
                if audit_result.findings:
                    for finding in audit_result.findings:
                        print(f"  - {finding.pattern_type.value} (confidence: {finding.confidence_score:.2f})")

                # Grade with judge
                print("Grading with LLM-as-a-Judge...")
                grade = await judge.grade_audit(ground_truth, audit_result)

                results.append(
                    {
                        "simulation": sim_file,
                        "ground_truth": ground_truth,
                        "audit_result": audit_result,
                        "grade": grade,
                    }
                )

                print(f"Grade: TP={grade['true_positives']}, "
                      f"FP={grade['false_positives']}, "
                      f"FN={grade['false_negatives']}, "
                      f"Reasoning={grade['reasoning_score']}/5")

            except Exception as e:
                print(f"Error evaluating {sim_file}: {e}")
                import traceback
                traceback.print_exc()

        # Calculate aggregate metrics
        print(f"\n{'=' * 70}")
        print("AGGREGATE RESULTS")
        print(f"{'=' * 70}")

        total_tp = sum(r["grade"]["true_positives"] for r in results)
        total_fp = sum(r["grade"]["false_positives"] for r in results)
        total_fn = sum(r["grade"]["false_negatives"] for r in results)
        avg_reasoning = sum(r["grade"]["reasoning_score"] for r in results) / len(results)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\nMetrics:")
        print(f"  True Positives:  {total_tp}")
        print(f"  False Positives: {total_fp}")
        print(f"  False Negatives: {total_fn}")
        print(f"  Precision:       {precision:.3f} (crucial for avoiding false accusations)")
        print(f"  Recall:          {recall:.3f} (crucial for safety)")
        print(f"  F1 Score:        {f1_score:.3f}")
        print(f"  Avg Reasoning:   {avg_reasoning:.2f}/5")

        # Detailed results table
        print(f"\n{'=' * 70}")
        print("DETAILED RESULTS")
        print(f"{'=' * 70}")
        print(f"{'Simulation':<25} {'TP':<5} {'FP':<5} {'FN':<5} {'Reasoning':<10}")
        print("-" * 70)
        for r in results:
            grade = r["grade"]
            print(
                f"{r['simulation']:<25} "
                f"{grade['true_positives']:<5} "
                f"{grade['false_positives']:<5} "
                f"{grade['false_negatives']:<5} "
                f"{grade['reasoning_score']}/5"
            )

        # Save results to file
        output_file = Path("evals/results.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(
                {
                    "aggregate": {
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "avg_reasoning": avg_reasoning,
                        "total_tp": total_tp,
                        "total_fp": total_fp,
                        "total_fn": total_fn,
                    },
                    "detailed": [
                        {
                            "simulation": r["simulation"],
                            "grade": r["grade"],
                            "summary": r["audit_result"].summary,
                        }
                        for r in results
                    ],
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nResults saved to {output_file}")

    finally:
        server.stop()


if __name__ == "__main__":
    asyncio.run(run_evaluation())
