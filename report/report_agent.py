from __future__ import annotations

from typing import TYPE_CHECKING

from report.cascade import detect_cascade
from report.curves import build_churn_by_segment, build_sentiment_curve

if TYPE_CHECKING:
    from core.simulation import SimulationRun


def generate_report(run: SimulationRun, all_runs: list[SimulationRun]) -> dict:
    """
    Assembles full report dict from simulation data.
    Does not call LLM - just assembles structured data.
    """
    sentiment_curve = build_sentiment_curve(run)
    churn_by_segment = build_churn_by_segment(run)
    cascade = detect_cascade(run)

    # Threshold comparison: one entry per run of the same policy type
    threshold_comparison = []
    for r in all_runs:
        if r.policy_type != run.policy_type:
            continue
        r_curve = build_sentiment_curve(r)
        r_churn = build_churn_by_segment(r)
        r_cascade = detect_cascade(r)
        final_sentiment = r_curve[-1]["avg_sentiment"] if r_curve else 0.0
        avg_churn = sum(r_churn.values()) / len(r_churn) if r_churn else 0.0

        threshold_comparison.append(
            {
                "run_id": r.run_id,
                "variables": r.policy_variables,
                "final_sentiment": round(final_sentiment, 4),
                "avg_churn": round(avg_churn, 4),
                "cascade": r_cascade["cascade"],
                "cascade_day": r_cascade["trigger_day"],
            }
        )

    return {
        "run_id": run.run_id,
        "policy_type": run.policy_type,
        "policy_variables": run.policy_variables,
        "sentiment_curve": sentiment_curve,
        "churn_by_segment": churn_by_segment,
        "cascade": cascade,
        "threshold_comparison": threshold_comparison,
    }
