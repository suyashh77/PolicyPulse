from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from report.cascade import detect_cascade
from report.curves import (
    aggregate_churn_by_segment,
    aggregate_sentiment_curve,
    build_churn_by_segment,
    build_sentiment_curve,
)

if TYPE_CHECKING:
    from core.simulation import SimulationRun


def _variables_key(variables: dict) -> tuple:
    return tuple(sorted(variables.items()))


def generate_report(run: SimulationRun, all_runs: list[SimulationRun]) -> dict:
    """
    Assembles full report dict from simulation data.
    Does not call an LLM - just assembles structured data.

    `threshold_comparison` groups runs of the same policy type by their variable
    settings and reports the mean and spread across each group's seeds. It used
    to emit one row per run, which compared random seeds against each other
    rather than comparing policies.
    """
    sentiment_curve = build_sentiment_curve(run)
    churn_by_segment = build_churn_by_segment(run)
    cascade = detect_cascade(run)

    # Group same-policy runs by variable setting.
    groups: dict[tuple, list[SimulationRun]] = {}
    for r in all_runs:
        if r.policy_type != run.policy_type:
            continue
        groups.setdefault(_variables_key(r.policy_variables), []).append(r)

    threshold_comparison = []
    for key, group in groups.items():
        finals = [r.round_summaries[-1]["avg_policy_sentiment"] for r in group]
        churns = [
            statistics.fmean(build_churn_by_segment(r).values()) for r in group
        ]
        cascades = [detect_cascade(r) for r in group]
        cascade_days = [c["trigger_day"] for c in cascades if c["trigger_day"] is not None]

        threshold_comparison.append(
            {
                "variables": dict(key),
                "n_runs": len(group),
                "final_sentiment": round(statistics.fmean(finals), 4),
                "final_sentiment_stdev": round(
                    statistics.stdev(finals) if len(finals) > 1 else 0.0, 4
                ),
                "avg_churn": round(statistics.fmean(churns), 4),
                "avg_churn_stdev": round(
                    statistics.stdev(churns) if len(churns) > 1 else 0.0, 4
                ),
                "cascade_rate": round(
                    sum(1 for c in cascades if c["cascade"]) / len(cascades), 3
                ),
                "median_cascade_day": (
                    statistics.median(cascade_days) if cascade_days else None
                ),
            }
        )

    threshold_comparison.sort(key=lambda row: row["final_sentiment"])

    # Runs sharing this run's exact configuration — the batch it belongs to.
    sibling_runs = groups.get(_variables_key(run.policy_variables), [run])

    return {
        "run_id": run.run_id,
        "policy_type": run.policy_type,
        "policy_variables": run.policy_variables,
        "seed": run.seed,
        "persona_shocks": run.persona_shocks,
        "sentiment_curve": sentiment_curve,
        "churn_by_segment": churn_by_segment,
        "cascade": cascade,
        "threshold_comparison": threshold_comparison,
        "aggregate_sentiment_curve": aggregate_sentiment_curve(sibling_runs),
        "aggregate_churn_by_segment": aggregate_churn_by_segment(sibling_runs),
        "n_sibling_runs": len(sibling_runs),
    }
