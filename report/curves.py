from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.simulation import SimulationRun


def build_sentiment_curve(run: SimulationRun) -> list[dict]:
    """Returns list of {day, avg_sentiment} for days 1-45."""
    return [
        {"day": s["round"], "avg_sentiment": s["avg_policy_sentiment"]}
        for s in run.round_summaries
    ]


def build_churn_by_segment(run: SimulationRun) -> dict[str, float]:
    """Returns {persona: avg_churn_intent} at round 45."""
    if not run.round_summaries:
        return {}
    final = run.round_summaries[-1]
    return {
        persona: data["avg_churn"]
        for persona, data in final["breakdown_by_persona"].items()
    }


def aggregate_sentiment_curve(runs: list[SimulationRun]) -> list[dict]:
    """Mean sentiment per day across a batch, with a +/-1 stdev band.

    A single run is one draw from a stochastic process. Charting it as a lone
    line invites reading noise as signal, so the UI plots this instead.
    """
    if not runs:
        return []

    n_days = min(len(r.round_summaries) for r in runs)
    curve = []
    for i in range(n_days):
        values = [r.round_summaries[i]["avg_policy_sentiment"] for r in runs]
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        curve.append(
            {
                "day": runs[0].round_summaries[i]["round"],
                "avg_sentiment": mean,
                "stdev": stdev,
                "lower": mean - stdev,
                "upper": mean + stdev,
                "n_runs": len(values),
            }
        )
    return curve


def aggregate_churn_by_segment(runs: list[SimulationRun]) -> dict[str, dict]:
    """Mean and stdev of day-45 churn per persona across a batch."""
    if not runs:
        return {}

    per_persona: dict[str, list[float]] = {}
    for run in runs:
        for persona, value in build_churn_by_segment(run).items():
            per_persona.setdefault(persona, []).append(value)

    return {
        persona: {
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for persona, values in per_persona.items()
    }
