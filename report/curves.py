from __future__ import annotations

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
