from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.simulation import SimulationRun


def detect_cascade(
    run: SimulationRun,
    drop_threshold: float = 0.4,
    window: int = 10,
) -> dict:
    """
    Sliding window over round_summaries avg_sentiment.
    If sentiment drops more than drop_threshold within any window of 10 consecutive
    rounds, cascade = True.

    Returns {cascade: bool, trigger_day: int | None}
    """
    summaries = run.round_summaries
    sentiments = [s["avg_policy_sentiment"] for s in summaries]

    for i in range(len(sentiments) - window + 1):
        window_start_val = sentiments[i]
        window_end_val = sentiments[i + window - 1]
        drop = window_start_val - window_end_val
        if drop >= drop_threshold:
            trigger_day = summaries[i]["round"]
            return {"cascade": True, "trigger_day": trigger_day}

    return {"cascade": False, "trigger_day": None}
