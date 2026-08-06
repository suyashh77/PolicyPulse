"""Save and reload completed runs.

Runs previously lived only in `st.session_state`, so a browser refresh destroyed
all history — including the threshold-comparison table, which needs several runs
to say anything.

Only the summary is persisted, not the 500 agents and their per-round memory
logs. Agent memory is what the interview feature reads, so a reloaded run
supports charts and comparison but not interviews.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.simulation import SimulationRun

DEFAULT_RUNS_DIR = Path("runs")


def run_to_dict(run: SimulationRun) -> dict:
    return {
        "run_id": run.run_id,
        "policy_type": run.policy_type,
        "policy_variables": run.policy_variables,
        "seed": run.seed,
        "persona_shocks": run.persona_shocks,
        "round_summaries": run.round_summaries,
        "completed": run.completed,
        "n_posts": len(run.posts),
        "n_agents": len(run.agents),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def save_batch(runs: list[SimulationRun], runs_dir: Path | str = DEFAULT_RUNS_DIR) -> Path:
    """Write one batch (all seeds for a single policy configuration) to disk."""
    if not runs:
        raise ValueError("cannot save an empty batch")

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    head = runs[0]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = runs_dir / f"{stamp}_{head.policy_type}_{head.run_id}.json"

    payload = {
        "policy_type": head.policy_type,
        "policy_variables": head.policy_variables,
        "runs": [run_to_dict(r) for r in runs],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_batches(runs_dir: Path | str = DEFAULT_RUNS_DIR) -> list[dict]:
    """Load every saved batch, newest first. Malformed files are skipped."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []

    batches = []
    for path in sorted(runs_dir.glob("*.json"), reverse=True):
        try:
            batches.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return batches
