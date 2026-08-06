from __future__ import annotations

import random
import statistics
import uuid
from dataclasses import dataclass, field

import yaml

from core.agent import (
    Agent,
    Post,
    generate_post,
    initialize_agents,
    reset_post_id_counter,
    update_agent_state,
)
from core.feed import build_feed_index, sample_feed
from core.policy_impact import policy_shocks_by_persona, population_shock
from core.policy_types import POLICY_TYPES

ROUNDS = 45
AGENT_COUNT = 500

# Posts older than this drop out of the pool entirely. Real feeds have a
# half-life; without one the pool grew to ~7,000 posts and day-45 agents were
# still reading day-2 opinions.
POST_TTL_ROUNDS = 12

# The announcement is a single pinned post, not a broadcast tower. Its old
# reach of 999,999 was 10x the largest influencer, so it never left the
# reach-weighted bucket and acted as a permanent anchor on every feed.
ANNOUNCEMENT_REACH = 50_000


@dataclass
class SimulationRun:
    run_id: str
    policy_type: str
    policy_variables: dict
    agents: list[Agent]
    posts: list[Post] = field(default_factory=list)
    round_summaries: list[dict] = field(default_factory=list)
    completed: bool = False
    seed: int | None = None
    persona_shocks: dict[str, float] = field(default_factory=dict)


def load_personas(personas_path: str = "config/personas.yaml") -> dict:
    with open(personas_path, "r") as f:
        return yaml.safe_load(f)


def record_round_summary(agents: list[Agent], round_num: int) -> dict:
    """Returns per-round summary with overall and per-persona breakdown."""
    n = len(agents)
    avg_sentiment = sum(a.policy_sentiment for a in agents) / n
    avg_churn = sum(a.churn_intent for a in agents) / n

    breakdown: dict[str, dict] = {}
    persona_groups: dict[str, list[Agent]] = {}
    for a in agents:
        persona_groups.setdefault(a.persona, []).append(a)

    for persona, group in persona_groups.items():
        gn = len(group)
        breakdown[persona] = {
            "avg_sentiment": sum(a.policy_sentiment for a in group) / gn,
            "avg_churn": sum(a.churn_intent for a in group) / gn,
        }

    return {
        "round": round_num,
        "avg_policy_sentiment": avg_sentiment,
        "avg_churn_intent": avg_churn,
        "breakdown_by_persona": breakdown,
    }


def announcement_text(policy_type: str, policy_variables: dict, date: str = "recently") -> str:
    return POLICY_TYPES[policy_type]["announcement_template"].format(
        date=date, **policy_variables
    )


def run_simulation(
    policy_type: str,
    policy_variables: dict,
    personas_path: str = "config/personas.yaml",
    seed: int | None = None,
    personas_config: dict | None = None,
) -> SimulationRun:
    """Orchestrates a full 45-round simulation.

    `seed` makes a run reproducible. The UI previously called `random.seed()`
    with no argument, so no run could ever be repeated or compared.
    """
    if seed is not None:
        random.seed(seed)

    reset_post_id_counter()

    if personas_config is None:
        personas_config = load_personas(personas_path)

    # The policy enters the simulation here and nowhere else.
    shocks = policy_shocks_by_persona(policy_type, policy_variables, personas_config)

    agents = initialize_agents(personas_config, shocks)
    run_id = str(uuid.uuid4())[:8]

    sim = SimulationRun(
        run_id=run_id,
        policy_type=policy_type,
        policy_variables=policy_variables,
        agents=agents,
        seed=seed,
        persona_shocks=shocks,
    )

    # --- Round 1: Announcement ---
    # Its sentiment is the population-weighted shock: the news itself reads as
    # bad or good in proportion to how the customer base receives it.
    announcement = Post(
        id=0,
        agent_id=-1,
        round=1,
        sentiment=population_shock(shocks, personas_config),
        reach=ANNOUNCEMENT_REACH,
        persona="announcement",
    )
    sim.posts.append(announcement)

    # Agents already hold their own shock as of initialization; round 1 records
    # that state without further social influence.
    for agent in agents:
        update_agent_state(agent, [], current_round=1)

    sim.round_summaries.append(record_round_summary(agents, round_num=1))

    live_posts: list[Post] = [announcement]

    # --- Rounds 2-45 ---
    for round_num in range(2, ROUNDS + 1):
        # Snapshot: posts written this round are not visible until the next one.
        post_pool = [p for p in live_posts if round_num - p.round <= POST_TTL_ROUNDS]
        index = build_feed_index(post_pool, round_num)

        new_posts: list[Post] = []
        for agent in agents:
            posts_seen = sample_feed(agent, post_pool, round_num, index=index)
            update_agent_state(agent, posts_seen, current_round=round_num)
            new_post = generate_post(agent, round_num)
            if new_post:
                new_posts.append(new_post)

        sim.posts.extend(new_posts)
        live_posts = post_pool + new_posts

        sim.round_summaries.append(record_round_summary(agents, round_num))

    sim.completed = True
    return sim


def run_batch(
    policy_type: str,
    policy_variables: dict,
    seeds: list[int],
    personas_path: str = "config/personas.yaml",
    personas_config: dict | None = None,
) -> list[SimulationRun]:
    """Run the same policy under several seeds.

    A single run is one draw from a stochastic process, not a result. Reporting
    should aggregate across seeds — see report.curves.aggregate_sentiment_curve.
    """
    if personas_config is None:
        personas_config = load_personas(personas_path)
    return [
        run_simulation(
            policy_type,
            policy_variables,
            seed=seed,
            personas_config=personas_config,
        )
        for seed in seeds
    ]


def summarize_batch(runs: list[SimulationRun]) -> dict:
    """Mean and spread of the day-45 outcome across a batch."""
    finals = [r.round_summaries[-1]["avg_policy_sentiment"] for r in runs]
    churns = [r.round_summaries[-1]["avg_churn_intent"] for r in runs]
    return {
        "n_runs": len(runs),
        "final_sentiment_mean": statistics.fmean(finals),
        "final_sentiment_stdev": statistics.stdev(finals) if len(finals) > 1 else 0.0,
        "final_churn_mean": statistics.fmean(churns),
        "final_churn_stdev": statistics.stdev(churns) if len(churns) > 1 else 0.0,
    }
