from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import yaml

from core.agent import Agent, Post, generate_post, initialize_agents, reset_post_id_counter, update_agent_state
from core.feed import sample_feed
from core.policy_types import POLICY_TYPES


@dataclass
class SimulationRun:
    run_id: str
    policy_type: str
    policy_variables: dict
    agents: list[Agent]
    posts: list[Post] = field(default_factory=list)
    round_summaries: list[dict] = field(default_factory=list)
    completed: bool = False


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


def run_simulation(
    policy_type: str,
    policy_variables: dict,
    personas_path: str = "config/personas.yaml",
) -> SimulationRun:
    """Orchestrates a full 45-round simulation."""
    reset_post_id_counter()

    with open(personas_path, "r") as f:
        personas_config = yaml.safe_load(f)

    agents = initialize_agents(personas_config)
    run_id = str(uuid.uuid4())[:8]

    sim = SimulationRun(
        run_id=run_id,
        policy_type=policy_type,
        policy_variables=policy_variables,
        agents=agents,
    )

    # --- Round 1: Announcement ---
    announcement = Post(
        id=0,
        agent_id=-1,
        round=1,
        sentiment=0.0,
        reach=999_999,
        persona="announcement",
    )
    sim.posts.append(announcement)

    # All agents see the announcement in round 1, no posting
    for agent in agents:
        update_agent_state(agent, [announcement], current_round=1)

    sim.round_summaries.append(record_round_summary(agents, round_num=1))

    # --- Rounds 2-45 ---
    for round_num in range(2, 46):
        post_pool = list(sim.posts)

        for agent in agents:
            posts_seen = sample_feed(agent, post_pool, round_num)
            update_agent_state(agent, posts_seen, current_round=round_num)
            new_post = generate_post(agent, round_num)
            if new_post:
                sim.posts.append(new_post)

        sim.round_summaries.append(record_round_summary(agents, round_num))

    sim.completed = True
    return sim
