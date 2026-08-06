from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Post:
    id: int
    agent_id: int
    round: int
    sentiment: float
    reach: int
    persona: str


@dataclass
class Agent:
    id: int
    persona: str
    policy_sentiment: float = 0.0
    churn_intent: float = 0.0
    reach: int = 0
    memory: list[dict] = field(default_factory=list)
    susceptibility: float = 0.0
    churn_elasticity: float = 0.0
    post_probability: float = 0.0
    post_variance: float = 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def initialize_agents(personas_config: dict) -> list[Agent]:
    """Create 500 agents from persona config."""
    agents: list[Agent] = []
    agent_id = 0
    personas = personas_config["personas"]
    total = 500

    counts: dict[str, int] = {}
    assigned = 0
    persona_names = list(personas.keys())

    for i, name in enumerate(persona_names):
        cfg = personas[name]
        if i == len(persona_names) - 1:
            counts[name] = total - assigned
        else:
            c = round(cfg["count_fraction"] * total)
            counts[name] = c
            assigned += c

    for name in persona_names:
        cfg = personas[name]
        lo, hi = cfg["reach_range"]
        for _ in range(counts[name]):
            agents.append(
                Agent(
                    id=agent_id,
                    persona=name,
                    reach=random.randint(lo, hi),
                    susceptibility=cfg["susceptibility"],
                    churn_elasticity=cfg["churn_elasticity"],
                    post_probability=cfg["post_probability"],
                    post_variance=cfg["post_variance"],
                )
            )
            agent_id += 1

    return agents


def update_agent_state(agent: Agent, posts_seen: list[Post], current_round: int) -> Agent:
    """Run one round of state update for a single agent."""
    if not posts_seen:
        return agent

    total_reach = sum(p.reach for p in posts_seen)
    if total_reach == 0:
        return agent

    weighted_feed_signal = sum(p.sentiment * p.reach for p in posts_seen) / total_reach

    prev_sentiment = agent.policy_sentiment
    agent.policy_sentiment += weighted_feed_signal * agent.susceptibility
    agent.policy_sentiment = _clamp(agent.policy_sentiment, -1.0, 1.0)

    sentiment_delta = agent.policy_sentiment - prev_sentiment
    agent.churn_intent += max(0, -sentiment_delta) * agent.churn_elasticity
    agent.churn_intent = _clamp(agent.churn_intent, 0.0, 1.0)

    agent.memory.append(
        {
            "round": current_round,
            "posts_seen_ids": [p.id for p in posts_seen],
            "sentiment": agent.policy_sentiment,
            "churn_intent": agent.churn_intent,
        }
    )

    return agent


_post_id_counter = 0


def _next_post_id() -> int:
    global _post_id_counter
    _post_id_counter += 1
    return _post_id_counter


def reset_post_id_counter() -> None:
    global _post_id_counter
    _post_id_counter = 0


def generate_post(agent: Agent, current_round: int) -> Post | None:
    """Roll post_probability. If posting, generate a post with noisy sentiment."""
    if random.random() > agent.post_probability:
        return None

    sentiment = _clamp(
        agent.policy_sentiment + random.gauss(0, agent.post_variance),
        -1.0,
        1.0,
    )

    return Post(
        id=_next_post_id(),
        agent_id=agent.id,
        round=current_round,
        sentiment=sentiment,
        reach=agent.reach,
        persona=agent.persona,
    )
