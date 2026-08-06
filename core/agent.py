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
    # Private reaction to the policy itself, set from core.policy_impact at
    # round 1. Sentiment is pulled back toward this as peer influence fades —
    # without it the population has no restoring force and saturates at +/-1.
    baseline_sentiment: float = 0.0
    anchor_strength: float = 0.0
    churn_recovery: float = 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def initialize_agents(personas_config: dict, shocks: dict[str, float] | None = None) -> list[Agent]:
    """Create 500 agents from persona config.

    `shocks` maps persona name to that persona's day-1 policy reaction (see
    core.policy_impact). Agents start at their persona's shock rather than at
    zero, which is what makes the simulation respond to its policy input.
    """
    agents: list[Agent] = []
    agent_id = 0
    personas = personas_config["personas"]
    total = 500
    shocks = shocks or {}

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
        baseline = shocks.get(name, 0.0)
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
                    policy_sentiment=baseline,
                    baseline_sentiment=baseline,
                    anchor_strength=cfg.get("anchor_strength", 0.0),
                    churn_recovery=cfg.get("churn_recovery", 0.0),
                )
            )
            agent_id += 1

    return agents


def update_agent_state(agent: Agent, posts_seen: list[Post], current_round: int) -> Agent:
    """Run one round of state update for a single agent.

    Sentiment moves *toward* the feed signal rather than accumulating it. The
    additive form (`sentiment += signal x susceptibility`) has no fixed point:
    a persistently positive feed marches an agent to +1 and pins it there, which
    is why every run used to terminate at the clamp. Moving a fraction of the
    remaining distance converges instead, so the population settles where the
    argument actually lands.

    Churn tracks the sentiment *level* and can fall again. Integrating only
    downward deltas made it a one-way ratchet that accumulated from noise, so a
    population could end up simultaneously delighted and increasingly likely to
    leave. Recovery is slower than escalation — people forgive gradually.
    """
    if posts_seen:
        total_reach = sum(p.reach for p in posts_seen)
        if total_reach > 0:
            feed_signal = sum(p.sentiment * p.reach for p in posts_seen) / total_reach

            social_pull = (feed_signal - agent.policy_sentiment) * agent.susceptibility
            anchor_pull = (agent.baseline_sentiment - agent.policy_sentiment) * agent.anchor_strength

            agent.policy_sentiment = _clamp(
                agent.policy_sentiment + social_pull + anchor_pull, -1.0, 1.0
            )

    # Churn is a function of how negative the agent currently feels.
    target_churn = max(0.0, -agent.policy_sentiment) * agent.churn_elasticity
    if target_churn > agent.churn_intent:
        rate = agent.churn_elasticity          # escalates at the persona's own elasticity
    else:
        rate = agent.churn_recovery            # decays more slowly
    agent.churn_intent = _clamp(
        agent.churn_intent + (target_churn - agent.churn_intent) * rate, 0.0, 1.0
    )

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
