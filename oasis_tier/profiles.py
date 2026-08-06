"""personas.yaml -> OASIS agent profiles and a follower graph.

The persona library is PolicyPulse's real contribution on top of a generic
social simulator, so it has to survive the port to Tier 2 intact. This module
translates each persona's numeric parameters into two things an LLM agent can
actually use:

  1. A natural-language bio. `susceptibility: 0.80` means nothing to a language
     model; "you're swayed by what other shoppers say" does.
  2. A position in a follower graph, so reach and homophily become real network
     structure instead of a sampling weight.

Tier 1 approximates homophily by matching persona strings. Here it is edges.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# Bio fragments per persona. These are the qualitative translation of the
# numeric parameters and are the main lever on Tier-2 agent behaviour.
PERSONA_BIOS = {
    "loyal": (
        "You have shopped with this brand for years and genuinely like it. "
        "You give them the benefit of the doubt and are slow to be swayed by "
        "online outrage. You rarely post, and when you do it is measured."
    ),
    "casual": (
        "You shop here occasionally, without strong feelings either way. "
        "You notice what other people are saying and it does move your opinion. "
        "You post now and then."
    ),
    "deal_seeker": (
        "You are highly price-sensitive and shop across many brands looking for "
        "value. Your orders are usually small, so any added fee feels like a large "
        "share of what you spent. You are quick to react, quick to compare against "
        "competitors, and quick to post about it."
    ),
    "influencer": (
        "You have a large following who look to you for opinions on retail brands. "
        "You post every day. You are measured and deliberate because your audience "
        "is watching, but your take spreads widely."
    ),
    "sustainability": (
        "You care about waste and the environmental cost of retail. You judge "
        "policies on whether they reduce or create waste, not only on price. "
        "You are sceptical of 'just keep it' offers because they encourage "
        "throwaway consumption."
    ),
}

BRAND_USER_NAME = "brand_official"


@dataclass
class AgentProfile:
    """One simulated consumer, ready to become an OASIS SocialAgent."""

    agent_id: int
    user_name: str
    name: str
    persona: str
    bio: str
    reach: int
    activation: float           # per-step probability this agent is asked to act
    baseline_sentiment: float   # the Tier-1 policy shock, carried across
    follows: list[int] = field(default_factory=list)

    def to_user_info_kwargs(self) -> dict:
        """Arguments for oasis.UserInfo."""
        return {
            "user_name": self.user_name,
            "name": self.name,
            "description": self.bio,
            "profile": {
                "other_info": {
                    "user_profile": self.bio,
                    "persona": self.persona,
                    "followers_count": self.reach,
                }
            },
        }


def _stance_sentence(shock: float) -> str:
    """Turn the numeric day-1 shock into an initial stance the agent can read."""
    if shock <= -0.45:
        return "Your first reaction to this policy was strongly negative."
    if shock <= -0.15:
        return "Your first reaction to this policy was fairly negative."
    if shock < -0.02:
        return "Your first reaction to this policy was mildly negative."
    if shock < 0.02:
        return "You feel neutral about this policy so far."
    if shock < 0.25:
        return "Your first reaction to this policy was mildly positive."
    return "Your first reaction to this policy was clearly positive."


def build_profiles(
    personas_config: dict,
    shocks: dict[str, float],
    n_agents: int = 60,
    rng: random.Random | None = None,
) -> list[AgentProfile]:
    """Create `n_agents` profiles with the persona mix from personas_config.

    `n_agents` is much smaller than Tier 1's 500 because every agent here costs
    LLM calls. The mix proportions are preserved so the two tiers stay
    comparable.
    """
    rng = rng or random.Random(0)
    personas = personas_config["personas"]

    # Allocate counts by count_fraction, last persona absorbs the remainder.
    names = list(personas)
    counts: dict[str, int] = {}
    assigned = 0
    for i, name in enumerate(names):
        if i == len(names) - 1:
            counts[name] = max(1, n_agents - assigned)
        else:
            c = max(1, round(personas[name]["count_fraction"] * n_agents))
            counts[name] = c
            assigned += c

    profiles: list[AgentProfile] = []
    agent_id = 0
    for persona in names:
        cfg = personas[persona]
        lo, hi = cfg["reach_range"]
        shock = shocks.get(persona, 0.0)
        bio_core = PERSONA_BIOS.get(persona, "")
        for k in range(counts[persona]):
            bio = f"{bio_core} {_stance_sentence(shock)}"
            profiles.append(
                AgentProfile(
                    agent_id=agent_id,
                    user_name=f"{persona}_{k}",
                    name=f"{persona.replace('_', ' ').title()} {k}",
                    persona=persona,
                    bio=bio,
                    reach=rng.randint(lo, hi),
                    activation=cfg["post_probability"],
                    baseline_sentiment=shock,
                )
            )
            agent_id += 1

    _wire_follower_graph(profiles, rng)
    return profiles


def _wire_follower_graph(
    profiles: list[AgentProfile],
    rng: random.Random,
    homophily_edges: int = 3,
    random_edges: int = 2,
) -> None:
    """Give every agent a follow list.

    Three structures, matching the mechanisms Tier 1 samples for:
      - everyone follows the high-reach accounts (reach bias)
      - everyone follows a few same-persona accounts (homophily / echo chambers)
      - everyone follows a couple of random accounts (cross-cohort exposure)
    """
    by_persona: dict[str, list[AgentProfile]] = {}
    for p in profiles:
        by_persona.setdefault(p.persona, []).append(p)

    # Hubs: the top decile by reach.
    ranked = sorted(profiles, key=lambda p: p.reach, reverse=True)
    hubs = ranked[: max(1, len(ranked) // 10)]
    hub_ids = {h.agent_id for h in hubs}

    for p in profiles:
        follows: set[int] = set()

        for h in hubs:
            if h.agent_id != p.agent_id:
                follows.add(h.agent_id)

        peers = [q for q in by_persona[p.persona] if q.agent_id != p.agent_id]
        for q in rng.sample(peers, min(homophily_edges, len(peers))):
            follows.add(q.agent_id)

        others = [q for q in profiles if q.agent_id != p.agent_id and q.agent_id not in hub_ids]
        for q in rng.sample(others, min(random_edges, len(others))):
            follows.add(q.agent_id)

        p.follows = sorted(follows)


def graph_stats(profiles: list[AgentProfile]) -> dict:
    """Summary of the wired graph, for the run manifest."""
    edges = sum(len(p.follows) for p in profiles)
    followers: dict[int, int] = {p.agent_id: 0 for p in profiles}
    for p in profiles:
        for t in p.follows:
            followers[t] = followers.get(t, 0) + 1
    counts = sorted(followers.values(), reverse=True)
    return {
        "n_agents": len(profiles),
        "n_edges": edges,
        "avg_following": round(edges / len(profiles), 2) if profiles else 0,
        "max_followers": counts[0] if counts else 0,
        "persona_mix": {
            persona: sum(1 for p in profiles if p.persona == persona)
            for persona in {p.persona for p in profiles}
        },
    }
