from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

from core.agent import Agent, Post

FEED_SIZE = 10
REACH_SLOTS = 6
HOMOPHILY_SLOTS = 2
RECENCY_SLOTS = 1
RANDOM_SLOTS = 1

# A post's pull on the reach-weighted bucket decays each round. Without this,
# day-45 feeds were still dominated by day-2 posts and the simulation carried
# enormous inertia from opinions nobody held any more.
REACH_DECAY_PER_ROUND = 0.75


@dataclass
class FeedIndex:
    """Per-round precomputation shared across all 500 agents.

    `sample_feed` previously rebuilt several filtered lists per agent per round
    over a pool that grew to ~7,000 posts — roughly 375M operations per run.
    Everything that does not depend on the agent is hoisted here and computed
    once per round instead.
    """

    round_num: int
    posts: list[Post]
    by_persona: dict[str, list[Post]] = field(default_factory=dict)
    reach_cum_weights: list[float] = field(default_factory=list)
    recency_candidates: list[Post] = field(default_factory=list)
    recency_cum_weights: list[float] = field(default_factory=list)


def build_feed_index(post_pool: list[Post], round_num: int) -> FeedIndex:
    index = FeedIndex(round_num=round_num, posts=list(post_pool))

    for post in index.posts:
        index.by_persona.setdefault(post.persona, []).append(post)

    # Reach weights, aged so recent posts dominate.
    weights = []
    for post in index.posts:
        age = max(0, round_num - post.round)
        weights.append(max(post.reach * (REACH_DECAY_PER_ROUND ** age), 1e-9))
    index.reach_cum_weights = list(itertools.accumulate(weights))

    # Recency bucket: current round weighted 2:1 against the prior round.
    rec_weights = []
    for post in index.posts:
        if post.round == round_num:
            rec_weights.append(2.0)
        elif post.round == round_num - 1:
            rec_weights.append(1.0)
        else:
            continue
        index.recency_candidates.append(post)
    index.recency_cum_weights = list(itertools.accumulate(rec_weights))

    return index


def sample_feed(agent: Agent, post_pool: list[Post], round_num: int,
                index: FeedIndex | None = None) -> list[Post]:
    """
    Return up to 10 posts for this agent this round.
    If post_pool has fewer than 10 posts, return all available.

    Sampling breakdown (10 posts total):
      6 posts  - reach-weighted (probability proportional to aged post.reach)
      2 posts  - homophily (same persona as agent, uniform sample)
      1 post   - recency-weighted (current and prior round weighted 2:1)
      1 post   - random uniform

    No duplicates across the 4 buckets.
    If a bucket can't be filled, remaining slots go to the reach-weighted bucket.

    `index` is the per-round precomputation from `build_feed_index`. It is
    optional so the function stays usable standalone (and in tests); the
    simulation always passes one.
    """
    if len(post_pool) <= FEED_SIZE:
        return list(post_pool)

    if index is None or index.round_num != round_num:
        index = build_feed_index(post_pool, round_num)

    selected: set[int] = set()
    result: list[Post] = []

    def _add(post: Post) -> bool:
        if post.id in selected:
            return False
        selected.add(post.id)
        result.append(post)
        return True

    # --- Homophily (2 posts) ---
    homophily_picked = 0
    same_persona = index.by_persona.get(agent.persona)
    if same_persona:
        for post in random.sample(same_persona, min(len(same_persona), HOMOPHILY_SLOTS * 3)):
            if homophily_picked >= HOMOPHILY_SLOTS:
                break
            if _add(post):
                homophily_picked += 1

    # --- Recency-weighted (1 post) ---
    recency_picked = 0
    if index.recency_candidates:
        for post in random.choices(
            index.recency_candidates,
            cum_weights=index.recency_cum_weights,
            k=RECENCY_SLOTS * 3,
        ):
            if recency_picked >= RECENCY_SLOTS:
                break
            if _add(post):
                recency_picked += 1

    # --- Random uniform (1 post) ---
    random_picked = 0
    for post in random.sample(index.posts, min(len(index.posts), RANDOM_SLOTS * 4)):
        if random_picked >= RANDOM_SLOTS:
            break
        if _add(post):
            random_picked += 1

    # --- Reach-weighted, absorbing any unfilled slots ---
    reach_target = (
        REACH_SLOTS
        + (HOMOPHILY_SLOTS - homophily_picked)
        + (RECENCY_SLOTS - recency_picked)
        + (RANDOM_SLOTS - random_picked)
    )
    reach_picked = 0
    if reach_target > 0:
        for post in random.choices(
            index.posts, cum_weights=index.reach_cum_weights, k=reach_target * 4
        ):
            if reach_picked >= reach_target:
                break
            if _add(post):
                reach_picked += 1

        # Still short (heavy duplicate draws) — top up uniformly.
        if reach_picked < reach_target:
            for post in index.posts:
                if reach_picked >= reach_target:
                    break
                if _add(post):
                    reach_picked += 1

    return result[:FEED_SIZE]
