from __future__ import annotations

import random

from core.agent import Agent, Post


def sample_feed(agent: Agent, post_pool: list[Post], round_num: int) -> list[Post]:
    """
    Return exactly 10 posts for this agent this round.
    If post_pool has fewer than 10 posts, return all available.

    Sampling breakdown (10 posts total):
      6 posts  - reach-weighted (probability proportional to post.reach)
      2 posts  - homophily (same persona as agent, uniform sample)
      1 post   - recency-weighted (current and prior round weighted 2:1)
      1 post   - random uniform

    No duplicates across the 4 buckets.
    If a bucket can't be filled, redistribute remaining slots to reach-weighted.
    """
    if len(post_pool) <= 10:
        return list(post_pool)

    selected: set[int] = set()  # post ids already picked
    result: list[Post] = []

    def _add(post: Post) -> bool:
        if post.id in selected:
            return False
        selected.add(post.id)
        result.append(post)
        return True

    # --- Bucket 2: Homophily (2 posts) ---
    homophily_target = 2
    homophily_posts = [p for p in post_pool if p.persona == agent.persona]
    homophily_picked = 0
    if homophily_posts:
        random.shuffle(homophily_posts)
        for p in homophily_posts:
            if homophily_picked >= homophily_target:
                break
            if _add(p):
                homophily_picked += 1

    # --- Bucket 3: Recency-weighted (1 post) ---
    recency_target = 1
    recency_picked = 0
    current_round_posts = [p for p in post_pool if p.id not in selected and p.round == round_num]
    prior_round_posts = [p for p in post_pool if p.id not in selected and p.round == round_num - 1]

    recency_candidates: list[Post] = []
    recency_weights: list[float] = []
    for p in current_round_posts:
        recency_candidates.append(p)
        recency_weights.append(2.0)
    for p in prior_round_posts:
        recency_candidates.append(p)
        recency_weights.append(1.0)

    if recency_candidates and recency_weights:
        picks = random.choices(recency_candidates, weights=recency_weights, k=1)
        for p in picks:
            if recency_picked >= recency_target:
                break
            if _add(p):
                recency_picked += 1

    # --- Bucket 4: Random uniform (1 post) ---
    random_target = 1
    random_picked = 0
    available = [p for p in post_pool if p.id not in selected]
    if available:
        random.shuffle(available)
        for p in available:
            if random_picked >= random_target:
                break
            if _add(p):
                random_picked += 1

    # --- Redistribute unfilled slots to reach-weighted ---
    reach_target = 6 + (homophily_target - homophily_picked) + (recency_target - recency_picked) + (random_target - random_picked)

    # --- Bucket 1: Reach-weighted (6+ posts) ---
    available = [p for p in post_pool if p.id not in selected]
    if available and reach_target > 0:
        reaches = [float(p.reach) for p in available]
        total_reach = sum(reaches)
        if total_reach > 0:
            weights = [r / total_reach for r in reaches]
        else:
            weights = [1.0 / len(available)] * len(available)

        # Sample with replacement then deduplicate
        picks = random.choices(available, weights=weights, k=reach_target * 3)
        reach_picked = 0
        for p in picks:
            if reach_picked >= reach_target:
                break
            if _add(p):
                reach_picked += 1

        # If still short, fill from remaining
        if reach_picked < reach_target:
            remaining = [p for p in post_pool if p.id not in selected]
            random.shuffle(remaining)
            for p in remaining:
                if reach_picked >= reach_target:
                    break
                if _add(p):
                    reach_picked += 1

    return result[:10]
