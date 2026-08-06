"""Cross-tier validation: does the cheap numeric model agree with LLM agents?

This is the reason both tiers exist. Tier 1 can be swept thousands of times for
free, but only if its abstraction is trustworthy. Tier 2 is expensive but reads
the actual announcement and produces real language. Running the same policy
through both, and quantifying the agreement, is what licenses using Tier 1 for
everything else.

**The comparison is not like-for-like, and pretending otherwise would be the
whole error.** Tier 1's sentiment is an agent's private internal state, averaged
over all 500 agents including the silent ones. Tier 2's is the sentiment of
posts that were actually written. People post when they feel strongly, so
Tier 2's population is selection-biased toward the intense end. Absolute levels
should *not* match, and a validator that demanded they match would fail a
correct model.

What should hold is structure:

  - **Sign agreement** - both tiers should call the policy negative or positive.
  - **Rank agreement** - both should order the personas the same way
    (deal_seeker angrier than loyal).
  - **Correlation** - the per-persona pattern should move together.

Rank and sign are the load-bearing checks. Level agreement is reported for
information, not used as a pass criterion.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation. Ties get average ranks."""
    if len(xs) < 2:
        return None

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    return _pearson(rank(xs), rank(ys))


def _pairwise_rank_agreement(xs: list[float], ys: list[float]) -> float | None:
    """Fraction of persona pairs ordered the same way by both tiers."""
    n = len(xs)
    if n < 2:
        return None
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            total += 1
            if (dx > 0) == (dy > 0):
                agree += 1
    return agree / total if total else None


@dataclass
class TierComparison:
    personas: list[str]
    tier1: list[float]
    tier2: list[float]
    n_posts: list[int]

    def to_dict(self) -> dict:
        pearson = _pearson(self.tier1, self.tier2)
        spearman = _spearman(self.tier1, self.tier2)
        rank_agreement = _pairwise_rank_agreement(self.tier1, self.tier2)
        sign_agreement = (
            sum(1 for a, b in zip(self.tier1, self.tier2) if (a < 0) == (b < 0))
            / len(self.tier1)
            if self.tier1
            else None
        )

        return {
            "personas_compared": self.personas,
            "n_personas": len(self.personas),
            "tier1_sentiment": [round(v, 4) for v in self.tier1],
            "tier2_sentiment": [round(v, 4) for v in self.tier2],
            "tier2_post_counts": self.n_posts,
            "pearson_r": round(pearson, 4) if pearson is not None else None,
            "spearman_rho": round(spearman, 4) if spearman is not None else None,
            "pairwise_rank_agreement": (
                round(rank_agreement, 4) if rank_agreement is not None else None
            ),
            "sign_agreement": round(sign_agreement, 4) if sign_agreement is not None else None,
            "tier1_mean": round(statistics.fmean(self.tier1), 4) if self.tier1 else None,
            "tier2_mean": round(statistics.fmean(self.tier2), 4) if self.tier2 else None,
            "level_gap": (
                round(statistics.fmean(self.tier2) - statistics.fmean(self.tier1), 4)
                if self.tier1
                else None
            ),
            "most_negative_tier1": self.personas[self.tier1.index(min(self.tier1))]
            if self.tier1
            else None,
            "most_negative_tier2": self.personas[self.tier2.index(min(self.tier2))]
            if self.tier2
            else None,
        }


def compare_tiers(
    tier1_run,
    tier2_report: dict,
    min_posts: int = 2,
) -> dict:
    """Compare a Tier-1 SimulationRun against a Tier-2 extracted report.

    Personas with fewer than `min_posts` Tier-2 posts are excluded: a cohort
    that posted once cannot support a mean, and including it would make the
    correlation a measure of sampling noise.
    """
    tier1_final = tier1_run.round_summaries[-1]["breakdown_by_persona"]
    tier2_breakdown = tier2_report.get("breakdown_by_persona", {})

    personas, t1, t2, counts, excluded = [], [], [], [], []
    for persona in sorted(tier1_final):
        entry = tier2_breakdown.get(persona)
        if entry is None:
            excluded.append({"persona": persona, "reason": "no Tier-2 posts", "n_posts": 0})
            continue
        if entry["n_posts"] < min_posts:
            excluded.append(
                {
                    "persona": persona,
                    "reason": f"fewer than {min_posts} Tier-2 posts",
                    "n_posts": entry["n_posts"],
                }
            )
            continue
        personas.append(persona)
        t1.append(tier1_final[persona]["avg_sentiment"])
        t2.append(entry["avg_sentiment"])
        counts.append(entry["n_posts"])

    comparison = TierComparison(personas, t1, t2, counts).to_dict()
    comparison["excluded_personas"] = excluded
    comparison["verdict"] = _verdict(comparison)
    comparison["caveat"] = (
        "Tier 1 averages private state over all agents; Tier 2 averages posts that "
        "were actually written, which is selection-biased toward strong feeling. "
        "Absolute levels are not expected to match - rank and sign are the checks."
    )
    return comparison


def _verdict(c: dict) -> dict:
    """Turn the metrics into a plain statement about whether Tier 1 is usable."""
    n = c["n_personas"]
    if n < 3:
        return {
            "status": "inconclusive",
            "reason": f"only {n} persona(s) had enough Tier-2 posts to compare; "
            "run more agents or more days",
        }

    rank = c["pairwise_rank_agreement"]
    sign = c["sign_agreement"]

    if rank is not None and rank >= 0.75 and sign is not None and sign >= 0.75:
        status, reason = "agrees", (
            f"{rank:.0%} of persona pairs ordered identically and {sign:.0%} sign "
            "agreement; Tier 1's segment structure is reproduced by LLM agents"
        )
    elif rank is not None and rank >= 0.5:
        status, reason = "partial", (
            f"only {rank:.0%} pairwise rank agreement; the tiers broadly agree on "
            "direction but disagree on which segments hurt most"
        )
    else:
        status, reason = "disagrees", (
            f"pairwise rank agreement {rank if rank is None else f'{rank:.0%}'}; "
            "the numeric model's segment ordering is not reproduced - investigate "
            "before trusting Tier-1 sweeps"
        )

    return {"status": status, "reason": reason}


def format_comparison(c: dict) -> str:
    """Human-readable table for the terminal and the run log."""
    lines = [
        "PERSONA           TIER 1     TIER 2   n_posts",
        "-------------------------------------------------",
    ]
    for persona, a, b, n in zip(
        c["personas_compared"], c["tier1_sentiment"], c["tier2_sentiment"], c["tier2_post_counts"]
    ):
        lines.append(f"{persona:<16} {a:+7.3f}   {b:+7.3f}   {n:>5}")

    lines += [
        "-------------------------------------------------",
        f"pearson r               {c['pearson_r']}",
        f"spearman rho            {c['spearman_rho']}",
        f"pairwise rank agreement {c['pairwise_rank_agreement']}",
        f"sign agreement          {c['sign_agreement']}",
        f"level gap (T2 - T1)     {c['level_gap']}",
        f"most negative  T1={c['most_negative_tier1']}  T2={c['most_negative_tier2']}",
        "",
        f"VERDICT: {c['verdict']['status'].upper()} - {c['verdict']['reason']}",
    ]
    if c.get("excluded_personas"):
        lines.append("")
        lines.append("excluded: " + ", ".join(
            f"{e['persona']}({e['n_posts']} posts)" for e in c["excluded_personas"]
        ))
    return "\n".join(lines)
