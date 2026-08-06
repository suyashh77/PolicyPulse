"""OASIS SQLite trace -> the same report shape Tier 1 emits.

Tier 1 knows each agent's sentiment because it is a number it computed. Tier 2
agents produce *text*, so sentiment has to be read back out of what they wrote.

Scoring is deliberately pluggable:
  - `lexicon`  free, no API calls, coarse. Used in tests and for smoke runs.
  - `llm`      one batched call, far better, costs money.

The extractor also reads likes/dislikes straight off the platform, which is a
free behavioural signal the lexicon can be checked against.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

# Small domain lexicon. Not a general sentiment model - it is tuned for return
# policy complaints and exists so the pipeline is testable without spending money.
NEGATIVE_TERMS = {
    "unfair", "ridiculous", "outrageous", "greedy", "cash grab", "nickel", "scam",
    "annoying", "frustrating", "disappointed", "disappointing", "terrible", "awful",
    "bad", "hate", "angry", "upset", "expensive", "costly", "rip off", "ripoff",
    "cancel", "canceling", "cancelling", "switch", "switching", "leaving", "elsewhere",
    "competitor", "never again", "done with", "boycott", "punish", "penalty", "fee",
    "charge", "charging", "worse", "downgrade", "no longer", "stop shopping",
}
POSITIVE_TERMS = {
    "great", "good", "fair", "reasonable", "love", "glad", "happy", "appreciate",
    "generous", "convenient", "easy", "helpful", "sensible", "understandable",
    "fine", "no problem", "makes sense", "still worth", "better", "nice", "thanks",
    "free", "keep it", "refund", "win",
}
NEGATORS = {"not", "isn't", "not really", "hardly", "no"}

_WORD = re.compile(r"[a-z']+")


def score_text_lexicon(text: str) -> float:
    """Crude sentiment in [-1, 1]. Phrase matches first, then unigrams."""
    if not text:
        return 0.0
    low = text.lower()

    score = 0
    for phrase in (t for t in NEGATIVE_TERMS if " " in t):
        if phrase in low:
            score -= 1
    for phrase in (t for t in POSITIVE_TERMS if " " in t):
        if phrase in low:
            score += 1

    tokens = _WORD.findall(low)
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i else ""
        flip = -1 if prev in NEGATORS else 1
        if tok in NEGATIVE_TERMS:
            score -= 1 * flip
        elif tok in POSITIVE_TERMS:
            score += 1 * flip

    if score == 0:
        return 0.0
    # Squash: a post with 6 complaints is not 6x angrier than one with 1.
    return max(-1.0, min(1.0, score / 3.0))


def load_run_tables(db_path: str | Path) -> dict:
    """Read the OASIS tables this package cares about."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"no OASIS database at {db_path}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        existing = {
            r["name"] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        out = {}
        for table in ("user", "post", "comment", "trace"):
            if table in existing:
                out[table] = [dict(r) for r in con.execute(f"SELECT * FROM {table}")]
            else:
                out[table] = []
        return out
    finally:
        con.close()


def _persona_from_username(user_name: str | None) -> str:
    """`deal_seeker_3` -> `deal_seeker`. Usernames encode the persona."""
    if not user_name:
        return "unknown"
    if user_name == "brand_official":
        return "brand"
    return user_name.rsplit("_", 1)[0]


def _day_from_timestamp(value, day_index: dict) -> int:
    """OASIS timestamps vary by clock config; rank distinct values into days."""
    if value not in day_index:
        day_index[value] = len(day_index) + 1
    return day_index[value]


def _apply_scores(posts: list[dict], scorer) -> None:
    """Attach sentiment (and churn/stance when available) to each post in place.

    Batches through `score_many` when the scorer offers it, so an LLM scorer
    makes one call per ~20 posts instead of one call per post.
    """
    if not posts:
        return

    texts = [p["content"] for p in posts]

    if hasattr(scorer, "score_many"):
        for post, scored in zip(posts, scorer.score_many(texts)):
            post["sentiment"] = scored.sentiment
            post["churn_intent"] = scored.churn_intent
            post["stance"] = scored.stance
    else:
        for post, text in zip(posts, texts):
            post["sentiment"] = scorer(text)


def extract_report(
    db_path: str | Path,
    manifest: dict | None = None,
    scorer=score_text_lexicon,
) -> dict:
    """Build a Tier-1-shaped report from an OASIS run.

    Returns sentiment_curve, churn_by_segment, cascade and the raw posts, so the
    existing reporting layer and UI can consume a Tier-2 run unchanged.

    `scorer` is either a per-text callable returning a float (the lexicon), or an
    object exposing `score_many(list[str])` (the LLM scorer), in which case all
    posts are scored in batches rather than one call each.
    """
    tables = load_run_tables(db_path)

    persona_by_user = {
        row["user_id"]: _persona_from_username(row.get("user_name"))
        for row in tables["user"]
    }

    day_index: dict = {}
    scored_posts = []
    n_reposts = 0

    # Posts AND comments. On a 60-agent run OASIS produced 20 posts against 158
    # comments - scoring only posts threw away 89% of what the agents said, and
    # left whole personas (loyal, sustainability) with no measurable opinion
    # because they reply far more than they post.
    utterances = [
        ("post", r, r.get("post_id")) for r in tables["post"]
    ] + [
        ("comment", r, r.get("comment_id")) for r in tables["comment"]
    ]

    for kind, row, row_id in sorted(
        utterances, key=lambda t: (t[1].get("created_at") or "", t[2] or 0)
    ):
        persona = persona_by_user.get(row["user_id"], "unknown")
        if persona == "brand":
            continue  # the brand's own posts are stimulus, not reaction
        content = row.get("content") or ""
        if not content.strip():
            # OASIS writes a post row with empty content for a repost. It is an
            # amplification signal, not an opinion - scoring it as 0.0 would drag
            # every persona mean toward neutral. Counted, not scored.
            n_reposts += 1
            continue
        scored_posts.append(
            {
                "kind": kind,
                "post_id": row_id,
                "user_id": row["user_id"],
                "persona": persona,
                "day": _day_from_timestamp(row.get("created_at"), day_index),
                "content": content,
                "num_likes": row.get("num_likes", 0) or 0,
                "num_dislikes": row.get("num_dislikes", 0) or 0,
            }
        )

    _apply_scores(scored_posts, scorer)

    # --- sentiment curve: mean of posts made up to and including each day ---
    by_day: dict[int, list[float]] = defaultdict(list)
    for p in scored_posts:
        by_day[p["day"]].append(p["sentiment"])

    curve = []
    running: list[float] = []
    for day in sorted(by_day) or [1]:
        running.extend(by_day.get(day, []))
        curve.append(
            {
                "day": day,
                "avg_sentiment": sum(running) / len(running) if running else 0.0,
                "posts_today": len(by_day.get(day, [])),
                "day_only_sentiment": (
                    sum(by_day[day]) / len(by_day[day]) if by_day.get(day) else 0.0
                ),
            }
        )

    # --- per-persona sentiment and a churn proxy ---
    persona_posts: dict[str, list[dict]] = defaultdict(list)
    for p in scored_posts:
        persona_posts[p["persona"]].append(p)

    shocks = (manifest or {}).get("persona_shocks", {})
    breakdown = {}
    for persona, posts in persona_posts.items():
        sentiments = [p["sentiment"] for p in posts]
        mean = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Churn: the LLM scorer judges intent directly. The keyword heuristic is
        # the fallback when scoring was done with the lexicon.
        if posts and "churn_intent" in posts[0]:
            churn = sum(p["churn_intent"] for p in posts) / len(posts)
        else:
            leaving = sum(
                1 for p in posts
                if any(t in p["content"].lower()
                       for t in ("cancel", "switch", "elsewhere", "never again",
                                 "done with", "stop shopping", "leaving"))
            )
            churn = leaving / len(posts) if posts else 0.0

        entry = {
            "avg_sentiment": round(mean, 4),
            "n_posts": len(posts),
            "churn_signal_rate": round(churn, 4),
            "tier1_baseline_shock": shocks.get(persona),
        }
        if posts and "stance" in posts[0]:
            stances: dict[str, int] = {}
            for p in posts:
                stances[p["stance"]] = stances.get(p["stance"], 0) + 1
            entry["stances"] = dict(sorted(stances.items(), key=lambda kv: -kv[1]))
        breakdown[persona] = entry

    final_sentiment = curve[-1]["avg_sentiment"] if curve else 0.0

    # --- cascade: a >0.4 fall in the running mean within any 10-day window ---
    values = [c["avg_sentiment"] for c in curve]
    cascade = {"cascade": False, "trigger_day": None}
    window = min(10, len(values))
    for i in range(len(values) - window + 1):
        if values[i] - values[i + window - 1] >= 0.4:
            cascade = {"cascade": True, "trigger_day": curve[i]["day"]}
            break

    return {
        "tier": 2,
        "engine": "camel-oasis",
        "run_id": (manifest or {}).get("run_id"),
        "policy_type": (manifest or {}).get("config", {}).get("policy_type"),
        "policy_variables": (manifest or {}).get("config", {}).get("policy_variables"),
        "sentiment_curve": curve,
        "final_sentiment": round(final_sentiment, 4),
        "breakdown_by_persona": breakdown,
        "cascade": cascade,
        "n_posts": len(scored_posts),
        "n_utterances": len(scored_posts),
        "n_original_posts": sum(1 for p in scored_posts if p["kind"] == "post"),
        "n_reposts": n_reposts,
        "n_comments": sum(1 for p in scored_posts if p["kind"] == "comment"),
        "n_trace_events": len(tables["trace"]),
        "posts": scored_posts,
        "cost": (manifest or {}).get("cost"),
    }


def top_quotes(report: dict, n: int = 5) -> list[dict]:
    """The most negative and most positive things agents actually said.

    Qualitative output like this is the main reason to run Tier 2 at all - it is
    what a human reads and what goes in a decision brief.
    """
    posts = [p for p in report.get("posts", []) if p["content"].strip()]
    posts.sort(key=lambda p: p["sentiment"])
    return {
        "most_negative": posts[:n],
        "most_positive": list(reversed(posts[-n:])),
    }


def save_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
