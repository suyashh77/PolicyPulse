"""Reddit scraper for seed data.

SCAFFOLD — this module has no call sites. `run_simulation` does not use it, and
`posts_to_seed_sentiment` returns a neutral 0.0 regardless of input. Wiring it
up (so a brand's existing Reddit sentiment sets the pre-policy baseline instead
of assuming neutrality) is listed under "What's left to build" in the README.

`praw` and `nltk` are not in the default requirements, so the imports are
guarded: importing this module never fails, but calling `scrape_reddit` without
those packages raises with a clear message.
"""
from __future__ import annotations

import os

try:
    import praw
except ImportError:  # pragma: no cover - optional dependency
    praw = None

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None


def scrape_reddit(
    brand_name: str,
    subreddits: list[str] | None = None,
    limit: int = 200,
) -> list[dict]:
    """
    Use PRAW to pull recent posts mentioning brand_name from given subreddits.
    Returns list of {title, body, score, created_utc, subreddit}.
    """
    if praw is None:
        raise ImportError("scrape_reddit requires `praw` (pip install praw)")

    if subreddits is None:
        subreddits = [
            "frugalmalefashion",
            "femalefashionadvice",
            "buyitforlife",
            "AskWomen",
            "minimalism",
        ]

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "PolicyPulse/1.0"),
    )

    results: list[dict] = []
    for sub_name in subreddits:
        subreddit = reddit.subreddit(sub_name)
        for submission in subreddit.search(brand_name, limit=limit // len(subreddits)):
            results.append(
                {
                    "title": submission.title,
                    "body": submission.selftext,
                    "score": submission.score,
                    "created_utc": submission.created_utc,
                    "subreddit": sub_name,
                }
            )

    return results


def posts_to_seed_sentiment(raw_posts: list[dict]) -> float:
    """
    Simple average sentiment of scraped posts using VADER.
    Returns float -1.0 to 1.0.

    For v1: default to 0.0 (neutral start) regardless of seed.
    Scaffolded but not activated.
    """
    # v1: always return neutral
    return 0.0

    # v2 activation:
    # sia = SentimentIntensityAnalyzer()
    # if not raw_posts:
    #     return 0.0
    # scores = []
    # for post in raw_posts:
    #     text = f"{post['title']} {post['body']}"
    #     compound = sia.polarity_scores(text)["compound"]
    #     scores.append(compound)
    # return sum(scores) / len(scores)
