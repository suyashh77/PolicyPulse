"""Reddit scraper for seed data. Scaffold only in v1 — uses neutral seed."""
from __future__ import annotations

import os

import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def scrape_reddit(
    brand_name: str,
    subreddits: list[str] | None = None,
    limit: int = 200,
) -> list[dict]:
    """
    Use PRAW to pull recent posts mentioning brand_name from given subreddits.
    Returns list of {title, body, score, created_utc, subreddit}.
    """
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
