"""LLM sentiment scoring for Tier-2 posts.

The lexicon scorer in `extract.py` counts words. On the P0 spike it read a post
containing "$12.95 return fee?? SERIOUSLY?? ... that's 30-50% of what I spent!
... 5 other retailers have FREE returns" as only -0.33, because it has no notion
of tone, sarcasm, or intensity. That flattening made cross-tier validation
meaningless: Tier 2 looked mild next to Tier 1's -0.569 shock purely as an
artefact of the scorer.

This module replaces it. Posts are scored in batches with structured outputs, so
one call handles ~20 posts and the response is schema-validated rather than
parsed out of prose.

It deliberately does **not** import OASIS. Scoring runs on the project's main
3.12 interpreter against a saved run database, so analysis is decoupled from the
fragile 3.11 simulation environment.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from oasis_tier.cost import CostMeter

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_BATCH_SIZE = 20

# Structured outputs reject numerical constraints (minimum/maximum), so ranges
# are stated in the prompt and clamped client-side.
SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "sentiment": {"type": "number"},
                    "churn_intent": {"type": "number"},
                    "stance": {
                        "type": "string",
                        "enum": [
                            "furious", "annoyed", "resigned", "neutral",
                            "understanding", "pleased", "delighted",
                        ],
                    },
                },
                "required": ["index", "sentiment", "churn_intent", "stance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You score social media posts about a retailer's return-policy change.

For each post return:
- sentiment: -1.0 (furious, hostile to the policy) to +1.0 (actively pleased by it).
  0.0 is genuinely neutral. Judge intensity, not just polarity: a post with
  capitals, repeated punctuation, sarcasm, or "this is why I shop elsewhere"
  should land near -0.8, not -0.2. A mild grumble is about -0.3.
- churn_intent: 0.0 to 1.0, how likely this person is to stop shopping with the
  brand. Explicit statements ("cancelling my order", "switching to X") are high.
  Complaining without any intent to leave is low.
- stance: the single closest label from the enum.

Score what the post actually expresses, not what a reasonable person would feel.
Return one entry per post, using the index given."""


@dataclass
class ScoredPost:
    sentiment: float
    churn_intent: float
    stance: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


@dataclass
class LLMScorer:
    """Batched LLM scorer. Callable per-post for API compatibility with the lexicon."""

    model: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    budget_usd: float = 1.00
    effort: str = "low"
    meter: CostMeter = field(default=None)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.meter is None:
            self.meter = CostMeter(model=self.model, budget_usd=self.budget_usd)

    # --- batch interface (what extract.py uses) ---

    def score_many(self, texts: list[str]) -> list[ScoredPost]:
        results: list[ScoredPost] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            results.extend(self._score_batch(chunk))
        return results

    def sentiments(self, texts: list[str]) -> list[float]:
        return [s.sentiment for s in self.score_many(texts)]

    # --- single-post interface (drop-in for score_text_lexicon) ---

    def __call__(self, text: str) -> float:
        if not text.strip():
            return 0.0
        if text not in self._cache:
            self._cache[text] = self._score_batch([text])[0]
        return self._cache[text].sentiment

    # --- internals ---

    def _score_batch(self, texts: list[str]) -> list[ScoredPost]:
        import anthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set; LLM scoring needs a model")

        numbered = "\n\n".join(
            f"[{i}] {t.strip()[:1500]}" for i, t in enumerate(texts)
        )

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model,
            # Thinking and output share this budget on Opus 5, and a 20-post
            # batch produces a sizeable JSON body.
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SCORE_SCHEMA},
            },
            messages=[
                {
                    "role": "user",
                    "content": f"Score these {len(texts)} posts:\n\n{numbered}",
                }
            ],
        )

        self.meter.record(response.usage)
        self.meter.check_budget()

        if response.stop_reason == "refusal":
            return [ScoredPost(0.0, 0.0, "neutral")] * len(texts)

        text = next((b.text for b in response.content if b.type == "text"), "")
        parsed = json.loads(text)

        by_index = {int(s["index"]): s for s in parsed.get("scores", [])}
        out: list[ScoredPost] = []
        for i in range(len(texts)):
            entry = by_index.get(i)
            if entry is None:  # model dropped one; fail soft rather than crash
                out.append(ScoredPost(0.0, 0.0, "neutral"))
                continue
            out.append(
                ScoredPost(
                    sentiment=_clamp(entry.get("sentiment", 0.0), -1.0, 1.0),
                    churn_intent=_clamp(entry.get("churn_intent", 0.0), 0.0, 1.0),
                    stance=str(entry.get("stance", "neutral")),
                )
            )
        return out


def make_scorer(kind: str = "llm", **kwargs):
    """Factory. `kind` is 'llm' or 'lexicon'.

    The lexicon is kept for tests and for smoke runs where spending money to
    score a handful of posts is not worth it.
    """
    if kind == "lexicon":
        from oasis_tier.extract import score_text_lexicon

        return score_text_lexicon
    if kind == "llm":
        return LLMScorer(**kwargs)
    raise ValueError(f"unknown scorer kind {kind!r}; expected 'llm' or 'lexicon'")
