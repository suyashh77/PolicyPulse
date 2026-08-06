from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from core.agent import Agent
    from core.simulation import SimulationRun

MODEL = "claude-opus-5"

# Thinking and response text share this budget. Opus 5 thinks by default, so a
# tight cap would truncate the answer mid-sentence; the effort setting keeps the
# spend down instead.
MAX_TOKENS = 2000

CHURN_THRESHOLD = 0.5

SYSTEM_PROMPT = (
    "You are a consumer who just experienced this return policy change. "
    "Respond in first person, grounded in your memory of what you saw "
    "and how you felt over the past 45 days. Be specific and human. "
    "Do not describe yourself as a simulation or an agent."
)


def get_interview_candidates(
    run: SimulationRun,
    persona: str,
    churned: bool,
) -> list[Agent]:
    """
    Filter agents by persona and churn status at round 45.
    churned=True: churn_intent >= 0.5
    churned=False: churn_intent < 0.5
    """
    candidates = []
    for agent in run.agents:
        if agent.persona != persona:
            continue
        if churned and agent.churn_intent >= CHURN_THRESHOLD:
            candidates.append(agent)
        elif not churned and agent.churn_intent < CHURN_THRESHOLD:
            candidates.append(agent)
    return candidates


def _describe_memory(agent: Agent, n: int = 10) -> str:
    """Render the agent's recent rounds as readable lines rather than raw dicts."""
    recent = agent.memory[-n:]
    return "\n".join(
        f"  Day {m['round']}: saw {len(m['posts_seen_ids'])} posts, "
        f"felt {m['sentiment']:+.2f}, likelihood of leaving {m['churn_intent']:.2f}"
        for m in recent
    )


def interview_agent(agent: Agent, announcement_text: str) -> str:
    """Ask Claude to voice one agent's 45-day experience of the policy.

    The only LLM call in the project. Everything in the reporting path is
    deterministic.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Policy announced: {announcement_text}\n\n"
                    f"Your recent experience:\n{_describe_memory(agent)}\n\n"
                    f"Your starting reaction to the policy was "
                    f"{agent.baseline_sentiment:+.2f} on a -1 to +1 scale.\n"
                    f"Final sentiment: {agent.policy_sentiment:+.2f}\n"
                    f"Final likelihood of leaving the brand: {agent.churn_intent:.2f}\n\n"
                    "Explain your reaction to this policy in 3-4 sentences."
                ),
            }
        ],
    )

    if response.stop_reason == "refusal":
        return "(The model declined to generate a response for this agent.)"

    text = next((b.text for b in response.content if b.type == "text"), "")
    return text or "(No response text returned.)"
