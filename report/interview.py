from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from core.agent import Agent
    from core.simulation import SimulationRun


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
        if churned and agent.churn_intent >= 0.5:
            candidates.append(agent)
        elif not churned and agent.churn_intent < 0.5:
            candidates.append(agent)
    return candidates


def interview_agent(
    agent: Agent,
    policy_variables: dict,
    announcement_text: str,
) -> str:
    """Call Claude Sonnet API to generate a first-person agent interview response."""
    last_memories = agent.memory[-10:] if len(agent.memory) >= 10 else agent.memory

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=(
            "You are a consumer who just experienced this return policy change. "
            "Respond in first person, grounded in your memory of what you saw "
            "and how you felt over the past 45 days. Be specific and human."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Policy announced: {announcement_text}\n"
                    f"Your memory log: {last_memories}\n"
                    f"Final sentiment: {agent.policy_sentiment:.2f}\n"
                    f"Final churn intent: {agent.churn_intent:.2f}\n"
                    f"Explain your reaction to this policy in 3-4 sentences."
                ),
            }
        ],
    )

    return message.content[0].text
