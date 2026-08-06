"""Runs a PolicyPulse scenario on OASIS.

The Tier-2 loop, mirroring Tier 1's shape:

  Day 1   the brand posts the announcement          (ManualAction)
  Day 2+  a sampled subset of agents act freely     (LLMAction)
  Day N   optional brand response                   (ManualAction)

The brand-response hook is the thing Tier 1 structurally cannot do. It turns
the tool from "will this cause backlash?" into "which response playbook
contains it fastest?" — a strictly more useful question.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from oasis_tier import require_oasis
from oasis_tier.cost import CostMeter, extrapolate, metered_backend
from oasis_tier.profiles import BRAND_USER_NAME, AgentProfile, build_profiles, graph_stats

DEFAULT_RUNS_DIR = Path("runs/oasis")

# Restricted to what a consumer plausibly does about a return policy. Every
# extra action widens the LLM's decision space and costs tokens.
DEFAULT_ACTIONS = [
    "CREATE_POST",
    "CREATE_COMMENT",
    "LIKE_POST",
    "DISLIKE_POST",
    "REPOST",
    "FOLLOW",
    "DO_NOTHING",
]


@dataclass
class BrandIntervention:
    """A brand message injected mid-simulation."""

    day: int
    content: str


@dataclass
class OasisRunConfig:
    policy_type: str
    policy_variables: dict
    announcement: str
    n_agents: int = 40
    n_days: int = 6
    agent_model: str = "claude-opus-5"
    model_platform: str = "anthropic"
    budget_usd: float = 1.00
    seed: int = 42
    activation_scale: float = 1.0      # multiplies each persona's post_probability
    max_agents_per_step: int = 25      # hard ceiling on per-step fan-out
    interventions: list[BrandIntervention] = field(default_factory=list)
    database_path: str | None = None
    available_actions: list[str] = field(default_factory=lambda: list(DEFAULT_ACTIONS))


def _build_model(cfg: OasisRunConfig, meter: CostMeter):
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    platform = {
        "anthropic": ModelPlatformType.ANTHROPIC,
        "openai": ModelPlatformType.OPENAI,
    }[cfg.model_platform]

    backend = ModelFactory.create(model_platform=platform, model_type=cfg.agent_model)
    return metered_backend(backend, meter)


def _resolve_actions(names: list[str]):
    from oasis import ActionType

    return [getattr(ActionType, n) for n in names]


async def _run_async(cfg: OasisRunConfig, personas_config: dict, shocks: dict[str, float]) -> dict:
    import oasis
    from oasis import ActionType, AgentGraph, LLMAction, ManualAction, SocialAgent, UserInfo

    rng = random.Random(cfg.seed)
    meter = CostMeter(model=cfg.agent_model, budget_usd=cfg.budget_usd)
    model = _build_model(cfg, meter)
    actions = _resolve_actions(cfg.available_actions)

    profiles = build_profiles(personas_config, shocks, n_agents=cfg.n_agents, rng=rng)

    db_path = cfg.database_path or str(
        DEFAULT_RUNS_DIR / f"oasis_{cfg.policy_type}_{cfg.seed}.db"
    )
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(db_path).exists():
        Path(db_path).unlink()  # OASIS appends; start clean so extraction is unambiguous

    # --- agents ---
    agent_graph = AgentGraph()

    brand = SocialAgent(
        agent_id=0,
        user_info=UserInfo(
            user_name=BRAND_USER_NAME,
            name="The Brand",
            description="Official brand account. Posts policy announcements.",
            profile={"other_info": {"user_profile": "Official brand account."}},
        ),
        agent_graph=agent_graph,
        model=model,
        available_actions=[ActionType.CREATE_POST],
    )
    agent_graph.add_agent(brand)

    consumers: list[tuple[SocialAgent, AgentProfile]] = []
    for p in profiles:
        agent = SocialAgent(
            agent_id=p.agent_id + 1,  # 0 is the brand
            user_info=UserInfo(**p.to_user_info_kwargs()),
            agent_graph=agent_graph,
            model=model,
            available_actions=actions,
        )
        agent_graph.add_agent(agent)
        consumers.append((agent, p))

    # Follower edges: everyone follows the brand, plus the wired graph.
    for agent, p in consumers:
        agent_graph.add_edge(agent.agent_id, 0)
        for target in p.follows:
            agent_graph.add_edge(agent.agent_id, target + 1)

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
    )
    await env.reset()

    timeline: list[dict] = []
    agent_steps = 0
    interventions_by_day = {i.day: i for i in cfg.interventions}

    try:
        # --- Day 1: the announcement ---
        await env.step(
            {brand: ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": cfg.announcement},
            )}
        )
        timeline.append({"day": 1, "kind": "announcement", "actors": 1})

        # --- Days 2..N ---
        for day in range(2, cfg.n_days + 1):
            step: dict = {}

            intervention = interventions_by_day.get(day)
            if intervention:
                step[brand] = ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": intervention.content},
                )

            # Activation: each agent acts with its persona's posting probability,
            # capped so a single step cannot blow the budget.
            active = [
                agent
                for agent, p in consumers
                if rng.random() < min(1.0, p.activation * cfg.activation_scale)
            ]
            if len(active) > cfg.max_agents_per_step:
                active = rng.sample(active, cfg.max_agents_per_step)

            for agent in active:
                step[agent] = LLMAction()

            if step:
                await env.step(step)
                agent_steps += len(active)

            timeline.append(
                {
                    "day": day,
                    "kind": "intervention+agents" if intervention else "agents",
                    "actors": len(active),
                    "cost_usd_running": round(meter.cost_usd, 6),
                }
            )
    finally:
        await env.close()

    manifest = {
        "run_id": f"oasis_{cfg.policy_type}_{cfg.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}",
        "tier": 2,
        "engine": "camel-oasis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **{k: v for k, v in asdict(cfg).items() if k != "interventions"},
            "interventions": [asdict(i) for i in cfg.interventions],
        },
        "persona_shocks": shocks,
        "graph": graph_stats(profiles),
        "timeline": timeline,
        "agent_steps": agent_steps,
        "cost": meter.snapshot(),
        "database_path": db_path,
    }
    manifest["projection"] = extrapolate(
        meter, agent_steps, target_agent_steps=500 * 44
    )
    return manifest


def run_oasis_simulation(
    cfg: OasisRunConfig,
    personas_config: dict,
    shocks: dict[str, float],
    write_manifest: bool = True,
) -> dict:
    """Run a Tier-2 simulation. Returns the run manifest."""
    require_oasis()

    if not os.getenv("ANTHROPIC_API_KEY") and cfg.model_platform == "anthropic":
        raise RuntimeError("ANTHROPIC_API_KEY is not set; Tier 2 needs a model to call")

    manifest = asyncio.run(_run_async(cfg, personas_config, shocks))

    if write_manifest:
        DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = DEFAULT_RUNS_DIR / f"{manifest['run_id']}.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(path)

    return manifest
