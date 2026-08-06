"""Maps a policy and its variables onto a per-persona sentiment shock.

This is the component the rest of the simulation exists to propagate. Without
it, `run_simulation` ignored its own `policy_variables` argument and every
policy produced identical output.

The shock is the agent's *private* reaction on day 1, before it has seen anyone
else's opinion. Social contagion then moves agents away from it; each agent's
`anchor_strength` pulls them partway back. So the shock sets the centre of
gravity the population argues around, not the final answer.

Functional form
---------------
Fees are judged relative to basket size, not in absolute dollars: $9.95 is a
fifth of a deal seeker's $45 order and a sixteenth of a loyal customer's $165
one. Relative cost passes through a saturating disutility curve so that pain
grows quickly at first and then plateaus — a $50 fee is not five times as
infuriating as a $10 one, it is simply unacceptable either way.

    disutility = fee_sensitivity * tanh(relative_cost / DISUTILITY_SCALE)

`DISUTILITY_SCALE` is the one free calibration constant. It sets where the
curve bends: at a relative cost equal to the scale, disutility is ~76% of the
persona's ceiling. It is a prior, not a fitted value — see README "Validation".
"""
from __future__ import annotations

import math

# Relative cost (fee / AOV) at which disutility reaches ~76% of its ceiling.
# Calibration target for historical backtesting.
DISUTILITY_SCALE = 0.25

# Distance scale for store-proximity coverage, in miles. A larger radius covers
# more customers with diminishing returns.
STORE_DISTANCE_SCALE = 18.0

# Controls how quickly a keepit threshold covers a persona's order distribution.
# threshold = 0.6 x AOV qualifies ~63% of orders.
KEEPIT_ORDER_SCALE = 0.6

_ECONOMIC_KEYS = (
    "aov_usd",
    "fee_sensitivity",
    "store_access",
    "in_store_friction",
    "keepit_affinity",
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _disutility(relative_cost: float, sensitivity: float) -> float:
    """Saturating pain curve. Returns 0..sensitivity."""
    if relative_cost <= 0:
        return 0.0
    return sensitivity * math.tanh(relative_cost / DISUTILITY_SCALE)


def store_coverage(distance_miles: float, store_access: float) -> float:
    """Fraction of this persona that can return in-store within `distance_miles`.

    Bounded above by `store_access` — living near a store does not help someone
    who never goes to one.
    """
    geographic = 1.0 - math.exp(-distance_miles / STORE_DISTANCE_SCALE)
    return store_access * geographic


def keepit_qualifying_fraction(threshold_usd: float, aov_usd: float) -> float:
    """Fraction of a persona's orders that fall under the keepit threshold."""
    if aov_usd <= 0:
        return 0.0
    return 1.0 - math.exp(-threshold_usd / (aov_usd * KEEPIT_ORDER_SCALE))


def _require_economics(persona_cfg: dict, persona_name: str) -> None:
    missing = [k for k in _ECONOMIC_KEYS if k not in persona_cfg]
    if missing:
        raise KeyError(
            f"persona {persona_name!r} is missing economic parameters {missing}; "
            "config/personas.yaml is out of date with core/policy_impact.py"
        )


def policy_shock(policy_type: str, policy_variables: dict, persona_cfg: dict,
                 persona_name: str = "?") -> float:
    """Day-1 sentiment reaction for one persona, in [-1, 1].

    Negative means the policy reads as a cost being pushed onto the customer;
    positive means it reads as a benefit.
    """
    _require_economics(persona_cfg, persona_name)

    aov = persona_cfg["aov_usd"]
    fee_sensitivity = persona_cfg["fee_sensitivity"]
    store_access = persona_cfg["store_access"]
    friction = persona_cfg["in_store_friction"]

    if policy_type == "A":
        # Distance-gated: free in-store within X miles, fee for mail-in.
        coverage = store_coverage(policy_variables["distance_miles"], store_access)
        fee = policy_variables["fee_usd"]
        shock = -(
            _disutility(fee * (1.0 - coverage) / aov, fee_sensitivity)
            + coverage * friction
        )

    elif policy_type == "B":
        # Flat fee on all mail-in returns; in-store still free but ungated, so
        # only a persona's baseline store access escapes the fee.
        coverage = store_access
        fee = policy_variables["fee_usd"]
        shock = -(
            _disutility(fee * (1.0 - coverage) / aov, fee_sensitivity)
            + coverage * friction
        )

    elif policy_type == "C":
        # Keepit: a giveaway, so the shock is positive for most personas.
        # `keepit_affinity` is negative for the sustainability cohort, who read
        # it as waste — the one policy that splits the population by sign.
        qualifying = keepit_qualifying_fraction(policy_variables["threshold_usd"], aov)
        shock = persona_cfg["keepit_affinity"] * qualifying

    else:
        raise ValueError(f"unknown policy_type {policy_type!r}; expected 'A', 'B' or 'C'")

    return _clamp(shock, -1.0, 1.0)


def policy_shocks_by_persona(policy_type: str, policy_variables: dict,
                             personas_config: dict) -> dict[str, float]:
    """Shock for every persona in the config."""
    personas = personas_config["personas"]
    return {
        name: policy_shock(policy_type, policy_variables, cfg, name)
        for name, cfg in personas.items()
    }


def population_shock(shocks: dict[str, float], personas_config: dict) -> float:
    """Population-weighted mean shock — the sentiment of the announcement itself."""
    personas = personas_config["personas"]
    total_weight = sum(personas[name]["count_fraction"] for name in shocks)
    if total_weight <= 0:
        return 0.0
    return sum(
        shocks[name] * personas[name]["count_fraction"] for name in shocks
    ) / total_weight
