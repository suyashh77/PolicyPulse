"""Translate sentiment and churn into money.

"Day-45 sentiment is -0.14" is not a decision. "Saves $2.1M in return logistics,
risks $3.4M in customer lifetime value" is. This module is the bridge, and it is
what lets PolicyPulse hand a number back to the returns-routing optimiser so
behavioural risk becomes a term in its EV calculation rather than an
externality.

Everything here is arithmetic over an explicit `BrandProfile`. There is no
fitting and no hidden constant: change a brand input and you can trace exactly
how the output moved. That matters more than sophistication, because the whole
point is to be auditable in front of a finance team who will push back.

**Churn intent is not churn.** It is a latent propensity in [0, 1] that has
never been mapped to observed behaviour. `churn_conversion` is the assumption
that turns one into the other, and it is the single most load-bearing and least
defensible number in this file. It is exposed as a parameter, and sensitivity to
it should be reported alongside any result.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BrandProfile:
    """The retailer's economics. Replace these with real figures per engagement."""

    active_customers: int = 250_000
    orders_per_customer_per_year: float = 3.2
    avg_order_value: float = 95.0
    gross_margin: float = 0.45           # fraction of revenue kept before returns
    return_rate: float = 0.28            # apparel/footwear runs high
    mail_in_share: float = 0.55          # of returns, the share not done in-store

    # What the brand pays today to process one mail-in return end to end:
    # inbound freight, handling, restocking, and shrink.
    cost_per_mail_in_return: float = 14.50
    cost_per_in_store_return: float = 6.00

    # How long a retained customer keeps generating margin.
    customer_lifetime_years: float = 3.0

    def annual_orders(self) -> float:
        return self.active_customers * self.orders_per_customer_per_year

    def annual_returns(self) -> float:
        return self.annual_orders() * self.return_rate

    def annual_mail_in_returns(self) -> float:
        return self.annual_returns() * self.mail_in_share

    def annual_margin_per_customer(self) -> float:
        return self.orders_per_customer_per_year * self.avg_order_value * self.gross_margin

    def customer_lifetime_value(self) -> float:
        return self.annual_margin_per_customer() * self.customer_lifetime_years


@dataclass
class EconomicAssumptions:
    """The behavioural-to-financial bridge. Every one of these is contestable."""

    # Fraction of a segment's churn_intent that becomes an actual lost customer
    # within the year. The least defensible number here - report sensitivity.
    churn_conversion: float = 0.35

    # Customers who keep shopping but shift returns in-store to dodge the fee.
    # Saves the fee-avoidance cost but adds store handling cost.
    behaviour_shift_rate: float = 0.30

    # Share of the fee the brand actually keeps after payment processing.
    fee_capture_rate: float = 0.95


def policy_savings(
    brand: BrandProfile,
    policy_type: str,
    policy_variables: dict,
    assumptions: EconomicAssumptions | None = None,
) -> dict:
    """Direct logistics effect of the policy, before any behavioural response."""
    a = assumptions or EconomicAssumptions()
    mail_in = brand.annual_mail_in_returns()

    if policy_type in ("A", "B"):
        fee = policy_variables["fee_usd"]
        shifted = mail_in * a.behaviour_shift_rate
        remaining = mail_in - shifted

        fee_revenue = remaining * fee * a.fee_capture_rate
        shift_saving = shifted * (brand.cost_per_mail_in_return - brand.cost_per_in_store_return)

        return {
            "mechanism": "fee revenue + in-store shift",
            "fee_revenue": round(fee_revenue, 2),
            "shift_saving": round(shift_saving, 2),
            "gross_saving": round(fee_revenue + shift_saving, 2),
            "returns_affected": round(mail_in, 0),
        }

    if policy_type == "C":
        # Keepit: skip return logistics entirely on qualifying items, but eat the
        # cost of goods, since nothing comes back to resell.
        threshold = policy_variables["threshold_usd"]
        qualifying_share = min(0.6, threshold / max(brand.avg_order_value, 1e-9))
        qualifying = brand.annual_returns() * qualifying_share

        logistics_saved = qualifying * brand.cost_per_mail_in_return
        goods_lost = qualifying * threshold * 0.5 * (1 - brand.gross_margin)

        return {
            "mechanism": "avoided return logistics minus unrecovered goods",
            "logistics_saved": round(logistics_saved, 2),
            "goods_lost": round(-goods_lost, 2),
            "gross_saving": round(logistics_saved - goods_lost, 2),
            "returns_affected": round(qualifying, 0),
        }

    raise ValueError(f"unknown policy_type {policy_type!r}")


def churn_cost(
    brand: BrandProfile,
    churn_by_segment: dict[str, float],
    persona_config: dict,
    assumptions: EconomicAssumptions | None = None,
) -> dict:
    """Lifetime value at risk, weighted by each segment's share of the base.

    Segment mix matters enormously: deal seekers churn hardest but are the
    cheapest customers to lose, while loyal customers barely react but each one
    lost is worth many times more.
    """
    a = assumptions or EconomicAssumptions()
    personas = persona_config["personas"]

    total_fraction = sum(personas[p]["count_fraction"] for p in churn_by_segment if p in personas)
    if total_fraction <= 0:
        return {"total_clv_at_risk": 0.0, "by_segment": {}}

    by_segment = {}
    total = 0.0
    for persona, intent in churn_by_segment.items():
        cfg = personas.get(persona)
        if cfg is None:
            continue

        share = cfg["count_fraction"] / total_fraction
        segment_customers = brand.active_customers * share

        # Segment-specific value: AOV varies a lot across personas.
        segment_aov = cfg.get("aov_usd", brand.avg_order_value)
        segment_clv = (
            brand.orders_per_customer_per_year
            * segment_aov
            * brand.gross_margin
            * brand.customer_lifetime_years
        )

        lost_customers = segment_customers * intent * a.churn_conversion
        value = lost_customers * segment_clv
        total += value

        by_segment[persona] = {
            "customers": round(segment_customers, 0),
            "churn_intent": round(intent, 4),
            "customers_lost": round(lost_customers, 0),
            "segment_clv": round(segment_clv, 2),
            "clv_at_risk": round(value, 2),
        }

    return {
        "total_clv_at_risk": round(total, 2),
        "by_segment": dict(
            sorted(by_segment.items(), key=lambda kv: -kv[1]["clv_at_risk"])
        ),
        "churn_conversion_used": a.churn_conversion,
    }


def evaluate_policy(
    brand: BrandProfile,
    policy_type: str,
    policy_variables: dict,
    churn_by_segment: dict[str, float],
    persona_config: dict,
    assumptions: EconomicAssumptions | None = None,
) -> dict:
    """The headline number: does this policy make or lose money once people react?"""
    a = assumptions or EconomicAssumptions()
    savings = policy_savings(brand, policy_type, policy_variables, a)
    risk = churn_cost(brand, churn_by_segment, persona_config, a)

    gross = savings["gross_saving"]
    at_risk = risk["total_clv_at_risk"]
    net = gross - at_risk

    return {
        "policy_type": policy_type,
        "policy_variables": policy_variables,
        "gross_annual_saving": round(gross, 2),
        "clv_at_risk": round(at_risk, 2),
        "net_value": round(net, 2),
        "verdict": (
            "accretive" if net > 0 else "value-destroying"
        ),
        "breakeven_churn_conversion": round(
            gross / at_risk * a.churn_conversion, 4
        ) if at_risk > 0 else None,
        "savings_detail": savings,
        "risk_detail": risk,
    }


def sensitivity_to_churn_conversion(
    brand: BrandProfile,
    policy_type: str,
    policy_variables: dict,
    churn_by_segment: dict[str, float],
    persona_config: dict,
    values: tuple[float, ...] = (0.10, 0.20, 0.35, 0.50, 0.75),
) -> list[dict]:
    """Net value across the assumption nobody can defend.

    If the sign of `net_value` flips inside this range, the recommendation is not
    robust and should be presented as a range rather than a number.
    """
    out = []
    for v in values:
        a = EconomicAssumptions(churn_conversion=v)
        result = evaluate_policy(
            brand, policy_type, policy_variables, churn_by_segment, persona_config, a
        )
        out.append(
            {
                "churn_conversion": v,
                "net_value": result["net_value"],
                "verdict": result["verdict"],
            }
        )
    return out


def build_frontier(
    brand: BrandProfile,
    policy_type: str,
    variable_name: str,
    levels: list,
    churn_by_level: dict,
    persona_config: dict,
    base_variables: dict | None = None,
    assumptions: EconomicAssumptions | None = None,
) -> list[dict]:
    """One row per policy level: savings, risk, and net.

    This is the data behind the frontier chart - the single image that shows
    where the safe fee ends and the value-destroying one begins.
    """
    rows = []
    for level in levels:
        variables = dict(base_variables or {})
        variables[variable_name] = level
        churn = churn_by_level.get(level, {})
        result = evaluate_policy(
            brand, policy_type, variables, churn, persona_config, assumptions
        )
        rows.append(
            {
                "level": level,
                "gross_annual_saving": result["gross_annual_saving"],
                "clv_at_risk": result["clv_at_risk"],
                "net_value": result["net_value"],
                "verdict": result["verdict"],
            }
        )
    return rows
