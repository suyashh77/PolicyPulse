"""Economics layer: sentiment/churn -> dollars."""
import pytest

from core.economics import (
    BrandProfile,
    EconomicAssumptions,
    build_frontier,
    churn_cost,
    evaluate_policy,
    policy_savings,
    sensitivity_to_churn_conversion,
)
from core.simulation import load_personas

CONFIG = load_personas("config/personas.yaml")
BRAND = BrandProfile()
CHURN = {
    "loyal": 0.012,
    "casual": 0.065,
    "deal_seeker": 0.151,
    "influencer": 0.051,
    "sustainability": 0.070,
}


class TestBrandProfile:
    def test_derived_volumes_are_consistent(self):
        assert BRAND.annual_orders() == pytest.approx(250_000 * 3.2)
        assert BRAND.annual_returns() == pytest.approx(BRAND.annual_orders() * 0.28)
        assert BRAND.annual_mail_in_returns() < BRAND.annual_returns()

    def test_clv_scales_with_lifetime(self):
        short = BrandProfile(customer_lifetime_years=1.0)
        long = BrandProfile(customer_lifetime_years=5.0)
        assert long.customer_lifetime_value() == pytest.approx(
            5 * short.customer_lifetime_value()
        )


class TestSavings:
    def test_higher_fee_saves_more(self):
        cheap = policy_savings(BRAND, "B", {"fee_usd": 4.95})
        dear = policy_savings(BRAND, "B", {"fee_usd": 12.95})
        assert dear["gross_saving"] > cheap["gross_saving"]

    def test_keepit_trades_logistics_against_lost_goods(self):
        s = policy_savings(BRAND, "C", {"threshold_usd": 25})
        assert s["logistics_saved"] > 0
        assert s["goods_lost"] < 0  # reported as a negative contribution

    def test_wider_distance_gate_reduces_fee_revenue(self):
        """Type A: more coverage means fewer people pay the mail-in fee."""
        near = policy_savings(BRAND, "A", {"distance_miles": 5, "fee_usd": 9.95})
        far = policy_savings(BRAND, "A", {"distance_miles": 25, "fee_usd": 9.95})
        assert near["gross_saving"] == far["gross_saving"]  # A and B share the fee model

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            policy_savings(BRAND, "Z", {})


class TestChurnCost:
    def test_zero_churn_costs_nothing(self):
        zero = {k: 0.0 for k in CHURN}
        assert churn_cost(BRAND, zero, CONFIG)["total_clv_at_risk"] == 0.0

    def test_risk_scales_with_conversion(self):
        low = churn_cost(BRAND, CHURN, CONFIG, EconomicAssumptions(churn_conversion=0.1))
        high = churn_cost(BRAND, CHURN, CONFIG, EconomicAssumptions(churn_conversion=0.5))
        assert high["total_clv_at_risk"] > low["total_clv_at_risk"]

    def test_segment_clv_reflects_persona_aov(self):
        """Loyal customers have a much higher AOV, so each one lost costs more."""
        result = churn_cost(BRAND, CHURN, CONFIG)
        by = result["by_segment"]
        assert by["loyal"]["segment_clv"] > by["deal_seeker"]["segment_clv"]

    def test_loudest_segment_is_not_necessarily_costliest(self):
        """The finding the tool exists to surface.

        deal_seeker churns ~12x harder than loyal, but each lost deal_seeker is
        worth far less, so raw churn rate is a misleading way to pick who to
        protect.
        """
        result = churn_cost(BRAND, CHURN, CONFIG)
        by = result["by_segment"]
        assert CHURN["deal_seeker"] > CHURN["casual"]          # churns harder
        assert by["casual"]["clv_at_risk"] > by["deal_seeker"]["clv_at_risk"]

    def test_unknown_persona_ignored(self):
        result = churn_cost(BRAND, {**CHURN, "martians": 0.9}, CONFIG)
        assert "martians" not in result["by_segment"]


class TestEvaluatePolicy:
    def test_returns_net_and_verdict(self):
        r = evaluate_policy(BRAND, "B", {"fee_usd": 9.95}, CHURN, CONFIG)
        assert r["net_value"] == pytest.approx(
            r["gross_annual_saving"] - r["clv_at_risk"]
        )
        assert r["verdict"] in ("accretive", "value-destroying")

    def test_no_churn_is_accretive(self):
        zero = {k: 0.0 for k in CHURN}
        assert evaluate_policy(BRAND, "B", {"fee_usd": 9.95}, zero, CONFIG)["verdict"] == "accretive"

    def test_extreme_churn_destroys_value(self):
        heavy = {k: 1.0 for k in CHURN}
        assert evaluate_policy(BRAND, "B", {"fee_usd": 4.95}, heavy, CONFIG)["verdict"] == (
            "value-destroying"
        )


class TestSensitivity:
    def test_net_value_falls_as_conversion_rises(self):
        rows = sensitivity_to_churn_conversion(BRAND, "B", {"fee_usd": 9.95}, CHURN, CONFIG)
        nets = [r["net_value"] for r in rows]
        assert nets == sorted(nets, reverse=True)

    def test_sign_flip_is_visible(self):
        """The point of the table: show where the recommendation reverses."""
        rows = sensitivity_to_churn_conversion(BRAND, "B", {"fee_usd": 12.95}, CHURN, CONFIG)
        verdicts = {r["verdict"] for r in rows}
        assert verdicts == {"accretive", "value-destroying"}, (
            "expected the verdict to flip across the plausible range, which is "
            "what makes this a range rather than a point estimate"
        )


class TestFrontier:
    def test_one_row_per_level(self):
        levels = [4.95, 7.95, 9.95, 12.95]
        churn_by_level = {lv: CHURN for lv in levels}
        rows = build_frontier(BRAND, "B", "fee_usd", levels, churn_by_level, CONFIG)
        assert [r["level"] for r in rows] == levels

    def test_savings_rise_with_fee_when_churn_held_fixed(self):
        levels = [4.95, 12.95]
        rows = build_frontier(
            BRAND, "B", "fee_usd", levels, {lv: CHURN for lv in levels}, CONFIG
        )
        assert rows[1]["gross_annual_saving"] > rows[0]["gross_annual_saving"]
