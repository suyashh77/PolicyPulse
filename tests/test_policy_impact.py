"""Tests for the policy -> sentiment mapping and its effect on the simulation.

The regression test that matters here is `test_higher_fee_ends_more_negative`.
Its absence is why the original build shipped green with a simulation that
ignored `policy_variables` entirely: every other test asserted structure
(500 agents, 45 rounds, no duplicate posts) and none asserted that changing the
policy changed the answer.
"""
import pytest

from core.policy_impact import (
    keepit_qualifying_fraction,
    policy_shock,
    policy_shocks_by_persona,
    population_shock,
    store_coverage,
)
from core.simulation import load_personas, run_simulation

CONFIG = load_personas("config/personas.yaml")


class TestShockShape:
    def test_fee_policies_are_negative(self):
        for policy_type, variables in [
            ("A", {"distance_miles": 10, "fee_usd": 9.95}),
            ("B", {"fee_usd": 9.95}),
        ]:
            shocks = policy_shocks_by_persona(policy_type, variables, CONFIG)
            assert all(v < 0 for v in shocks.values()), (policy_type, shocks)

    def test_keepit_is_positive_except_sustainability(self):
        shocks = policy_shocks_by_persona("C", {"threshold_usd": 35}, CONFIG)
        assert shocks["sustainability"] < 0, "waste aversion should make keepit unwelcome"
        for persona, value in shocks.items():
            if persona != "sustainability":
                assert value > 0, (persona, value)

    def test_shock_stays_in_range(self):
        for policy_type, variables in [
            ("A", {"distance_miles": 5, "fee_usd": 12.95}),
            ("B", {"fee_usd": 12.95}),
            ("C", {"threshold_usd": 50}),
        ]:
            for value in policy_shocks_by_persona(policy_type, variables, CONFIG).values():
                assert -1.0 <= value <= 1.0

    def test_unknown_policy_type_raises(self):
        with pytest.raises(ValueError):
            policy_shock("Z", {}, CONFIG["personas"]["loyal"], "loyal")

    def test_missing_economic_params_raises(self):
        with pytest.raises(KeyError):
            policy_shock("B", {"fee_usd": 9.95}, {"susceptibility": 0.5}, "broken")


class TestShockOrdering:
    def test_higher_fee_hurts_more(self):
        cheap = policy_shocks_by_persona("B", {"fee_usd": 4.95}, CONFIG)
        dear = policy_shocks_by_persona("B", {"fee_usd": 12.95}, CONFIG)
        for persona in cheap:
            assert dear[persona] < cheap[persona], persona

    def test_deal_seeker_hurts_more_than_loyal(self):
        """Same fee, smaller basket, higher sensitivity -> sharper reaction."""
        shocks = policy_shocks_by_persona("B", {"fee_usd": 9.95}, CONFIG)
        assert shocks["deal_seeker"] < shocks["loyal"]

    def test_wider_radius_softens_distance_gated_policy(self):
        near = policy_shocks_by_persona("A", {"distance_miles": 5, "fee_usd": 9.95}, CONFIG)
        far = policy_shocks_by_persona("A", {"distance_miles": 25, "fee_usd": 9.95}, CONFIG)
        for persona in near:
            assert far[persona] > near[persona], persona

    def test_higher_keepit_threshold_is_more_generous(self):
        low = policy_shocks_by_persona("C", {"threshold_usd": 15}, CONFIG)
        high = policy_shocks_by_persona("C", {"threshold_usd": 50}, CONFIG)
        assert high["deal_seeker"] > low["deal_seeker"]
        # And more waste for the cohort that dislikes it.
        assert high["sustainability"] < low["sustainability"]


class TestHelpers:
    def test_store_coverage_monotonic_and_bounded(self):
        access = 0.6
        values = [store_coverage(d, access) for d in (0, 5, 10, 25, 100)]
        assert values == sorted(values)
        assert values[0] == pytest.approx(0.0)
        assert all(v <= access for v in values)

    def test_keepit_fraction_bounded(self):
        assert keepit_qualifying_fraction(0, 50) == pytest.approx(0.0)
        assert 0.0 < keepit_qualifying_fraction(25, 50) < 1.0
        assert keepit_qualifying_fraction(10_000, 50) == pytest.approx(1.0, abs=1e-6)

    def test_population_shock_is_weighted_mean(self):
        shocks = policy_shocks_by_persona("B", {"fee_usd": 9.95}, CONFIG)
        pop = population_shock(shocks, CONFIG)
        assert min(shocks.values()) <= pop <= max(shocks.values())


class TestSimulationRespondsToPolicy:
    """The tests that would have caught the original defect."""

    def test_higher_fee_ends_more_negative(self):
        cheap = run_simulation("B", {"fee_usd": 4.95}, seed=7, personas_config=CONFIG)
        dear = run_simulation("B", {"fee_usd": 12.95}, seed=7, personas_config=CONFIG)

        cheap_final = cheap.round_summaries[-1]["avg_policy_sentiment"]
        dear_final = dear.round_summaries[-1]["avg_policy_sentiment"]

        assert dear_final < cheap_final, (
            f"a $12.95 fee ended at {dear_final:+.4f} but a $4.95 fee ended at "
            f"{cheap_final:+.4f} - the simulation is not reading its policy input"
        )

    def test_higher_fee_drives_more_churn(self):
        cheap = run_simulation("B", {"fee_usd": 4.95}, seed=7, personas_config=CONFIG)
        dear = run_simulation("B", {"fee_usd": 12.95}, seed=7, personas_config=CONFIG)
        assert (
            dear.round_summaries[-1]["avg_churn_intent"]
            > cheap.round_summaries[-1]["avg_churn_intent"]
        )

    def test_keepit_ends_positive_while_fee_ends_negative(self):
        keepit = run_simulation("C", {"threshold_usd": 50}, seed=7, personas_config=CONFIG)
        fee = run_simulation("B", {"fee_usd": 12.95}, seed=7, personas_config=CONFIG)
        assert keepit.round_summaries[-1]["avg_policy_sentiment"] > 0
        assert fee.round_summaries[-1]["avg_policy_sentiment"] < 0

    def test_distance_gate_matters(self):
        near = run_simulation(
            "A", {"distance_miles": 5, "fee_usd": 9.95}, seed=7, personas_config=CONFIG
        )
        far = run_simulation(
            "A", {"distance_miles": 25, "fee_usd": 9.95}, seed=7, personas_config=CONFIG
        )
        assert (
            far.round_summaries[-1]["avg_policy_sentiment"]
            > near.round_summaries[-1]["avg_policy_sentiment"]
        )

    def test_deal_seekers_churn_more_than_loyal_under_a_fee(self):
        run = run_simulation("B", {"fee_usd": 12.95}, seed=7, personas_config=CONFIG)
        final = run.round_summaries[-1]["breakdown_by_persona"]
        assert final["deal_seeker"]["avg_churn"] > final["loyal"]["avg_churn"]
