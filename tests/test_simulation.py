import random

import pytest

from core.agent import Agent
from core.simulation import (
    ANNOUNCEMENT_REACH,
    POST_TTL_ROUNDS,
    ROUNDS,
    load_personas,
    record_round_summary,
    run_batch,
    run_simulation,
    summarize_batch,
)

CONFIG = load_personas("config/personas.yaml")


class TestRecordRoundSummary:
    def test_returns_correct_structure(self):
        agents = [
            Agent(id=0, persona="loyal", reach=100, susceptibility=0.2,
                  churn_elasticity=0.1, policy_sentiment=-0.2, churn_intent=0.1),
            Agent(id=1, persona="casual", reach=200, susceptibility=0.45,
                  churn_elasticity=0.35, policy_sentiment=0.3, churn_intent=0.0),
        ]
        summary = record_round_summary(agents, round_num=5)
        assert summary["round"] == 5
        assert "avg_policy_sentiment" in summary
        assert "avg_churn_intent" in summary
        assert "breakdown_by_persona" in summary
        assert "loyal" in summary["breakdown_by_persona"]
        assert "casual" in summary["breakdown_by_persona"]

    def test_averages_correct(self):
        agents = [
            Agent(id=0, persona="loyal", reach=100, policy_sentiment=-0.4, churn_intent=0.2),
            Agent(id=1, persona="loyal", reach=100, policy_sentiment=-0.6, churn_intent=0.4),
        ]
        summary = record_round_summary(agents, round_num=1)
        assert summary["avg_policy_sentiment"] == pytest.approx(-0.5)
        assert summary["avg_churn_intent"] == pytest.approx(0.3)


class TestRunSimulation:
    def test_completes_45_rounds(self):
        sim = run_simulation("B", {"fee_usd": 9.95}, seed=42, personas_config=CONFIG)
        assert sim.completed is True
        assert len(sim.round_summaries) == ROUNDS
        assert sim.round_summaries[0]["round"] == 1
        assert sim.round_summaries[-1]["round"] == ROUNDS

    def test_has_500_agents(self):
        sim = run_simulation(
            "A", {"distance_miles": 10, "fee_usd": 9.95}, seed=42, personas_config=CONFIG
        )
        assert len(sim.agents) == 500

    def test_posts_generated(self):
        sim = run_simulation("C", {"threshold_usd": 25}, seed=42, personas_config=CONFIG)
        assert len(sim.posts) > 100

    def test_announcement_post_present(self):
        sim = run_simulation("B", {"fee_usd": 4.95}, seed=42, personas_config=CONFIG)
        announcement = sim.posts[0]
        assert announcement.agent_id == -1
        assert announcement.round == 1
        assert announcement.reach == ANNOUNCEMENT_REACH
        # No longer hardcoded to 0.0 - it carries the population's reaction.
        assert announcement.sentiment < 0.0

    def test_announcement_reach_below_top_influencer(self):
        """A 999,999-reach announcement never left the reach-weighted bucket."""
        max_influencer_reach = CONFIG["personas"]["influencer"]["reach_range"][1]
        assert ANNOUNCEMENT_REACH <= max_influencer_reach

    def test_sentiment_never_saturates_at_the_clamp(self):
        sim = run_simulation("B", {"fee_usd": 12.95}, seed=3, personas_config=CONFIG)
        finals = [s["avg_policy_sentiment"] for s in sim.round_summaries]
        assert all(abs(v) < 0.99 for v in finals), (
            "population pinned at the clamp - the herding loop has no restoring force"
        )

    def test_records_persona_shocks(self):
        sim = run_simulation("B", {"fee_usd": 9.95}, seed=1, personas_config=CONFIG)
        assert set(sim.persona_shocks) == set(CONFIG["personas"])
        assert all(v < 0 for v in sim.persona_shocks.values())


class TestReproducibility:
    def test_same_seed_reproduces_exactly(self):
        a = run_simulation("B", {"fee_usd": 9.95}, seed=123, personas_config=CONFIG)
        b = run_simulation("B", {"fee_usd": 9.95}, seed=123, personas_config=CONFIG)
        assert [s["avg_policy_sentiment"] for s in a.round_summaries] == [
            s["avg_policy_sentiment"] for s in b.round_summaries
        ]

    def test_different_seeds_differ(self):
        a = run_simulation("B", {"fee_usd": 9.95}, seed=1, personas_config=CONFIG)
        b = run_simulation("B", {"fee_usd": 9.95}, seed=2, personas_config=CONFIG)
        assert (
            a.round_summaries[-1]["avg_policy_sentiment"]
            != b.round_summaries[-1]["avg_policy_sentiment"]
        )

    def test_seed_is_recorded(self):
        sim = run_simulation("B", {"fee_usd": 9.95}, seed=77, personas_config=CONFIG)
        assert sim.seed == 77


class TestPostPool:
    def test_pool_is_pruned_by_ttl(self):
        """Stale posts must leave the pool or day-45 feeds read day-2 opinions."""
        sim = run_simulation("B", {"fee_usd": 9.95}, seed=5, personas_config=CONFIG)
        seen_ids = set()
        for agent in sim.agents:
            for entry in agent.memory[-1:]:
                seen_ids.update(entry["posts_seen_ids"])

        posts_by_id = {p.id: p for p in sim.posts}
        for pid in seen_ids:
            post = posts_by_id[pid]
            assert ROUNDS - post.round <= POST_TTL_ROUNDS


class TestBatch:
    def test_run_batch_returns_one_run_per_seed(self):
        runs = run_batch("B", {"fee_usd": 9.95}, seeds=[1, 2, 3], personas_config=CONFIG)
        assert len(runs) == 3
        assert [r.seed for r in runs] == [1, 2, 3]

    def test_summarize_batch_reports_spread(self):
        runs = run_batch("B", {"fee_usd": 9.95}, seeds=[1, 2, 3, 4], personas_config=CONFIG)
        stats = summarize_batch(runs)
        assert stats["n_runs"] == 4
        assert stats["final_sentiment_stdev"] >= 0.0
        assert -1.0 <= stats["final_sentiment_mean"] <= 1.0
