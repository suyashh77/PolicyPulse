import random

from core.simulation import SimulationRun, record_round_summary, run_simulation
from core.agent import Agent


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
        assert abs(summary["avg_policy_sentiment"] - (-0.5)) < 1e-9
        assert abs(summary["avg_churn_intent"] - 0.3) < 1e-9


class TestRunSimulation:
    def test_completes_45_rounds(self):
        random.seed(42)
        sim = run_simulation(
            policy_type="B",
            policy_variables={"fee_usd": 9.95},
        )
        assert sim.completed is True
        assert len(sim.round_summaries) == 45
        assert sim.round_summaries[0]["round"] == 1
        assert sim.round_summaries[-1]["round"] == 45

    def test_has_500_agents(self):
        random.seed(42)
        sim = run_simulation(
            policy_type="A",
            policy_variables={"distance_miles": 10, "fee_usd": 9.95},
        )
        assert len(sim.agents) == 500

    def test_posts_generated(self):
        random.seed(42)
        sim = run_simulation(
            policy_type="C",
            policy_variables={"threshold_usd": 25},
        )
        # Should have at least the announcement + many agent posts
        assert len(sim.posts) > 100

    def test_announcement_post_present(self):
        random.seed(42)
        sim = run_simulation(
            policy_type="B",
            policy_variables={"fee_usd": 4.95},
        )
        announcement = sim.posts[0]
        assert announcement.agent_id == -1
        assert announcement.round == 1
        assert announcement.reach == 999_999
        assert announcement.sentiment == 0.0
