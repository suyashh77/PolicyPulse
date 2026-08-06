import random

import pytest
import yaml

from core.agent import Agent, Post, generate_post, initialize_agents, reset_post_id_counter, update_agent_state


def _load_personas():
    with open("config/personas.yaml") as f:
        return yaml.safe_load(f)


class TestInitializeAgents:
    def test_creates_500_agents(self):
        agents = initialize_agents(_load_personas())
        assert len(agents) == 500

    def test_persona_distribution(self):
        agents = initialize_agents(_load_personas())
        counts = {}
        for a in agents:
            counts[a.persona] = counts.get(a.persona, 0) + 1
        assert counts["loyal"] == 150
        assert counts["casual"] == 125
        assert counts["deal_seeker"] == 100
        assert counts["influencer"] == 50
        # sustainability gets the remainder to hit 500
        assert counts["sustainability"] == 75

    def test_reach_within_range(self):
        random.seed(42)
        agents = initialize_agents(_load_personas())
        personas = _load_personas()["personas"]
        for a in agents:
            lo, hi = personas[a.persona]["reach_range"]
            assert lo <= a.reach <= hi

    def test_initial_state_is_neutral_without_a_policy_shock(self):
        agents = initialize_agents(_load_personas())
        for a in agents:
            assert a.policy_sentiment == 0.0
            assert a.churn_intent == 0.0

    def test_agents_start_at_their_persona_shock(self):
        shocks = {"loyal": -0.1, "casual": -0.3, "deal_seeker": -0.6,
                  "influencer": -0.2, "sustainability": -0.25}
        agents = initialize_agents(_load_personas(), shocks)
        for a in agents:
            assert a.policy_sentiment == shocks[a.persona]
            assert a.baseline_sentiment == shocks[a.persona]

    def test_behavioral_params_are_loaded(self):
        agents = initialize_agents(_load_personas())
        personas = _load_personas()["personas"]
        for a in agents:
            assert a.anchor_strength == personas[a.persona]["anchor_strength"]
            assert a.churn_recovery == personas[a.persona]["churn_recovery"]


class TestUpdateAgentState:
    def test_positive_signal_increases_sentiment(self):
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        posts = [Post(id=1, agent_id=1, round=1, sentiment=0.8, reach=100, persona="casual")]
        update_agent_state(agent, posts, current_round=1)
        assert agent.policy_sentiment > 0.0

    def test_negative_signal_increases_churn(self):
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.5)
        posts = [Post(id=1, agent_id=1, round=1, sentiment=-0.8, reach=100, persona="casual")]
        update_agent_state(agent, posts, current_round=1)
        assert agent.churn_intent > 0.0

    def test_sentiment_converges_toward_feed_rather_than_saturating(self):
        """The additive update marched agents to the clamp; this one converges."""
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5,
                      churn_elasticity=0.3)
        posts = [Post(id=1, agent_id=1, round=1, sentiment=0.4, reach=100, persona="casual")]
        for i in range(200):
            update_agent_state(agent, posts, current_round=i)
        assert agent.policy_sentiment == pytest.approx(0.4, abs=0.01)
        assert agent.policy_sentiment < 1.0

    def test_anchor_pulls_back_toward_baseline(self):
        """With no feed at all, sentiment holds at the policy-implied baseline."""
        agent = Agent(id=0, persona="loyal", reach=100, susceptibility=0.2,
                      churn_elasticity=0.1, policy_sentiment=-0.5,
                      baseline_sentiment=-0.5, anchor_strength=0.15)
        for i in range(20):
            update_agent_state(agent, [], current_round=i)
        assert agent.policy_sentiment == pytest.approx(-0.5)

    def test_churn_recovers_when_sentiment_recovers(self):
        """Churn used to be a one-way ratchet that only ever accumulated."""
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5,
                      churn_elasticity=0.5, churn_recovery=0.2)
        angry = [Post(id=1, agent_id=1, round=1, sentiment=-0.9, reach=100, persona="casual")]
        for i in range(40):
            update_agent_state(agent, angry, current_round=i)
        peak = agent.churn_intent
        assert peak > 0.0

        happy = [Post(id=2, agent_id=2, round=1, sentiment=0.9, reach=100, persona="casual")]
        for i in range(80):
            update_agent_state(agent, happy, current_round=40 + i)
        assert agent.churn_intent < peak

    def test_state_stays_clamped(self):
        agent = Agent(id=0, persona="deal_seeker", reach=100, susceptibility=1.0,
                      churn_elasticity=0.5)
        posts = [Post(id=1, agent_id=1, round=1, sentiment=-1.0, reach=100, persona="deal_seeker")]
        for i in range(50):
            update_agent_state(agent, posts, current_round=i)
        assert -1.0 <= agent.policy_sentiment <= 1.0
        assert 0.0 <= agent.churn_intent <= 1.0

    def test_memory_appended(self):
        agent = Agent(id=0, persona="loyal", reach=100, susceptibility=0.2, churn_elasticity=0.1)
        posts = [Post(id=1, agent_id=1, round=3, sentiment=0.5, reach=100, persona="loyal")]
        update_agent_state(agent, posts, current_round=3)
        assert len(agent.memory) == 1
        assert agent.memory[0]["round"] == 3

    def test_empty_posts_leave_sentiment_unchanged(self):
        agent = Agent(id=0, persona="loyal", reach=100, susceptibility=0.2, churn_elasticity=0.1)
        update_agent_state(agent, [], current_round=1)
        assert agent.policy_sentiment == 0.0
        assert agent.churn_intent == 0.0
        assert len(agent.memory) == 1  # the round is still recorded


class TestGeneratePost:
    def test_always_posts_for_influencer(self):
        reset_post_id_counter()
        random.seed(42)
        agent = Agent(
            id=0, persona="influencer", reach=50000,
            susceptibility=0.5, churn_elasticity=0.3,
            post_probability=1.0, post_variance=0.1,
            policy_sentiment=-0.3,
        )
        post = generate_post(agent, current_round=5)
        assert post is not None
        assert post.agent_id == 0
        assert post.round == 5
        assert -1.0 <= post.sentiment <= 1.0

    def test_never_posts_at_zero_probability(self):
        agent = Agent(
            id=0, persona="test", reach=100,
            post_probability=0.0, post_variance=0.1,
        )
        for _ in range(100):
            assert generate_post(agent, current_round=1) is None
