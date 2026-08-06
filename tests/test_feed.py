import random

from core.agent import Agent, Post
from core.feed import sample_feed


def _make_post(pid, sentiment=0.0, reach=100, persona="casual", rnd=2):
    return Post(id=pid, agent_id=pid, round=rnd, sentiment=sentiment, reach=reach, persona=persona)


class TestSampleFeed:
    def test_returns_10_posts(self):
        random.seed(42)
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        pool = [_make_post(i) for i in range(50)]
        feed = sample_feed(agent, pool, round_num=2)
        assert len(feed) == 10

    def test_returns_all_when_pool_small(self):
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        pool = [_make_post(i) for i in range(5)]
        feed = sample_feed(agent, pool, round_num=2)
        assert len(feed) == 5

    def test_no_duplicates(self):
        random.seed(42)
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        pool = [_make_post(i, reach=random.randint(1, 10000)) for i in range(100)]
        feed = sample_feed(agent, pool, round_num=2)
        ids = [p.id for p in feed]
        assert len(ids) == len(set(ids))

    def test_homophily_bias(self):
        """Posts from same persona should appear more often than pure random."""
        random.seed(42)
        agent = Agent(id=0, persona="deal_seeker", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        pool = []
        for i in range(80):
            pool.append(_make_post(i, persona="casual"))
        for i in range(80, 100):
            pool.append(_make_post(i, persona="deal_seeker"))

        same_count = 0
        trials = 100
        for _ in range(trials):
            feed = sample_feed(agent, pool, round_num=2)
            same_count += sum(1 for p in feed if p.persona == "deal_seeker")

        # With 20% same-persona in pool, homophily should push above 20% in feed
        avg_same = same_count / trials
        assert avg_same > 2.0  # expect > 20% of 10

    def test_empty_pool(self):
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        feed = sample_feed(agent, [], round_num=2)
        assert len(feed) == 0
