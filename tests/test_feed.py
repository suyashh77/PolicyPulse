import random

from core.agent import Agent, Post
from core.feed import FEED_SIZE, build_feed_index, sample_feed


def _make_post(pid, sentiment=0.0, reach=100, persona="casual", rnd=2):
    return Post(id=pid, agent_id=pid, round=rnd, sentiment=sentiment, reach=reach, persona=persona)


class TestSampleFeed:
    def test_returns_10_posts(self):
        random.seed(42)
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        pool = [_make_post(i) for i in range(50)]
        feed = sample_feed(agent, pool, round_num=2)
        assert len(feed) == FEED_SIZE

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
        assert same_count / trials > 2.0

    def test_reach_bias(self):
        """High-reach posts should dominate the feed."""
        random.seed(42)
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        pool = [_make_post(i, reach=10, persona="loyal") for i in range(90)]
        pool += [_make_post(i, reach=100_000, persona="loyal") for i in range(90, 100)]

        loud = 0
        trials = 50
        for _ in range(trials):
            feed = sample_feed(agent, pool, round_num=2)
            loud += sum(1 for p in feed if p.reach == 100_000)
        # 10% of the pool by count, but should take well over 10% of feed slots.
        assert loud / trials > 3.0

    def test_empty_pool(self):
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        feed = sample_feed(agent, [], round_num=2)
        assert len(feed) == 0


class TestFeedIndex:
    def test_index_produces_same_shape_as_standalone(self):
        random.seed(1)
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        pool = [_make_post(i, reach=random.randint(1, 5000)) for i in range(200)]
        index = build_feed_index(pool, round_num=3)

        for _ in range(20):
            feed = sample_feed(agent, pool, round_num=3, index=index)
            assert len(feed) == FEED_SIZE
            assert len({p.id for p in feed}) == FEED_SIZE

    def test_stale_index_is_rebuilt(self):
        """A mismatched round must not silently produce a wrong-round feed."""
        random.seed(1)
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        pool = [_make_post(i) for i in range(50)]
        stale = build_feed_index(pool, round_num=2)
        feed = sample_feed(agent, pool, round_num=9, index=stale)
        assert len(feed) == FEED_SIZE

    def test_recent_posts_outweigh_old_ones(self):
        """Reach decays with age, so a stale post loses to a fresh one of equal reach."""
        random.seed(7)
        agent = Agent(id=0, persona="casual", reach=100, susceptibility=0.5, churn_elasticity=0.3)
        pool = [_make_post(i, reach=1000, persona="loyal", rnd=1) for i in range(50)]
        pool += [_make_post(i, reach=1000, persona="loyal", rnd=10) for i in range(50, 100)]

        fresh = 0
        trials = 50
        for _ in range(trials):
            feed = sample_feed(agent, pool, round_num=10)
            fresh += sum(1 for p in feed if p.round == 10)
        assert fresh / trials > 5.0
