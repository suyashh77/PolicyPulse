"""Tier-2 tests that never call an API.

These run on the project's 3.12 interpreter alongside the Tier-1 suite, even
though OASIS itself needs 3.11 and is not installed there. Everything tested
here — profile generation, the follower graph, cost accounting, budget
enforcement, and report extraction — is pure Python that does not import
`oasis`. The parts that do are exercised by `scripts/run_oasis_spike.py`, which
costs money and is therefore run by hand, never in CI.
"""
import sqlite3

import pytest
import yaml

from core.policy_impact import policy_shocks_by_persona
from oasis_tier.cost import BudgetExceeded, CostMeter, extrapolate, metered_backend
from oasis_tier.extract import extract_report, score_text_lexicon, top_quotes
from oasis_tier.profiles import build_profiles, graph_stats

CONFIG = yaml.safe_load(open("config/personas.yaml"))
SHOCKS = policy_shocks_by_persona("B", {"fee_usd": 12.95}, CONFIG)


class TestProfiles:
    def test_builds_requested_number_of_agents(self):
        profiles = build_profiles(CONFIG, SHOCKS, n_agents=40)
        assert len(profiles) == 40

    def test_preserves_persona_mix_proportions(self):
        profiles = build_profiles(CONFIG, SHOCKS, n_agents=100)
        counts = {p: 0 for p in CONFIG["personas"]}
        for prof in profiles:
            counts[prof.persona] += 1
        # loyal is the largest cohort (0.30) and influencer the smallest (0.10)
        assert counts["loyal"] > counts["influencer"]
        assert all(c > 0 for c in counts.values())

    def test_bio_carries_the_numeric_shock_as_language(self):
        """An LLM cannot read `susceptibility: 0.8`; it can read a stance."""
        profiles = build_profiles(CONFIG, SHOCKS, n_agents=40)
        deal = next(p for p in profiles if p.persona == "deal_seeker")
        loyal = next(p for p in profiles if p.persona == "loyal")
        assert "strongly negative" in deal.bio      # shock -0.569
        assert "strongly negative" not in loyal.bio  # shock -0.067
        assert deal.baseline_sentiment < loyal.baseline_sentiment

    def test_reach_within_persona_range(self):
        profiles = build_profiles(CONFIG, SHOCKS, n_agents=60)
        for prof in profiles:
            lo, hi = CONFIG["personas"][prof.persona]["reach_range"]
            assert lo <= prof.reach <= hi

    def test_deterministic_for_a_given_seed(self):
        import random

        a = build_profiles(CONFIG, SHOCKS, n_agents=30, rng=random.Random(5))
        b = build_profiles(CONFIG, SHOCKS, n_agents=30, rng=random.Random(5))
        assert [p.reach for p in a] == [p.reach for p in b]
        assert [p.follows for p in a] == [p.follows for p in b]


class TestFollowerGraph:
    def test_every_agent_follows_someone(self):
        profiles = build_profiles(CONFIG, SHOCKS, n_agents=40)
        assert all(p.follows for p in profiles)

    def test_nobody_follows_themselves(self):
        profiles = build_profiles(CONFIG, SHOCKS, n_agents=40)
        for p in profiles:
            assert p.agent_id not in p.follows

    def test_high_reach_agents_become_hubs(self):
        """Reach must translate into actual network structure, not a weight."""
        profiles = build_profiles(CONFIG, SHOCKS, n_agents=60)
        followers = {p.agent_id: 0 for p in profiles}
        for p in profiles:
            for t in p.follows:
                followers[t] += 1

        by_reach = sorted(profiles, key=lambda p: p.reach, reverse=True)
        top = by_reach[0]
        median = by_reach[len(by_reach) // 2]
        assert followers[top.agent_id] > followers[median.agent_id]

    def test_homophily_edges_exist(self):
        profiles = build_profiles(CONFIG, SHOCKS, n_agents=60)
        by_id = {p.agent_id: p for p in profiles}
        same = sum(
            1 for p in profiles for t in p.follows if by_id[t].persona == p.persona
        )
        assert same > 0

    def test_graph_stats_shape(self):
        profiles = build_profiles(CONFIG, SHOCKS, n_agents=40)
        stats = graph_stats(profiles)
        assert stats["n_agents"] == 40
        assert stats["n_edges"] > 0
        assert sum(stats["persona_mix"].values()) == 40


class TestCostMeter:
    def test_accumulates_openai_style_usage(self):
        meter = CostMeter(model="claude-haiku-4-5", budget_usd=10)
        meter.record({"prompt_tokens": 1000, "completion_tokens": 500})
        assert meter.calls == 1
        assert meter.input_tokens == 1000
        assert meter.output_tokens == 500

    def test_accumulates_anthropic_style_usage(self):
        meter = CostMeter(model="claude-haiku-4-5", budget_usd=10)
        meter.record({"input_tokens": 2000, "output_tokens": 1000})
        assert meter.input_tokens == 2000
        assert meter.output_tokens == 1000

    def test_cost_matches_published_rates(self):
        meter = CostMeter(model="claude-haiku-4-5", budget_usd=10)
        meter.record({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
        assert meter.cost_usd == pytest.approx(1.00 + 5.00)

    def test_none_usage_is_ignored(self):
        meter = CostMeter(model="claude-haiku-4-5")
        meter.record(None)
        assert meter.calls == 0

    def test_budget_is_enforced(self):
        meter = CostMeter(model="claude-opus-5", budget_usd=0.01)
        meter.record({"input_tokens": 1_000_000, "output_tokens": 0})  # $5.00
        with pytest.raises(BudgetExceeded):
            meter.check_budget()

    def test_unknown_model_is_flagged_not_silently_free(self):
        meter = CostMeter(model="some-new-model")
        meter.record({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
        assert meter.snapshot()["priced"] is False

    def test_metered_backend_records_and_preserves_identity(self):
        """CAMEL isinstance-checks the backend, so it must be the same object."""

        class FakeUsage:
            input_tokens, output_tokens = 100, 50

        class FakeResult:
            usage = FakeUsage()

        class FakeBackend:
            def run(self, *a, **k):
                return FakeResult()

        meter = CostMeter(model="claude-haiku-4-5", budget_usd=10)
        backend = FakeBackend()
        wrapped = metered_backend(backend, meter)

        assert wrapped is backend
        wrapped.run()
        wrapped.run()
        assert meter.calls == 2
        assert meter.input_tokens == 200

    def test_metered_backend_raises_when_over_budget(self):
        class FakeResult:
            usage = {"input_tokens": 1_000_000, "output_tokens": 0}

        class FakeBackend:
            def run(self, *a, **k):
                return FakeResult()

        meter = CostMeter(model="claude-opus-5", budget_usd=0.01)
        wrapped = metered_backend(FakeBackend(), meter)
        with pytest.raises(BudgetExceeded):
            wrapped.run()

    def test_extrapolation_scales_linearly(self):
        meter = CostMeter(model="claude-haiku-4-5", budget_usd=100)
        meter.record({"input_tokens": 100_000, "output_tokens": 10_000})  # $0.15
        proj = extrapolate(meter, agent_steps_done=10, target_agent_steps=1000)
        assert proj["cost_per_agent_step_usd"] == pytest.approx(0.015)
        assert proj["projected_cost_usd"] == pytest.approx(15.0)

    def test_extrapolation_guards_zero_steps(self):
        assert "error" in extrapolate(CostMeter(model="x"), 0, 100)


class TestLexiconScorer:
    def test_complaint_scores_negative(self):
        assert score_text_lexicon("This fee is a ridiculous cash grab") < 0

    def test_approval_scores_positive(self):
        assert score_text_lexicon("Honestly this seems fair and reasonable") > 0

    def test_neutral_scores_zero(self):
        assert score_text_lexicon("I saw the announcement this morning.") == 0.0

    def test_empty_is_zero(self):
        assert score_text_lexicon("") == 0.0

    def test_bounded(self):
        rant = "unfair ridiculous outrageous greedy scam terrible awful hate " * 5
        assert -1.0 <= score_text_lexicon(rant) <= 1.0

    def test_negation_flips_polarity(self):
        assert score_text_lexicon("this is not fair") < score_text_lexicon("this is fair")


def _make_oasis_db(path, rows):
    """Build a minimal OASIS-shaped database for extraction tests."""
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE user (user_id INTEGER PRIMARY KEY, agent_id INTEGER, "
        "user_name TEXT, name TEXT, bio TEXT, created_at DATETIME)"
    )
    con.execute(
        "CREATE TABLE post (post_id INTEGER PRIMARY KEY, user_id INTEGER, "
        "original_post_id INTEGER, content TEXT, quote_content TEXT, "
        "created_at DATETIME, num_likes INTEGER, num_dislikes INTEGER, "
        "num_shares INTEGER, num_reports INTEGER)"
    )
    con.execute("CREATE TABLE comment (comment_id INTEGER PRIMARY KEY, user_id INTEGER)")
    con.execute("CREATE TABLE trace (user_id INTEGER, created_at DATETIME, action TEXT, info TEXT)")

    for uid, uname in rows["users"]:
        con.execute(
            "INSERT INTO user (user_id, agent_id, user_name, name, bio, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (uid, uid, uname, uname, "", "2026-01-01"),
        )
    for pid, uid, content, created in rows["posts"]:
        con.execute(
            "INSERT INTO post (post_id, user_id, original_post_id, content, "
            "quote_content, created_at, num_likes, num_dislikes, num_shares, num_reports) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (pid, uid, None, content, None, created, 0, 0, 0, 0),
        )
    con.commit()
    con.close()


@pytest.fixture
def sample_db(tmp_path):
    path = tmp_path / "oasis.db"
    _make_oasis_db(
        path,
        {
            "users": [
                (1, "brand_official"),
                (2, "deal_seeker_0"),
                (3, "loyal_0"),
                (4, "deal_seeker_1"),
            ],
            "posts": [
                (1, 1, "We are introducing a $12.95 mail-in return fee.", "2026-01-01"),
                (2, 2, "This fee is a ridiculous cash grab, I'm switching to a competitor",
                 "2026-01-02"),
                (3, 3, "Seems reasonable to me, in-store is still free", "2026-01-02"),
                (4, 4, "Outrageous and unfair, canceling my order", "2026-01-03"),
            ],
        },
    )
    return path


class TestExtraction:
    def test_brand_posts_excluded_from_sentiment(self, sample_db):
        report = extract_report(sample_db)
        assert report["n_posts"] == 3  # 4 posts, 1 is the brand's
        assert "brand" not in report["breakdown_by_persona"]

    def test_persona_recovered_from_username(self, sample_db):
        report = extract_report(sample_db)
        assert set(report["breakdown_by_persona"]) == {"deal_seeker", "loyal"}
        assert report["breakdown_by_persona"]["deal_seeker"]["n_posts"] == 2

    def test_sentiment_direction_is_right(self, sample_db):
        report = extract_report(sample_db)
        breakdown = report["breakdown_by_persona"]
        assert breakdown["deal_seeker"]["avg_sentiment"] < 0
        assert breakdown["loyal"]["avg_sentiment"] > 0

    def test_churn_signal_detects_leaving_language(self, sample_db):
        report = extract_report(sample_db)
        assert report["breakdown_by_persona"]["deal_seeker"]["churn_signal_rate"] == 1.0
        assert report["breakdown_by_persona"]["loyal"]["churn_signal_rate"] == 0.0

    def test_emits_tier1_compatible_keys(self, sample_db):
        report = extract_report(sample_db)
        for key in ("sentiment_curve", "breakdown_by_persona", "cascade", "policy_type"):
            assert key in report
        for point in report["sentiment_curve"]:
            assert "day" in point and "avg_sentiment" in point

    def test_manifest_shocks_are_carried_through(self, sample_db):
        manifest = {
            "run_id": "test",
            "persona_shocks": {"deal_seeker": -0.569, "loyal": -0.067},
            "config": {"policy_type": "B", "policy_variables": {"fee_usd": 12.95}},
        }
        report = extract_report(sample_db, manifest)
        assert report["policy_type"] == "B"
        assert report["breakdown_by_persona"]["deal_seeker"]["tier1_baseline_shock"] == -0.569

    def test_top_quotes_orders_by_sentiment(self, sample_db):
        report = extract_report(sample_db)
        quotes = top_quotes(report, n=2)
        assert quotes["most_negative"][0]["sentiment"] <= quotes["most_positive"][0]["sentiment"]

    def test_missing_database_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_report(tmp_path / "nope.db")
