import json

import pytest

from core.simulation import load_personas, run_batch, run_simulation
from report.cascade import detect_cascade
from report.curves import (
    aggregate_churn_by_segment,
    aggregate_sentiment_curve,
    build_churn_by_segment,
    build_sentiment_curve,
)
from report.persistence import load_batches, save_batch
from report.report_agent import generate_report

CONFIG = load_personas("config/personas.yaml")


@pytest.fixture(scope="module")
def batch():
    return run_batch("B", {"fee_usd": 12.95}, seeds=[1, 2, 3], personas_config=CONFIG)


class TestCurves:
    def test_sentiment_curve_covers_every_round(self, batch):
        curve = build_sentiment_curve(batch[0])
        assert len(curve) == 45
        assert curve[0]["day"] == 1
        assert curve[-1]["day"] == 45

    def test_churn_by_segment_has_all_personas(self, batch):
        churn = build_churn_by_segment(batch[0])
        assert set(churn) == set(CONFIG["personas"])

    def test_aggregate_curve_reports_a_band(self, batch):
        curve = aggregate_sentiment_curve(batch)
        assert len(curve) == 45
        for point in curve:
            assert point["n_runs"] == 3
            assert point["lower"] <= point["avg_sentiment"] <= point["upper"]

    def test_aggregate_curve_handles_empty_input(self):
        assert aggregate_sentiment_curve([]) == []

    def test_aggregate_churn_reports_mean_and_stdev(self, batch):
        agg = aggregate_churn_by_segment(batch)
        assert set(agg) == set(CONFIG["personas"])
        for stats in agg.values():
            assert stats["mean"] >= 0.0
            assert stats["stdev"] >= 0.0


class TestCascade:
    def test_no_cascade_on_a_stable_run(self, batch):
        # A flat-fee policy produces a step at day 1, not a 0.4 drop mid-window.
        assert detect_cascade(batch[0])["cascade"] is False

    def test_detects_an_engineered_drop(self):
        run = run_simulation("B", {"fee_usd": 4.95}, seed=1, personas_config=CONFIG)
        for i, summary in enumerate(run.round_summaries):
            summary["avg_policy_sentiment"] = 0.5 - (i * 0.05)
        result = detect_cascade(run)
        assert result["cascade"] is True
        assert result["trigger_day"] == 1


class TestReportAgent:
    def test_report_contains_expected_keys(self, batch):
        report = generate_report(batch[0], batch)
        for key in (
            "run_id", "policy_type", "policy_variables", "seed", "persona_shocks",
            "sentiment_curve", "churn_by_segment", "cascade",
            "threshold_comparison", "aggregate_sentiment_curve",
        ):
            assert key in report

    def test_threshold_comparison_groups_by_variables_not_by_run(self, batch):
        """One row per policy setting, not one row per random seed."""
        other = run_batch("B", {"fee_usd": 4.95}, seeds=[1, 2, 3], personas_config=CONFIG)
        report = generate_report(batch[0], batch + other)
        comparison = report["threshold_comparison"]

        assert len(comparison) == 2, "expected one row per fee level, not per seed"
        assert all(row["n_runs"] == 3 for row in comparison)

        # Sorted worst-first, so the expensive fee leads.
        assert comparison[0]["variables"]["fee_usd"] == 12.95
        assert comparison[0]["final_sentiment"] < comparison[1]["final_sentiment"]

    def test_excludes_other_policy_types(self, batch):
        keepit = run_batch("C", {"threshold_usd": 25}, seeds=[1], personas_config=CONFIG)
        report = generate_report(batch[0], batch + keepit)
        for row in report["threshold_comparison"]:
            assert "threshold_usd" not in row["variables"]


class TestPersistence:
    def test_save_and_reload_roundtrip(self, batch, tmp_path):
        path = save_batch(batch, runs_dir=tmp_path)
        assert path.exists()

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["policy_type"] == "B"
        assert payload["policy_variables"] == {"fee_usd": 12.95}
        assert len(payload["runs"]) == 3
        assert len(payload["runs"][0]["round_summaries"]) == 45

        loaded = load_batches(tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["runs"][0]["seed"] == 1

    def test_missing_directory_returns_empty(self, tmp_path):
        assert load_batches(tmp_path / "does-not-exist") == []

    def test_malformed_file_is_skipped(self, batch, tmp_path):
        save_batch(batch, runs_dir=tmp_path)
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        assert len(load_batches(tmp_path)) == 1

    def test_empty_batch_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            save_batch([], runs_dir=tmp_path)
