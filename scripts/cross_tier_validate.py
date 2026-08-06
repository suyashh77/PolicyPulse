"""Cross-tier validation: run the same policy through both tiers and compare.

Runs on the project's main 3.12 interpreter. It reads a *saved* OASIS database
rather than driving OASIS, so it does not need the 3.11 environment:

    python scripts/run_oasis_spike.py ...        # in the 3.11 env, writes a .db
    python scripts/cross_tier_validate.py --db runs/oasis/oasis_B_11.db
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="OASIS run database")
    parser.add_argument("--manifest", default=None, help="matching run manifest json")
    parser.add_argument("--scorer", default="llm", choices=["llm", "lexicon"])
    parser.add_argument("--scorer-model", default="claude-opus-5")
    parser.add_argument("--scorer-budget", type=float, default=1.00)
    parser.add_argument("--seeds", type=int, default=10, help="Tier-1 seeds to average")
    parser.add_argument("--min-posts", type=int, default=2)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    from core.simulation import load_personas, run_batch
    from oasis_tier.extract import extract_report
    from oasis_tier.scoring import make_scorer
    from oasis_tier.validate import compare_tiers, format_comparison

    manifest = {}
    manifest_path = args.manifest
    if manifest_path is None:
        # Convention: the manifest sits beside the db with a matching policy/seed.
        candidates = sorted(Path(args.db).parent.glob("*.json"))
        candidates = [c for c in candidates if not c.name.endswith("_report.json")]
        if candidates:
            manifest_path = str(candidates[-1])
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    cfg = manifest.get("config", {})
    policy_type = cfg.get("policy_type", "B")
    policy_variables = cfg.get("policy_variables", {"fee_usd": 12.95})

    print(f"Tier 2 db:     {args.db}")
    print(f"policy:        Type {policy_type}  {policy_variables}")
    print(f"scorer:        {args.scorer}"
          + (f" ({args.scorer_model})" if args.scorer == "llm" else ""))
    print()

    # --- Tier 2 ---
    scorer = (
        make_scorer("llm", model=args.scorer_model, budget_usd=args.scorer_budget)
        if args.scorer == "llm"
        else make_scorer("lexicon")
    )
    tier2 = extract_report(args.db, manifest, scorer=scorer)
    print(f"Tier 2: {tier2['n_utterances']} utterances "
          f"({tier2['n_original_posts']} posts, {tier2['n_comments']} comments, "
          f"{tier2['n_reposts']} reposts)")
    if hasattr(scorer, "meter"):
        print(f"        scoring cost ${scorer.meter.cost_usd:.4f} "
              f"in {scorer.meter.calls} call(s)")

    # --- Tier 1 ---
    personas_config = load_personas(str(ROOT / "config" / "personas.yaml"))
    runs = run_batch(
        policy_type,
        policy_variables,
        seeds=list(range(1, args.seeds + 1)),
        personas_config=personas_config,
    )
    print(f"Tier 1: {len(runs)} seeds x 500 agents x 45 rounds\n")

    # Average Tier 1 across seeds by rewriting the final breakdown in place.
    merged = runs[0]
    final = merged.round_summaries[-1]["breakdown_by_persona"]
    for persona in final:
        final[persona]["avg_sentiment"] = sum(
            r.round_summaries[-1]["breakdown_by_persona"][persona]["avg_sentiment"]
            for r in runs
        ) / len(runs)

    comparison = compare_tiers(merged, tier2, min_posts=args.min_posts)
    print(format_comparison(comparison))

    out = ROOT / "runs" / "oasis" / f"cross_tier_{Path(args.db).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
