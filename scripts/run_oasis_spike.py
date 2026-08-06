"""Tier-2 spike: run a small OASIS simulation and report measured cost.

Must run under the Python 3.11 env that has camel-oasis:

    conda activate policypulse-oasis
    python scripts/run_oasis_spike.py --agents 12 --days 4 --budget 0.50

The point of this script is the *measured* cost per agent-step. Everything
downstream about scale depends on that number being real rather than estimated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Agent posts contain emoji and smart quotes; the Windows console defaults to
# cp1252 and raises on them mid-report.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass


def _load_env() -> None:
    """Minimal .env loader (python-dotenv may not be in the OASIS env)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=12)
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--platform", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--budget", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-type", default="B")
    parser.add_argument("--fee", type=float, default=12.95)
    parser.add_argument("--activation-scale", type=float, default=1.0)
    parser.add_argument("--max-per-step", type=int, default=25)
    parser.add_argument(
        "--intervene-day", type=int, default=None,
        help="Day for a brand clarification post (tests the response playbook)",
    )
    args = parser.parse_args()

    _load_env()

    import yaml

    from core.policy_impact import policy_shocks_by_persona
    from core.simulation import announcement_text
    from oasis_tier.extract import extract_report, save_report, top_quotes
    from oasis_tier.runner import BrandIntervention, OasisRunConfig, run_oasis_simulation

    personas_config = yaml.safe_load((ROOT / "config" / "personas.yaml").read_text())

    policy_variables = (
        {"fee_usd": args.fee}
        if args.policy_type == "B"
        else {"distance_miles": 10, "fee_usd": args.fee}
        if args.policy_type == "A"
        else {"threshold_usd": 25}
    )

    shocks = policy_shocks_by_persona(args.policy_type, policy_variables, personas_config)
    announcement = announcement_text(args.policy_type, policy_variables)

    interventions = []
    if args.intervene_day:
        interventions.append(
            BrandIntervention(
                day=args.intervene_day,
                content=(
                    "We hear you. To be clear: in-store returns remain completely "
                    "free, and the fee only applies to mail-in returns. We're "
                    "reviewing the feedback and will share an update this week."
                ),
            )
        )

    cfg = OasisRunConfig(
        policy_type=args.policy_type,
        policy_variables=policy_variables,
        announcement=announcement,
        n_agents=args.agents,
        n_days=args.days,
        agent_model=args.model,
        model_platform=args.platform,
        budget_usd=args.budget,
        seed=args.seed,
        activation_scale=args.activation_scale,
        max_agents_per_step=args.max_per_step,
        interventions=interventions,
    )

    print(f"Announcement: {announcement}")
    print(f"Day-1 shocks: " + ", ".join(f"{k}={v:+.3f}" for k, v in shocks.items()))
    print(f"Running {args.agents} agents x {args.days} days on {args.model} "
          f"(budget ${args.budget:.2f})...\n")

    manifest = run_oasis_simulation(cfg, personas_config, shocks)

    print("=== RUN MANIFEST ===")
    print(json.dumps(
        {k: manifest[k] for k in ("run_id", "graph", "agent_steps", "cost", "projection")},
        indent=2,
    ))

    report = extract_report(manifest["database_path"], manifest)
    out = save_report(report, ROOT / "runs" / "oasis" / f"{manifest['run_id']}_report.json")

    print("\n=== EXTRACTED REPORT ===")
    print(f"posts={report['n_posts']}  comments={report['n_comments']}  "
          f"trace_events={report['n_trace_events']}")
    print(f"final_sentiment={report['final_sentiment']:+.4f}  cascade={report['cascade']}")
    print("per-persona:")
    for persona, stats in sorted(report["breakdown_by_persona"].items()):
        base = stats["tier1_baseline_shock"]
        base_s = f"{base:+.3f}" if base is not None else "  n/a"
        print(f"  {persona:16s} n={stats['n_posts']:<3} "
              f"sentiment={stats['avg_sentiment']:+.3f}  "
              f"tier1_shock={base_s}  churn_signal={stats['churn_signal_rate']:.2f}")

    quotes = top_quotes(report, n=3)
    print("\n=== WHAT AGENTS ACTUALLY SAID (most negative) ===")
    for p in quotes["most_negative"]:
        print(f"  [{p['persona']}, {p['sentiment']:+.2f}] {p['content'][:220]}")
    print("\n=== (most positive) ===")
    for p in quotes["most_positive"]:
        print(f"  [{p['persona']}, {p['sentiment']:+.2f}] {p['content'][:220]}")

    print(f"\nreport saved: {out}")
    print(f"manifest:     {manifest.get('manifest_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
