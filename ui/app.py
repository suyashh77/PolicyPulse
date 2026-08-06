"""PolicyPulse Streamlit UI."""
from __future__ import annotations

import json
import os
import random
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from core.policy_impact import policy_shocks_by_persona
from core.policy_types import POLICY_TYPES
from core.simulation import announcement_text, load_personas, run_batch, summarize_batch
from report.interview import get_interview_candidates, interview_agent
from report.persistence import load_batches, save_batch
from report.report_agent import generate_report

load_dotenv()

st.set_page_config(page_title="PolicyPulse", layout="wide")
st.title("PolicyPulse — Retail Policy Simulation")
st.caption(
    "Simulates how consumer segments react to a return-policy change over 45 days. "
    "Results are directional, not validated — see README."
)

PERSONA_COLORS = {
    "loyal": "#636EFA",
    "casual": "#EF553B",
    "deal_seeker": "#00CC96",
    "influencer": "#AB63FA",
    "sustainability": "#FFA15A",
}

if "runs" not in st.session_state:
    st.session_state.runs = []
if "current_batch" not in st.session_state:
    st.session_state.current_batch = None

tab_config, tab_results, tab_economics, tab_interview = st.tabs(
    ["Configure Simulation", "Results", "Economics", "Agent Interview"]
)

# ===================== PAGE 1: CONFIGURE =====================
with tab_config:
    st.header("Configure Simulation")

    policy_type = st.selectbox(
        "Policy Type",
        options=list(POLICY_TYPES.keys()),
        format_func=lambda k: f"Type {k}: {POLICY_TYPES[k]['name']}",
    )

    policy_cfg = POLICY_TYPES[policy_type]
    st.caption(policy_cfg["description"])

    policy_variables: dict = {}
    for var_name, options in policy_cfg["variables"].items():
        label = var_name.replace("_", " ").title()
        policy_variables[var_name] = st.selectbox(
            label, options=options, key=f"var_{var_name}"
        )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        brand_name = st.text_input("Brand Name", value="Everlane")
    with col_b:
        n_seeds = st.number_input(
            "Runs (seeds)",
            min_value=1,
            max_value=25,
            value=5,
            help="Each run is one draw. Several runs give a mean and a spread; "
            "a single run cannot distinguish a result from noise.",
        )
    with col_c:
        base_seed = st.number_input(
            "Base seed", min_value=0, value=42,
            help="Runs use base_seed, base_seed+1, ... Same seed reproduces exactly.",
        )

    # Show the day-1 reaction before running anything — this is the part of the
    # model that carries the actual judgment.
    personas_config = load_personas()
    shocks = policy_shocks_by_persona(policy_type, policy_variables, personas_config)
    st.subheader("Day-1 reaction by segment (before any social contagion)")
    st.caption(announcement_text(policy_type, policy_variables))
    shock_df = pd.DataFrame(
        [{"persona": k, "initial_sentiment": round(v, 3)} for k, v in shocks.items()]
    ).sort_values("initial_sentiment")
    st.dataframe(shock_df, width='stretch', hide_index=True)

    if st.button("Run Simulation", type="primary"):
        seeds = [int(base_seed) + i for i in range(int(n_seeds))]
        with st.spinner(f"Running {len(seeds)} x 45-round simulations with 500 agents..."):
            batch = run_batch(
                policy_type=policy_type,
                policy_variables=policy_variables,
                seeds=seeds,
                personas_config=personas_config,
            )
            st.session_state.runs.extend(batch)
            st.session_state.current_batch = batch
            try:
                save_batch(batch)
            except OSError as exc:
                st.warning(f"Could not save run to disk: {exc}")

        stats = summarize_batch(batch)
        st.success(
            f"Complete. Day-45 sentiment {stats['final_sentiment_mean']:+.3f} "
            f"± {stats['final_sentiment_stdev']:.3f} across {stats['n_runs']} runs."
        )
        st.rerun()

    saved = load_batches()
    if saved:
        st.caption(f"{len(saved)} batch(es) saved on disk in `runs/`.")

# ===================== PAGE 2: RESULTS =====================
with tab_results:
    st.header("Simulation Results")

    if not st.session_state.current_batch:
        st.info("Run a simulation first to see results.")
    else:
        batch = st.session_state.current_batch
        run = batch[0]
        report = generate_report(run, st.session_state.runs)

        stats = summarize_batch(batch)
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Day-45 sentiment",
            f"{stats['final_sentiment_mean']:+.3f}",
            f"± {stats['final_sentiment_stdev']:.3f}",
        )
        m2.metric(
            "Day-45 churn intent",
            f"{stats['final_churn_mean']:.3f}",
            f"± {stats['final_churn_stdev']:.3f}",
        )
        m3.metric("Runs in batch", stats["n_runs"])

        # --- Sentiment curve with spread band ---
        st.subheader("Sentiment Curve (45 Days)")
        curve = report["aggregate_sentiment_curve"]
        days = [p["day"] for p in curve]

        fig_sent = go.Figure()
        if len(batch) > 1:
            fig_sent.add_trace(
                go.Scatter(
                    x=days + days[::-1],
                    y=[p["upper"] for p in curve] + [p["lower"] for p in curve][::-1],
                    fill="toself",
                    fillcolor="rgba(239,85,59,0.18)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    name="±1 stdev",
                )
            )
        fig_sent.add_trace(
            go.Scatter(
                x=days,
                y=[p["avg_sentiment"] for p in curve],
                mode="lines",
                name=f"Mean of {len(batch)} run(s)",
                line=dict(color="#EF553B", width=2),
            )
        )
        fig_sent.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_sent.update_layout(
            xaxis_title="Day",
            yaxis_title="Average Policy Sentiment",
            yaxis_range=[-1, 1],
            height=420,
        )
        st.plotly_chart(fig_sent, width='stretch')

        # --- Churn by segment ---
        st.subheader("Churn Intent by Segment (Day 45)")
        churn = report["aggregate_churn_by_segment"]
        personas = list(churn.keys())
        fig_churn = go.Figure(
            go.Bar(
                x=personas,
                y=[churn[p]["mean"] for p in personas],
                error_y=dict(type="data", array=[churn[p]["stdev"] for p in personas]),
                marker_color=[PERSONA_COLORS.get(p, "#888") for p in personas],
            )
        )
        fig_churn.update_layout(
            xaxis_title="Persona",
            yaxis_title="Avg Churn Intent",
            yaxis_range=[0, 1],
            height=420,
        )
        st.plotly_chart(fig_churn, width='stretch')

        # --- Cascade ---
        cascades = [generate_report(r, [r])["cascade"] for r in batch]
        n_cascade = sum(1 for c in cascades if c["cascade"])
        if n_cascade:
            days_hit = [c["trigger_day"] for c in cascades if c["trigger_day"]]
            st.error(
                f"⚠ Cascade detected in {n_cascade} of {len(batch)} runs "
                f"(median trigger day {int(sorted(days_hit)[len(days_hit) // 2])}). "
                "Sentiment dropped more than 0.4 within a 10-day window."
            )
        else:
            st.success(f"No cascade event detected in any of {len(batch)} runs.")

        # --- Threshold comparison ---
        comparison = report["threshold_comparison"]
        if len(comparison) > 1:
            st.subheader("Threshold Comparison")
            st.caption(
                "Grouped by variable setting, averaged across seeds. Rows sorted "
                "worst sentiment first."
            )
            df = pd.DataFrame(comparison)
            df["variables"] = df["variables"].apply(
                lambda d: ", ".join(f"{k}={v}" for k, v in d.items())
            )
            st.dataframe(
                df[
                    [
                        "variables",
                        "n_runs",
                        "final_sentiment",
                        "final_sentiment_stdev",
                        "avg_churn",
                        "cascade_rate",
                        "median_cascade_day",
                    ]
                ],
                width='stretch',
                hide_index=True,
            )
        else:
            st.caption(
                "Run another fee level of the same policy type to populate the "
                "threshold comparison."
            )

        # --- Tier 2: what agents actually said ---
        st.subheader("What consumers actually said (Tier 2)")
        tier2_reports = sorted(
            Path("runs/oasis").glob("*_report.json"), reverse=True
        ) if Path("runs/oasis").is_dir() else []

        if not tier2_reports:
            st.caption(
                "No Tier-2 run found. Tier 2 puts LLM agents on a simulated Reddit "
                "so you can read what people say, not just a curve. "
                "See docs/OASIS_INTEGRATION_PLAN.md."
            )
        else:
            t2 = json.loads(tier2_reports[0].read_text(encoding="utf-8"))
            posts = [p for p in t2.get("posts", []) if p.get("content", "").strip()]
            st.caption(
                f"From `{tier2_reports[0].name}` — {len(posts)} utterances, "
                f"Type {t2.get('policy_type')} {t2.get('policy_variables')}"
            )
            posts.sort(key=lambda p: p.get("sentiment", 0))
            col_neg, col_pos = st.columns(2)
            with col_neg:
                st.markdown("**Most negative**")
                for p in posts[:4]:
                    st.markdown(
                        f"> {p['content'][:400]}\n\n"
                        f"— *{p['persona']}* · {p.get('sentiment', 0):+.2f}"
                    )
            with col_pos:
                st.markdown("**Most positive**")
                for p in reversed(posts[-4:]):
                    st.markdown(
                        f"> {p['content'][:400]}\n\n"
                        f"— *{p['persona']}* · {p.get('sentiment', 0):+.2f}"
                    )

# ===================== PAGE 3: ECONOMICS =====================
with tab_economics:
    st.header("Economics")
    st.caption(
        "Translates sentiment and churn into money. Every figure below is "
        "arithmetic over the brand inputs — change one and you can trace how the "
        "output moved."
    )

    if not st.session_state.current_batch:
        st.info("Run a simulation first to price it.")
    else:
        from core.economics import (
            BrandProfile,
            EconomicAssumptions,
            build_frontier,
            evaluate_policy,
            sensitivity_to_churn_conversion,
        )
        from report.curves import aggregate_churn_by_segment

        st.subheader("Brand inputs")
        b1, b2, b3 = st.columns(3)
        with b1:
            customers = st.number_input("Active customers", 1_000, 50_000_000, 250_000, step=10_000)
            aov = st.number_input("Average order value ($)", 10.0, 1000.0, 95.0)
        with b2:
            orders_py = st.number_input("Orders / customer / year", 0.5, 30.0, 3.2)
            margin = st.slider("Gross margin", 0.05, 0.90, 0.45)
        with b3:
            return_rate = st.slider("Return rate", 0.01, 0.70, 0.28)
            mail_share = st.slider("Mail-in share of returns", 0.05, 1.0, 0.55)

        brand = BrandProfile(
            active_customers=int(customers),
            orders_per_customer_per_year=orders_py,
            avg_order_value=aov,
            gross_margin=margin,
            return_rate=return_rate,
            mail_in_share=mail_share,
        )

        conversion = st.slider(
            "Churn conversion — share of churn *intent* that becomes a lost customer",
            0.05, 1.0, 0.35,
            help="The least defensible number in the model. It has never been "
                 "mapped to observed behaviour. Check the sensitivity table below.",
        )
        assumptions = EconomicAssumptions(churn_conversion=conversion)

        churn_now = {
            k: v["mean"] for k, v in aggregate_churn_by_segment(batch).items()
        }
        result = evaluate_policy(
            brand, run.policy_type, run.policy_variables,
            churn_now, personas_config, assumptions,
        )

        e1, e2, e3 = st.columns(3)
        e1.metric("Gross annual saving", f"${result['gross_annual_saving']:,.0f}")
        e2.metric("Customer value at risk", f"${result['clv_at_risk']:,.0f}")
        e3.metric(
            "Net", f"${result['net_value']:,.0f}",
            delta=result["verdict"],
            delta_color="normal" if result["net_value"] > 0 else "inverse",
        )

        # --- Frontier ---
        st.subheader("Policy frontier")
        st.caption("Where the savings stop covering the behavioural risk.")

        var_name = list(POLICY_TYPES[run.policy_type]["variables"])[-1]
        levels = POLICY_TYPES[run.policy_type]["variables"][var_name]

        with st.spinner(f"Simulating each {var_name.replace('_',' ')} level..."):
            churn_by_level = {}
            for level in levels:
                variables = dict(run.policy_variables)
                variables[var_name] = level
                level_runs = run_batch(
                    run.policy_type, variables, seeds=[1, 2, 3, 4, 5],
                    personas_config=personas_config,
                )
                churn_by_level[level] = {
                    k: v["mean"] for k, v in aggregate_churn_by_segment(level_runs).items()
                }

            frontier = build_frontier(
                brand, run.policy_type, var_name, levels, churn_by_level,
                personas_config, base_variables=run.policy_variables,
                assumptions=assumptions,
            )

        xs = [f["level"] for f in frontier]
        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(
            x=xs, y=[f["gross_annual_saving"] for f in frontier],
            name="Gross saving", marker_color="#00CC96",
        ))
        fig_f.add_trace(go.Bar(
            x=xs, y=[-f["clv_at_risk"] for f in frontier],
            name="Customer value at risk", marker_color="#EF553B",
        ))
        fig_f.add_trace(go.Scatter(
            x=xs, y=[f["net_value"] for f in frontier],
            name="Net", mode="lines+markers",
            line=dict(color="#111", width=3),
        ))
        fig_f.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_f.update_layout(
            barmode="relative", height=440,
            xaxis_title=var_name.replace("_", " ").title(),
            yaxis_title="Annual $",
        )
        st.plotly_chart(fig_f, width="stretch")

        safe = [f["level"] for f in frontier if f["net_value"] > 0]
        if safe:
            st.success(f"Value-accretive at: {', '.join(str(s) for s in safe)}")
        else:
            st.error(
                f"No {var_name.replace('_',' ')} level is accretive at "
                f"{conversion:.0%} churn conversion. Check the sensitivity below "
                "before concluding the policy is unviable."
            )

        # --- Sensitivity ---
        st.subheader("Sensitivity to the churn-conversion assumption")
        st.caption(
            "If the sign of Net flips inside this range, the recommendation is "
            "not robust and should be presented as a range, not a number."
        )
        sens = sensitivity_to_churn_conversion(
            brand, run.policy_type, run.policy_variables, churn_now, personas_config
        )
        st.dataframe(pd.DataFrame(sens), width="stretch", hide_index=True)

        # --- Who carries the risk ---
        st.subheader("Who carries the risk")
        risk_rows = [
            {"persona": p, **d} for p, d in result["risk_detail"]["by_segment"].items()
        ]
        st.dataframe(pd.DataFrame(risk_rows), width="stretch", hide_index=True)
        st.caption(
            "Loudest is not costliest: the segment that churns hardest may be the "
            "cheapest to lose, while a quieter, higher-value cohort carries more."
        )


# ===================== PAGE 4: AGENT INTERVIEW =====================
with tab_interview:
    st.header("Agent Interview")

    if not st.session_state.current_batch:
        st.info("Run a simulation first to interview agents.")
    else:
        run = st.session_state.current_batch[0]

        persona_options = list(load_personas()["personas"].keys())
        selected_persona = st.selectbox("Persona Type", options=persona_options)
        churned = st.toggle("Churned (churn_intent ≥ 0.5)", value=True)

        if st.button("Interview Agent"):
            candidates = get_interview_candidates(run, selected_persona, churned)

            if not candidates:
                st.warning(
                    f"No {'churned' if churned else 'non-churned'} "
                    f"{selected_persona} agents in this run."
                )
            else:
                agent = random.choice(candidates)
                st.write(f"**Agent #{agent.id}** — {agent.persona}")
                st.write(
                    f"Day-1 reaction: `{agent.baseline_sentiment:+.2f}` | "
                    f"Final sentiment: `{agent.policy_sentiment:+.2f}` | "
                    f"Churn intent: `{agent.churn_intent:.2f}`"
                )

                announcement = announcement_text(run.policy_type, run.policy_variables)

                if not os.getenv("ANTHROPIC_API_KEY"):
                    st.error("Set ANTHROPIC_API_KEY in .env to use agent interviews.")
                else:
                    with st.spinner("Generating agent perspective..."):
                        try:
                            response = interview_agent(agent, announcement)
                        except Exception as exc:  # surface API errors in the UI
                            st.error(f"Interview failed: {exc}")
                            response = None
                    if response:
                        st.markdown(f"> {response}")
