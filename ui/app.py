"""PolicyPulse Streamlit UI."""
from __future__ import annotations

import random
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from core.policy_types import POLICY_TYPES
from core.simulation import SimulationRun, run_simulation
from report.curves import build_churn_by_segment, build_sentiment_curve
from report.cascade import detect_cascade
from report.report_agent import generate_report
from report.interview import get_interview_candidates, interview_agent

load_dotenv()

st.set_page_config(page_title="PolicyPulse", layout="wide")
st.title("PolicyPulse — Retail Policy Simulation")

# Session state for storing runs
if "runs" not in st.session_state:
    st.session_state.runs = []
if "current_run" not in st.session_state:
    st.session_state.current_run = None

# --- Sidebar: Configure Simulation ---
tab_config, tab_results, tab_interview = st.tabs(
    ["Configure Simulation", "Results", "Agent Interview"]
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
        selected = st.selectbox(f"{label}", options=options, key=f"var_{var_name}")
        policy_variables[var_name] = selected

    brand_name = st.text_input("Brand Name", value="Everlane")

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running 45-round simulation with 500 agents..."):
            random.seed()
            sim = run_simulation(
                policy_type=policy_type,
                policy_variables=policy_variables,
            )
            st.session_state.runs.append(sim)
            st.session_state.current_run = sim
        st.success(f"Simulation complete! Run ID: {sim.run_id}")
        st.rerun()

# ===================== PAGE 2: RESULTS =====================
with tab_results:
    st.header("Simulation Results")

    if st.session_state.current_run is None:
        st.info("Run a simulation first to see results.")
    else:
        run = st.session_state.current_run
        report = generate_report(run, st.session_state.runs)

        # Sentiment Curve
        st.subheader("Sentiment Curve (45 Days)")
        curve = report["sentiment_curve"]
        fig_sent = go.Figure()
        fig_sent.add_trace(
            go.Scatter(
                x=[p["day"] for p in curve],
                y=[p["avg_sentiment"] for p in curve],
                mode="lines+markers",
                name="Avg Sentiment",
                line=dict(color="#EF553B"),
            )
        )
        fig_sent.update_layout(
            xaxis_title="Day",
            yaxis_title="Average Policy Sentiment",
            yaxis_range=[-1, 1],
            height=400,
        )
        st.plotly_chart(fig_sent, use_container_width=True)

        # Churn by Segment
        st.subheader("Churn Intent by Segment (Day 45)")
        churn = report["churn_by_segment"]
        personas = list(churn.keys())
        values = [churn[p] for p in personas]

        fig_churn = go.Figure()
        fig_churn.add_trace(
            go.Bar(
                x=personas,
                y=values,
                marker_color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"],
            )
        )
        fig_churn.update_layout(
            xaxis_title="Persona",
            yaxis_title="Avg Churn Intent",
            yaxis_range=[0, 1],
            height=400,
        )
        st.plotly_chart(fig_churn, use_container_width=True)

        # Cascade Alert
        cascade = report["cascade"]
        if cascade["cascade"]:
            st.error(
                f"⚠ CASCADE EVENT DETECTED — triggered on Day {cascade['trigger_day']}. "
                f"Sentiment dropped more than 0.4 within a 10-day window."
            )
        else:
            st.success("No cascade event detected.")

        # Threshold Comparison
        comparison = report["threshold_comparison"]
        if len(comparison) > 1:
            st.subheader("Threshold Comparison")
            import pandas as pd

            df = pd.DataFrame(comparison)
            df["variables"] = df["variables"].apply(str)
            st.dataframe(
                df[["run_id", "variables", "final_sentiment", "avg_churn", "cascade", "cascade_day"]],
                use_container_width=True,
            )

# ===================== PAGE 3: AGENT INTERVIEW =====================
with tab_interview:
    st.header("Agent Interview")

    if st.session_state.current_run is None:
        st.info("Run a simulation first to interview agents.")
    else:
        run = st.session_state.current_run

        persona_options = ["loyal", "casual", "deal_seeker", "influencer", "sustainability"]
        selected_persona = st.selectbox("Persona Type", options=persona_options)
        churned = st.toggle("Churned (churn_intent ≥ 0.5)", value=True)

        if st.button("Interview Agent"):
            candidates = get_interview_candidates(run, selected_persona, churned)

            if not candidates:
                st.warning(f"No {'churned' if churned else 'non-churned'} {selected_persona} agents found.")
            else:
                agent = random.choice(candidates)
                st.write(f"**Interviewing Agent #{agent.id}** — {agent.persona}")
                st.write(f"Final sentiment: `{agent.policy_sentiment:.2f}` | Churn intent: `{agent.churn_intent:.2f}`")

                policy_cfg = POLICY_TYPES[run.policy_type]
                announcement = policy_cfg["announcement_template"].format(
                    date="recently", **run.policy_variables
                )

                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    st.error("Set ANTHROPIC_API_KEY in .env to use agent interviews.")
                else:
                    with st.spinner("Generating agent perspective..."):
                        response = interview_agent(agent, run.policy_variables, announcement)
                    st.markdown(f"> {response}")
