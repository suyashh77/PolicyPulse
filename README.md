# PolicyPulse

**A multi-agent consumer simulation framework for pre-launch retail policy validation**

> Built as a companion to my Returns Routing Project — a pre-movement return routing system for mid-market DTC apparel and footwear brands. PolicyPulse answers the question that optimization models can't: *will consumers actually accept the policy you're recommending?*

---

## Background & Motivation

My return routing project proposes a 3-layer pre-movement system that fires at return label generation to intelligently route returns across a brand's fulfillment network. The model recommends actions like:

- **In-store routing** — if the customer is within 10 miles of a store, route the return there (free for the customer, cheaper for the brand)
- **Paid non-store returns** — if the nearest node is a DC or 3PL, charge a nominal shipping fee rather than absorbing it
- **Keepit gate** — if the item's expected recovery value is below the cost of return logistics, offer the customer a keepit instead

These recommendations are operationally sound. But they carry an implicit assumption: *that consumers will accept them.* A policy that is financially optimal but behaviorally unacceptable is not actually optimal — it drives churn, social backlash, and brand damage that never shows up in an EV calculator.

To validate these assumptions before recommending them to brands, I built **PolicyPulse**: a multi-agent social simulation platform that models how different consumer segments respond to retail policy changes in a simulated social environment — before those policies go live.

---

## What PolicyPulse Does

PolicyPulse takes a proposed retail policy change as input, generates 500 AI agents with distinct consumer personas, and simulates 45 days of social reaction on Reddit. It produces a structured report showing:

- Sentiment trajectory over 45 days post-announcement
- Churn probability by consumer segment
- Whether a negative cascade event occurred, and on which day
- How different fee levels compare across independent runs
- Why individual agents made the decisions they made (agent interview)

The core claim is that emergent group dynamics — the pile-on, the counter-narrative, the influencer amplification — are fundamentally different from the sum of individual stated preferences. Traditional research tools (focus groups, surveys, A/B tests) capture the latter. PolicyPulse models the former.

---

## Simulation Architecture

PolicyPulse is built on top of [MiroFish](https://github.com/666ghj/MiroFish), an open-source multi-agent prediction engine developed by Guo Hongjian and backed by Shanda Group. MiroFish's simulation engine is [OASIS](https://arxiv.org/abs/2411.11581) (Open Agent Social Interaction Simulations) from the CAMEL-AI team, which supports up to 1 million agents and 23 distinct social actions.

PolicyPulse adds a retail-specific configuration layer on top:

```
Input Layer
  └── Policy type selection (v1: 3 supported types)
  └── Policy variables (fee amount, distance threshold, item value threshold)
  └── Brand context (positioning, current policy)
  └── Seed data (Reddit sentiment scrape for brand)

Knowledge Graph (Neo4j)
  └── Entity extraction from seed material
  └── Relationship mapping: brand ↔ policy ↔ consumer segments ↔ competitors

Agent Generation (500 agents per run)
  └── Persona library (5 retail archetypes)
  └── Individual trait assignment per archetype
  └── Persistent memory initialization (Zep Cloud)
  └── Initial state: policy_sentiment = 0, churn_intent = 0

Simulation Engine — 45 rounds (1 round = 1 day)
  └── Round 1: all agents see pinned announcement post (sentiment = 0)
  └── Rounds 2–45: feed algorithm + posting + state update
  └── Platform: Reddit only (v1)

ReportAgent (Claude Sonnet)
  └── Sentiment curve (per-round average across all agents)
  └── Churn probability by segment (round 45 snapshot)
  └── Cascade detection (sentiment drop >0.4 in 10 consecutive rounds)
  └── Threshold comparison across fee-level runs
  └── Agent interview (filter by persona type + churn status)
```

---

## Core Simulation Logic

### Agent State

Each agent tracks two independent dimensions updated every round:

| Dimension | Range | Description |
|---|---|---|
| `policy_sentiment` | −1.0 to +1.0 | Opinion of the policy itself |
| `churn_intent` | 0.0 to 1.0 | Likelihood to stop shopping with the brand |

The two dimensions are independent. An agent can find a fee unfair (policy_sentiment: −0.6) while still having low churn intent (0.1) due to brand loyalty. In v1, churn_intent only increases — brand response mechanics that bring it back down are a v2 feature.

### State Update Rules (per round)

```
policy_sentiment += weighted_feed_signal × persona_susceptibility
churn_intent     += max(0, −sentiment_delta) × churn_elasticity

# Bounds
policy_sentiment = clamp(policy_sentiment, −1.0, 1.0)
churn_intent     = clamp(churn_intent, 0.0, 1.0)
```

Where:
- `weighted_feed_signal` = sentiment-weighted average of posts seen this round
- `sentiment_delta` = change in policy_sentiment this round (negative = got worse)
- `persona_susceptibility` = how much peer opinion moves this archetype
- `churn_elasticity` = how quickly negative sentiment converts to churn intent

### Feed Algorithm (per agent per round)

Each agent sees 10 posts per round, sampled from the post pool as follows:

| Source | Weight | Rationale |
|---|---|---|
| Reach-weighted | 60% | High-follower posts dominate feeds naturally |
| Homophily | 20% | Agents cluster with similar personas, echo chambers form |
| Recency-weighted | 10% | Newest posts get a boost (Reddit hot/new algorithm) |
| Random | 10% | Cross-cohort exposure, prevents total isolation |

### Posting Behavior

Each round, agents decide whether to post based on their archetype's frequency, then generate a post with sentiment drawn from their current state plus persona-specific noise:

```
post_sentiment = current_policy_sentiment + noise(persona_variance)
```

| Persona | Post probability / round | Variance |
|---|---|---|
| Micro-influencer | 100% | 0.1 (measured, deliberate) |
| Deal-seeker | 40% | 0.3 (amplified, reactive) |
| Sustainability buyer | 25% | 0.2 |
| Casual shopper | 20% | 0.15 |
| Loyal customer | 15% | 0.1 (moderate, restrained) |

### Initial Conditions

- All agents: `policy_sentiment = 0`, `churn_intent = 0`
- Round 1: pinned announcement post (sentiment = 0) visible to all agents
- Feed algorithm active from round 2 onward

---

## Consumer Persona Library (v1)

The persona library is PolicyPulse's primary contribution on top of raw MiroFish. Five archetypes, each with distinct susceptibility and elasticity parameters:

| Archetype | Susceptibility | Churn Elasticity | Social Reach |
|---|---|---|---|
| Loyal customer | 0.2 | 0.1 | Low |
| Casual shopper | 0.45 | 0.35 | Low–medium |
| Deal-seeker | 0.8 | 0.6 | Medium |
| Micro-influencer | 0.5 | 0.3 | High (10K–100K followers) |
| Sustainability buyer | 0.55 | 0.4 | Medium |

Each archetype is also configured with purchase behavior parameters (frequency, AOV, return rate) and social behavior parameters (follower count range, platform preference) that ground the agent's persona in realistic consumer behavior.

---

## v1 Policy Types

v1 supports three fixed policy types with structured variable inputs. v2 will introduce a generalized framework for arbitrary policy text once proof of concept is established.

### Type A: Distance-gated routing

**Policy**: Free in-store return if customer is within X miles of a store. Non-store (mail-in) return costs $Y.

**Variables**:
- Distance threshold: 5 / 10 / 15 / 25 miles
- Fee amount: $4.95 / $7.95 / $9.95 / $12.95

**Connection to returns project**: Directly validates the 10-mile routing rule in v4.0 L3 optimizer.

### Type B: Flat return fee

**Policy**: All mail-in returns carry a $Y fee deducted from refund. In-store returns remain free.

**Variables**:
- Fee amount: $4.95 / $7.95 / $9.95 / $12.95

**Connection to returns project**: Validates the cost-sharing assumption in the EV calculator.

### Type C: Keepit gate

**Policy**: Items under $Z in value receive a full refund with no return required (keepit).

**Variables**:
- Item value threshold: $15 / $25 / $35 / $50

**Connection to returns project**: Directly validates the keepit pre-gate condition: `max(EV) < shipping cost`.

### Independent runs

Each variable combination is a separate, independent simulation run. There is no bleed between runs. Comparison across runs (e.g. $4.95 vs $9.95 vs $12.95) is handled by the ReportAgent's threshold analysis.

---

## ReportAgent Output Spec

After each run, the ReportAgent produces:

1. **Sentiment curve** — per-round average `policy_sentiment` across all 500 agents, plotted over 45 days
2. **Churn by segment** — average `churn_intent` per persona archetype at round 45
3. **Cascade flag** — boolean: did `policy_sentiment` drop >0.4 within any 10 consecutive rounds? If yes, which day did it trigger?
4. **Threshold comparison** — across all fee-level runs for this policy type, side-by-side sentiment and churn at round 45
5. **Agent interview** — user filters by persona type and churn status ("show me a deal-seeker who churned"), ReportAgent returns that agent's memory-grounded explanation of their behavior

---

## Sample Output

```
PolicyPulse Report — Type A: Distance-gated routing ($9.95 non-store fee, 10-mile threshold)
Generated: 2025-04-01 | Agents: 500 | Rounds: 45 | Platform: Reddit

SENTIMENT CURVE
  Day 1:   0.00  (announcement, neutral baseline)
  Day 10: −0.18  (deal-seeker cohort vocal)
  Day 20: −0.22  (peak negativity)
  Day 35: −0.14  (stabilization)
  Day 45: −0.11  (net slightly negative, no cascade)

CASCADE EVENT: None triggered

CHURN INTENT BY SEGMENT (day 45)
  Loyal customer:       4.2%
  Casual shopper:       11.7%
  Deal-seeker:          34.1%   ← primary risk cohort
  Micro-influencer:     8.9%
  Sustainability buyer: 6.1%

THRESHOLD COMPARISON (Type A, 10-mile threshold)
  $4.95  — sentiment: −0.04 | avg churn: 3.1%  | cascade: No
  $7.95  — sentiment: −0.09 | avg churn: 7.4%  | cascade: No
  $9.95  — sentiment: −0.11 | avg churn: 11.2% | cascade: No
  $12.95 — sentiment: −0.31 | avg churn: 24.6% | cascade: Yes (day 18)

AGENT INTERVIEW
  Filter: deal-seeker, churned

  Agent #247: "I saw multiple people saying the same thing — why should I
  pay $10 to return something that didn't fit? I already paid for shipping
  to get it here. I checked Vuori and they still do free returns. I placed
  an order there instead."
```

---

## Versioning Roadmap

| Version | Scope |
|---|---|
| v1 | 3 policy types, 500 agents, 45 rounds, Reddit only, structured variable input |
| v2 | Brand response mid-simulation (churn recovery), Twitter + news sources added |
| v3 | Generalized policy input (arbitrary text), broader social ecosystem |

---

## Validation Approach

A simulation is only useful if it tracks reality. PolicyPulse's validation strategy:

1. **Historical backtesting** — run simulations against known policy changes with documented social responses:
   - H&M paid returns launch (2022) — significant documented backlash
   - Zara return fee introduction (2022) — measured brand sentiment shift
   - Lululemon Like New program — largely positive response
   - Amazon keepit policy expansion — minimal consumer backlash

2. **Accuracy metric** — compare simulated sentiment direction and cascade occurrence against observed Reddit data from the actual launch. Target: directional accuracy ≥ 80% across backtested cases.

3. **Calibration loop** — use backtest deltas to adjust persona susceptibility and elasticity parameters iteratively.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Simulation engine | MiroFish / OASIS (CAMEL-AI) |
| Agent memory | Zep Cloud |
| Knowledge graph | Neo4j |
| Seed data | PRAW (Reddit scraper) |
| Backend | Python / FastAPI |
| UI | Streamlit |
| Visualization | Plotly |
| ReportAgent LLM | Claude Sonnet (Anthropic API) |
| Local fallback | Ollama + Qwen2.5:32B |
| Embeddings | nomic-embed-text |

## Connection to Return Routing Project

PolicyPulse is a validation layer that sits above the operational recommendation engine in the return routing project:

```
Return Routing v4.0
  L1: Condition estimator (multinomial logistic, F1=0.58)
  L2: EV calculator: EV(j) = Σₖ P(k) × [R(j,k) − T(j) − P(j,k)]
  L3: ILP optimizer (PuLP)
  Pre-gate: keepit if max(EV) < shipping cost

          ↓  generates policy recommendation

PolicyPulse
  "Will consumers accept this recommendation?"
  Type A → validates distance-gated routing rule
  Type B → validates cost-sharing fee assumption
  Type C → validates keepit pre-gate threshold
```

The routing optimizer determines what to do economically. PolicyPulse determines whether consumers will accept it behaviorally.

---

## Acknowledgements

- **MiroFish** by Guo Hongjian (666ghj) — the simulation engine this project builds on
- **OASIS** by the CAMEL-AI team — the underlying multi-agent social interaction framework
- **Zep Cloud** — persistent agent memory infrastructure
- Stanford HAI Generative Agents paper (Park et al., 2023) — foundational research on believable agent behavior

---