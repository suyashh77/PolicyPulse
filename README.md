# PolicyPulse

**A multi-agent consumer simulation for pre-launch retail return-policy validation.**

A returns-routing optimiser tells you what is *cheapest*. It produces recommendations like "charge $9.95 for mail-in returns beyond 10 miles" that are correct on an expected-value basis and can still be commercially catastrophic, because an EV calculator has no term for a Reddit thread. PolicyPulse models the other half of the question: **will customers accept it?**

It generates 500 persona-driven consumer agents, hits them with a proposed policy, and simulates 45 days of social reaction — feed exposure, opinion contagion, influencer amplification, churn.

> **Status: working prototype, not validated.** The simulation responds correctly to its inputs and the dynamics are coherent, but the parameters are priors rather than fitted values and no backtest against real policy launches exists yet. Output is directional intuition, not evidence. See [Honest limitations](#honest-limitations).

---

## Contents

- [Quick start](#quick-start)
- [The problem it solves](#the-problem-it-solves)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [What's good about it](#whats-good-about-it)
- [What's bad about it](#whats-bad-about-it)
- [Honest limitations](#honest-limitations)
- [What's left to build](#whats-left-to-build)
- [Lessons learned](#lessons-learned)
- [Project layout](#project-layout)

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY for the interview tab

streamlit run ui/app.py       # ~0.6s per simulation run

python -m pytest -q           # 105 tests, ~36s (Tier 1 + Tier 2 logic; no API calls)
```

Note `python -m pytest`, not bare `pytest` — the executable is not on PATH in most setups here.

Only `ANTHROPIC_API_KEY` is needed for anything currently wired up. The Reddit and Neo4j variables in `.env.example` correspond to scaffolded modules with no call sites.

---

## The problem it solves

A survey asks 500 people what they think of a $9.95 return fee and averages the answers. That captures stated individual preference. It cannot capture the mechanism that actually determines outcomes: one micro-influencer post reaching 80,000 people, a deal-seeker cohort amplifying it, and a pile-on producing a brand-level result that nothing in the individual responses predicted.

PolicyPulse models *dynamics* rather than *preferences* — amplification, echo chambers, and threshold effects. That is the part traditional research genuinely cannot do.

### Where it fits

```
Return Routing optimiser
  L1: condition estimator      →  what state will the item come back in?
  L2: EV calculator             →  what is each disposition worth?
  L3: ILP optimiser             →  what is the cheapest routing?
  Pre-gate: keepit if max(EV) < shipping cost

          ↓  emits a policy recommendation

PolicyPulse
  "Will consumers accept this recommendation?"
  Type A → validates the distance-gated routing rule
  Type B → validates the cost-sharing fee assumption
  Type C → validates the keepit threshold
```

### Realistic applications

- **Fee-level selection** — find the sentiment cliff between $7.95 and $9.95 before committing.
- **Segment risk** — identify which cohort carries the exposure, and whether it is one you can afford to lose.
- **Comparative screening** — rank policy types A/B/C against each other under the same assumptions.
- **Internal argument** — a defensible artefact for pushing back on a finance-driven fee proposal.

---

## How it works

### 1. The policy becomes a per-segment shock

This is the core of the model, in [`core/policy_impact.py`](core/policy_impact.py). Fees are judged **relative to basket size**, not in absolute dollars — $9.95 is a fifth of a deal seeker's $45 order and a sixteenth of a loyal customer's $165 one. Relative cost passes through a saturating disutility curve, because a $50 fee is not five times as infuriating as a $10 one, it is simply unacceptable either way.

```
disutility = fee_sensitivity × tanh(relative_cost / DISUTILITY_SCALE)
```

Each policy type has its own structure:

| Type | Structure | Shock |
|---|---|---|
| **A** Distance-gated | Free in-store within X miles, $Y mail-in | Store coverage rises with radius; the uncovered fraction pays the fee, the covered fraction pays a small trip friction |
| **B** Flat fee | $Y on all mail-in, in-store free | Same form, but coverage is capped at the persona's baseline store access — no distance gate to widen |
| **C** Keepit | Items under $Z refunded, no return | **Positive** for most personas, scaled by the fraction of orders that qualify |

Type C is the interesting one: `keepit_affinity` is **negative** for the sustainability cohort, who read "keep it, we'll refund you anyway" as manufactured waste rather than a gift. It is the one policy whose day-1 shock splits the population by sign — and watching what happens next is the point. Under a $50 threshold the sustainability cohort starts at −0.19 while everyone else starts positive; by day 45 social pressure has dragged them to **+0.17**, still the lowest segment and the only one with any churn at all. A dissenting minority getting partially converted by a positive majority is exactly the emergent dynamic a survey cannot produce.

### 2. 500 agents are generated

From [`config/personas.yaml`](config/personas.yaml), each starting at their persona's shock rather than at zero:

| Archetype | Count | Susceptibility | Churn elasticity | AOV | Reach | Posts/round |
|---|---|---|---|---|---|---|
| Loyal | 150 | 0.20 | 0.10 | $165 | 50–500 | 15% |
| Casual | 125 | 0.45 | 0.35 | $85 | 50–1K | 20% |
| Deal seeker | 100 | 0.80 | 0.60 | $45 | 100–5K | 40% |
| Micro-influencer | 50 | 0.50 | 0.30 | $120 | 10K–100K | **100%** |
| Sustainability | 75 | 0.55 | 0.40 | $110 | 200–8K | 25% |

### 3. Day 1 — the announcement

A single pinned post whose sentiment is the population-weighted shock. The news itself reads as bad or good in proportion to how the customer base receives it.

### 4. Days 2–45 — the contagion loop

Each day, every agent: **samples a 10-post feed → updates state → maybe posts**. Posts written on day *N* are not visible until day *N+1*.

**Feed algorithm** ([`core/feed.py`](core/feed.py)) — four mechanisms, mirroring how a real feed distributes attention:

| Source | Slots | Rationale |
|---|---|---|
| Reach-weighted | 6 | Loud voices dominate feeds naturally |
| Homophily | 2 | Agents cluster with similar personas; echo chambers form |
| Recency | 1 | Newest posts get a boost |
| Random | 1 | Cross-cohort exposure prevents total isolation |

Reach decays 0.75× per round of age, and posts older than 12 rounds leave the pool entirely — real feeds have a half-life.

**State update** ([`core/agent.py`](core/agent.py)) — two independent dimensions:

```
social_pull  = (feed_signal − sentiment) × susceptibility
anchor_pull  = (baseline   − sentiment) × anchor_strength
sentiment   += social_pull + anchor_pull            → clamp [−1, +1]

target_churn = max(0, −sentiment) × churn_elasticity
churn       += (target_churn − churn) × rate         → clamp [0, 1]
               (rate = elasticity when rising, churn_recovery when falling)
```

Sentiment moves *toward* the feed signal rather than accumulating it, and each agent is pulled partway back toward its policy-implied baseline. Churn tracks the sentiment **level** and can recover, escalating faster than it decays — people forgive gradually.

Keeping sentiment and churn separate is deliberate: an agent can find a fee unfair (−0.6) and still not leave (churn 0.1) out of brand loyalty.

### 5. Reporting

Deterministic, no LLM. Sentiment curve with a ±1σ band across seeds, day-45 churn by segment, cascade detection (a >0.4 drop within any 10-day window), and a threshold comparison grouped by policy setting.

### 6. Agent interview — the only LLM call

Filter for "a deal seeker who churned", and Claude reads that agent's 45-day memory log and voices it in first person.

---

## Architecture

```
Input          Policy type + variables (A/B/C)
                 ↓
core/policy_impact.py   → per-persona sentiment shock      ← the model's judgment layer
                 ↓
       ┌─────────┴──────────────────────────────┐
       ↓                                        ↓
TIER 1 — numeric (default)            TIER 2 — OASIS LLM agents (optional)
core/agent.py    500 agents           oasis_tier/profiles.py  personas → agents + follower graph
core/feed.py     6/2/1/1 sampling     oasis_tier/runner.py    announcement + brand response
core/simulation  45 rounds, 0.55s     oasis_tier/extract.py   posts → report
                                      oasis_tier/cost.py      token metering + budget cap
       └─────────┬──────────────────────────────┘
                 ↓  (both emit the same report shape)
report/                 → curves, cascade, threshold comparison   (deterministic)
   └── interview.py     → Claude voices one agent
                 ↓
ui/app.py               → Streamlit + Plotly
report/persistence.py   → runs saved to runs/*.json
```

Keeping the LLM out of the *aggregation* path is deliberate. Deterministic numbers with narrow generative surfaces means results are reproducible; most projects in this space over-use the model and end up unable to reproduce their own figures.

### The two tiers

|  | **Tier 1 — numeric** | **Tier 2 — OASIS** |
|---|---|---|
| Cost / run | free | $174–$529 at equivalent scale |
| Speed | 0.55s | minutes |
| Reproducible | bit-identical by seed | trace-logged only |
| Reads announcement *text* | no | **yes** |
| Social graph | persona-matched homophily | explicit follower edges |
| Actions | post | post, comment, like, repost, follow… |
| Use for | sweeps, calibration, CI | realism, copy testing, quotable output |

Tier 1 is the workhorse; Tier 2 is the microscope. Running the same policy through both is the point — where they agree, Tier 1 can be swept cheaply; where they diverge is where the numeric model is wrong.

**Tier 2 needs Python 3.11 and costs money**, so it lives in its own environment and is entirely optional:

```bash
conda create -n policypulse-oasis python=3.11 -y
conda activate policypulse-oasis
pip install -r requirements-oasis.txt
python scripts/run_oasis_spike.py --agents 20 --days 5 --model claude-haiku-4-5 --budget 1.00
```

Full design rationale and measured cost findings: [docs/OASIS_INTEGRATION_PLAN.md](docs/OASIS_INTEGRATION_PLAN.md).

**An early validation signal.** Told only *"your orders are usually small, so any added fee feels like a large share of what you spent"*, a Tier-2 deal-seeker agent wrote: *"$12.95 return fee?? When my average order is $25–40, that's 30–50% of what I spent!"* — that is `policy_impact.py`'s relative-cost disutility, re-derived independently by a language model. Two different methods converging on the same mechanism is the best evidence the abstraction is sound.

---

## What's good about it

- **The abstraction is right.** Separating `policy_sentiment` from `churn_intent` is the single best modelling decision here. Collapsing them into one number — the obvious shortcut — would destroy the model's usefulness.
- **The feed algorithm is the strongest code in the project.** The 6/2/1/1 split across reach, homophily, recency, and random is well-reasoned, handles degenerate pools, redistributes unfilled slots, and guarantees no duplicates.
- **Type C reveals a genuine segment split, then resolves it.** Sustainability buyers start negative on keepit while every other segment starts positive, and the majority partly converts them over 45 days. That trajectory is emergent, not special-cased.
- **The signal now dominates the noise.** Across 10 seeds, the fee ladder moves sentiment monotonically from −0.080 ($4.95) to −0.144 ($12.95) with a per-seed standard deviation of 0.008 — roughly an 8:1 signal-to-noise ratio, and every seed agrees on the sign.
- **Clean module boundaries.** `core` / `report` / `data` / `ui` separate cleanly with `TYPE_CHECKING` imports avoiding circularity. The reporting layer or UI could be swapped without touching `core`.
- **Reproducible by construction.** Every run carries its seed; the same seed reproduces bit-identically.
- **Results are reported with spread, not as point estimates.** Given a per-run standard deviation around 0.02–0.08, a single run is not evidence, and the UI does not present it as one.
- **Fast.** ~0.6s per 45-round run, so a 10-seed batch is interactive.
- **The tests assert behaviour, not just shape.** `tests/test_policy_impact.py` checks that a $12.95 fee ends more negative than a $4.95 one — the property the whole system exists to model.

## What's bad about it

- **Every parameter is a prior.** `DISUTILITY_SCALE = 0.25`, the AOVs, the sensitivities, the store-access fractions — all are reasoned guesses, not fitted values. The model's *structure* is defensible; its *calibration* is not yet evidence-based.
- **No validation against reality.** H&M and Zara both introduced return fees in 2022 with a public record of the response. Until the tool reproduces known outcomes, it is an argument, not a measurement.
- **The social graph is implicit.** Homophily is approximated by persona matching rather than a real follower network. Agents have reach but no edges, so there are no communities, no bridges, and no structural holes.
- **Reach is static.** A post that goes viral does not gain reach, and an influencer whose take is ignored does not lose any.
- **No brand response.** The brand announces once and then goes silent for 45 days. Real brands walk policies back, clarify, or issue apologies — the single most important missing dynamic.
- **Personas are homogeneous within a segment.** All 100 deal seekers share identical susceptibility and elasticity; only reach and the noise draw vary. Real segments have internal distributions.
- **Two scaffolded modules with zero call sites.** `data/seeder.py` and `data/graph_builder.py` are defined, documented, and never called.
- **Sentiment is one-dimensional.** "How do you feel about this policy" collapses fairness, convenience, and value into a single scalar, which is why an agent can't be annoyed at the fee but pleased about the in-store option.

---

## Honest limitations

**Read this before quoting a number from this tool.**

1. **Unvalidated.** No backtest exists. The parameters were reasoned, not measured.
2. **Directional, not quantitative.** "Deal seekers react ~3× more sharply than loyal customers to a flat fee" is the kind of claim the model supports. "11.2% of deal seekers will churn" is not — that number is an artefact of the elasticity constants fed in.
3. **The rank ordering is more trustworthy than the magnitudes.** That $12.95 hurts more than $4.95, and that deal seekers absorb the most damage, are structural results. The size of the gap is a parameter choice.
4. **Churn intent is not churn.** It is a latent propensity in [0, 1], not a probability, and it has never been mapped to observed behaviour.
5. **45 days is an assumption.** So are 500 agents, the 10-post feed, and the 12-round post lifetime.

---

## What's left to build

Ordered by what unblocks the most.

### Validation — the highest-value work remaining

1. **Backtesting harness.** Run against documented policy changes with known outcomes: H&M paid returns (2022, significant backlash), Zara return fee (2022, measured sentiment shift), Lululemon Like New (largely positive), Amazon keepit expansion (minimal backlash). Target: ≥80% directional accuracy.
2. **Parameter fitting.** Use backtest error to fit `DISUTILITY_SCALE`, the per-persona sensitivities, and the AOVs instead of asserting them.
3. **Churn calibration.** Map `churn_intent` onto observed post-policy retention so the number means something.

### Modelling

4. **Brand response mid-simulation** — walk-backs, clarifications, apologies. The biggest missing dynamic.
5. **An explicit social graph** — replace persona-matched homophily with a real follower network so communities and bridges exist.
6. **Within-segment heterogeneity** — draw susceptibility and elasticity from distributions rather than assigning a constant.
7. **Engagement-driven reach** — let posts gain or lose reach based on reaction.
8. **Multi-dimensional sentiment** — separate fairness, convenience, and value.

### Data pipeline

9. **Wire up `data/seeder.py`.** Uncomment the VADER path and use scraped brand sentiment as the pre-policy baseline instead of assuming neutrality.
10. **Wire up or delete `data/graph_builder.py`.** It currently writes a graph nothing reads. Either query it during simulation or cut it and drop the Neo4j dependency.

### Product

11. **Sensitivity sweeps** — run the full fee ladder automatically and plot the response curve rather than making the user run each level by hand.
12. **Export** — CSV/PDF reports for sharing outside the app.
13. **Confidence guidance in the UI** — surface when the seed spread is wide enough that the run count is too low to conclude anything.

---

## Lessons learned

**Passing tests measured the wrong thing.** An earlier version had a green suite alongside a simulation that ignored its primary input entirely — every test asserted structure (500 agents, 45 rounds, no duplicate posts) and none asserted the causal property the system exists to model. The single most valuable test here is three lines: run at $4.95, run at $12.95, assert the results differ. Structural tests prove code runs; behavioural tests prove it means something.

**A parameter that is stored but never read is invisible to every check.** `policy_variables` flowed through the function signature, onto the dataclass, into the report, and onto the screen — present everywhere it could be *seen* and absent from the one place it had to be *used*. Type checkers passed, tests passed, the UI displayed the value back to you. Data flowing *into* a system is not the same as data affecting a system, and only executing with varied inputs distinguishes them.

**Plausible output is more dangerous than broken output.** A crash gets fixed immediately. A smooth sentiment curve that renders beautifully and is pure RNG can be screenshotted into a deck. Polish actively concealed the defect, because a chart that looks like a result gets read as one. Anything producing numbers for decisions needs a way to fail loudly — which is why runs now carry seeds and charts now carry error bands.

**Verify with a distinguishing experiment, not by reading.** The simulation loop *looked* correct: state updates matched the spec, the feed algorithm was genuinely well-built. The defect only became undeniable on running the same seed against three different policies and getting byte-identical output. The question is never "does this look right" but "what observation would differ if it were broken" — then make that observation.

**Building the periphery before the core inverts the risk order.** The feed algorithm, persona library, UI, interview, and cascade detector were all completed to spec while the policy→sentiment mapping — the one component that is genuinely hard and can't be copied from a spec — was deferred to a hardcoded `0.0`. That is the natural order to build in, because the periphery is well-defined and satisfying. It also means a lot of finished work whose value was gated on a piece that was never started. Prototype the riskiest, least-specified component first, even crudely.

**The fix for a runaway feedback loop is usually the update rule, not a bound.** Sentiment used to *accumulate* the feed signal (`+= signal × susceptibility`), which has no fixed point — a persistently positive feed marched every agent to +1 and the clamp was the only thing stopping divergence. Moving a *fraction of the remaining distance* toward the signal converges instead. A model whose only equilibrium is its own boundary is not modelling anything.

**One-directional state variables accumulate noise into signal.** Churn integrated only downward sentiment deltas and never recovered, so symmetric noise ratcheted it upward indefinitely and a population could end up simultaneously delighted and increasingly likely to leave. Anything that can only rise will eventually max out on nothing at all; if a quantity should be able to recover in reality, it needs a decay path in the model.

**Aspirational documentation decays into inaccurate documentation.** An earlier README described the intended system in the present tense — claiming a MiroFish-backed, Zep-memory, Neo4j-graph, LLM-agent platform, next to a fabricated sample output. Nothing was dishonest at the time of writing; the roadmap and the state diverged and the tense never moved. Roadmap belongs in the future tense or a separate section, and illustrative output needs to be labelled at the point of reading.

**Graceful degradation can hide non-existence.** `build_knowledge_graph()` returns immediately without `NEO4J_URI`, which reads as robust engineering. Combined with having no call sites at all, the effect was a component that appeared operational, never errored, and did nothing. A no-op fallback should still be reachable and observable, or the absence becomes structurally undetectable.

**Performance work paid for correctness work.** Feed sampling rescanned a growing pool once per agent per round — ~375M operations, 70 seconds per run. Hoisting the per-round work into a shared index cut it to 0.6s. That 117× speedup is what makes multi-seed batches viable, and multi-seed batches are what make the output honest. The optimisation wasn't a nicety; it was the precondition for reporting uncertainty at all.

---

## Project layout

```
config/personas.yaml      5 archetypes: social + economic parameters
core/
  policy_impact.py        policy + variables + persona → sentiment shock   ← the judgment layer
  policy_types.py         A/B/C definitions and announcement templates
  agent.py                Agent/Post models, state update, posting
  feed.py                 6/2/1/1 feed sampling with per-round index
  simulation.py           45-round orchestration, seeding, batches
report/
  curves.py               sentiment curves, churn by segment, seed aggregation
  cascade.py              sliding-window collapse detection
  report_agent.py         assembles the report dict (no LLM)
  interview.py            the only LLM call — voices one agent
  persistence.py          save/load runs to runs/*.json
data/                     SCAFFOLDS, no call sites
  seeder.py               Reddit scraper (returns neutral 0.0)
  graph_builder.py        Neo4j graph (nothing reads it)
oasis_tier/               TIER 2 - LLM agents on OASIS (optional, needs py3.11)
  profiles.py             personas.yaml -> OASIS agents + follower graph
  runner.py               announcement injection, brand response, manifest
  extract.py              OASIS sqlite -> Tier-1-shaped report
  cost.py                 token metering + hard budget cap
scripts/run_oasis_spike.py  metered Tier-2 spike runner
docs/OASIS_INTEGRATION_PLAN.md  design rationale + measured costs
ui/app.py                 Streamlit: configure / results / interview
tests/                    105 tests (no API calls)
```

### Acknowledgements

- OASIS (CAMEL-AI) and the Stanford HAI Generative Agents paper (Park et al., 2023) as conceptual influences on agent-based social simulation. Neither is a dependency — the simulation engine here is homegrown.
