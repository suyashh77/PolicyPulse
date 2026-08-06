# PolicyPulse

Agent-based simulator for retail return-policy changes. Generates 500 consumer
agents across five personas, applies a proposed policy, and simulates 45 days of
social reaction to estimate sentiment and churn by segment.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # ANTHROPIC_API_KEY, optional
```

## Run

```bash
python -m streamlit run ui/app.py
python -m pytest -q
```

## Layout

```
core/       policy impact model, agents, feed algorithm, simulation, economics
report/     curves, cascade detection, persistence, agent interview
oasis_tier/ optional LLM-agent simulation on OASIS (needs Python 3.11)
ui/         Streamlit app
tests/      132 tests
```

## Optional LLM tier

Requires Python 3.11 and an API key. Costs money per run.

```bash
conda create -n policypulse-oasis python=3.11 -y
conda activate policypulse-oasis
pip install -r requirements-oasis.txt
python scripts/run_oasis_spike.py --agents 20 --days 5 --budget 1.00
```

## Note

Research prototype. Parameters are priors, not fitted to data, and no backtest
against real policy launches has been run. Output is directional.
