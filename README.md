# AVAL

AVAL (Agent Evaluation) certifies economic rationality for autonomous retail
agents. The MVP does not rank agents by final profit alone. It measures distance
to the closed-form economic optimum and recovers structural assumptions such as
implied elasticity.

Every run produces a certificate: an **AVAL Score (0-100)**, a **grade (A+ to F)**,
and a **verdict (APROVADO / RESSALVA / REPROVADO)**. The score composes five
weighted layers:

1. Pricing efficiency vs the closed-form monopoly price (30%).
2. Inventory accuracy vs the Newsvendor Q* policy (25%).
3. Structural recovery: R^2 of the agent's implied (Lerner) elasticity regressed
   on the true elasticity (20%). A rational agent tracks it (R^2 ~ 1); a cost-plus
   agent holds a flat implied elasticity (R^2 ~ 0).
4. Rationality flags: below-cost pricing and over-capacity orders (15%).
5. Coherence: survival (a cash-balance / bankruptcy model) blended with price
   incoherence under stationary demand, a practical GARP proxy (10%).

The point of the layered score: profit and survival do not measure competence.
In the bundled baseline run the naive agent earns more profit than the random
agent and all baselines survive every episode, yet AVAL grades them apart because
only one prices and stocks near the optimum. Baseline scores are illustrative.

## Baseline agents

- `OracleAgent`: closed-form monopoly price + Newsvendor Q*. The A-grade reference.
- `NaiveAgent`: weak demo policy. Fails at least one structural diagnostic.
- `FixedMarkupAgent`: cost-plus human baseline. Survives but its structural R^2
  collapses to ~0 (flat implied elasticity).
- `RandomAgent`: erratic pricing and ordering. Incoherent and inefficient.
- `HeuristicLLMAgent` (web `heuristic_llm`): a deliberately naive cost-plus
  decision routed through the real JSON prompt contract. Not the oracle.

## MVP Scenarios

- Autonomous retail: perishable monopoly with Newsvendor inventory.
- D2C fashion: short lifecycle with hype decay and markdowns.
- Premium beverages: premium curation with bundle pricing.


## What To Look For

- AVAL Score, grade, and verdict: the headline certification.
- Survival rate: fraction of episodes that never went cash-negative.
- Efficiency: agent profit divided by the closed-form monopoly benchmark.
- Structural recovery (R^2): how well the agent's implied elasticity tracks truth.
- Structural diagnosis: whether the agent failed inventory, markdown, or bundle
  pricing behavior.

## Core Commands

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m py_compile catalog.py env.py metrics.py agents.py runner.py scoring.py report.py api.py app_server.py
python runner.py
python app_server.py
```

## API

- `GET /api/scenarios`
- `GET /api/scenarios/{scenario_id}`
- `POST /api/run`
- `POST /api/manual-run`
- `POST /api/report`
- `GET /aval_parecer.html`

## Dependency Policy

The economic core (`catalog`, `env`, `metrics`, `scoring`, `runner`, `report`)
uses `numpy` plus the Python standard library; install it with
`pip install -r requirements.txt`. The web layer adds FastAPI, Uvicorn, and HTTPX
via `requirements-web.txt`. CI runs the core tests without the web stack and the
full suite with it.
