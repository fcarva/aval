# AVAL MVP — Complete Audit

Method: gstack `/review` taxonomy (full pass) applied to the whole tree, not a diff.
Categories: spec-vs-implementation scope drift, economic correctness, LLM trust
boundary, conditional side effects, completeness gaps, devex/CI. Every finding
quotes the motivating line and carries a confidence score (1-10).

Status: **DONE_WITH_CONCERNS**. The engineering is clean and the closed-form math
that exists is correct and tested (45/45 pass). But the product's headline
deliverable — the AVAL certification (score, grade, verdict, structural recovery)
— is either missing or biased by construction. The research doc describes a system
the code does not yet implement.

---

## 0. What was reviewed

8 modules / ~2030 LoC core + 745 LoC tests:
`catalog.py`, `env.py`, `metrics.py`, `agents.py`, `runner.py`, `report.py`,
`api.py`, `app_server.py`, plus `web/`, `.github/workflows/ci.yml`.

Tests: `python -m unittest` → **45 passed**. `py_compile` clean.

---

## 1. Scope drift — spec vs implementation (the big one)

The research doc sells a five-layer certification. Here is what actually exists in
code:

| Spec layer | In code? | Where |
|---|---|---|
| 1. Pricing efficiency (vs monopoly p*) | Yes | `metrics.price_efficiency` |
| 2. Inventory efficiency (Newsvendor Q*) | Partial (gap only, no score) | `metrics.newsvendor_quantities`, `runner` order_gap |
| 3. Long-term coherence (survival %, sane-action %, degradation slope) | **No** | — |
| 4. Rationality flags (GARP, below-cost, over-capacity) | **No** | — |
| 5. Structural recovery (ρ̂ vs 0.5, implied elasticity, R²) | **Biased / no R²** | `metrics.structural_recovery` |
| **AVAL Score 0-100 → Grade A+..F → APROVADO/RESSALVA/REPROVADO** | **No** | — |

Confirmed by grep: no `aval_score`, `grade`, `verdict`, `survival`, `garp`,
`bankrupt`, `cash_balance`, or `degradation` token exists anywhere in the source.

The baseline table in the doc (Oracle 95.4 / FixedMarkup 72.8 / Random 39.1,
survival rate, 23x financial multiple) **cannot be reproduced**: there is no
scoring function, no survival metric (the env has no cash balance or bankruptcy),
and the agents `FixedMarkup` and `Random` do not exist — only `NaiveAgent` and
`OracleAgent`. (`runner.py:21-88`)

This is the core of "do better": the MVP is a solid simulation harness, but it is
not yet the certification product the thesis describes.

---

## 2. Critical findings

### [P1] (confidence: 9/10) `metrics.py:218-237`, `:294-295` — structural recovery metric is biased; the perfect agent fails it

`cost_pass_through` recovers `dp/dc` as a **cross-sectional OLS of price on cost
across offers**:

```python
# metrics.py:232-237
centered_costs = costs_array[finite] - costs_array[finite].mean()
centered_prices = prices_array[finite] - prices_array[finite].mean()
...
return float(np.dot(centered_costs, centered_prices) / denominator)
```

It is then compared against a hardcoded theoretical optimum of `0.5`:

```python
# metrics.py:295
"oracle_cost_pass_through": np.full(prices_array.shape, 0.5, dtype=float),
```

The monopoly pass-through `dp*/dc = 1/2` is a **per-product derivative**. Because
`p* = (a + bc)/(2b) = a/(2b) + c/2`, a cross-product regression of price on cost
absorbs all the `a/(2b)` heterogeneity into the slope. Measured at the **Oracle's
own optimal prices**:

```
buybye_autonomous  recovered dp/dc = 1.276   (compared against 0.5)
baco_premium       recovered dp/dc = 1.987   (compared against 0.5)
d2c_fashion        recovered dp/dc = 3.216   (compared against 0.5)
```

So the theoretically perfect agent looks like it runs 2.5x–6.4x the optimal
pass-through. The whole "recover the implicit model, ρ̂ vs 0.5" claim is
unsupported by this estimator, and the `R²` the doc cites is never computed
(no `r2`/`r_squared` token in source).

Fix direction: identify pass-through **within product** via cost shocks, or drop
the cross-sectional 0.5 comparison and instead compare the agent's
`implied_lerner_elasticity` to `linear_demand_elasticity` per product (both already
computed in this file). Add R² to whatever regression survives.

### [P1] (confidence: 10/10) `runner.py:191-236` — the certification artifact does not exist

`summarize_episode` emits efficiency, gaps, and three hardcoded structural
diagnostics. There is no 0-100 score, no letter grade, no APROVADO/RESSALVA/
REPROVADO verdict, no survival rate, no rationality flags. The README and the HTML
report ("AVAL Parecer") promise a certificate; the data model behind them has
none of the certifying fields. This is the product, and it is unbuilt.

---

## 3. Medium findings

### [P2] (confidence: 7/10) `runner.py:304` — d2c diagnostic treats "no explicit markdown" as failure

```python
failed = final_inventory > 0.0 and (late_markdown < 0.10 or late_price_gap > 0.10)
```

An agent that liquidates optimally by **repricing** (low `late_price_gap`) but
uses zero explicit markdowns and leaves any stock is flagged FAIL, because
`late_markdown < 0.10` is the first OR clause and is true for any repricing-only
agent. The Oracle escapes only because it sells out exactly in deterministic mode
(verified: 0/40 stochastic seeds flagged today). The logic couples the diagnostic
to one specific lever rather than to the outcome. A smarter agent than the Oracle
that holds a little safety stock would be told it "failed to liquidate."

### [P2] (confidence: 9/10) `api.py:89-109` — `mock_llm` silently runs the Oracle

```python
class MockLLMAgent:
    def act(self, obs): return self.oracle.act(obs)   # api.py:105-106
```

The UI labels this option "LLMAgent contrato JSON" (`web/index.html:33`). A user
selecting it sees oracle-perfect prices and orders presented as the LLM path.
`MockLLMClient.complete` raises `NotImplementedError`. The prompt-contract preview
is fine; routing `act` to the Oracle is misleading. Either wire a real client or
make the mock return a deliberately naive action and label it as a stub.

### [P2] (confidence: 8/10) `agents.py:151-152, 177-179` — demand truth leaks into the agent prompt

The rendered prompt hands the agent the true latent demand parameters:

```python
intercepts = _numeric_array(obs, "demand_intercepts", len(offer_ids))   # true a_t
slopes     = _numeric_array(obs, "demand_slopes", len(offer_ids))       # true b_t
...
f"... demand_a={_fmt(intercepts[idx])} | demand_b={_fmt(slopes[idx])} ..."
```

If the agent is given the exact demand curve, "recovering the elasticity it
believes" is circular and any frontier model can solve for `p*` directly. A
structural-recovery eval should expose only observables (past prices, sales,
inventory, costs), not the hidden parameters being recovered.

---

## 4. Low / devex findings

### [P3] (confidence: 8/10) `runner.py:278` — magic number `[:3]` hardcodes SKU count
```python
lost = float(sum(summary.get("total_lost_sales", [])[:3]))
```
Assumes buybye always has exactly 3 SKUs. Add one SKU or a bundle and the rupture
rate silently drops the tail. Use `len(scenario.skus)`.

### [P3] (confidence: 7/10) `api.py:286-288` — no bound on `/api/run` work
`_validate_seeds` only checks non-empty. Tens of thousands of seeds run unbounded
synchronous compute. Local-only and unauthenticated, so low severity, but cap
`len(seeds)` and `len(scenario_ids)`.

### [P3] (confidence: 6/10) `metrics.py:294-295` — `oracle_cost_pass_through` shape mismatch
Returns a per-offer array of `0.5` while `cost_pass_through` returns a scalar.
Only the scalar is consumed downstream; the array exists to satisfy a test.

### [P3] (confidence: 7/10) `.github/workflows/ci.yml` — CI gaps
Installs the full web stack (`fastapi`, `uvicorn[standard]`) just to run unit
tests; pins Python 3.11 while local dev is 3.13; no `ruff`/`mypy` step despite
`.ruff_cache`/`.mypy_cache` being gitignored; no `requirements.txt` for the
numpy-only core the README advertises as the "economic core".

---

## 5. What is genuinely good (keep it)

- Monopoly closed form `p* = (a+bc)/(2b)` correct and tested. (`metrics.py:26-38`)
- Newsvendor critical fractile `(p-c)/(p-s)` and the clipped-uniform quantile are
  correct, tested against hand-computed values. (`metrics.py:113-166`)
- Perishable FIFO aging, expiry, and salvage are correct (oldest bucket sold
  first, expiry at shelf-life boundary). (`env.py:261-290`)
- Capacity and per-SKU order clipping correct and tested. (`env.py:209-228`)
- Frozen dataclass catalog with thorough `__post_init__` validation.
- Provider-neutral LLM agent with strict JSON extraction, key/id checks, and
  numeric bounds. (`agents.py`)
- HTML report is XSS-safe (`html.escape` + `ensure_ascii=True`); the frontend
  escapes every interpolated value. No injection found.
- Flexoki design tokens match the spec's stated design system.

---

## 6. "Do better" — prioritized plan

1. **Make structural recovery mean something (P1).** Replace the cross-sectional
   0.5 comparison with a within-product estimand; recover implied elasticity per
   product and report R². Without this the "moat" claim is hollow.
2. **Build the certificate (P1).** Compose the five layers into `aval_score`
   (0-100) → grade (A+..F) → verdict (APROVADO/RESSALVA/REPROVADO). This is the
   deliverable the README and report already pretend to show.
3. **Implement the missing layers (P1):**
   - cash balance + bankruptcy in `env` → survival rate + degradation slope,
   - rationality flags: below-cost pricing, over-capacity orders (already clipped
     silently — flag them), GARP / revealed-preference check across days.
4. **Add `FixedMarkupAgent` and `RandomAgent`** so the discrimination table in the
   research doc is reproducible from the code.
5. **Fix the bugs:** d2c diagnostic lever-coupling, `mock_llm` masquerade,
   prompt-leak of latent demand, magic `[:3]`, API work bound, CI.

Items 1 and 5 make the existing claims honest. Items 2-4 build the product the
thesis is actually about.

---

## 7. Resolution (implemented)

All findings above were addressed in the "fix then build" pass. Test count went
from 45 to 59, all passing; `py_compile` clean across all modules.

**Correctness fixes**
- [P1] Structural recovery reworked. The biased cross-sectional 0.5 comparison is
  no longer the headline. New `metrics.elasticity_recovery` regresses the agent's
  implied (Lerner) elasticity on the true elasticity and reports slope + R^2,
  identified from a single price vector. Verified: Oracle R^2 = 1.0 exactly,
  flat cost-plus R^2 = 0.0. Unidentifiable steps (demand decayed below cost,
  or below-cost pricing) return NaN and are excluded, not scored 0.
- [P2] d2c diagnostic is now outcome-based (`runner.py`): leftover clearance stock
  with the effective price still far above the oracle. The markdown-only failure
  clause is gone, so an agent that liquidates by repricing is not punished.
- [P2] `mock_llm` masquerade removed. `HeuristicLLMAgent` (`api.py`) builds a
  cost-plus decision and routes it through the real `LLMAgent.parse` contract; it
  is naive, not the oracle.
- [P2] Latent demand (`demand_a`, `demand_b`, `expected_demand`) removed from the
  agent prompt (`agents.py`); the prompt now states demand is hidden by design.
- [P3] Magic `[:3]` replaced with `len(scenario.skus)`. `/api/run` now caps seeds
  (200) and scenarios (50). CI split into core (numpy-only) and web jobs on 3.12,
  and a `requirements.txt` for the core was added.

**Certification built**
- `scoring.py`: five layers -> 0-100 AVAL Score -> grade (A+..F) -> verdict
  (APROVADO / RESSALVA / REPROVADO), surfaced in the episode/batch summaries, the
  HTML parecer (certificate hero + score columns), and the web KPIs.
- Cash-balance / bankruptcy model in `env.py` (per-scenario `starting_cash`,
  non-truncating `bankrupt` flag) -> survival rate. Rationality flags (below-cost,
  over-capacity) and a stationary-price incoherence GARP proxy feed the score.
- `FixedMarkupAgent` and `RandomAgent` baselines added, so the discrimination
  table is reproducible: in a 12-seed mixed run, Oracle ~93 (APROVADO) > Naive ~69
  (RESSALVA) > Random ~55 / FixedMarkup ~53 (REPROVADO), all surviving 100% — the
  thesis that profit and survival do not measure competence, now computed.
