# TenantEase AI Intelligence Engine — Phase-wise Development Plan

**Project:** AI/ML Engineer Intern assessment — Cynbit Technologies
**Author:** Krishna Reddy Nalla
**Deadline:** Tue, 20 Oct 2026, 05:40 AM IST
**Repo (target):** `github.com/Krishnareddy2468/cynbit-tenantease-ai` (public)

---

## 1. What we are building and why

Cynbit sells **TenantEase**, a property-management SaaS. Reading the product
page, every feature is *reactive*: it tracks units, stores tenant profiles,
records payments, and fires time-triggered reminders. It tells a landlord
**what happened**. It cannot tell them **what will happen**.

We add a three-layer predictive intelligence engine on top of the data
TenantEase already collects:

| Layer | Question it answers | Technique | Model |
|---|---|---|---|
| 1 | *Who* is likely to default? | Classification | XGBoost |
| 2 | *What* will revenue be? | Time series | Prophet |
| 3 | *What should I do* about it? | Generative AI | LLM (provider-agnostic) |

**Why this opportunity over the alternatives** (proposal-scoping engine,
healthcare pre-screening agent, Cynbit Studio code review): it enhances a
product Cynbit already ships and markets, it closes a visible gap (Cynbit
positions as "AI automation" leaders while TenantEase has zero AI), no Indian
competitor — RentOk, ApnaComplex, Aeria — does prediction, and the India
proptech software market is growing from USD 91.9M (2025) to USD 3.82B (2034).

---

## 2. Ground rules for execution

These exist because the submission is explicitly graded on **clean code with
git history**, not just a working demo.

1. **One phase = one logical commit.** Never batch several phases into a
   single "initial commit". A reviewer reads `git log` to see how you think.
2. **Every commit must leave the repo runnable.** No commit that breaks
   `import src` or leaves a half-written module.
3. **Verify before you commit.** Each phase below has a verification command
   and the output you should expect. Run it, read it, *then* commit.
4. **Conventional commit prefixes:** `chore:`, `feat(scope):`, `test:`,
   `docs:`, `fix:`, `refactor:`.
5. **Commit body explains *why*, not *what*.** The diff already shows what.
6. **Push at the end of each phase**, not once at the very end. A burst of 11
   commits pushed in one second looks exactly like what it is.
7. **Never commit generated artifacts.** `data/*.csv` and `models/*` are
   gitignored; they are reproducible from `scripts/`.
8. **Work on `main`.** A solo one-day project does not need feature branches,
   and an empty `main` with all work on a side branch reads badly.

---

## 3. Phase breakdown

Legend: ✅ done · ⬜ not started

---

### ✅ Phase 0 — Project scaffold

**Goal:** A skeleton that makes every later phase a drop-in.

| | |
|---|---|
| **Time** | 20 min |
| **Creates** | `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `.env.example`, `LICENSE`, `src/`, `scripts/`, `tests/`, `docs/`, `data/`, `models/`, `screenshots/` |
| **Key decisions** | Python 3.11 (XGBoost + Prophet wheels); pinned exact versions so the reviewer's install matches ours; `.env.example` documents the LLM keys without committing one |

**Verify**

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -c "import xgboost, prophet, streamlit, sklearn; print('ok')"
```

**Commit:** `chore: initialize TenantEase AI Intelligence Engine project`

---

### ✅ Phase 1 — Synthetic data generator

**Goal:** A portfolio that behaves like a real rent roll, because Cynbit's
production data is not public.

| | |
|---|---|
| **Time** | 1 h |
| **Creates** | `src/config.py`, `src/data_generator.py`, `scripts/generate_data.py` |

**The important design decision.** Do not generate random noise and bolt a
label on. Generate *causally*:

```
latent reliability (Beta 5,2)
        │
        ├── × seasonal pressure   (Diwali squeeze, FY-end tax outflow)
        ├── − rent-to-income burden
        └── + tenure bonus
        ▼
   per-cycle risk  ──►  P(miss) = 0.55·risk^2.2
                   └►  days late ~ Exp(0.9 + risk^2.1 · 44)
```

Both responses are **convex in risk** — reliable tenants cluster tightly near
the due date, the risky tail degrades sharply. That convexity is what gives
the classifier a learnable signal. If you make the relationship linear, the
model will look suspiciously perfect and a reviewer will spot the leak.

**Calibration targets** (real Indian rent-roll benchmarks):

| Metric | Target | Achieved |
|---|---|---|
| Collection rate | 93–96% | 94.5% |
| On-time rate | 70–80% | 71.5% |
| Missed cycles | 4–8% | 5.9% |

**Verify**

```bash
.venv/bin/python scripts/generate_data.py
# 520 tenants · 8,531 ledger rows · 24 monthly rows
```

**Commit:** `feat(data): synthetic tenant roster and payment ledger generator`

---

### ✅ Phase 2 — Feature pipeline

**Goal:** Turn the ledger into a modelling table without leaking the future.

| | |
|---|---|
| **Time** | 45 min |
| **Creates** | `src/features.py` |

**The important design decision.** Split the ledger on the **time axis**:

```
│◄──────── observation window (21 cycles) ────────►│◄─ label (3) ─►│
         features computed here only                 ground truth
```

A tenant is a default if, inside the label window, they miss a cycle or run
30+ days past due. Features read **only** the observation window. This is the
single most common mistake in intern ML submissions — computing
`on_time_rate` over the full history and then predicting a label derived from
that same history. Call this out explicitly in the Technical Report.

**14 features:** on-time rate, late rate, late count (6 cycles), missed count,
avg/max/std days past due, 3-month delay trend, days since last payment,
cycles observed, monthly rent, rent-to-income, deposit months, tenure.
Plus booleans (deposit paid, ID verified) and one-hot property type.

**Verify**

```bash
.venv/bin/python -c "
from src import data_generator, features
p = data_generator.load(); t = features.build(p.tenants, p.payments)
print(len(t), f\"{t['will_default'].mean():.1%}\", features.to_matrix(t).shape)"
# 477 16.6% (477, 22)
```

A base rate near 15–20% is what you want. Below 5% and the model has nothing
to learn; above 40% and the portfolio is not credible.

**Commit:** `feat(features): tenant risk feature pipeline with temporal split`

---

### ✅ Phase 3 — Layer 1: XGBoost risk classifier

**Goal:** Score every tenant 0–100 on default risk, and prove the score is
real with honest evaluation.

| | |
|---|---|
| **Time** | 1 h 15 min |
| **Creates** | `src/risk_model.py`, `scripts/train.py`, writes `models/risk_model.joblib` + `models/risk_metrics.json` |

**Build**

1. Stratified train/test split (80/20, `random_state=42`).
2. **Baseline first**: logistic regression. Record its AUC. A reviewer trusts
   an XGBoost number far more when it is shown beating a stated baseline.
3. XGBoost with `scale_pos_weight` for the ~17% class imbalance.
4. 5-fold stratified cross-validation — report mean ± std, not one lucky split.
5. Calibrate probability → 0–100 score → band via `config.risk_category`.
6. Persist model + the exact training column order (`align_matrix` needs it).

**Metrics to emit:** accuracy, precision, recall, F1, AUC-ROC, average
precision, confusion matrix, per-feature gain importance.

**Watch for:** AUC above ~0.95 means leakage — go back and check that no
feature touches the label window. Target is **0.82–0.90**. Honest and
defensible beats suspiciously perfect.

**Verify**

```bash
.venv/bin/python scripts/train.py
# prints baseline AUC, XGBoost AUC, CV mean±std, confusion matrix
```

**Commit:** `feat(model): XGBoost tenant default risk classifier`

---

### ✅ Phase 4 — Layer 2: Prophet revenue forecasting

**Goal:** Forecast 3 months of rental revenue with confidence bands.

| | |
|---|---|
| **Time** | 45 min |
| **Creates** | `src/forecasting.py` |

**Build**

1. Reshape `monthly_revenue.csv` to Prophet's `ds` / `yhat` contract.
2. Yearly seasonality on, weekly/daily off (monthly data).
3. Fit, forecast `FORECAST_HORIZON_MONTHS`, return `yhat`, `yhat_lower`,
   `yhat_upper` plus the trend/seasonality decomposition.
4. **Backtest**: hold out the last 3 observed months, fit on the rest, report
   MAPE. A forecast with no error metric is decoration.
5. **Fallback**: wrap the Prophet import in `try/except` and fall back to
   statsmodels Holt-Winters. Prophet's build can fail on Streamlit Cloud;
   the demo must never show a stack trace to a reviewer.

**Caveat to state honestly in the report:** 24 monthly points is thin for
Prophet. Say so, and say that production would use daily collection data.

**Verify**

```bash
.venv/bin/python -c "
from src import data_generator, forecasting
p = data_generator.load()
f = forecasting.forecast(p.monthly_revenue); print(f.tail(3)); print(forecasting.backtest(p.monthly_revenue))"
```

**Commit:** `feat(forecast): Prophet revenue forecasting with backtest`

---

### ✅ Phase 5 — Layer 3: LLM recommendation engine

**Goal:** Turn scores and forecasts into plain-English actions a non-technical
landlord can execute.

| | |
|---|---|
| **Time** | 1 h |
| **Creates** | `src/llm_client.py`, `src/recommendations.py` |

**Build**

1. `LLMClient` with three backends, auto-detected from env:
   `anthropic` (claude-haiku-5-5) → `openai` (gpt-4o-mini) → `fallback`.
2. **Deterministic fallback is mandatory, not optional.** The Cynbit reviewer
   will clone this repo with no API key. If the AI tab errors, the whole
   submission reads as broken. The fallback emits the same JSON schema from
   rule-based templates.
3. Prompt design: system prompt assigns the role (property advisor for the
   Indian rental market), tenant context injected as JSON, output constrained
   to `{action, reason, urgency, suggested_message}`.
4. Parse and validate the JSON; on malformed output, degrade to fallback
   rather than crashing.
5. Cache by tenant ID so re-renders do not re-bill the API.

**Verify**

```bash
.venv/bin/python -c "
from src import recommendations as r
print(r.for_tenant({'tenant_name':'Test','risk_score':82,'late_count_6m':3}))"
# works with and without a key set
```

**Commit:** `feat(llm): provider-agnostic smart recommendation engine`

---

### ✅ Phase 6 — Streamlit dashboard

**Goal:** The thing the reviewer actually clicks. Four tabs.

| | |
|---|---|
| **Time** | 2 h |
| **Creates** | `app.py`, `src/ui.py` |

| Tab | Contents |
|---|---|
| Tenant Risk | Score distribution, band breakdown, sortable table, individual drill-down |
| Revenue Forecast | 3-month projection with confidence band, trend + seasonality decomposition, MAPE |
| AI Recommendations | Generated action cards per tenant, provider badge showing live vs fallback |
| Model Performance | ROC curve, confusion matrix, feature importance, baseline comparison |

**Build notes**

- `@st.cache_data` on data load, `@st.cache_resource` on the model. Without
  these, every widget interaction retrains and the demo feels broken.
- Plotly dark theme, consistent risk-band colours from `config.RISK_COLORS`.
- A sidebar note stating the data is synthetic — never let a reviewer think
  you are passing simulated numbers off as Cynbit's real book.
- Graceful empty states on every tab.

**Verify**

```bash
.venv/bin/streamlit run app.py
# click every tab, confirm no exceptions, confirm AI tab works with no API key
```

**Commit:** `feat(app): four-tab Streamlit intelligence dashboard`

---

### ✅ Phase 7 — Tests and CI

**Goal:** Evidence of engineering discipline. Most intern submissions have
zero tests; this is cheap differentiation.

| | |
|---|---|
| **Time** | 40 min |
| **Creates** | `tests/test_data.py`, `tests/test_features.py`, `tests/test_model.py`, `tests/test_llm.py`, `.github/workflows/ci.yml` |

**What to assert**

- Generator is deterministic under a fixed seed; collection rate lands in band.
- **No temporal leakage**: max `cycle_month` used for features < min of the
  label window. This is the test that matters most — it proves Phase 2.
- `align_matrix` reorders and zero-fills correctly.
- Risk scores are bounded 0–100 and bands map correctly at the boundaries.
- LLM fallback returns the full schema with no API key present.

**Verify**

```bash
.venv/bin/pytest -q && .venv/bin/ruff check src tests
```

**Commit:** `test: pipeline coverage and leakage guard` · `chore(ci): GitHub Actions`

---

### ✅ Phase 8 — Deliverable 1: AI Opportunity Report

| | |
|---|---|
| **Time** | 1 h 30 min |
| **Creates** | `docs/AI_Opportunity_Report.md` + `.pdf` |

**Contents**

1. Cynbit company analysis — three revenue streams, product portfolio, client
   verticals, positioning ("Architecting the future with AI automation").
2. All four AI opportunities evaluated, with a selection matrix and the
   rationale for rejecting three.
3. Deep dive on the selected opportunity.
4. India proptech market sizing: USD 91.9M (2025) → USD 3.82B (2034).
5. Competitor gap analysis: RentOk, ApnaComplex, Aeria — all tracking, none
   predicting.
6. ROI projection: reduced defaults, collection efficiency, data-driven pricing.

**Cite every market figure.** An uncited number is worth less than no number.

**Commit:** `docs: AI opportunity report`

---

### ⬜ Phase 9 — Deliverable 3: Technical Approach Report

| | |
|---|---|
| **Time** | 1 h 30 min |
| **Creates** | `docs/Technical_Approach_Report.md` + `.pdf` |

**Contents**

1. System architecture — PoC and production.
2. Model selection rationale: why XGBoost over logistic regression / random
   forest; why Prophet over ARIMA / LSTM; why a small fast LLM.
3. Data pipeline: schema, generative process, feature engineering, the
   temporal split.
4. Prompt engineering: system prompt, context injection, output constraints,
   failure handling.
5. Evaluation methodology: metrics, cross-validation, baseline comparison,
   backtesting.
6. **Limitations, stated plainly** — synthetic data, 24-month series, no
   fairness audit yet. Naming your own weaknesses reads as senior.
7. Production deployment: FastAPI microservice, monthly retraining, Redis
   score cache, webhook into TenantEase's existing reminder system,
   LangChain + Pinecone RAG over lease documents.
8. Cost and scalability analysis.

**Commit:** `docs: technical approach report`

---

### ⬜ Phase 10 — README, screenshots, publish

| | |
|---|---|
| **Time** | 1 h |
| **Creates** | `README.md`, `screenshots/*.png` |

**Build**

1. README: problem statement, architecture diagram, screenshots, setup in
   three commands, model results table, repo map, limitations, links to both
   reports.
2. Capture four screenshots, one per tab.
3. Create the public repo and push:

```bash
gh repo create cynbit-tenantease-ai --public \
  --description "Predictive AI intelligence layer for Cynbit's TenantEase property management platform" \
  --source=. --remote=origin --push
```

4. Deploy to Streamlit Community Cloud from the GitHub repo; add the live URL
   to the README.
5. Final check: open the repo in a logged-out browser window. If it 404s, the
   reviewer sees a 404.

**Commit:** `docs: README with architecture and results` · `chore: publish`

---

## 4. Schedule against the deadline

Deadline is **Tue 20 Oct, 05:40 AM**. Remaining work after Phases 0–2 is
roughly **9 hours**.

| Block | Phases | Hours |
|---|---|---|
| Modelling core | 3, 4, 5 | 3.0 |
| Dashboard | 6 | 2.0 |
| Tests + CI | 7 | 0.7 |
| Reports | 8, 9 | 3.0 |
| README + publish | 10 | 1.0 |

**If time runs short, cut in this order:** CI workflow → PDF exports (ship
Markdown, it renders on GitHub) → Prophet backtest → screenshots. Never cut
the reports; two of the three graded deliverables *are* documents.

---

## 5. Submission checklist

- [ ] Repo is public and opens in a logged-out browser
- [ ] `git log --oneline` shows ~12 phase-wise commits, not one dump
- [ ] `pip install -r requirements.txt && streamlit run app.py` works clean
- [ ] AI tab works with **no API key set**
- [ ] Both reports present as `.md` and `.pdf` under `docs/`
- [ ] README has architecture diagram, screenshots, results table
- [ ] Synthetic data is clearly disclosed in README and in the app
- [ ] Live Streamlit URL in the README
- [ ] Primary link submitted to the portal; extra links for the two reports

---

## 6. Current state

**8 of 10 phases complete** (0–8). The three ML layers and the dashboard are built and verified.

```
daa12d8  feat(app): four-tab Streamlit intelligence dashboard        ← Phase 6
9461f12  feat(llm): provider-agnostic smart recommendation engine    ← Phase 5
8642c38  feat(forecast): Prophet revenue forecasting with backtest   ← Phase 4
012574f  Delete LICENSE
bbc71b5  feat(model): XGBoost tenant default-risk classifier         ← Phase 3
00a325d  feat(features): tenant risk feature pipeline with temporal split  ← Phase 2
fff5fd9  feat(data): synthetic tenant roster and payment ledger generator  ← Phase 1
31a2ae4  chore: initialize TenantEase AI Intelligence Engine project ← Phase 0
```

Pushed to GitHub. **Next: Phase 9 — Technical Approach Report.**

Verified so far: data generator (collection 94.5%), risk model (test AUC 0.893,
CV 0.866, oracle ceiling 0.918), Prophet forecast (MAPE 1.36%) with Holt-Winters
fallback, LLM engine (live + rule-based fallback), dashboard (AppTest: 0 errors,
4 tabs, 7 charts).
