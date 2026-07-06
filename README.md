# iFood Data Science Case — Coupon & Offer Targeting via Incremental Effects

A causal machine learning solution to the case question — **which offer should be sent to each
customer?** — built on the heterogeneous treatment effect (CATE) of each offer, estimated with
Double Machine Learning on a randomized experiment, and converted into an offer-assignment policy
that maximizes net incremental value.

**Headline result:** the targeted policy yields **+R$ 17.3k per 10k customers** (net of discount
costs), versus **−R$ 12.4k** under indiscriminate coupon distribution — a ~R$ 30k swing — while
sending no discount at all to 16% of the base.

---

## Repository structure

```
ifood-case/
├── data/
│   ├── raw/                       # original JSONs (downloaded by notebook 1)
│   └── processed/                 # unified dataset (parquet, written by notebook 1)
├── notebooks/
│   ├── 1_data_processing.ipynb    # PySpark: ingestion, EDA, features, outcomes
│   └── 2_modeling.ipynb           # CATE models, validation, policy, business impact
├── presentation/
│   ├── apresentacao_ifood.pptx    # 5 slides for business leaders (in Portuguese — target audience)
│   └── dashboard_executivo.html   # self-contained interactive dashboard (open in any browser)
├── README.md
└── requirements.txt
```

> The slide deck and dashboard are in Portuguese by design: their audience is the Brazilian
> business leadership specified in the case brief. All code, notebooks and documentation are in
> English.

## How to run

Developed on **Databricks Free Edition** (serverless compute), as recommended by the case brief.

1. Import both notebooks (`Workspace → Import`).
2. Run `notebooks/1_data_processing.ipynb` end-to-end. It will:
   - create the Unity Catalog Volume `workspace.default.ifood_case` (if permissions block
     `CREATE VOLUME`, create it once via the UI: `Catalog → workspace → default → Create → Volume`);
   - download the data from S3 and stage it in the Volume;
   - write `dataset_unified.parquet` to `/Volumes/workspace/default/ifood_case/processed/`.
3. Run `notebooks/2_modeling.ipynb` (the first cell installs `econml`/`lightgbm` via `%pip` and
   restarts the Python kernel automatically). End-to-end runtime: ~15 min on serverless.

Outside Databricks, both notebooks run on vanilla PySpark by pointing the `/Volumes/...` paths to
a local directory; dependencies are pinned in `requirements.txt`.

## The solution in six steps

1. **Raw data → customer-offer instances** (76,277 sends): each `offer received` event becomes one
   row with strictly pre-send features (RFM, offer history, demographics) and fixed-horizon
   outcomes.
2. **Identification:** offer allocation in the test is randomized (balance checks and ~0 covariate
   correlations, Notebook 1), so cross-offer comparisons are causal by design. `informational`
   offers (zero discount) form the comparison group, isolating the discount effect from the effect
   of receiving a communication. No inverse-propensity weighting is used for identification —
   with known uniform propensities it would add variance and remove no bias.
3. **CATE model:** CausalForestDML (econml) with a discrete multi-valued treatment — 8 active
   offers vs. control — LightGBM nuisances, customer-grouped cross-fitting. The DML layer serves
   regularized heterogeneity estimation; average-effect inference uses saturated OLS with
   customer-clustered standard errors (exact under randomization).
4. **Model selection, not a model zoo:** S-, T-, X- and DR-learners are fitted with identical base
   learners and scored on held-out data via the Nie–Wager R-loss (`econml.RScorer`) and uplift
   AUUC — the causal forest's role is earned, not assumed.
5. **Validation:** DML points within the OLS confidence intervals for all arms; monotonic decile
   calibration on unseen customers; Qini well above random; placebo test (treatment permuted
   within send waves) statistically zero across all 8 arms; window-overlap sensitivity analysis.
6. **Policy:** per customer, `argmax_k [CATE_k(x) × ticket − cost_k]`, with a no-discount option.
   Policy value is estimated twice — plug-in and **doubly robust off-policy evaluation** (the one
   legitimate use of propensity weights here, since they are known by design) — and distilled
   into an interpretable depth-3 policy tree for deployment audits.

## Key findings

| Finding | Evidence |
|---|---|
| Most conversions would happen without a coupon | 77.5% 7-day conversion in the control group; 25.8% of offer completions occur without a prior view |
| Average effects are small and highly heterogeneous | ATEs from −1.9 to +8.4 p.p. (clustered 95% CIs); individual CATEs from −10 to +24 p.p. |
| Delivery decides | The two offers without the social channel have effects ≈ 0 (54% view rate vs. 96%) — same discount, ~8 p.p. smaller effect |
| One offer destroys value | R$5 discount, R$20 minimum, weak channels: −1.9 p.p. (95% CI [−3.1, −0.7]) while still paying rewards |
| Discounts pull purchases forward — they do not build habit | Post-expiry retention effect ≤ 0 in all 8 arms (two significantly negative); the lightest BOGO is the only one near zero, with 53% of customers showing a positive individual effect |
| Offers change purchase *timing*, not purchase *size* | Mean ticket inside vs. outside active offer windows: R$ 12.78 vs. R$ 12.74 |
| Coupons activate dormant customers | In the most sensitive quartile, 55% had never purchased; the least sensitive quartile has 2.3× more prior purchases |
| Targeting beats mass sending | +R$ 17.3k (policy) vs. +R$ 14.8k (best single offer) vs. −R$ 12.4k (random) per 10k customers |
| Policy value survives doubly-robust scrutiny | 884 (plug-in) vs. 864 (DR) incremental conversions per 10k — a 2.3% gap |
| The model choice is criterion-driven and the policy is learner-robust | S-learner edges the causal forest on R-loss (0.0061 vs. 0.0039) and champion-arm AUUC (52.6 vs. 44.5); LinearDML's weak AUUC confirms the heterogeneity is non-linear |

## Assumptions and documented decisions

1. **Unit of analysis:** instance (customer, offer received); train/test split, cross-fitting and
   standard errors all grouped by customer.
2. **Treatment = receipt (intention-to-treat):** viewing is post-treatment; conditioning on it
   would introduce selection bias. Consequence: each offer's estimated effect embeds its delivery
   probability — exactly the quantity needed for send decisions.
3. **Fixed-horizon outcomes** (7-day conversion; repurchase in days 10–17 after receipt) eliminate
   the mechanical confounding from offer durations varying with treatment (a ~20 p.p. artifact
   before the fix). Retention-window censoring depends only on the send wave — administrative,
   not selective.
4. **Transactions attributed by time window** (they carry no offer id); 82% of instances have
   overlapping windows — handled as a nuisance control plus a clean-subset sensitivity analysis
   (directions preserved).
5. **`age = 118` is coded missingness** (it coincides exactly with null gender and credit limit);
   kept with an explicit flag — the pattern is informative.
6. **Monetization:** conversion-conditional mean ticket (R$ 33.57) held constant across segments;
   offer cost = mean observed reward per send. Natural refinement: model 7-day spend as a second
   continuous outcome.
7. **Channels are part of the offer package** (not a customer attribute): effects are estimated
   for offers *as delivered*. Disentangling price elasticity from channel effects would require
   experimental channel variation within the same offer.
8. **Off-policy evaluation** on the randomized holdout (plug-in + doubly robust); definitive
   validation is the production A/B test proposed in the presentation.

## Known limitations

- ~30-day test: long-run retention is unobservable; the proxy is repurchase in the week after
  offer expiry.
- No per-order margin data: net value uses spend − reward; true margins would refine the policy
  thresholds.
- Individual CATE confidence intervals are not reported (the forest bootstrap does not cluster by
  customer); decisions rely on segment-level averages plus a ranking validated by calibration and
  Qini.

## Next steps

1. **A/B test** of the targeted policy against the current rule (conversion, net margin,
   repurchase; 4–6 week read).
2. **Retention layer:** prioritize, among targets, the ~1/3 of customers whose retention effect is
   also positive; explore light offers (the only arm with a positive mean retention effect).
3. **Production scoring:** weekly per-customer incremental-effect scores integrated with CRM to
   decide offer, channel and timing.
