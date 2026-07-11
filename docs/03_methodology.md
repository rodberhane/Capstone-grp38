# Methodology and model choices

The project builds a composed, multi-layer framework: a **hazard** layer (where and how deep it floods), an **exposure** layer (what dwelling types are there), and a **vulnerability** layer (the depth-damage curve). Each layer is its own ML task. This document states the model choices and maps them to the report Chapter 3 (Data & Model).

## Guiding constraints (from the literature review)

1. No object-level Irish flood-loss records exist, so supervision is synthetic (harmonised international curves as labels; Wagenaar et al. 2021 transfer).
2. Multivariable methods beat univariable depth-only curves, but must stay interpretable and auditable for a risk-advisory audience.
3. Every output is a band, not a point (McGrath et al. 2019; Gnan et al. 2022).

## Vulnerability layer (the core deliverable): staged architecture

Implemented in `src/vulnerability_model.py`, matching Chapter 3.3.3:

| Stage | Method | Role | Implemented as |
|---|---|---|---|
| 1 Baseline | univariable depth-damage family: linear, polynomial, power, square-root | reference to beat | best spec by cross-validated RMSE |
| 2 Primary | Random Forest regressor, multivariable | main predictor | `RandomForestRegressor` |
| 3 Benchmark | XGBoost / gradient boosting | like-for-like vs prior GT model | `xgboost` (falls back to sklearn gradient boosting if libomp absent) |
| 4 Curve output | GAM, smooth monotone curves | hazard-curve reporting form | `pygam` monotone spline (isotonic fallback) |

**Transfer layer (Wagenaar et al. 2021):** training rows are reweighted to the Irish dwelling-stock proportions, so the aggregate curve reflects the national mix, not the training sample. This needs only the Irish predictor distribution, not Irish loss records.

**Uncertainty layer (McGrath et al. 2019):** the label is not a single line. Each dwelling is assigned one plausible reference curve at random, so the training labels carry the true cross-source scatter; the model learns the conditional mean and the band is the residual spread (low/median/high).

**Feature set:** dwelling type, floor area, storeys, construction age, wall masonry, floor elevation, flood depth, and **depth above floor** (Paulik et al. 2024; the dominant predictor). Depth above floor is where dwelling type enters physically: an upper-floor apartment has a high floor datum and so takes near-zero damage from ground flooding.

**Validation (Chapter 3.3.4):** repeated grouped train-test splits by dwelling (Paulik et al. 2024), reporting MAE, RMSE, R2 and mean bias. Note that R2 here measures fidelity to the synthetic curve family, not to real losses, so it is not directly comparable to the Bles et al. (2026) R2 of 0.39 on real NFIP data. The meaningful comparison is the primary model against the univariable baseline.

## Hazard layer

Implemented in `src/hazard_coastal.py` and `src/hazard_fluvial.py`. Predicts flood occurrence from terrain, comparing logistic regression, random forest and gradient boosting (ROC AUC). Coastal flooding is elevation-driven; fluvial flooding needs a distance-to-river feature (from OpenStreetMap), because rivers run through elevated ground. Rainfall (Met Éireann) is a national-scale feature. Pluvial flooding is out of scope: no open pluvial flood-extent labels exist in Ireland.

## Exposure layer

Implemented in `src/exposure_classifier.py`. A fine dwelling-type classifier (logistic / random forest / gradient boosting) that disaggregates the coarse CSO small-area type split into canonical types, so the vulnerability curve can be placed spatially.

## Deployment

Implemented in `src/deployment_dublin.py`. Applies the vulnerability curve to real Dublin small areas under the real OPW 100-year coastal flood, producing a per-area damaged-floor-area-equivalent and a euro band. Euro is a band across illustrative rebuild rates because the SCSI rebuild-cost table is not openly machine-readable; the damaged-area and spatial pattern are the firm outputs.

## Two documented exclusions (needs-first discipline)

- **Pluvial flooding:** the OPW open portal distributes only fluvial and coastal; the pluvial layer is view-only. Scoped out, rainfall kept as a national feature.
- **SCSI rebuild cost:** not openly machine-readable, so euro is a band rather than a single figure.

Both are logged as verified "cannot obtain, using proxy X because Y", not silent omissions.
