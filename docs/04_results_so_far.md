# Results so far

All numbers below come from running the code in this repo on real Irish data. Re-run any script to reproduce them; they are written to `artifacts/`.

## Vulnerability model (the core deliverable)

Trained depth-damage model, staged architecture, validated with 15 repeated grouped train-test splits:

| Model | MAE | RMSE | R2 | bias |
|---|---|---|---|---|
| Baseline (univariable, depth only) | 0.195 | 0.235 | 0.356 | 0.000 |
| Random Forest (primary) | 0.154 | 0.200 | 0.532 | -0.003 |
| XGBoost (benchmark) | 0.153 | 0.197 | 0.545 | -0.002 |

The multivariable models beat the depth-only baseline by about 21 percent in MAE (0.195 to 0.153), consistent with Wagenaar et al. (2017). **Depth above floor dominates the feature importance (0.69)**, exactly as Paulik et al. (2024) report, followed by floor area (0.15) and age (0.08). R2 here measures fidelity to the synthetic curve family (loss at a given depth is a distribution, McGrath 2019), not to real losses, so it is not comparable to the Bles et al. (2026) R2 of 0.39 on real NFIP data; the meaningful result is primary-beats-baseline.

Trained-model Irish curve, damage ratio at 1 m depth (ground-floor dwelling), with band:

| Type | band low | model | band high |
|---|---|---|---|
| Detached | 0.11 | 0.23 | 0.59 |
| Bungalow | 0.16 | 0.34 | 0.59 |
| Semi-Detached | 0.11 | 0.38 | 0.59 |
| Terrace | 0.11 | 0.36 | 0.59 |
| Apartment | 0.10 | 0.38 | 0.59 |

## Hazard layer

| Task | Logistic | Random forest | Gradient boosting |
|---|---|---|---|
| Coastal flood from terrain (ROC AUC) | 0.776 | 0.977 | 0.978 |
| Fluvial, terrain only (ROC AUC) | 0.709 | 0.753 | 0.800 |
| Fluvial + distance-to-river (ROC AUC) | 0.937 | 0.939 | 0.929 |

Coastal flooding is elevation-driven (trees win decisively). Fluvial flooding needs a river-proximity feature: adding distance-to-river lifts AUC from 0.80 to 0.94, because Dublin's rivers run through elevated ground so elevation alone fails.

## Exposure layer

Fine dwelling-type classifier, macro-F1: logistic 0.726, random forest 0.771, gradient boosting 0.766. Reliable for Bungalow, Apartment and Detached; the Semi-Detached vs Terrace boundary is weaker and carries extra uncertainty into the exposure weights.

## Deployment (Dublin worked example)

Applying the curve to real Dublin small areas under the OPW 100-year coastal flood: 379 small areas flood; 286,000 m2 of damaged floor-area equivalent (the firm, data-derived output), curve band 136,000 to 782,400 m2. Euro is reported as a band across illustrative rebuild rates (SCSI not openly available): envelope about EUR 272M to 2,739M across 2,000 to 3,500 EUR/m2, illustrative headline EUR 715M at 2,500 EUR/m2. Damage concentrates on Sandymount, Ringsend, the Liffey mouth and Clontarf, Dublin's real coastal flood-risk areas.

## Honest limitations

- Supervision is synthetic (no paired Irish loss data), so the vulnerability model reproduces the harmonised curve surface with realistic scatter; the real test comes when Irish loss labels (Storm Babet wrack-marks, future claims) replace the synthetic ones.
- Type resolution is strongest through floor level (apartments) and single-storey (bungalow); Detached/Semi/Terrace curves are similar because the reference curves are.
- Pluvial flooding and SCSI rebuild cost are out of scope (verified unobtainable), documented in `docs/03_methodology.md`.
