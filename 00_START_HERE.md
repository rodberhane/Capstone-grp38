# Project 38: Irish residential flood depth-damage modelling

Standalone, runnable version of the Grant Thornton capstone (Group 6). This repo lets a tester download the data, run the pipeline, and see the models and results end to end.

**What it does:** builds dwelling-type-resolved Irish residential flood **depth-damage curves** and a supporting **hazard** and **exposure** model from open data, supervised by harmonised international curves (no paired Irish loss data exists), with every output reported as an uncertainty band. It then applies the curve to Dublin under a real OPW flood as a worked example.

## Quick start

```
# 1. install dependencies (use a fresh virtual environment)
pip install -r requirements.txt
#    macOS + XGBoost also needs OpenMP:  brew install libomp

# 2. put the raw data in ./data/ (see docs/01_data_sources.md for links + folder layout)
#    data/ is gitignored (1.7GB), so it is not on GitHub: download it into data/ once.

# 3. run the whole pipeline
python run_all.py
#    or run one layer:
python src/vulnerability_model.py
```

Data lives in `./data/` inside the repo (config.DATA_DIR points there by default; no external paths). It is gitignored, so after cloning from GitHub you download the files into `data/` once, keeping the subfolder layout in `docs/01_data_sources.md`. Outputs (CSV tables) are written to `artifacts/`; the harmonised inputs are shipped in `artifacts/harmonised/` so the vulnerability model runs immediately.

## What to read, in order

1. `docs/01_data_sources.md` - every dataset, direct download link, licence, where to put it.
2. `docs/02_harmonisation_method.md` - how raw sources become one comparable set (the preprocessing method).
3. `docs/03_methodology.md` - model choices for all three layers, mapped to the report Chapter 3.
4. `docs/04_results_so_far.md` - the current results, with real numbers you can reproduce.

## Repo layout

```
config.py                     data paths (edit DATA_DIR here)
run_all.py                    run the whole pipeline
requirements.txt
src/
  harmonisation.py            build canonical taxonomy + reference-curve family   [config-driven]
  vulnerability_model.py      trained depth-damage model (Ch.3 staged pipeline)   [config-driven]
  exposure_classifier.py      dwelling-type classifier
  hazard_coastal.py           coastal flood from terrain
  hazard_fluvial.py           fluvial flood + distance-to-river
  deployment_dublin.py        apply curve to Dublin, euro band
docs/                         the four guides above
artifacts/                    outputs (harmonised inputs shipped here)
```

## The model in one paragraph

The vulnerability core is a **staged pipeline** (report Chapter 3): a univariable depth-only baseline, a Random Forest primary model, an XGBoost benchmark, and a GAM smooth-curve output, supervised by the reference-curve family as synthetic labels (Wagenaar et al. 2021 transfer, since Ireland has no loss records), evaluated at **depth above floor** (Paulik et al. 2024), reweighted to the Irish dwelling stock, and reported as a band (McGrath et al. 2019). The primary model beats the depth-only baseline by about 21 percent MAE. Hazard and exposure are separate trained classifiers.

## Honest status (so a tester knows what they are looking at)

- **Trained on our data:** hazard, exposure, and the vulnerability model all train on real Irish data (BER, OPW rasters, DEM). Supervision for vulnerability is synthetic because no Irish loss records exist; the model reproduces the harmonised curve surface with realistic scatter and will accept real loss labels later without redesign.
- **Config coverage:** all six scripts are fully config-driven. Every data path is read from `config.py` and every output is written under `artifacts/`; there are no absolute paths, so the pipeline runs unchanged on any machine once the data is downloaded into `data/`.
- **Out of scope (verified, not silent):** pluvial flood labels (not in the OPW open portal) and SCSI rebuild cost (not openly machine-readable). Euro is reported as a band.
