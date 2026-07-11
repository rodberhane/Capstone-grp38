# Harmonisation method

Harmonisation is the preprocessing step that turns raw, incompatible sources into one comparable set: a common dwelling-type taxonomy, a common depth grid, and reference curves aligned to both. It is implemented in `src/harmonisation.py` and produces two files the rest of the pipeline consumes. This document is the method a teammate can cite or reproduce.

## Why it is needed

The raw inputs do not line up. The SEAI BER file describes dwellings with fifteen-plus free-text type labels; the CSO census uses two coarse classes; the international curves each use their own building categories and depth points. Nothing can be modelled until they share a taxonomy and a grid.

## Step 1: canonical dwelling-type taxonomy

Every dwelling is mapped to one of five canonical Irish types. The mapping from the BER `dwellingtypedescr` field:

| BER description | Canonical type |
|---|---|
| Detached house (two or more storeys) | Detached |
| Detached house (single storey) | Bungalow |
| Semi-detached house | Semi-Detached |
| Mid-terrace house, End of terrace house | Terrace |
| Ground/Mid/Top-floor apartment, Apartment, Maisonette, Basement Dwelling | Apartment |
| House (unspecified) | Detached (low confidence, flagged) |

**Bungalow is derived, not labelled:** BER has no bungalow class, so a single-storey detached house is reclassified as Bungalow. This is a large and real share of Irish stock and matters because a single-storey dwelling exposes more of its living area at a given flood depth.

## Step 2: exposure profile aggregation

Dwellings are aggregated to **county x canonical-type profiles**, because BER carries only county geography, not a small-area code. Each profile records the count, the median ground-floor area, the masonry share (from wall type), and the pre-1945 share (from construction year, banded into pre-1945 / 1945-1980 / post-1980). Output: `irish_exposure_ber_profiles.csv`.

**Geolocation bridge:** because BER is county-only, the deployment step splits the coarse CSO small-area dwelling counts into fine canonical types using each county's BER proportions. This is a documented proxy, not a claim of dwelling-level geolocation.

## Step 3: reference-curve harmonisation

Each international curve is interpolated onto a common depth grid (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0 m) and mapped to the canonical types, so all curves are directly comparable as damage ratios (0 to 1). The family:

- **Middlesex FHRC (D20):** Irish anchor, one curve per canonical type.
- **JRC Huizinga 2017 (D19):** three developed-context continental residential curves read directly from the source Excel, Europe, North America, Oceania. The Asia, Africa and Central/South America curves are excluded as contextually inappropriate.

Output: `reference_curve_family.csv` (columns: source, canonical_type, metric, depth_m, damage_ratio).

## Step 4: the depth datum (used by the model, not the table)

The vulnerability model evaluates curves at **depth above floor**, not raw flood depth (Paulik et al. 2024; Gnan et al. 2022). Houses use a 0.15 m floor threshold; apartments use a floor height by storey (ground 0.15 m, mid 4.5 m, top 7.5 m), so an upper-floor apartment takes near-zero structural damage from ground-level flooding. This is where dwelling type enters physically.

## What harmonisation deliberately does not do

- It does not invent a small-area code for BER (uses the county-proportion bridge instead).
- It does not rescale curves to euro (that needs rebuild cost, reported as a band, see methodology).
- It does not include curves that ship only as documentation or absolute-currency tables (Hazus, Welsh), because those lack clean depth-vs-ratio values.

## Reproduce it

```
python src/harmonisation.py
```

Reads BER, the JRC Excel and the Middlesex CSV (paths in `config.py`); writes the two harmonised files. Everything downstream (vulnerability, deployment) reads those two files.
