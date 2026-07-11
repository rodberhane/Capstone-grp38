"""D0: harmonisation and preprocessing. Emits two tables:
  1. irish_exposure_ber_profiles.csv  (BER county x canonical dwelling type profiles)
  2. reference_curve_family.csv       (reference curves on a common depth grid, per canonical type)
All values read from the actual data; gaps carried explicitly; nothing invented.
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import config
import numpy as np, pandas as pd
try:
    import psycopg2   # only needed for the optional OS2 Danish shape prior
except Exception:
    psycopg2 = None

DATA = config.DATA_DIR
OUT = config.HARMONISED_DIR
OUT.mkdir(parents=True, exist_ok=True)
GRID = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])

# ---------- 1. BER county x canonical-type exposure profiles ----------
BER = DATA / "irish_buildings/d05_seai_ber/building_energy_ratings.csv"
cols = ["countyname", "dwellingtypedescr", "groundfloorarea(sq_m)", "nostoreys",
        "year_of_construction", "firstwalltype_description"]
ber = pd.read_csv(BER, usecols=cols, dtype=str)
ber["area"] = pd.to_numeric(ber["groundfloorarea(sq_m)"], errors="coerce")
ber["storeys"] = pd.to_numeric(ber["nostoreys"], errors="coerce")
ber["year"] = pd.to_numeric(ber["year_of_construction"], errors="coerce")

def canonical(row):
    t = str(row["dwellingtypedescr"]).strip()
    if t == "Detached house":
        return "Bungalow" if row["storeys"] == 1 else "Detached"
    if t == "Semi-detached house":
        return "Semi-Detached"
    if t in ("Mid-terrace house", "End of terrace house"):
        return "Terrace"
    if t in ("Ground-floor apartment", "Mid-floor apartment", "Top-floor apartment",
             "Apartment", "Maisonette", "Basement Dwelling"):
        return "Apartment"
    if t == "House":
        return "Detached"   # low confidence, flagged below
    return "Other"
ber["canonical"] = ber.apply(canonical, axis=1)
ber["age_band"] = pd.cut(ber["year"], [0, 1945, 1980, 2100],
                         labels=["pre_1945", "1945_1980", "post_1980"])

prof = ber.groupby(["countyname", "canonical"]).agg(
    n=("canonical", "size"),
    median_floor_area_m2=("area", "median"),
    pct_masonry=("firstwalltype_description", lambda s: 100*s.astype(str).str.contains("Cavity|Stone|Block|brick|Brick|Concrete", case=False, na=False).mean()),
    pct_pre1945=("age_band", lambda s: 100*(s == "pre_1945").mean()),
).reset_index()
prof.to_csv(OUT / "irish_exposure_ber_profiles.csv", index=False)
print(f"[BER] {len(ber):,} dwellings -> {len(prof)} county x type profiles")
print(ber["canonical"].value_counts().to_string())

# ---------- 2. reference-curve family on the common grid ----------
def interp(depths, ratios):
    d = np.asarray(depths, float); r = np.asarray(ratios, float)
    ok = ~np.isnan(d) & ~np.isnan(r); d, r = d[ok], r[ok]
    order = np.argsort(d)
    return np.clip(np.interp(GRID, d[order], r[order], left=0), 0, 1)

fam = []

# 2a. Middlesex D20 (Irish anchor, per-type % of value)
mid = pd.read_csv(DATA / "europe/curves/d20_middlesex_irish_curves/middlesex_irish_residential.csv",
                  skiprows=2, encoding="utf-8")
mid.columns = [str(c).replace("​", "").strip() for c in mid.columns]
def num(x):
    return pd.to_numeric(str(x).replace("​", "").replace("%", "").strip(), errors="coerce")
mid = mid.rename(columns={mid.columns[0]: "depth"})
mid["depth"] = mid["depth"].apply(num)
mid = mid[mid["depth"].notna()]
midmap = {"Detached": "Detached", "Semi-Detached": "Semi-Detached", "Terrace": "Terrace",
          "Bungalow": "Bungalow", "Flat": "Apartment"}
for col, canon in midmap.items():
    if col in mid.columns:
        r = interp(mid["depth"].values, mid[col].apply(num).values / 100.0)
        for d, v in zip(GRID, r):
            fam.append(("Middlesex_D20", canon, "value_ratio", d, round(float(v), 4)))

# 2b. JRC Huizinga D19 continental residential curves, read directly from the supplementary
# Excel (verified, not hardcoded). We add the developed-context curves that are defensible
# priors for Irish residential: Europe, North America, Oceania. The other continental curves
# (Centr&South America, Asia, Africa) exist in the same sheet but are excluded from the Irish
# band as contextually inappropriate. Normalised damage factors 0 to 1, so already value_ratio.
XLS = DATA / "europe/curves/d19_jrc_huizinga_curves/copy_of_global_flood_depth-damage_functions__30102017.xlsx"
hz = pd.read_excel(XLS, sheet_name="Damage functions", header=None)
start = hz.index[hz[0].astype(str).str.strip() == "Residential buildings"][0]
block = hz.iloc[start:start + 9]            # depths 0,0.5,1,1.5,2,3,4,5,6
hz_depth = pd.to_numeric(block[1], errors="coerce").values
jrc_cols = {"JRC_Europe_D19": 2, "JRC_NorthAmerica_D19": 3, "JRC_Oceania_D19": 7}
for src, col in jrc_cols.items():
    vals = pd.to_numeric(block[col], errors="coerce").values
    rj = interp(hz_depth, vals)
    for canon in ["Detached", "Bungalow", "Semi-Detached", "Terrace", "Apartment"]:
        for d, v in zip(GRID, rj):
            fam.append((src, canon, "value_ratio", d, round(float(v), 4)))

# 2c. OS2 Danish residential, normalised to a shape prior (absolute DKK, so shape only)
try:
    con = psycopg2.connect(host="localhost", port=5432, user="postgres", dbname="skadesokonomi")
    os2 = pd.read_sql("SELECT skade_type, b1, b2 FROM fdc_lookup.skadefunktioner "
                      "WHERE skade_kategori='Helårsbeboelse'", con); con.close()
    for _, row in os2.iterrows():
        perm2 = row.b1 * np.log(np.clip(GRID * 100, 1, None)) + row.b2
        perm2 = np.clip(perm2, 0, None)
        shape = perm2 / perm2.max() if perm2.max() > 0 else perm2
        ft = {"Stormflod": "coastal", "Skybrud": "pluvial", "Vandløb": "fluvial"}.get(row.skade_type, row.skade_type)
        for canon in ["Detached", "Semi-Detached", "Terrace", "Apartment"]:
            for d, v in zip(GRID, shape):
                fam.append((f"OS2_{ft}_shape", canon, "shape_normalised", d, round(float(v), 4)))
except Exception as e:
    print("[OS2] skipped (DB down?):", e)

fam_df = pd.DataFrame(fam, columns=["source", "canonical_type", "metric", "depth_m", "damage_ratio"])
fam_df.to_csv(OUT / "reference_curve_family.csv", index=False)
print(f"\n[curves] family rows: {len(fam_df)}  sources: {sorted(fam_df.source.unique())}")

# spread (uncertainty prior) among value_ratio curves, per type at 1 m
vr = fam_df[(fam_df.metric == "value_ratio") & (fam_df.depth_m == 1.0)]
print("\nvalue-ratio spread at 1.0 m depth (the uncertainty prior):")
print(vr.groupby("canonical_type").damage_ratio.agg(["min", "median", "max"]).round(3).to_string())
print("\nNot in clean curve form (flag to confirm): Hazus D22 (documentation export), Welsh D18 (AAD tables).")
