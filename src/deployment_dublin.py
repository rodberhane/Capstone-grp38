"""E4/E5/E6: apply OUR Irish curve to real Dublin small areas under the OPW 100-year
coastal flood. Bridges the coarse SAPS dwelling types to our fine types via BER county
proportions, samples real flood depth per small area, applies our composed curve + band,
and estimates damage. The Irish analog of the Koge map, using our own curve.
"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import geometry_mask
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import config

D0 = config.HARMONISED_DIR                     # irish_exposure_ber_profiles.csv
E2 = config.HARMONISED_DIR                      # irish_composed_curves.csv (shipped here)
OUT = config.ARTIFACTS / "deployment"; OUT.mkdir(parents=True, exist_ok=True)
# Rebuild cost is reported as a band across illustrative EUR/m2 levels, NOT a single sourced
# figure (SCSI is not openly accessible; scope decision 6 July 2026). These are clearly-labelled
# illustrative rate levels, not attributed to any source; a verified free figure can slot in later.
RATES = [2000, 2500, 3000, 3500]
RATE_CENTRAL = 2500.0     # illustrative headline level only

# --- our curve + band, per type, interpolated over depth ---
curve = pd.read_csv(E2 / "irish_composed_curves.csv")
def curve_at(t, depth, col):
    s = curve[curve.canonical_type == t].sort_values("depth_m")
    return float(np.interp(depth, s.depth_m, s[col]))

# --- BER Dublin fine-type split (to break coarse SAPS House/Bungalow into our types) ---
prof = pd.read_csv(D0 / "irish_exposure_ber_profiles.csv")
dub = prof[prof.countyname.astype(str).str.contains("Dublin", case=False, na=False)]
house = dub[dub.canonical.isin(["Detached", "Semi-Detached", "Terrace", "Bungalow"])]
hb_share = (house.groupby("canonical").n.sum() / house.n.sum()).to_dict()   # split of houses+bungalows
area_by_type = dub.groupby("canonical").median_floor_area_m2.median().to_dict()
print("Dublin house/bungalow split:", {k: round(v, 3) for k, v in hb_share.items()})

# --- Dublin coastal small areas (bbox in ITM) + SAPS dwelling counts ---
GPKG = config.SAPS_GPKG
BBOX = (712000, 728000, 730000, 745000)  # E,N Dublin city + coast
sa = gpd.read_file(GPKG, bbox=BBOX, engine="pyogrio")
saps = pd.read_csv(config.SAPS_CSV,
                   usecols=["GEOGID", "T6_1_HB_H", "T6_1_FA_H"], dtype={"GEOGID": str})
saps["T6_1_HB_H"] = pd.to_numeric(saps["T6_1_HB_H"], errors="coerce")
saps["T6_1_FA_H"] = pd.to_numeric(saps["T6_1_FA_H"], errors="coerce")
sa["key"] = sa["SA_PUB2022"].astype(str)
sa = sa.merge(saps, left_on="key", right_on="GEOGID", how="left").fillna({"T6_1_HB_H": 0, "T6_1_FA_H": 0})
print(f"Dublin small areas in bbox: {len(sa)}")

# --- sample real OPW 100-yr coastal depth per small area ---
RAST = config.COASTAL_TIF
with rasterio.open(RAST) as src:
    win = src.window(*BBOX)
    arr = src.read(1, window=win); tr = src.window_transform(win)
    nod = src.nodata
arr = np.where((arr == nod) | (arr < 0), np.nan, arr)
depths, ffrac = [], []
for geom in sa.geometry:
    try:
        m = geometry_mask([geom], out_shape=arr.shape, transform=tr, invert=True)
        cell = arr[m]
        flooded = cell[(~np.isnan(cell)) & (cell >= 0.2)]
        depths.append(float(flooded.mean()) if flooded.size else 0.0)
        ffrac.append(float(flooded.size / max(cell.size, 1)))
    except Exception:
        depths.append(0.0); ffrac.append(0.0)
sa["flood_depth_m"] = depths; sa["flood_frac"] = ffrac

# --- apply OUR curve: rate-INDEPENDENT damaged floor-area equivalent (m2) per small area ---
# damaged_m2 = sum over dwellings of (count x floor area x damage ratio). This is the firm,
# fully data-derived output; euro is damaged_m2 x rebuild rate, applied afterwards as a band.
def damaged_m2(row, col):
    d = row.flood_depth_m
    if d < 0.2 or row.flood_frac <= 0:
        return 0.0
    tot = 0.0
    for t, share in hb_share.items():                 # houses/bungalows
        n = row.T6_1_HB_H * share * row.flood_frac
        tot += n * area_by_type.get(t, 90) * curve_at(t, d, col)
    n_ap = row.T6_1_FA_H * row.flood_frac              # apartments
    tot += n_ap * area_by_type.get("Apartment", 70) * curve_at("Apartment", d, col)
    return tot
for col, name in [("our_curve", "m2_mid"), ("band_lower", "m2_lo"), ("band_upper", "m2_hi")]:
    sa[name] = sa.apply(lambda r: damaged_m2(r, col), axis=1)
# euro at the illustrative central rate, per small area (for the map + CSV)
for m2c, eurc in [("m2_lo", "dmg_lo"), ("m2_mid", "dmg_eur"), ("m2_hi", "dmg_hi")]:
    sa[eurc] = sa[m2c] * RATE_CENTRAL

flooded = sa[sa.m2_mid > 0]
m2_mid, m2_lo, m2_hi = sa.m2_mid.sum(), sa.m2_lo.sum(), sa.m2_hi.sum()
print(f"\nflooded small areas: {len(flooded)}")
print(f"damaged floor-area equivalent (firm, data-derived): {m2_mid/1e3:.1f}k m2  "
      f"(curve band {m2_lo/1e3:.1f}k to {m2_hi/1e3:.1f}k m2)")
print("\nEuro by illustrative rebuild rate (each with curve band), EUR M:")
print(f"  {'rate EUR/m2':>12} {'lower':>8} {'central':>8} {'upper':>8}")
for r in RATES:
    print(f"  {r:>12} {m2_lo*r/1e6:>8.0f} {m2_mid*r/1e6:>8.0f} {m2_hi*r/1e6:>8.0f}")
print(f"\nFull envelope: EUR {m2_lo*min(RATES)/1e6:.0f}M (curve-lower x lowest rate) "
      f"to EUR {m2_hi*max(RATES)/1e6:.0f}M (curve-upper x highest rate).")
print(f"Illustrative headline at EUR {RATE_CENTRAL:.0f}/m2: "
      f"EUR {m2_mid*RATE_CENTRAL/1e6:.0f}M (curve band {m2_lo*RATE_CENTRAL/1e6:.0f} to {m2_hi*RATE_CENTRAL/1e6:.0f}M).")
sa[["key", "T6_1_HB_H", "T6_1_FA_H", "flood_depth_m", "flood_frac",
    "m2_lo", "m2_mid", "m2_hi", "dmg_lo", "dmg_eur", "dmg_hi"]].to_csv(OUT / "dublin_damage_by_sa.csv", index=False)

# --- map ---
fig, ax = plt.subplots(figsize=(11, 11))
sa.plot(ax=ax, color="#eeeeee", edgecolor="#cccccc", linewidth=0.2)
if len(flooded):
    flooded.plot(ax=ax, column="dmg_eur", cmap="Reds", legend=True, edgecolor="#600", linewidth=0.3,
                 legend_kwds={"label": "estimated residential damage (EUR)", "shrink": 0.5})
ax.set_title(f"E6: OUR Irish curve applied to Dublin, 100-year coastal flood\n"
             f"{len(flooded)} small areas flood; {m2_mid/1e3:.0f}k m2 damaged-equivalent "
             f"(curve band {m2_lo/1e3:.0f}k to {m2_hi/1e3:.0f}k). "
             f"Euro envelope EUR {m2_lo*min(RATES)/1e6:.0f}M to {m2_hi*max(RATES)/1e6:.0f}M "
             f"(rebuild {min(RATES)}-{max(RATES)}/m2, illustrative)", fontsize=10)
ax.set_axis_off()
out = OUT / "e6_dublin_our_curve.png"
plt.savefig(str(out), dpi=130, bbox_inches="tight"); print("wrote", out)
