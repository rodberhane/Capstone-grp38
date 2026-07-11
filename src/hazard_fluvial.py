"""H2b: test the H2 hypothesis that the fluvial layer needs a river-proximity feature.

H2 showed terrain alone predicts fluvial flooding poorly (AUC 0.80) because Dublin's rivers
run through elevated ground. Here we add distance-to-watercourse (from OpenStreetMap) as a
third feature and re-run the method comparison: elevation+slope alone vs elevation+slope+
distance-to-river. If the hypothesis holds, the river feature should lift fluvial AUC.

OSM waterways pulled live from the Overpass API (free, no key). River vertices reprojected to
ITM; distance = nearest-vertex distance via a KD-tree. Same fluvial label and DEM features as H2.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, rasterio, glob, requests
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import config
BASE = str(config.DATA_DIR)
DEM = f"{BASE}/irish_hazard/d24_copernicus_dem/dem_dublin_itm.tif"
FLUV = sorted(glob.glob(f"{BASE}/irish_hazard/d02_nifm_fluvial/*_0100_*.tif"))
BBOX = (712000, 728000, 730000, 745000)  # Dublin, ITM

# --- OSM watercourses for the Dublin bbox (Overpass) ---
to_ll = Transformer.from_crs(2157, 4326, always_xy=True)
lon0, lat0 = to_ll.transform(BBOX[0], BBOX[1]); lon1, lat1 = to_ll.transform(BBOX[2], BBOX[3])
south, north = min(lat0, lat1), max(lat0, lat1); west, east = min(lon0, lon1), max(lon0, lon1)
q = (f'[out:json][timeout:90];way["waterway"~"river|stream|canal"]'
     f'({south},{west},{north},{east});out geom;')
hdr = {"User-Agent": "capstone-flood-research/1.0 (academic; contact znwayooo@gmail.com)"}
r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, headers=hdr, timeout=120)
r.raise_for_status()
els = r.json()["elements"]
lons, lats = [], []
for e in els:
    for p in e.get("geometry", []):
        lons.append(p["lon"]); lats.append(p["lat"])
print(f"OSM waterway ways: {len(els)}  vertices: {len(lons)}")
to_itm = Transformer.from_crs(4326, 2157, always_xy=True)
rx, ry = to_itm.transform(np.array(lons), np.array(lats))
river_tree = cKDTree(np.c_[rx, ry])

# --- terrain features from DEM ---
with rasterio.open(DEM) as d:
    dem = d.read(1).astype(float); dtr = d.transform; dnod = d.nodata
dem[dem == dnod] = np.nan
gy, gx = np.gradient(dem, 30.0)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))

# --- sample points; fluvial label (reproject to Irish Grid) ---
rng = np.random.default_rng(42)
n = 20000
xs = rng.uniform(BBOX[0], BBOX[2], n); ys = rng.uniform(BBOX[1], BBOX[3], n)
to_ig = Transformer.from_crs(2157, 29903, always_xy=True)
xi, yi = to_ig.transform(xs, ys)
depth = np.zeros(n); igbb = (xi.min(), yi.min(), xi.max(), yi.max())
for f in FLUV:
    with rasterio.open(f) as rr:
        b = rr.bounds
        if b.right < igbb[0] or b.left > igbb[2] or b.top < igbb[1] or b.bottom > igbb[3]:
            continue
        inside = (xi >= b.left) & (xi <= b.right) & (yi >= b.bottom) & (yi <= b.top)
        if inside.sum() == 0:
            continue
        vals = np.array([v[0] for v in rr.sample(np.c_[xi[inside], yi[inside]])], float)
        nod = rr.nodata
        if nod is not None:
            vals[vals == nod] = 0
        vals[vals < 0] = 0
        depth[inside] = np.maximum(depth[inside], np.nan_to_num(vals))
label = (depth >= 0.2).astype(int)

# DEM features + distance-to-river at each point
inv = ~dtr
cols = np.floor((inv.a * xs + inv.b * ys + inv.c)).astype(int)
rows = np.floor((inv.d * xs + inv.e * ys + inv.f)).astype(int)
ok = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])
elev = np.full(n, np.nan); slp = np.full(n, np.nan)
elev[ok] = dem[rows[ok], cols[ok]]; slp[ok] = slope[rows[ok], cols[ok]]
dist_river, _ = river_tree.query(np.c_[xs, ys])

m = ~np.isnan(elev) & ~np.isnan(slp)
y = label[m]
X_terrain = np.c_[elev[m], slp[m]]
X_river = np.c_[elev[m], slp[m], dist_river[m]]
print(f"samples: {len(y)}  flooded (fluvial): {y.sum()} ({100*y.mean():.2f}%)")
print(f"mean distance-to-river flooded vs dry: {dist_river[m][y==1].mean():.0f} m vs {dist_river[m][y==0].mean():.0f} m")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "Random forest":       RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=0),
    "Gradient boosting":   HistGradientBoostingClassifier(random_state=0),
}
for tag, Xf in [("terrain only (elev+slope)", X_terrain), ("terrain + distance-to-river", X_river)]:
    print(f"\nMethod comparison, {tag} (ROC AUC, 5-fold):")
    for name, mdl in models.items():
        auc = cross_val_score(mdl, Xf, y, cv=cv, scoring="roc_auc")
        print(f"  {name:22s} AUC {auc.mean():.3f} +/- {auc.std():.3f}")

rf = models["Random forest"].fit(X_river, y)
print("\nRF feature importance (elev, slope, dist_river): %.2f, %.2f, %.2f" % tuple(rf.feature_importances_))
