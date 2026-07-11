"""Hazard task, method comparison: predict 100-year coastal flooding from terrain.
Label = OPW coastal depth >= 0.2 m (flooded vs not). Features = elevation, slope (from
the Copernicus DEM). Tests logistic regression vs random forest vs gradient boosting,
a genuine ML task where the method choice matters.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, rasterio
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
FLOOD = f"{BASE}/irish_hazard/d03_coastal_hazard/ncfhm_itm_dep_c_c_0100_f_00.tif"
BBOX = (712000, 728000, 730000, 745000)

# --- terrain features from DEM ---
with rasterio.open(DEM) as d:
    dem = d.read(1).astype(float); dtr = d.transform; dnod = d.nodata
dem[dem == dnod] = np.nan
gy, gx = np.gradient(dem, 30.0)              # 30 m cells
slope = np.degrees(np.arctan(np.hypot(gx, gy)))

# --- sample points, label from flood, features from DEM ---
rng = np.random.default_rng(42)
n = 20000
xs = rng.uniform(BBOX[0], BBOX[2], n); ys = rng.uniform(BBOX[1], BBOX[3], n)
with rasterio.open(FLOOD) as f:
    depth = np.array([v[0] for v in f.sample(np.c_[xs, ys])], float)
    fnod = f.nodata
depth[depth == fnod] = 0
label = (depth >= 0.2).astype(int)

# DEM elevation + slope at the same points (via array index)
inv = ~dtr
cols = np.floor((inv.a * xs + inv.b * ys + inv.c)).astype(int)
rows = np.floor((inv.d * xs + inv.e * ys + inv.f)).astype(int)
ok = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])
elev = np.full(n, np.nan); slp = np.full(n, np.nan)
elev[ok] = dem[rows[ok], cols[ok]]; slp[ok] = slope[rows[ok], cols[ok]]

m = ~np.isnan(elev) & ~np.isnan(slp)
X = np.c_[elev[m], slp[m]]; y = label[m]
print(f"samples: {len(y)}  flooded: {y.sum()} ({100*y.mean():.1f}%)")
print(f"mean elevation flooded vs dry: {X[y==1,0].mean():.1f} m vs {X[y==0,0].mean():.1f} m")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "Random forest":       RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=0),
    "Gradient boosting":   HistGradientBoostingClassifier(random_state=0),
}
print("\nMethod comparison (ROC AUC, 5-fold):")
for name, mdl in models.items():
    auc = cross_val_score(mdl, X, y, cv=cv, scoring="roc_auc")
    print(f"  {name:22s} AUC {auc.mean():.3f} +/- {auc.std():.3f}")

rf = models["Random forest"].fit(X, y)
print("\nRF feature importance: elevation %.2f, slope %.2f" % tuple(rf.feature_importances_))
