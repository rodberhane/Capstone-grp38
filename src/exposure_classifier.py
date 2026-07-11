"""Exposure task, method comparison: fine dwelling-type classifier.

Problem this solves: SAPS (D06) gives only a coarse split (House/Bungalow vs Flat/
Apartment) per small area, but our curve needs the fine canonical type (Detached,
Bungalow, Semi-Detached, Terrace, Apartment). This classifier learns, from real BER
attributes that a small area also exposes (floor area, storeys, construction year,
wall type, county), how to disaggregate a coarse class into fine types.

Label = canonical type (5 classes). Features EXCLUDE dwellingtypedescr (that is what the
label is derived from, so using it would be leakage). We test logistic regression vs
random forest vs gradient boosting: a genuine third ML task where method choice matters.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

import sys; sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent)); import config
DATA = config.DATA_DIR
OUT = config.ARTIFACTS / "exposure"
OUT.mkdir(parents=True, exist_ok=True)

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
        return "Detached"
    return "Other"
ber["canonical"] = ber.apply(canonical, axis=1)
ber = ber[ber["canonical"] != "Other"].copy()

# masonry flag from wall description (same rule as D0), keep raw wall as categorical too
ber["masonry"] = ber["firstwalltype_description"].astype(str).str.contains(
    "Cavity|Stone|Block|brick|Brick|Concrete", case=False, na=False).astype(int)

# plausibility clip (drop obvious data-entry outliers, keep the mass)
ber = ber[(ber["area"].between(20, 1000)) & (ber["year"].between(1700, 2026))]

# --- realistic sample: full set is 1.4M rows; take a stratified 120k for speed ---
ber = pd.concat([g.sample(min(len(g), 24000), random_state=0)
                 for _, g in ber.groupby("canonical")], ignore_index=True)
print(f"[X] training rows: {len(ber):,}")
print(ber["canonical"].value_counts().to_string())

num = ["area", "storeys", "year", "masonry"]
cat = ["countyname"]
for c in num:
    ber[c] = ber[c].astype("float64")
ber["countyname"] = ber["countyname"].astype(str)
y = ber["canonical"].to_numpy(dtype=object)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
])
X = ber[num + cat].reset_index(drop=True)

cv = StratifiedKFold(5, shuffle=True, random_state=0)
models = {
    "Logistic regression": Pipeline([("pre", pre), ("clf", LogisticRegression(
        max_iter=2000, class_weight="balanced"))]),
    "Random forest": Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0))]),
    "Gradient boosting": Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(random_state=0))]),
}

print("\nMethod comparison (5-fold, macro-F1 and accuracy):")
best_name, best_f1, best_pred = None, -1, None
for name, mdl in models.items():
    pred = cross_val_predict(mdl, X, y, cv=cv, n_jobs=-1)
    f1 = f1_score(y, pred, average="macro"); acc = accuracy_score(y, pred)
    print(f"  {name:22s} macro-F1 {f1:.3f}   accuracy {acc:.3f}")
    if f1 > best_f1:
        best_name, best_f1, best_pred = name, f1, pred

print(f"\nBest method: {best_name} (macro-F1 {best_f1:.3f})")
print("\nPer-class (best method):")
print(classification_report(y, best_pred, digits=3))

labels = sorted(np.unique(y))
cm = confusion_matrix(y, best_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(OUT / "exposure_confusion_matrix.csv")
print("Confusion matrix (rows true, cols predicted):")
print(cm_df.to_string())

# permutation-free feature signal: RF importances mapped back to feature names
rf = models["Random forest"].fit(X, y)
ohe = rf.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
feat_names = num + list(ohe.get_feature_names_out(cat))
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=feat_names).sort_values(ascending=False)
print("\nTop feature signals (RF):")
print(imp.head(8).round(3).to_string())
imp.to_csv(OUT / "exposure_feature_importance.csv", header=["importance"])
