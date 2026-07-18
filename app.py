"""Irish residential flood depth-damage explorer.

This app reads the existing harmonised labels, model artifacts, and deployment outputs only.
It does not retrain anything; it visualises the synthetic supervision setup and the trained
curves so the results stay comparable to the rest of Project 38.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import config


st.set_page_config(
    page_title="Irish residential flood depth-damage explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    :root {
        color-scheme: dark;
    }
    html, body, .stApp {
        background: #050608;
        color: #e8edf2;
        font-family: "Aptos", "Segoe UI", sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(13, 76, 146, 0.16), transparent 34%),
            radial-gradient(circle at right top, rgba(0, 168, 168, 0.12), transparent 26%),
            linear-gradient(180deg, #050608 0%, #090c12 100%);
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f15 0%, #070b10 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    h1, h2, h3, h4 {
        letter-spacing: -0.02em;
    }
    .soft-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }
    .muted {
        color: rgba(232,237,242,0.68);
    }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


MODEL_COLORS = {
    "selected": "#7dd3fc",
    "reference": "#7c8796",
    "band": "rgba(124, 135, 150, 0.18)",
    "band_line": "rgba(125, 211, 252, 0.18)",
}

DEPTH_GRID = np.linspace(0.0, 3.0, 121)
EUR_GRID = np.linspace(0.0, 3500.0, 121)


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return read_csv_cached(str(path), file_mtime(path)).copy()
    except Exception:
        return None


def load_reference_family() -> pd.DataFrame | None:
    return load_optional_csv(config.CURVE_FAMILY)


def load_metrics() -> pd.DataFrame | None:
    return load_optional_csv(config.ARTIFACTS / "vulnerability_extended" / "metrics_comparison.csv")


def load_curves() -> pd.DataFrame | None:
    return load_optional_csv(config.ARTIFACTS / "vulnerability_extended" / "curves_by_model.csv")


def load_quantiles() -> pd.DataFrame | None:
    return load_optional_csv(config.ARTIFACTS / "vulnerability_extended" / "quantile_bands.csv")


def load_feature_importance() -> pd.DataFrame | None:
    return load_optional_csv(config.ARTIFACTS / "vulnerability_extended" / "feature_importance.csv")


def load_shap_values() -> pd.DataFrame | None:
    return load_optional_csv(config.ARTIFACTS / "vulnerability_extended" / "shap_values.csv")


def canonical_types_from_reference(ref: pd.DataFrame | None) -> list[str]:
    if ref is None or ref.empty or "canonical_type" not in ref.columns:
        return []
    return list(pd.unique(ref["canonical_type"]))


def model_options(metrics: pd.DataFrame | None) -> list[str]:
    if metrics is None or metrics.empty or "model" not in metrics.columns:
        return []
    ordered = metrics.sort_values("R2", ascending=False)["model"].dropna().astype(str).tolist()
    return ordered


def source_summary(ref: pd.DataFrame, canonical_type: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    sub = ref[(ref["metric"] == "value_ratio") & (ref["canonical_type"] == canonical_type)].copy()
    if sub.empty:
        return np.array([]), np.array([]), {}
    sources = [s for s in pd.unique(sub["source"]) if pd.notna(s)]
    curves: dict[str, np.ndarray] = {}
    x = np.sort(pd.unique(sub["depth_m"].astype(float)))
    for source in sources:
        sdf = sub[sub["source"] == source].sort_values("depth_m")
        curves[source] = np.interp(DEPTH_GRID, sdf["depth_m"].astype(float).to_numpy(), sdf["damage_ratio"].astype(float).to_numpy())
    if not curves:
        return x, np.array([]), {}
    stacked = np.vstack(list(curves.values()))
    return x, stacked.min(axis=0), curves


def curve_for_model(curves: pd.DataFrame, model: str, canonical_type: str) -> pd.DataFrame:
    if curves is None or curves.empty:
        return pd.DataFrame()
    sub = curves[(curves["model"].astype(str) == model) & (curves["canonical_type"].astype(str) == canonical_type)].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("depth_m")
    return sub


def interpolate_curve(curve_df: pd.DataFrame, x_col: str, y_col: str, grid: np.ndarray) -> np.ndarray:
    if curve_df.empty or x_col not in curve_df.columns or y_col not in curve_df.columns:
        return np.full_like(grid, np.nan, dtype=float)
    x = curve_df[x_col].astype(float).to_numpy()
    y = curve_df[y_col].astype(float).to_numpy()
    if len(x) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, x, y, left=y[0], right=y[-1])


def selected_model_band(curve_df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if curve_df.empty:
        return None, None, "no curve"
    if {"band_low", "band_high"}.issubset(curve_df.columns) and curve_df[["band_low", "band_high"]].notna().any().any():
        low = interpolate_curve(curve_df.dropna(subset=["band_low", "band_high"]), "depth_m", "band_low", DEPTH_GRID)
        high = interpolate_curve(curve_df.dropna(subset=["band_low", "band_high"]), "depth_m", "band_high", DEPTH_GRID)
        if np.isfinite(low).any() and np.isfinite(high).any():
            return low, high, "model band"
    return None, None, "no model band"


def quantile_band_for_type(quantiles: pd.DataFrame | None, canonical_type: str) -> pd.DataFrame:
    if quantiles is None or quantiles.empty:
        return pd.DataFrame()
    sub = quantiles[quantiles["canonical_type"].astype(str) == canonical_type].copy()
    return sub.sort_values("depth_m")


def style_metrics_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    if df.empty:
        return df.style
    best = df["R2"].idxmax()

    def highlight(row: pd.Series) -> list[str]:
        return ["background-color: rgba(125, 211, 252, 0.15)" if row.name == best else "" for _ in row]

    return (
        df.style.apply(highlight, axis=1)
        .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}", "bias": "{:.4f}"})
    )


def bar_figure_from_table(df: pd.DataFrame, x_col: str, y_col: str, color_col: str, title: str, horizontal: bool = True) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    if horizontal:
        for color_name in pd.unique(df[color_col]):
            sub = df[df[color_col] == color_name]
            fig.add_trace(go.Bar(
                x=sub[y_col],
                y=sub[x_col],
                name=str(color_name),
                orientation="h",
            ))
        fig.update_layout(barmode="group")
    else:
        for color_name in pd.unique(df[color_col]):
            sub = df[df[color_col] == color_name]
            fig.add_trace(go.Bar(
                x=sub[x_col],
                y=sub[y_col],
                name=str(color_name),
            ))
        fig.update_layout(barmode="group")
    fig.update_layout(title=title, template="plotly_dark", paper_bgcolor="#050608", plot_bgcolor="#050608")
    return fig


def build_harmonisation_figure(ref: pd.DataFrame, canonical_type: str) -> go.Figure:
    sub = ref[(ref["metric"] == "value_ratio") & (ref["canonical_type"] == canonical_type)].copy()
    fig = go.Figure()
    if sub.empty:
        return fig
    sources = list(pd.unique(sub["source"]))
    for source in sources:
        sdf = sub[sub["source"] == source].sort_values("depth_m")
        fig.add_trace(go.Scatter(
            x=sdf["depth_m"],
            y=sdf["damage_ratio"],
            mode="lines",
            name=str(source),
            line=dict(width=1.8),
        ))
    agg = sub.groupby("depth_m", as_index=False)["damage_ratio"].agg(["min", "max"]).reset_index()
    depth = agg["depth_m"].to_numpy(dtype=float)
    low = agg["min"].to_numpy(dtype=float)
    high = agg["max"].to_numpy(dtype=float)
    fig.add_trace(go.Scatter(x=depth, y=high, line=dict(width=0), showlegend=False, hoverinfo="skip", name="spread upper"))
    fig.add_trace(
        go.Scatter(
            x=depth,
            y=low,
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(125, 211, 252, 0.18)",
            showlegend=True,
            name="source spread",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=f"Reference-curve family: {canonical_type}",
        template="plotly_dark",
        paper_bgcolor="#050608",
        plot_bgcolor="#050608",
        xaxis_title="Depth above floor (m)",
        yaxis_title="Damage ratio",
        legend_title_text="Published sources",
    )
    fig.update_xaxes(range=[0, 3.0])
    fig.update_yaxes(range=[0, 1.0])
    return fig


def build_model_curves_figure(metrics: pd.DataFrame, curves: pd.DataFrame, ref: pd.DataFrame, selected_model: str, canonical_types: list[str]) -> go.Figure:
    rows = 2
    cols = 3
    subplot_titles = canonical_types + [""]
    while len(subplot_titles) < rows * cols:
        subplot_titles.append("")
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles[: rows * cols])
    model_order = metrics["model"].astype(str).tolist() if not metrics.empty else []
    if selected_model not in model_order:
        return go.Figure()
    selected_color = MODEL_COLORS["selected"]
    for idx, canonical_type in enumerate(canonical_types, start=1):
        row = 1 if idx <= 3 else 2
        col = idx if idx <= 3 else idx - 3
        ref_sub = ref[(ref["metric"] == "value_ratio") & (ref["canonical_type"] == canonical_type)].copy()
        if not ref_sub.empty:
            agg = ref_sub.groupby("depth_m", as_index=False)["damage_ratio"].agg(["min", "max"]).reset_index()
            fig.add_trace(
                go.Scatter(x=agg["depth_m"], y=agg["max"], line=dict(width=0), showlegend=False, hoverinfo="skip"),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=agg["depth_m"],
                    y=agg["min"],
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=MODEL_COLORS["band"],
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )

        for model in model_order:
            msub = curve_for_model(curves, model, canonical_type)
            if msub.empty:
                continue
            color = selected_color if model == selected_model else MODEL_COLORS["reference"]
            width = 3.5 if model == selected_model else 1.0
            opacity = 1.0 if model == selected_model else 0.25
            fig.add_trace(
                go.Scatter(
                    x=msub["depth_m"],
                    y=msub["damage_ratio"],
                    mode="lines",
                    name=model,
                    line=dict(color=color, width=width),
                    opacity=opacity,
                    showlegend=(idx == 1),
                    hovertemplate=f"{model}<br>depth=%{{x:.2f}} m<br>damage=%{{y:.3f}}<extra></extra>",
                ),
                row=row, col=col,
            )
        fig.update_xaxes(range=[0, 3.0], row=row, col=col)
        fig.update_yaxes(range=[0, 1.0], row=row, col=col)

    fig.update_layout(
        title=f"Model curves: {selected_model}",
        template="plotly_dark",
        paper_bgcolor="#050608",
        plot_bgcolor="#050608",
        height=780,
        legend_title_text="Model references",
    )
    return fig


def deployment_inputs_present() -> bool:
    required = [
        config.ARTIFACTS / "deployment" / "dublin_damage_by_sa.csv",
        config.ARTIFACTS / "deployment" / "dublin_sa_geometry.json",
        config.EXPOSURE_PROFILES,
    ]
    return all(path.exists() for path in required)


def dublin_key_column(gdf: gpd.GeoDataFrame) -> str | None:
    candidates = ["SA_PUB2022", "SA_CODE", "GEOGID", "key"]
    for column in candidates:
        if column in gdf.columns:
            return column
    return None


def load_dublin_geometry_and_inputs() -> tuple[gpd.GeoDataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    dmg_path = config.ARTIFACTS / "deployment" / "dublin_damage_by_sa.csv"
    geom_path = config.ARTIFACTS / "deployment" / "dublin_sa_geometry.json"
    if not dmg_path.exists() or not geom_path.exists() or not config.EXPOSURE_PROFILES.exists():
        return None, None, None
    try:
        damage = load_optional_csv(dmg_path)
        exposure = load_optional_csv(config.EXPOSURE_PROFILES)
        gdf = gpd.read_file(geom_path)
        return gdf, damage, exposure
    except Exception:
        return None, None, None


def dublin_type_shares(exposure: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    dub = exposure[exposure["countyname"].astype(str).str.contains("Dublin", case=False, na=False)].copy()
    if dub.empty:
        dub = exposure.copy()
    house = dub[dub["canonical"].isin(["Detached", "Semi-Detached", "Terrace", "Bungalow"])].copy()
    if house.empty or house["n"].sum() == 0:
        hb_share = {k: 0.25 for k in ["Detached", "Semi-Detached", "Terrace", "Bungalow"]}
    else:
        hb_share = (house.groupby("canonical")["n"].sum() / house["n"].sum()).to_dict()
    area_by_type = dub.groupby("canonical")["median_floor_area_m2"].median().to_dict()
    return hb_share, area_by_type


def model_curve_value(curves: pd.DataFrame, model: str, canonical_type: str, depth: float) -> float:
    sub = curve_for_model(curves, model, canonical_type)
    if sub.empty:
        return float("nan")
    return float(np.interp(depth, sub["depth_m"].astype(float).to_numpy(), sub["damage_ratio"].astype(float).to_numpy(), left=sub["damage_ratio"].iloc[0], right=sub["damage_ratio"].iloc[-1]))


def model_curve_band(curves: pd.DataFrame, quantiles: pd.DataFrame | None, ref: pd.DataFrame | None, model: str, canonical_type: str, depth: float) -> tuple[float, float, str]:
    sub = curve_for_model(curves, model, canonical_type)
    if sub.empty:
        return float("nan"), float("nan"), "no curve"
    if model == "QuantileRF" and quantiles is not None and not quantiles.empty:
        qsub = quantile_band_for_type(quantiles, canonical_type)
        if not qsub.empty:
            lo = np.interp(depth, qsub["depth_m"].astype(float).to_numpy(), qsub["q10"].astype(float).to_numpy())
            hi = np.interp(depth, qsub["depth_m"].astype(float).to_numpy(), qsub["q90"].astype(float).to_numpy())
            return float(lo), float(hi), "QuantileRF band"
    if {"band_low", "band_high"}.issubset(sub.columns) and sub[["band_low", "band_high"]].notna().any().any():
        band_sub = sub.dropna(subset=["band_low", "band_high"])
        lo = np.interp(depth, band_sub["depth_m"].astype(float).to_numpy(), band_sub["band_low"].astype(float).to_numpy())
        hi = np.interp(depth, band_sub["depth_m"].astype(float).to_numpy(), band_sub["band_high"].astype(float).to_numpy())
        return float(lo), float(hi), "model band"
    if ref is not None and not ref.empty:
        ref_sub = ref[(ref["metric"] == "value_ratio") & (ref["canonical_type"] == canonical_type)].copy()
        if not ref_sub.empty:
            agg = ref_sub.groupby("depth_m", as_index=False)["damage_ratio"].agg(["min", "max"]).reset_index()
            lo = np.interp(depth, agg["depth_m"].astype(float).to_numpy(), agg["min"].astype(float).to_numpy())
            hi = np.interp(depth, agg["depth_m"].astype(float).to_numpy(), agg["max"].astype(float).to_numpy())
            return float(lo), float(hi), "source spread"
    mid = model_curve_value(curves, model, canonical_type, depth)
    return mid, mid, "no band"


def deployment_costs_for_model(
    damage_df: pd.DataFrame,
    exposure: pd.DataFrame,
    curves: pd.DataFrame,
    quantiles: pd.DataFrame | None,
    ref: pd.DataFrame | None,
    model: str,
    rate: float,
) -> pd.DataFrame:
    hb_share, area_by_type = dublin_type_shares(exposure)
    rows = []
    for _, row in damage_df.iterrows():
        if float(row.get("flood_frac", 0.0)) <= 0.0 or float(row.get("flood_depth_m", 0.0)) <= 0.0:
            rows.append({"key": row["key"], "m2_lo": 0.0, "m2_mid": 0.0, "m2_hi": 0.0, "dmg_lo": 0.0, "dmg_eur": 0.0, "dmg_hi": 0.0})
            continue
        depth = float(row["flood_depth_m"])
        frac = float(row["flood_frac"])
        hb_count = float(row.get("T6_1_HB_H", 0.0)) * frac
        fa_count = float(row.get("T6_1_FA_H", 0.0)) * frac
        m2 = {"lo": 0.0, "mid": 0.0, "hi": 0.0}
        for t, share in hb_share.items():
            n = hb_count * float(share)
            lo, mid, hi, label = None, None, None, None
            lo, hi, label = model_curve_band(curves, quantiles, ref, model, t, depth)
            mid = model_curve_value(curves, model, t, depth)
            m2["lo"] += n * area_by_type.get(t, 90.0) * lo
            m2["mid"] += n * area_by_type.get(t, 90.0) * mid
            m2["hi"] += n * area_by_type.get(t, 90.0) * hi
        lo, hi, label = model_curve_band(curves, quantiles, ref, model, "Apartment", depth)
        mid_val = model_curve_value(curves, model, "Apartment", depth)
        m2["lo"] += fa_count * area_by_type.get("Apartment", 70.0) * lo
        m2["mid"] += fa_count * area_by_type.get("Apartment", 70.0) * mid_val
        m2["hi"] += fa_count * area_by_type.get("Apartment", 70.0) * hi
        rows.append({
            "key": row["key"],
            "m2_lo": m2["lo"],
            "m2_mid": m2["mid"],
            "m2_hi": m2["hi"],
            "dmg_lo": m2["lo"] * rate,
            "dmg_eur": m2["mid"] * rate,
            "dmg_hi": m2["hi"] * rate,
        })
    return pd.DataFrame(rows)


def deployment_figure(geometry: gpd.GeoDataFrame, damage: pd.DataFrame, metric_col: str) -> go.Figure:
    key_col = dublin_key_column(geometry)
    if key_col is None:
        return go.Figure()
    merged = geometry.merge(damage, left_on=key_col, right_on="key", how="left")
    merged[metric_col] = pd.to_numeric(merged[metric_col], errors="coerce").fillna(0.0)
    flooded = merged[merged[metric_col] > 0].copy()
    geojson = json.loads(merged.to_json())
    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            geojson=geojson,
                locations=merged[key_col].astype(str),
            z=np.zeros(len(merged)),
                featureidkey=f"properties.{key_col}",
            colorscale=[[0, "#111827"], [1, "#111827"]],
            showscale=False,
            marker_line_color="#333",
            marker_line_width=0.25,
            hoverinfo="skip",
            name="All small areas",
        )
    )
    if not flooded.empty:
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=flooded[key_col].astype(str),
                z=flooded[metric_col],
                featureidkey=f"properties.{key_col}",
                colorscale="Reds",
                colorbar_title=metric_col,
                marker_line_color="rgba(255,255,255,0.2)",
                marker_line_width=0.35,
                name="Flooded small areas",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#050608",
        plot_bgcolor="#050608",
        geo=dict(
            bgcolor="#050608",
            showland=True,
            landcolor="#050608",
            showocean=False,
            showlakes=False,
            showcountries=False,
            showframe=False,
            fitbounds="locations",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=760,
        title="Dublin small areas under the OPW 100-year coastal flood",
    )
    return fig


def live_prediction_curve(curves: pd.DataFrame, quantiles: pd.DataFrame | None, selected_model: str, dwelling_type: str) -> pd.DataFrame:
    sub = curve_for_model(curves, selected_model, dwelling_type)
    if sub.empty:
        return pd.DataFrame()
    out = sub[["depth_m", "damage_ratio"]].copy()
    lo, hi, label = model_curve_band(curves, quantiles, load_reference_family(), selected_model, dwelling_type, 0.0)
    if selected_model == "QuantileRF" and quantiles is not None and not quantiles.empty:
        qsub = quantile_band_for_type(quantiles, dwelling_type)
        if not qsub.empty:
            out = out.merge(qsub[["depth_m", "q10", "q50", "q90"]], on="depth_m", how="left")
            return out
    out["q10"] = np.nan
    out["q50"] = out["damage_ratio"]
    out["q90"] = np.nan
    return out


def representative_floor_elev(dwelling_type: str, apartment_storey: str) -> tuple[float, str]:
    if dwelling_type != "Apartment":
        return 0.15, "Houses use the pipeline threshold of 0.15 m above ground-floor datum."
    mapping = {
        "Ground-floor apartment": 0.15,
        "Mid-floor apartment": 4.5,
        "Top-floor apartment": 7.5,
    }
    return mapping.get(apartment_storey, 4.5), f"Apartment threshold set to {apartment_storey}."


def download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    ref = load_reference_family()
    metrics = load_metrics()
    curves = load_curves()
    quantiles = load_quantiles()
    feature_importance = load_feature_importance()
    shap_values = load_shap_values()
    canonical_types = canonical_types_from_reference(ref)
    models = model_options(metrics)

    st.title("Irish residential flood depth-damage explorer")
    st.caption(
        "Project 38. Scenario estimates under the OPW 100-year coastal flood; not predictions. "
        "Trained on synthetic harmonised labels; real Irish loss data is the missing ingredient."
    )

    if not models:
        st.error("No trained models were found in artifacts/vulnerability_extended/metrics_comparison.csv.")
        return
    if not canonical_types:
        st.error("No harmonised dwelling types were found in artifacts/harmonised/reference_curve_family.csv.")
        return

    selected_model = st.sidebar.radio("Model selection", models, index=0)
    rebuild_rate = st.sidebar.slider("Illustrative rebuild rate (EUR/m²)", 2000, 3500, 3000, 50)
    st.sidebar.caption(
        "Rates are illustrative; no open Irish rebuild-cost source exists. JRC max-damage table (EUR 826/m², 2010 prices) is a citable alternative pending inflation adjustment."
    )

    tabs = st.tabs([
        "Harmonisation",
        "Model curves",
        "Dublin deployment",
        "Method comparison",
        "Feature importance",
        "Uncertainty explorer",
        "Live prediction + export",
    ])

    with tabs[0]:
        st.subheader("Harmonised reference-curve family")
        if ref is None or ref.empty:
            st.info("The harmonised reference curve family is missing, so this view cannot be drawn.")
        else:
            dwell = st.selectbox("Dwelling type", canonical_types, key="harmonisation_type")
            fig = build_harmonisation_figure(ref, dwell)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "These are published tables, not model outputs; the spread is the uncertainty band. "
                "Ireland has no measured loss records, so these source curves fill the label column."
            )

    with tabs[1]:
        st.subheader(f"Model curves for {selected_model}")
        if curves is None or curves.empty:
            st.info("The trained curves are missing, so model comparison cannot be shown.")
        else:
            fig = build_model_curves_figure(metrics, curves, ref, selected_model, canonical_types)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Grey lines are the other trained models; the highlighted line is the active sidebar model.")

    with tabs[2]:
        st.subheader("Dublin deployment")
        geometry, damage_base, exposure = load_dublin_geometry_and_inputs()
        if geometry is None or damage_base is None or exposure is None:
            st.info(
                "Deployment inputs are incomplete. This tab needs artifacts/deployment/dublin_damage_by_sa.csv "
                "plus the Dublin geometry and exposure profile tables to render the map."
            )
        else:
            metric_choice = st.radio(
                "Colour map by",
                ["damaged floor-area equivalent (m²)", "estimated loss (EUR)"],
                horizontal=True,
                index=1,
                key="deployment_metric_choice",
            )
            selected_damage = deployment_costs_for_model(
                damage_base,
                exposure,
                curves if curves is not None else pd.DataFrame(),
                quantiles,
                ref,
                selected_model,
                rebuild_rate,
            )
            value_col = "m2_mid" if metric_choice.startswith("damaged") else "dmg_eur"
            band_low = selected_damage["m2_lo"] if value_col == "m2_mid" else selected_damage["dmg_lo"]
            band_high = selected_damage["m2_hi"] if value_col == "m2_mid" else selected_damage["dmg_hi"]
            flooded_mask = selected_damage[value_col] > 0
            flooded_count = int(flooded_mask.sum())
            total_count = int(len(selected_damage))
            dwellings_flooded = float((selected_damage["m2_mid"] >= 0).sum())
            if "flood_frac" in damage_base.columns:
                dwellings_in_flooded_areas = float(((damage_base["T6_1_HB_H"].fillna(0) + damage_base["T6_1_FA_H"].fillna(0)) * damage_base["flood_frac"].fillna(0)).sum())
            else:
                dwellings_in_flooded_areas = float((damage_base["T6_1_HB_H"].fillna(0) + damage_base["T6_1_FA_H"].fillna(0)).sum())
            m2_lo = float(selected_damage["m2_lo"].sum())
            m2_mid = float(selected_damage["m2_mid"].sum())
            m2_hi = float(selected_damage["m2_hi"].sum())
            eur_lo = m2_lo * rebuild_rate
            eur_mid = m2_mid * rebuild_rate
            eur_hi = m2_hi * rebuild_rate

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Flooded small areas", f"{flooded_count:,}", f"of {total_count:,}")
            c2.metric("Dwellings in flooded areas", f"{dwellings_in_flooded_areas:,.0f}")
            c3.metric("Damaged floor-area equivalent", f"{m2_mid:,.0f} m²", f"band {m2_lo:,.0f}–{m2_hi:,.0f} m²")
            c4.metric("Estimated cost", f"EUR {eur_mid/1e6:,.1f}M", f"band EUR {eur_lo/1e6:,.1f}M–{eur_hi/1e6:,.1f}M at EUR {rebuild_rate:,}/m²")

            st.markdown('<div class="soft-card muted">Black background, flooded small areas only, coloured by the selected model and the chosen output unit.</div>', unsafe_allow_html=True)
            fig = deployment_figure(geometry, selected_damage if metric_choice.startswith("damaged") else selected_damage.assign(dmg_eur=selected_damage["dmg_eur"]), value_col)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The underlying flood footprint and exposure are held fixed; only the depth-damage curve changes with the selected model.")

    with tabs[3]:
        st.subheader("Method comparison")
        if metrics is None or metrics.empty:
            st.info("The metrics table is missing, so method comparison is unavailable.")
        else:
            metrics_sorted = metrics.sort_values("R2", ascending=False).reset_index(drop=True)
            st.dataframe(style_metrics_table(metrics_sorted), use_container_width=True, hide_index=True)
            st.caption(
                "R2 measures fidelity to the synthetic harmonised curve family, not prediction of real losses, so it is not comparable to the Bles et al. (2026) 0.39 on real NFIP data. "
                "All multivariable architectures sit within noise of each other under synthetic supervision; real Irish loss data is what would separate them."
            )
            fig = make_subplots(rows=2, cols=2, subplot_titles=["MAE", "RMSE", "R2", "bias"])
            metrics_long = metrics_sorted.melt(id_vars="model", value_vars=["MAE", "RMSE", "R2", "bias"], var_name="metric", value_name="value")
            for idx, metric_name in enumerate(["MAE", "RMSE", "R2", "bias"], start=1):
                sub = metrics_long[metrics_long["metric"] == metric_name]
                fig.add_trace(go.Bar(x=sub["model"], y=sub["value"], name=metric_name, showlegend=(idx == 1)), row=1 if idx <= 2 else 2, col=1 if idx % 2 else 2)
            fig.update_layout(template="plotly_dark", paper_bgcolor="#050608", plot_bgcolor="#050608", barmode="group", height=700)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Feature importance")
        if feature_importance is None or feature_importance.empty:
            st.info("Feature importance is not available in the artifact folder yet.")
        else:
            available_models = pd.unique(feature_importance["model"].astype(str)).tolist()
            if not available_models:
                st.info("No tree or boosting feature importance rows were found.")
            else:
                mode = st.radio("View", ["Tree/boosting feature importance", "SHAP"], horizontal=True, key="feature_view")
                model_choice = st.selectbox("Model", available_models, key="feature_model_choice")
                if mode.startswith("Tree"):
                    sub = feature_importance[feature_importance["model"].astype(str) == model_choice].copy()
                    if sub.empty:
                        st.info("Feature importance is not available for this model.")
                    else:
                        sub = sub.sort_values("importance", ascending=True)
                        fig = go.Figure(go.Bar(x=sub["importance"], y=sub["feature"], orientation="h", marker_color="#7dd3fc"))
                        fig.update_layout(template="plotly_dark", paper_bgcolor="#050608", plot_bgcolor="#050608", title=f"Feature importance: {model_choice}")
                        st.plotly_chart(fig, use_container_width=True)
                        top_feature = sub.iloc[-1]["feature"]
                        if str(top_feature) == "depth_above_floor":
                            st.success("depth_above_floor is the dominant feature, which is consistent with Paulik et al. (2024).")
                        else:
                            st.info(f"Top feature: {top_feature}. depth_above_floor should still dominate in the synthetic setup.")
                else:
                    if shap_values is None or shap_values.empty:
                        st.info("SHAP values not available for this model — rerun the trainer with the shap library installed.")
                    else:
                        sub = shap_values[shap_values["model"].astype(str) == model_choice].copy()
                        if sub.empty:
                            st.info("SHAP values not available for this model — rerun the trainer with the shap library installed.")
                        else:
                            sub = sub.sort_values("mean_abs_shap", ascending=True)
                            fig = go.Figure(go.Bar(x=sub["mean_abs_shap"], y=sub["feature"], orientation="h", marker_color="#f59e0b"))
                            fig.update_layout(template="plotly_dark", paper_bgcolor="#050608", plot_bgcolor="#050608", title=f"Mean |SHAP|: {model_choice}")
                            st.plotly_chart(fig, use_container_width=True)
                            if "depth_above_floor" in sub["feature"].astype(str).tolist():
                                st.success("depth_above_floor appears in the SHAP export, consistent with the expected dominant driver.")

    with tabs[5]:
        st.subheader("Uncertainty explorer")
        if quantiles is None or quantiles.empty:
            st.info("quantile_bands.csv is missing or empty, so the uncertainty explorer cannot be shown.")
        else:
            dwell = st.selectbox("Dwelling type", canonical_types, key="quantile_type")
            sub = quantile_band_for_type(quantiles, dwell)
            if sub.empty:
                st.info("No quantile bands were found for the selected dwelling type.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sub["depth_m"], y=sub["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=sub["depth_m"], y=sub["q10"], line=dict(width=0), fill="tonexty", fillcolor="rgba(245, 158, 11, 0.18)", name="q10–q90 band"))
                fig.add_trace(go.Scatter(x=sub["depth_m"], y=sub["q50"], mode="lines", line=dict(color="#f59e0b", width=3), name="median (q50)"))
                fig.update_layout(template="plotly_dark", paper_bgcolor="#050608", plot_bgcolor="#050608", title=f"Quantile random forest band: {dwell}", xaxis_title="Depth above floor (m)", yaxis_title="Damage ratio")
                fig.update_xaxes(range=[0, 3.0])
                fig.update_yaxes(range=[0, 1.0])
                st.plotly_chart(fig, use_container_width=True)
                st.caption("This is the model's own predictive band from Quantile Random Forest, distinct from the source-disagreement band in the harmonisation tab.")

    with tabs[6]:
        st.subheader("Live prediction + export")
        if curves is None or curves.empty:
            st.info("The trained curves are missing, so live prediction cannot be shown.")
        else:
            left, right = st.columns([1, 1])
            with left:
                dwelling_type = st.selectbox("Dwelling type", canonical_types, key="live_type")
                floor_area = st.number_input("Floor area (m²)", min_value=20.0, max_value=1000.0, value=120.0, step=5.0)
                age_band = st.radio("Age band", ["pre-1945", "1945–1980", "post-1980"], horizontal=True, key="live_age")
                flood_depth = st.slider("Flood depth (m)", 0.0, 3.0, 1.0, 0.05)
                apartment_storey = "Mid-floor apartment"
                if dwelling_type == "Apartment":
                    apartment_storey = st.selectbox("Apartment floor threshold", ["Ground-floor apartment", "Mid-floor apartment", "Top-floor apartment"], key="apartment_storey")
                floor_elev, floor_note = representative_floor_elev(dwelling_type, apartment_storey)
                depth_above_floor = max(0.0, flood_depth - floor_elev)
                st.caption(f"Depth above floor = max(0, flood depth - floor threshold) = {depth_above_floor:.2f} m. {floor_note}")
                st.caption(f"Age band selected: {age_band}. The curve-based predictor itself does not vary by age band.")

            with right:
                curve_sub = curve_for_model(curves, selected_model, dwelling_type)
                if curve_sub.empty:
                    st.info("The selected model does not have a curve for this dwelling type.")
                else:
                    pred_ratio = float(np.interp(depth_above_floor, curve_sub["depth_m"].astype(float).to_numpy(), curve_sub["damage_ratio"].astype(float).to_numpy(), left=curve_sub["damage_ratio"].iloc[0], right=curve_sub["damage_ratio"].iloc[-1]))
                    lo, hi, band_label = model_curve_band(curves, quantiles, ref, selected_model, dwelling_type, depth_above_floor)
                    if not np.isfinite(lo) or not np.isfinite(hi):
                        lo, hi = pred_ratio, pred_ratio
                    loss_mid = pred_ratio * floor_area * rebuild_rate
                    loss_lo = lo * floor_area * rebuild_rate
                    loss_hi = hi * floor_area * rebuild_rate
                    k1, k2 = st.columns(2)
                    k1.metric("Predicted damage ratio", f"{pred_ratio:.3f}", f"band {lo:.3f}–{hi:.3f}")
                    k2.metric("Illustrative euro loss", f"EUR {loss_mid:,.0f}", f"band EUR {loss_lo:,.0f}–{loss_hi:,.0f}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=curve_sub["depth_m"], y=curve_sub["damage_ratio"], mode="lines", line=dict(color="#7dd3fc", width=3), name=selected_model))
                    if selected_model == "QuantileRF" and quantiles is not None and not quantiles.empty:
                        qsub = quantile_band_for_type(quantiles, dwelling_type)
                        if not qsub.empty:
                            fig.add_trace(go.Scatter(x=qsub["depth_m"], y=qsub["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                            fig.add_trace(go.Scatter(x=qsub["depth_m"], y=qsub["q10"], line=dict(width=0), fill="tonexty", fillcolor="rgba(245, 158, 11, 0.16)", name="q10–q90 band"))
                    else:
                        fig.add_trace(go.Scatter(x=curve_sub["depth_m"], y=curve_sub["band_high"] if "band_high" in curve_sub.columns else curve_sub["damage_ratio"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                        fig.add_trace(go.Scatter(x=curve_sub["depth_m"], y=curve_sub["band_low"] if "band_low" in curve_sub.columns else curve_sub["damage_ratio"], line=dict(width=0), fill="tonexty", fillcolor="rgba(125, 211, 252, 0.14)", name=band_label))
                    fig.add_vline(x=depth_above_floor, line_dash="dash", line_color="#e2e8f0")
                    fig.update_layout(template="plotly_dark", paper_bgcolor="#050608", plot_bgcolor="#050608", title=f"{selected_model}: {dwelling_type}", xaxis_title="Depth above floor (m)", yaxis_title="Damage ratio")
                    fig.update_xaxes(range=[0, 3.0])
                    fig.update_yaxes(range=[0, 1.0])
                    st.plotly_chart(fig, use_container_width=True)

                    curve_export = curve_sub.copy()
                    if selected_model == "QuantileRF" and quantiles is not None and not quantiles.empty:
                        qsub = quantile_band_for_type(quantiles, dwelling_type)
                        if not qsub.empty:
                            curve_export = curve_export.merge(qsub, on="depth_m", how="left")
                    if st.download_button(
                        "Download curve (CSV)",
                        data=download_bytes(curve_export),
                        file_name=f"{selected_model}_{dwelling_type}_curve.csv".replace(" ", "_"),
                        mime="text/csv",
                    ):
                        st.toast("Curve CSV prepared for download.")
                    all_curves_path = config.ARTIFACTS / "vulnerability_extended" / "curves_by_model.csv"
                    if all_curves_path.exists():
                        st.download_button(
                            "Download all curves",
                            data=all_curves_path.read_bytes(),
                            file_name="curves_by_model.csv",
                            mime="text/csv",
                        )


if __name__ == "__main__":
    main()