# app.py
# SPER Allocation & Store Performance Dashboard
# Creator: Ryan Childs / xQUANDT-style retail analytics build

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="SPER Allocation Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "📦 SPER Allocation & Store Performance Dashboard"
APP_SUBTITLE = "Upload a pivot-style CSV/Excel export and diagnose sales, margin, inventory, in-stock health, allocation risk, and store-level opportunities."

MONEY_COLS_KEYWORDS = ["sales", "margin", "receipt", "inv", "in-tran", "allocated"]
BASE_ID_COLS = ["Region", "Store", "Site No", "Volume Band", "Size Band"]


# -----------------------------
# Utility functions
# -----------------------------
def normalize_col(c: object) -> str:
    s = str(c).strip()
    s = re.sub(r"^Sum of\s+", "", s, flags=re.I)
    s = s.replace("$", "Dollars")
    s = s.replace("#", "No")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("IN-Tran", "In-Tran").replace("IN-TRAN", "In-Tran")
    replacements = {
        "Row Labels": "Region",
        "STORE_NO_NAME": "Store",
        "SITE_NO": "Site No",
        "VOLUME_BAND": "Volume Band",
        "SIZE_BAND": "Size Band",
        "Sales Dollars LW": "Sales LW TY",
        "Sales Dollars LW LY": "Sales LW LY",
        "Margin Dollars LW": "Margin LW TY",
        "Margin Dollars LW LY": "Margin LW LY",
        "Receipt Dollars LW": "Receipt LW TY",
        "Receipt Dollars LW LY": "Receipt LW LY",
        "Sales Dollars QTD": "Sales QTD TY",
        "Sales Dollars QTD LY": "Sales QTD LY",
        "Margin Dollars QTD": "Margin QTD TY",
        "Margin Dollars QTD LY": "Margin QTD LY",
        "Receipt Dollars QTD": "Receipt QTD TY",
        "Receipt Dollars QTD LY": "Receipt QTD LY",
        "Sales Dollars YTD": "Sales YTD TY",
        "Sales Dollars YTD LY": "Sales YTD LY",
        "Margin Dollars YTD": "Margin YTD TY",
        "Margin Dollars YTD LY": "Margin YTD LY",
        "Receipt Dollars YTD": "Receipt YTD TY",
        "Receipt Dollars YTD LY": "Receipt YTD LY",
        "Total Inv LW TY": "Inventory LW TY",
        "Total Inv LW LY": "Inventory LW LY",
        "In-Tran LW TY": "In Transit LW TY",
        "In-Tran LW LY": "In Transit LW LY",
        "Allocated LW TY": "Allocated LW TY",
        "Allocated LW LY": "Allocated LW LY",
        "SKU Count": "SKU Count",
        "In-Stock Stores": "In-Stock Stores",
    }
    return replacements.get(s, s)


def clean_numeric(x: object) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s in {"", "-", "--", "nan", "None"}:
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").replace("%", "")
    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return np.nan


def fmt_money(v: float) -> str:
    if pd.isna(v):
        return "—"
    sign = "-" if v < 0 else ""
    v = abs(float(v))
    if v >= 1_000_000:
        return f"{sign}${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{sign}${v/1_000:.1f}K"
    return f"{sign}${v:,.0f}"


def fmt_num(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:,.0f}"


def fmt_pct(v: float, decimals: int = 1) -> str:
    if pd.isna(v) or np.isinf(v):
        return "—"
    return f"{v*100:.{decimals}f}%"


def safe_div(num, den):
    return np.where((pd.notna(den)) & (den != 0), num / den, np.nan)


def yoy(ty, ly):
    return safe_div(ty - ly, ly)


def zscore(s: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(skipna=True)
    if pd.isna(std) or std == 0:
        z = pd.Series(0.0, index=s.index)
    else:
        z = (s - s.mean(skipna=True)) / std
    z = z.clip(-3, 3)
    return -z if invert else z


def winsorize(s: pd.Series, lower=0.02, upper=0.98) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def detect_header_row(raw: pd.DataFrame) -> int:
    best_idx, best_score = 0, -1
    target_terms = ["row labels", "store", "site", "sales", "margin", "receipt", "inventory", "allocated", "sku"]
    for idx in range(min(len(raw), 30)):
        row = raw.iloc[idx].astype(str).str.lower().fillna("").tolist()
        joined = " | ".join(row)
        non_null = raw.iloc[idx].notna().sum()
        score = non_null + sum(8 for t in target_terms if t in joined)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def read_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, int]:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(uploaded_file, header=None, dtype=str)
    else:
        raw = pd.read_csv(uploaded_file, header=None, dtype=str)
    header_idx = detect_header_row(raw)
    headers = [normalize_col(x) for x in raw.iloc[header_idx].tolist()]
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = headers
    df = df.dropna(how="all")
    df = df.loc[:, ~pd.Index(df.columns).astype(str).str.startswith("nan")]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df, header_idx


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]

    # Remove grand total/subtotal rows and blank site rows.
    if "Store" in df.columns:
        df = df[df["Store"].notna()]
        df = df[~df["Store"].astype(str).str.contains("grand total|total", case=False, na=False)]
    if "Site No" in df.columns:
        df["Site No"] = df["Site No"].astype(str).str.extract(r"(\d+)", expand=False)

    if "Region" in df.columns:
        df["Region"] = df["Region"].fillna("Unassigned").replace({"(blank)": "Ecommerce", "nan": "Unassigned"})
    else:
        df["Region"] = "Unassigned"

    # Numeric conversion for all likely metric fields.
    id_cols = set(BASE_ID_COLS)
    for c in df.columns:
        if c not in id_cols:
            converted = df[c].map(clean_numeric)
            if converted.notna().sum() >= max(3, int(0.2 * len(df))):
                df[c] = converted

    # Make missing expected columns available so the dashboard never hard-crashes.
    expected = [
        "Sales LW TY", "Sales LW LY", "Margin LW TY", "Margin LW LY", "Receipt LW TY", "Receipt LW LY",
        "Sales QTD TY", "Sales QTD LY", "Margin QTD TY", "Margin QTD LY", "Receipt QTD TY", "Receipt QTD LY",
        "Sales YTD TY", "Sales YTD LY", "Margin YTD TY", "Margin YTD LY", "Receipt YTD TY", "Receipt YTD LY",
        "Inventory LW TY", "Inventory LW LY", "In Transit LW TY", "In Transit LW LY",
        "Allocated LW TY", "Allocated LW LY", "SKU Count", "In-Stock Stores"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    df["Sales LW YoY"] = yoy(df["Sales LW TY"], df["Sales LW LY"])
    df["Sales QTD YoY"] = yoy(df["Sales QTD TY"], df["Sales QTD LY"])
    df["Sales YTD YoY"] = yoy(df["Sales YTD TY"], df["Sales YTD LY"])
    df["Margin LW YoY"] = yoy(df["Margin LW TY"], df["Margin LW LY"])
    df["Margin YTD YoY"] = yoy(df["Margin YTD TY"], df["Margin YTD LY"])
    df["Receipt LW YoY"] = yoy(df["Receipt LW TY"], df["Receipt LW LY"])
    df["Receipt YTD YoY"] = yoy(df["Receipt YTD TY"], df["Receipt YTD LY"])
    df["Inventory YoY"] = yoy(df["Inventory LW TY"], df["Inventory LW LY"])
    df["Allocated YoY"] = yoy(df["Allocated LW TY"], df["Allocated LW LY"])
    df["In Transit YoY"] = yoy(df["In Transit LW TY"], df["In Transit LW LY"])

    df["GM LW TY"] = safe_div(df["Margin LW TY"], df["Sales LW TY"])
    df["GM LW LY"] = safe_div(df["Margin LW LY"], df["Sales LW LY"])
    df["GM YTD TY"] = safe_div(df["Margin YTD TY"], df["Sales YTD TY"])
    df["GM YTD LY"] = safe_div(df["Margin YTD LY"], df["Sales YTD LY"])
    df["GM YTD Change"] = df["GM YTD TY"] - df["GM YTD LY"]

    df["In-Stock Rate"] = safe_div(df["In-Stock Stores"], df["SKU Count"])
    df["Inventory/Sales YTD"] = safe_div(df["Inventory LW TY"], df["Sales YTD TY"])
    df["Sales per SKU"] = safe_div(df["Sales YTD TY"], df["SKU Count"])
    df["Allocated/Sales LW"] = safe_div(df["Allocated LW TY"], df["Sales LW TY"])
    df["Allocated/Inventory"] = safe_div(df["Allocated LW TY"], df["Inventory LW TY"])
    df["In Transit/Inventory"] = safe_div(df["In Transit LW TY"], df["Inventory LW TY"])
    df["Inventory Productivity"] = safe_div(df["Sales YTD TY"], df["Inventory LW TY"])
    df["Sales Gap YTD"] = df["Sales YTD TY"] - df["Sales YTD LY"]
    df["Margin Gap YTD"] = df["Margin YTD TY"] - df["Margin YTD LY"]
    df["Momentum Gap"] = df["Sales LW YoY"] - df["Sales YTD YoY"]

    # Advanced scoring. Higher is better for allocation support.
    instock_gap = (0.90 - df["In-Stock Rate"]).clip(lower=0)
    df["Allocation Priority Score"] = (
        28 * zscore(winsorize(df["Sales YTD YoY"]))
        + 25 * zscore(winsorize(df["Sales LW YoY"]))
        + 18 * zscore(winsorize(instock_gap))
        + 16 * zscore(winsorize(df["Inventory/Sales YTD"]), invert=True)
        + 8 * zscore(winsorize(df["GM YTD TY"]))
        + 5 * zscore(winsorize(df["Momentum Gap"]))
    )
    score = df["Allocation Priority Score"]
    if score.notna().sum() > 0 and score.max() != score.min():
        df["Allocation Priority Score"] = 100 * (score - score.min()) / (score.max() - score.min())
    else:
        df["Allocation Priority Score"] = 50.0

    # Risk score: higher means be careful before sending more product.
    df["Allocation Risk Score"] = (
        30 * zscore(winsorize(df["Sales YTD YoY"]), invert=True)
        + 25 * zscore(winsorize(df["Sales LW YoY"]), invert=True)
        + 20 * zscore(winsorize(df["Inventory/Sales YTD"]))
        + 15 * zscore(winsorize(df["Allocated/Sales LW"]))
        + 10 * zscore(winsorize(df["GM YTD Change"]), invert=True)
    )
    risk = df["Allocation Risk Score"]
    if risk.notna().sum() > 0 and risk.max() != risk.min():
        df["Allocation Risk Score"] = 100 * (risk - risk.min()) / (risk.max() - risk.min())
    else:
        df["Allocation Risk Score"] = 50.0

    conditions = [
        (df["Allocation Priority Score"] >= 75) & (df["Allocation Risk Score"] < 60),
        (df["Allocation Priority Score"] >= 60) & (df["Allocation Risk Score"] < 70),
        (df["Allocation Risk Score"] >= 75),
        (df["Sales YTD YoY"] < -0.20) & (df["Inventory/Sales YTD"] > df["Inventory/Sales YTD"].median(skipna=True)),
    ]
    choices = ["Increase / Protect", "Selective Support", "Hold / Review", "Reduce / Avoid"]
    df["Allocation Action"] = np.select(conditions, choices, default="Maintain")

    df["Store Label"] = df["Store"].astype(str)
    if "Site No" in df.columns:
        df["Store Label"] = df["Site No"].fillna("").astype(str) + " - " + df["Store"].astype(str).str.replace(r"^\d+\s*-\s*", "", regex=True)
        df["Store Label"] = df["Store Label"].str.replace(r"^\s*-\s*", "", regex=True)

    return df.reset_index(drop=True)


def aggregate_region(df: pd.DataFrame) -> pd.DataFrame:
    sum_cols = [c for c in df.columns if any(x in c for x in ["Sales", "Margin", "Receipt", "Inventory", "In Transit", "Allocated", "SKU Count", "In-Stock Stores"]) and pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby("Region", dropna=False)[sum_cols].sum(min_count=1).reset_index()
    agg["Stores"] = df.groupby("Region")["Store"].count().values
    agg = prepare_derived_only(agg)
    return agg


def prepare_derived_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["Sales LW TY", "Sales LW LY", "Margin LW TY", "Margin LW LY", "Sales YTD TY", "Sales YTD LY", "Margin YTD TY", "Margin YTD LY", "Inventory LW TY", "Inventory LW LY", "Allocated LW TY", "Allocated LW LY", "In Transit LW TY", "In Transit LW LY", "SKU Count", "In-Stock Stores"]:
        if c not in df.columns:
            df[c] = np.nan
    df["Sales LW YoY"] = yoy(df["Sales LW TY"], df["Sales LW LY"])
    df["Sales YTD YoY"] = yoy(df["Sales YTD TY"], df["Sales YTD LY"])
    df["Margin YTD YoY"] = yoy(df["Margin YTD TY"], df["Margin YTD LY"])
    df["Inventory YoY"] = yoy(df["Inventory LW TY"], df["Inventory LW LY"])
    df["Allocated YoY"] = yoy(df["Allocated LW TY"], df["Allocated LW LY"])
    df["GM YTD TY"] = safe_div(df["Margin YTD TY"], df["Sales YTD TY"])
    df["GM YTD LY"] = safe_div(df["Margin YTD LY"], df["Sales YTD LY"])
    df["In-Stock Rate"] = safe_div(df["In-Stock Stores"], df["SKU Count"])
    df["Inventory/Sales YTD"] = safe_div(df["Inventory LW TY"], df["Sales YTD TY"])
    df["Sales Gap YTD"] = df["Sales YTD TY"] - df["Sales YTD LY"]
    return df


def metric_card(label: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
    st.metric(label, value, delta=delta, help=help_text)


def styled_df(df: pd.DataFrame, currency_cols=None, pct_cols=None, score_cols=None):
    currency_cols = currency_cols or []
    pct_cols = pct_cols or []
    score_cols = score_cols or []
    fmt = {c: "${:,.0f}" for c in currency_cols if c in df.columns}
    fmt.update({c: "{:.1%}" for c in pct_cols if c in df.columns})
    fmt.update({c: "{:.1f}" for c in score_cols if c in df.columns})
    return df.style.format(fmt, na_rep="—")


def download_button_for_df(df: pd.DataFrame, label: str, file_name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=file_name, mime="text/csv", use_container_width=True)


def plot_bar(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, title: str = "", text_auto: bool = False, orientation: str = "v"):
    if orientation == "h":
        fig = px.bar(df, x=y, y=x, color=color, title=title, text_auto=text_auto, orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(df, x=x, y=y, color=color, title=title, text_auto=text_auto)
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def make_flags(row: pd.Series) -> str:
    flags = []
    if row.get("Sales YTD YoY", np.nan) < -0.20:
        flags.append("YTD down >20%")
    if row.get("Sales LW YoY", np.nan) < -0.25:
        flags.append("LW down >25%")
    if row.get("In-Stock Rate", np.nan) < 0.85:
        flags.append("Low in-stock")
    if row.get("Inventory/Sales YTD", np.nan) > 2.0:
        flags.append("Inventory heavy")
    if row.get("Allocated/Sales LW", np.nan) > 0.75:
        flags.append("Allocation high vs LW sales")
    if row.get("GM YTD Change", np.nan) < -0.03:
        flags.append("GM pressure")
    return "; ".join(flags) if flags else "No major flag"


def add_clusters(df: pd.DataFrame, k: int) -> pd.DataFrame:
    out = df.copy()
    features = ["Sales YTD YoY", "Sales LW YoY", "GM YTD TY", "In-Stock Rate", "Inventory/Sales YTD", "Allocated/Sales LW", "Inventory Productivity"]
    X = out[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    if SKLEARN_AVAILABLE and len(out) >= k:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        out["Cluster"] = model.fit_predict(Xs).astype(str)
    else:
        out["Cluster"] = pd.qcut(out["Allocation Priority Score"].rank(method="first"), q=min(k, len(out)), labels=False, duplicates="drop").astype(str)
    return out


# -----------------------------
# Sidebar + data load
# -----------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("1) Upload Data")
    uploaded = st.file_uploader("Upload SPER pivot CSV or Excel file", type=["csv", "xlsx", "xls"])
    st.caption("The app automatically detects pivot headers like `Row Labels`, store, sales, margin, inventory, allocation, SKU count, and in-stock stores.")

    st.header("2) Dashboard Controls")
    min_sales = st.number_input("Minimum YTD sales included", min_value=0, value=0, step=1000)
    top_n = st.slider("Top / bottom N", min_value=5, max_value=50, value=15, step=5)
    benchmark_instock = st.slider("Target in-stock rate", 0.70, 0.99, 0.90, 0.01)
    cluster_k = st.slider("Store clusters", 2, 8, 4, 1)

    st.header("3) Score Weights")
    st.caption("These are used in the interactive what-if score on the Allocation tab.")
    w_ytd = st.slider("YTD sales trend", 0, 50, 25)
    w_lw = st.slider("LW momentum", 0, 50, 25)
    w_instock = st.slider("In-stock gap", 0, 50, 20)
    w_inv = st.slider("Inventory efficiency", 0, 50, 20)
    w_margin = st.slider("Margin strength", 0, 30, 10)

if uploaded is None:
    st.info("Upload the SPER pivot file to begin. A sample file is included in the ZIP as `sample_sper_week14.csv`.")
    st.stop()

try:
    raw_loaded, header_row = read_uploaded_file(uploaded)
    data = prepare_data(raw_loaded)
except Exception as e:
    st.error("The file could not be parsed. Check that it is a CSV/Excel export with a pivot header row.")
    st.exception(e)
    st.stop()

# Apply basic filters
regions = sorted(data["Region"].dropna().unique().tolist())
with st.sidebar:
    selected_regions = st.multiselect("Regions", regions, default=regions)
    action_filter = st.multiselect("Allocation actions", sorted(data["Allocation Action"].dropna().unique().tolist()), default=sorted(data["Allocation Action"].dropna().unique().tolist()))

f = data[(data["Region"].isin(selected_regions)) & (data["Sales YTD TY"].fillna(0) >= min_sales) & (data["Allocation Action"].isin(action_filter))].copy()
f["Flags"] = f.apply(make_flags, axis=1)
region_df = aggregate_region(f)
clustered = add_clusters(f, cluster_k)

# Custom what-if score using sidebar weights
custom_score = (
    w_ytd * zscore(winsorize(f["Sales YTD YoY"]))
    + w_lw * zscore(winsorize(f["Sales LW YoY"]))
    + w_instock * zscore(winsorize((benchmark_instock - f["In-Stock Rate"]).clip(lower=0)))
    + w_inv * zscore(winsorize(f["Inventory/Sales YTD"]), invert=True)
    + w_margin * zscore(winsorize(f["GM YTD TY"]))
)
if custom_score.notna().sum() and custom_score.max() != custom_score.min():
    f["Custom Allocation Score"] = 100 * (custom_score - custom_score.min()) / (custom_score.max() - custom_score.min())
else:
    f["Custom Allocation Score"] = 50

# -----------------------------
# Header metrics
# -----------------------------
st.success(f"Loaded {len(data):,} rows. Detected header row: {header_row + 1}. Current filter: {len(f):,} stores/sites.")

sales_ty = f["Sales YTD TY"].sum()
sales_ly = f["Sales YTD LY"].sum()
margin_ty = f["Margin YTD TY"].sum()
margin_ly = f["Margin YTD LY"].sum()
inv_ty = f["Inventory LW TY"].sum()
inv_ly = f["Inventory LW LY"].sum()
alloc_ty = f["Allocated LW TY"].sum()
alloc_ly = f["Allocated LW LY"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("YTD Sales", fmt_money(sales_ty), fmt_pct(yoy(sales_ty, sales_ly)))
with c2:
    metric_card("YTD Margin", fmt_money(margin_ty), fmt_pct(yoy(margin_ty, margin_ly)))
with c3:
    metric_card("YTD GM%", fmt_pct(safe_div(margin_ty, sales_ty)), f"{(safe_div(margin_ty, sales_ty)-safe_div(margin_ly, sales_ly))*10000:.0f} bps")
with c4:
    metric_card("LW Inventory", fmt_money(inv_ty), fmt_pct(yoy(inv_ty, inv_ly)))
with c5:
    metric_card("LW Allocated", fmt_money(alloc_ty), fmt_pct(yoy(alloc_ty, alloc_ly)))

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "Executive Summary",
    "Regional Diagnostics",
    "Store Leaderboard",
    "Allocation Engine",
    "Inventory & In-Stock",
    "Margin & Receipts",
    "Outliers & Clusters",
    "Data Explorer",
])

# Executive Summary
with tabs[0]:
    st.subheader("Executive Summary")
    st.write("This tab gives a fast read on total performance, the largest contributors to sales movement, and stores that deserve allocation review.")

    a, b = st.columns([1.2, 1])
    with a:
        waterfall = region_df.sort_values("Sales Gap YTD")[["Region", "Sales Gap YTD", "Sales YTD TY", "Sales YTD YoY"]]
        fig = px.bar(waterfall, x="Region", y="Sales Gap YTD", title="YTD Sales Gap by Region: TY - LY", text_auto=".2s")
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        action_counts = f["Allocation Action"].value_counts().reset_index()
        action_counts.columns = ["Allocation Action", "Stores"]
        fig = px.pie(action_counts, names="Allocation Action", values="Stores", title="Stores by Recommended Allocation Action", hole=0.45)
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Fastest Opportunities")
    opp = f.sort_values("Allocation Priority Score", ascending=False).head(top_n)[[
        "Region", "Store Label", "Sales YTD TY", "Sales YTD YoY", "Sales LW YoY", "GM YTD TY", "In-Stock Rate", "Inventory/Sales YTD", "Allocated LW TY", "Allocation Priority Score", "Allocation Risk Score", "Allocation Action", "Flags"
    ]]
    st.dataframe(styled_df(opp, currency_cols=["Sales YTD TY", "Allocated LW TY"], pct_cols=["Sales YTD YoY", "Sales LW YoY", "GM YTD TY", "In-Stock Rate"], score_cols=["Inventory/Sales YTD", "Allocation Priority Score", "Allocation Risk Score"]), use_container_width=True, height=430)

# Regional Diagnostics
with tabs[1]:
    st.subheader("Regional Diagnostics")
    st.write("Compare regions across sales, margin, inventory, in-stock rate, and allocation intensity.")
    r = region_df.sort_values("Sales YTD TY", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_bar(r, "Region", "Sales YTD TY", color="Sales YTD YoY", title="YTD Sales by Region"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_bar(r.sort_values("Sales YTD YoY"), "Region", "Sales YTD YoY", color="Sales YTD YoY", title="YTD Sales YoY by Region"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(r, x="Inventory/Sales YTD", y="Sales YTD YoY", size="Sales YTD TY", color="Region", hover_name="Region", title="Region Productivity Map: Inventory Load vs Sales Trend")
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.scatter(r, x="In-Stock Rate", y="Allocated YoY", size="Allocated LW TY", color="Sales YTD YoY", hover_name="Region", title="Allocation vs In-Stock by Region")
        fig.add_vline(x=benchmark_instock, line_dash="dash")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Regional Table")
    st.dataframe(styled_df(r[["Region", "Stores", "Sales YTD TY", "Sales YTD LY", "Sales YTD YoY", "Margin YTD YoY", "GM YTD TY", "Inventory LW TY", "Inventory YoY", "Allocated LW TY", "Allocated YoY", "In-Stock Rate", "Inventory/Sales YTD"]], currency_cols=["Sales YTD TY", "Sales YTD LY", "Inventory LW TY", "Allocated LW TY"], pct_cols=["Sales YTD YoY", "Margin YTD YoY", "GM YTD TY", "Inventory YoY", "Allocated YoY", "In-Stock Rate"], score_cols=["Inventory/Sales YTD"]), use_container_width=True)

# Store Leaderboard
with tabs[2]:
    st.subheader("Store Leaderboard")
    st.write("Rank stores by growth, productivity, margin, and current momentum.")
    metric = st.selectbox("Leaderboard metric", ["Sales YTD TY", "Sales YTD YoY", "Sales LW YoY", "GM YTD TY", "In-Stock Rate", "Inventory Productivity", "Allocation Priority Score", "Allocation Risk Score"], index=6)
    asc = st.checkbox("Show lowest values instead", value=False)
    lb = f.sort_values(metric, ascending=asc).head(top_n)
    fig = px.bar(lb, x=metric, y="Store Label", color="Region", orientation="h", title=f"{'Lowest' if asc else 'Highest'} {top_n}: {metric}", hover_data=["Sales YTD TY", "Sales YTD YoY", "Sales LW YoY", "In-Stock Rate", "Inventory/Sales YTD", "Allocation Action"])
    fig.update_layout(height=max(500, 28 * len(lb)), yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Best YTD Growth")
        best = f.sort_values("Sales YTD YoY", ascending=False).head(top_n)
        st.dataframe(styled_df(best[["Region", "Store Label", "Sales YTD TY", "Sales YTD YoY", "Sales LW YoY", "In-Stock Rate", "Inventory/Sales YTD", "Allocation Action"]], currency_cols=["Sales YTD TY"], pct_cols=["Sales YTD YoY", "Sales LW YoY", "In-Stock Rate"], score_cols=["Inventory/Sales YTD"]), use_container_width=True, height=430)
    with c2:
        st.markdown("### Weakest YTD Growth")
        worst = f.sort_values("Sales YTD YoY", ascending=True).head(top_n)
        st.dataframe(styled_df(worst[["Region", "Store Label", "Sales YTD TY", "Sales YTD YoY", "Sales LW YoY", "In-Stock Rate", "Inventory/Sales YTD", "Allocation Action"]], currency_cols=["Sales YTD TY"], pct_cols=["Sales YTD YoY", "Sales LW YoY", "In-Stock Rate"], score_cols=["Inventory/Sales YTD"]), use_container_width=True, height=430)

# Allocation Engine
with tabs[3]:
    st.subheader("Allocation Engine")
    st.write("The engine scores stores using sales trend, recent momentum, in-stock pressure, inventory load, and margin health. Use the sidebar to tune the custom score.")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(f, x="Allocation Risk Score", y="Allocation Priority Score", color="Allocation Action", size="Sales YTD TY", hover_name="Store Label", hover_data=["Region", "Sales YTD YoY", "Sales LW YoY", "In-Stock Rate", "Inventory/Sales YTD", "Allocated LW TY"], title="Priority vs Risk Matrix")
        fig.add_hline(y=75, line_dash="dash")
        fig.add_vline(x=75, line_dash="dash")
        fig.update_layout(height=540)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(f, x="Custom Allocation Score", y="Allocated LW TY", color="Allocation Action", size="Sales YTD TY", hover_name="Store Label", title="Custom What-If Score vs Current Allocation")
        fig.update_layout(height=540)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recommended Allocation Actions")
    rec = f.sort_values(["Allocation Action", "Allocation Priority Score"], ascending=[True, False])[[
        "Region", "Store Label", "Sales YTD TY", "Sales YTD YoY", "Sales LW YoY", "In-Stock Rate", "Inventory/Sales YTD", "Allocated LW TY", "Allocated/Sales LW", "Allocation Priority Score", "Allocation Risk Score", "Custom Allocation Score", "Allocation Action", "Flags"
    ]]
    st.dataframe(styled_df(rec, currency_cols=["Sales YTD TY", "Allocated LW TY"], pct_cols=["Sales YTD YoY", "Sales LW YoY", "In-Stock Rate", "Allocated/Sales LW"], score_cols=["Inventory/Sales YTD", "Allocation Priority Score", "Allocation Risk Score", "Custom Allocation Score"]), use_container_width=True, height=560)
    download_button_for_df(rec, "Download allocation recommendations", "allocation_recommendations.csv")

# Inventory & In-Stock
with tabs[4]:
    st.subheader("Inventory & In-Stock")
    st.write("Identify understocked winners, inventory-heavy laggards, and stores where allocation may duplicate inbound supply.")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(f, x="In-Stock Rate", y="Sales YTD YoY", size="Sales YTD TY", color="Region", hover_name="Store Label", title="In-Stock vs YTD Growth")
        fig.add_vline(x=benchmark_instock, line_dash="dash")
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(f, x="Inventory/Sales YTD", y="Sales YTD YoY", size="Inventory LW TY", color="Allocation Action", hover_name="Store Label", title="Inventory Load vs YTD Growth")
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### Low In-Stock Stores")
        low = f.sort_values("In-Stock Rate").head(top_n)
        st.dataframe(styled_df(low[["Region", "Store Label", "In-Stock Rate", "SKU Count", "In-Stock Stores", "Sales YTD YoY", "Sales LW YoY", "Allocated LW TY", "Allocation Action"]], currency_cols=["Allocated LW TY"], pct_cols=["In-Stock Rate", "Sales YTD YoY", "Sales LW YoY"]), use_container_width=True, height=420)
    with c4:
        st.markdown("### Inventory-Heavy Stores")
        heavy = f.sort_values("Inventory/Sales YTD", ascending=False).head(top_n)
        st.dataframe(styled_df(heavy[["Region", "Store Label", "Inventory LW TY", "Sales YTD TY", "Inventory/Sales YTD", "Sales YTD YoY", "Sales LW YoY", "Allocation Risk Score", "Flags"]], currency_cols=["Inventory LW TY", "Sales YTD TY"], pct_cols=["Sales YTD YoY", "Sales LW YoY"], score_cols=["Inventory/Sales YTD", "Allocation Risk Score"]), use_container_width=True, height=420)

# Margin & Receipts
with tabs[5]:
    st.subheader("Margin & Receipts")
    st.write("Review whether performance issues are coming from volume, margin rate, receipts, or inventory timing.")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(f, x="Sales YTD YoY", y="GM YTD Change", size="Sales YTD TY", color="Region", hover_name="Store Label", title="Sales Growth vs GM Rate Change")
        fig.add_hline(y=0, line_dash="dash")
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(f, x="Receipt YTD YoY", y="Sales YTD YoY", size="Receipt YTD TY", color="Allocation Action", hover_name="Store Label", title="Receipts vs Sales Trend")
        fig.add_hline(y=0, line_dash="dash")
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Margin Pressure")
    margin_pressure = f.sort_values("GM YTD Change").head(top_n)[["Region", "Store Label", "Sales YTD TY", "Sales YTD YoY", "GM YTD TY", "GM YTD LY", "GM YTD Change", "Margin Gap YTD", "Receipt YTD YoY", "Flags"]]
    st.dataframe(styled_df(margin_pressure, currency_cols=["Sales YTD TY", "Margin Gap YTD"], pct_cols=["Sales YTD YoY", "GM YTD TY", "GM YTD LY", "GM YTD Change", "Receipt YTD YoY"]), use_container_width=True, height=460)

# Outliers & Clusters
with tabs[6]:
    st.subheader("Outliers & Clusters")
    st.write("Cluster stores into operating profiles and detect unusual combinations of growth, inventory, allocation, and in-stock rate.")
    clustered = add_clusters(f, cluster_k)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(clustered, x="Inventory/Sales YTD", y="Sales YTD YoY", color="Cluster", size="Sales YTD TY", hover_name="Store Label", title="Store Clusters: Inventory Load vs Growth")
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(clustered, x="In-Stock Rate", y="Allocated/Sales LW", color="Cluster", size="Allocated LW TY", hover_name="Store Label", title="Store Clusters: In-Stock vs Allocation Intensity")
        fig.add_vline(x=benchmark_instock, line_dash="dash")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    cluster_summary = clustered.groupby("Cluster").agg(
        Stores=("Store", "count"),
        Sales_YTD=("Sales YTD TY", "sum"),
        Sales_YTD_YoY=("Sales YTD YoY", "mean"),
        Sales_LW_YoY=("Sales LW YoY", "mean"),
        GM_YTD=("GM YTD TY", "mean"),
        In_Stock=("In-Stock Rate", "mean"),
        Inv_to_Sales=("Inventory/Sales YTD", "mean"),
        Allocated=("Allocated LW TY", "sum"),
        Priority=("Allocation Priority Score", "mean"),
        Risk=("Allocation Risk Score", "mean"),
    ).reset_index()
    st.markdown("### Cluster Summary")
    st.dataframe(styled_df(cluster_summary, currency_cols=["Sales_YTD", "Allocated"], pct_cols=["Sales_YTD_YoY", "Sales_LW_YoY", "GM_YTD", "In_Stock"], score_cols=["Inv_to_Sales", "Priority", "Risk"]), use_container_width=True)

    st.markdown("### Flagged Outliers")
    flagged = clustered[clustered["Flags"] != "No major flag"].sort_values("Allocation Risk Score", ascending=False)
    st.dataframe(styled_df(flagged[["Cluster", "Region", "Store Label", "Sales YTD TY", "Sales YTD YoY", "Sales LW YoY", "In-Stock Rate", "Inventory/Sales YTD", "Allocated/Sales LW", "Allocation Risk Score", "Allocation Action", "Flags"]], currency_cols=["Sales YTD TY"], pct_cols=["Sales YTD YoY", "Sales LW YoY", "In-Stock Rate", "Allocated/Sales LW"], score_cols=["Inventory/Sales YTD", "Allocation Risk Score"]), use_container_width=True, height=480)

# Data Explorer
with tabs[7]:
    st.subheader("Data Explorer")
    st.write("Search, filter, download, and inspect the fully enriched dataset.")
    search = st.text_input("Search store, site, region, action, or flags")
    explorer = f.copy()
    if search:
        mask = pd.Series(False, index=explorer.index)
        for c in ["Region", "Store Label", "Site No", "Allocation Action", "Flags"]:
            if c in explorer.columns:
                mask = mask | explorer[c].astype(str).str.contains(search, case=False, na=False)
        explorer = explorer[mask]

    default_cols = [
        "Region", "Store Label", "Site No", "Volume Band", "Size Band", "Sales LW TY", "Sales LW YoY", "Sales YTD TY", "Sales YTD YoY", "Margin YTD TY", "GM YTD TY", "Inventory LW TY", "Inventory YoY", "In Transit LW TY", "Allocated LW TY", "Allocated YoY", "SKU Count", "In-Stock Rate", "Inventory/Sales YTD", "Allocation Priority Score", "Allocation Risk Score", "Custom Allocation Score", "Allocation Action", "Flags"
    ]
    available_cols = [c for c in data.columns if c not in default_cols]
    selected_cols = st.multiselect("Columns", default_cols + available_cols, default=[c for c in default_cols if c in explorer.columns])
    view = explorer[selected_cols] if selected_cols else explorer
    st.dataframe(view, use_container_width=True, height=620)
    download_button_for_df(f, "Download enriched filtered data", "sper_enriched_filtered_data.csv")
    download_button_for_df(data, "Download enriched full data", "sper_enriched_full_data.csv")

st.caption("Built for weekly retail allocation review: pivot upload → cleaned metrics → store diagnostics → allocation recommendation workflow.")
