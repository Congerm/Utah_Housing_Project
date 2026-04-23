import re
import json
from urllib.request import urlopen
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
# from utah_housing import OUTCOME, PREDICTORS 
from utah_housing_package.utah_housing import OUTCOME, PREDICTORS

st.set_page_config(page_title="Utah Housing Affordability Explorer", layout="wide")

st.title("Utah Housing Affordability Explorer")
st.markdown(
    """
    A user-guided exploration of American Community Survey estimates for Utah census tracts, 2009–2023.

    Use the sidebar to filter by year range and county, then explore trends, maps, and statistics.
    """
)

DATA_PATH = "data/utah_housing.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["county_name"] = df["NAME"].str.extract(r",\s*(.+?)\s+County", expand=False)
    return df

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"`{DATA_PATH}` not found. Run `getting_started.ipynb` first to generate it."
    )
    st.stop()

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

all_years = sorted(df["year"].unique())
year_range = st.sidebar.slider(
    "Year Range",
    min_value=int(min(all_years)),
    max_value=int(max(all_years)),
    value=(int(min(all_years)), int(max(all_years))),
    step=1,
)
selected_years = list(range(year_range[0], year_range[1] + 1))

metric = st.sidebar.selectbox(
    "Outcome metric",
    options=[OUTCOME] + COMPLEX_PREDICTORS,
    format_func=lambda x: x.replace("_", " ").title(),
)

all_county_names = sorted(df["county_name"].dropna().unique())
st.sidebar.markdown("**Counties**")
with st.sidebar:
    btn_col1, btn_col2 = st.columns(2)
    if btn_col1.button("Select All", key="select_all"):
        for c in all_county_names:
            st.session_state[f"county_{c}"] = True
    if btn_col2.button("Deselect All", key="deselect_all"):
        for c in all_county_names:
            st.session_state[f"county_{c}"] = False
    col_a, col_b = st.columns(2)
    selected_county_names = []
    for i, c in enumerate(all_county_names):
        checked = (col_a if i % 2 == 0 else col_b).checkbox(c, value=True, key=f"county_{c}")
        if checked:
            selected_county_names.append(c)

# ── Filter ───────────────────────────────────────────────────────────────────
mask = (
    df["year"].isin(selected_years)
    & df["county_name"].isin(selected_county_names)
    & df["county_name"].notna()
)
filtered = df[mask].copy()

if filtered.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ── Data Preview ──────────────────────────────────────────────────────────────
st.subheader("Data Preview")
st.dataframe(filtered.head(50))
st.caption(f"{len(filtered):,} rows shown after filtering")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Trends Over Time", "County Comparison", "Variable Statistics"])

with tab1:
    st.markdown(f"### {metric.replace('_', ' ').title()} — Statewide Median by Year")

    yearly = (
        filtered.groupby("year")[metric]
        .median()
        .reset_index()
        .rename(columns={metric: "median_value"})
    )

    fig = px.line(
        yearly,
        x="year",
        y="median_value",
        markers=True,
        labels={"year": "Year", "median_value": metric.replace("_", " ").title()},
        title=f"Statewide Median {metric.replace('_', ' ').title()} Over Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Matplotlib version")
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yearly["year"], yearly["median_value"], marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Statewide Median {metric.replace('_', ' ').title()} Over Time")
    ax.grid(alpha=0.3)
    st.pyplot(fig2)

with tab2:
    st.markdown(f"### {metric.replace('_', ' ').title()} by County")

    all_metrics = [OUTCOME] + COMPLEX_PREDICTORS
    county_avg = (
        filtered.groupby(["county", "county_name"])[all_metrics]
        .median()
        .reset_index()
        .rename(columns={metric: "median_value"})
    )
    county_avg["fips"] = "49" + county_avg["county"].astype(str).str.zfill(3)

    @st.cache_data
    def load_geojson():
        with urlopen(
            "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        ) as r:
            return json.load(r)

    counties_geojson = load_geojson()

    hover_cols = {m: ":.2f" for m in all_metrics if m != metric}
    hover_cols["fips"] = False

    st.markdown("#### Utah County Map")
    fig_map = px.choropleth_map(
        county_avg,
        geojson=counties_geojson,
        locations="fips",
        color="median_value",
        color_continuous_scale="Viridis",
        map_style="open-street-map",
        center={"lat": 39.5, "lon": -111.5},
        zoom=5.85,
        height=650,
        hover_name="county_name",
        hover_data=hover_cols,
        labels={m: m.replace("_", " ").title() for m in all_metrics}
        | {"median_value": metric.replace("_", " ").title()},
        title=f"Median {metric.replace('_', ' ').title()} by County",
        opacity=0.7,
    )
    fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": False})

    county_yearly = (
        filtered.groupby(["year", "county_name"])[metric]
        .median()
        .reset_index()
        .rename(columns={metric: "median_value"})
    )

    fig3 = px.line(
        county_yearly,
        x="year",
        y="median_value",
        color="county_name",
        markers=True,
        labels={
            "year": "Year",
            "median_value": metric.replace("_", " ").title(),
            "county_name": "County",
        },
        title=f"{metric.replace('_', ' ').title()} by County Over Time",
    )
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        county_avg_sorted = county_avg.sort_values("median_value", ascending=False)
        fig4 = px.bar(
            county_avg_sorted,
            x="county_name",
            y="median_value",
            labels={
                "county_name": "County",
                "median_value": metric.replace("_", " ").title(),
            },
            title="Overall Median by County",
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.markdown("### County Summary Table")
        st.dataframe(
            county_avg_sorted[["county_name", "median_value"]].rename(
                columns={
                    "county_name": "County",
                    "median_value": metric.replace("_", " ").title(),
                }
            )
        )

with tab3:
    st.markdown("### Descriptive Statistics")

    analysis_cols = [OUTCOME] + COMPLEX_PREDICTORS
    available_cols = [c for c in analysis_cols if c in filtered.columns]

    st.write(filtered[available_cols].describe().round(3))

    st.markdown("### Missing Values")
    null_counts = filtered[available_cols].isnull().sum().rename("null_count").reset_index()
    null_counts.columns = ["variable", "null_count"]
    null_counts["pct_missing"] = (null_counts["null_count"] / len(filtered) * 100).round(2)
    st.dataframe(null_counts)

    st.markdown("### Correlation Matrix")
    corr_cols = st.multiselect(
        "Select variables",
        options=available_cols,
        default=available_cols,
    )
    if len(corr_cols) >= 2:
        corr = filtered[corr_cols].corr().round(3)
        fig5 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix",
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Select at least 2 variables to show the correlation matrix.")
