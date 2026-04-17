"""
Fixed effects model for Utah housing affordability analysis.

Outcome: median_owner_costs_with_mortgage

Usage:
    from utah_housing import run_model
    import pandas as pd

    df = pd.read_csv("utah_housing_2009_2023.csv")
    results, coefs = run_model(df)
"""

from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS
from linearmodels.iv.absorbing import AbsorbingLS
from .variables import OUTCOME, PREDICTORS

warnings.filterwarnings("ignore")

def run_diagnostics(df: pd.DataFrame, predictors: list[str]) -> None:
    # Print correlation matrix and VIF diagnostics for predictors. 

    print("PREDICTOR CORRELATIONS")
    corr = df[predictors].corr().round(2)
    print(corr.to_string())

    high_corr = [
        (predictors[i], predictors[j], corr.iloc[i, j])
        for i in range(len(predictors))
        for j in range(i + 1, len(predictors))
        if abs(corr.iloc[i, j]) > 0.8
    ]

    if high_corr:
        print("high correlations: ")
        for a, b, r in high_corr:
            print(f"   {a} <-> {b}: r = {r}")
    else:
        print("correlations are acceptable and we can move forward with analysis")

    print("VIF DIAGNOSTICS")

    X_vif = df[predictors].dropna()
    vif_df = pd.DataFrame({
        "variable": predictors,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(len(predictors))],
    }).round(2)
    print(vif_df.to_string(index=False))
    high_vif = vif_df[vif_df["VIF"] > 5]
    if len(high_vif):
        print(f"High VIFs: {list(high_vif['variable'])}")
    else:
        print("VIFs are acceptable and we can move forward with analysis")

def _prepare(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Select, drop NaN rows, and filter to tracts with 2+ years."""
    cols = ["GEOID", "county", "year", OUTCOME] + predictors
    out = df[cols].dropna().copy()
    counts = out.groupby("GEOID")["year"].count()
    valid = counts[counts > 1].index
    out = out[out["GEOID"].isin(valid)]
    return out

def _coef_table(result) -> pd.DataFrame:
    return pd.DataFrame({
        "coef":   result.params,
        "se":     result.std_errors,
        "t":      result.tstats,
        "p":      result.pvalues,
        "ci_low": result.params - 1.96 * result.std_errors,
        "ci_high":result.params + 1.96 * result.std_errors,
    }).drop(index="const", errors="ignore").round(4)


def run_model(df: pd.DataFrame, verbose: bool = True,) -> tuple:
    """
    Run the model (tract FE + county×year FE) via AbsorbingLS.

    County×year FE absorbs any shock that hits a county in a specific year
    (e.g. a city rezoning, a local employer expanding, COVID hitting SLC harder).
    Adds pop_in_occupied_total as a direct within-tract demand signal.

    Parameters
    ----------
    df : pd.DataFrame
        Output of fetch_all_years() or a saved CSV.
    verbose : bool
        Print summary table. Default True.

    Returns
    -------
    (result, coef_df)
        result  — linearmodels AbsorbingLS result object
        coef_df — pd.DataFrame with coef, se, t, p, ci_low, ci_high
    """
    clean = _prepare(df, PREDICTORS)
    clean["county_year"] = clean["county"].astype(str) + "_" + clean["year"].astype(str)

    if verbose:
        n_drop = len(df) - len(clean)
        print(f"Fitted model: {len(clean):,} rows ({n_drop:,} dropped for missing values), "
              f"{clean['GEOID'].nunique():,} tracts\n")

    X = sm.add_constant(clean[PREDICTORS])
    absorb = pd.DataFrame({
        "tract_fe":       pd.Categorical(clean["GEOID"]),
        "county_year_fe": pd.Categorical(clean["county_year"]),
    })

    model = AbsorbingLS(clean[OUTCOME], X, absorb=absorb)
    result = model.fit(cov_type="clustered", clusters=pd.Categorical(clean["GEOID"]),)

    if verbose:
        print("FITTED MODEL (tract fixed effect + county * year fixed effect)")
        print(result.summary.tables[1])
        print(f"\n  Absorbed R²:  {result.rsquared:.4f}")
        print(f"  Observations: {result.nobs:,}\n")

    return result, _coef_table(result)

