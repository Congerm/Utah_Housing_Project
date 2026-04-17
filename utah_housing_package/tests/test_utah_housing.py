"""
Basic tests for utah_housing package.
Run with: pytest tests/
"""

import os
import pandas as pd
import numpy as np
import sys
import pytest

from utah_housing.variables import ALL_VARS, RENAME_MAP, SENTINEL_COLS, PREDICTORS, OUTCOME
from utah_housing.fixed_effects_model import run_model, run_diagnostics


# Variable tests - check that the variable configuration in variables.py is internally consistent. 

def test_all_vars_unique():
    # make sure no column name appears twice
    assert len(ALL_VARS) == len(set(ALL_VARS)), "Duplicate variables in ALL_VARS"


def test_rename_map_covers_all_vars():
    # verify that every column code has a corresponding descriptive name in RENAME_MAP
    missing = [v for v in ALL_VARS if v not in RENAME_MAP]
    assert not missing, f"Variables missing from RENAME_MAP: {missing}"


def test_sentinel_cols_use_renamed_names():
    # confirm that SENTINEL_COLS references the renamed column names and not the codes 
    renamed = set(RENAME_MAP.values())
    bad = [c for c in SENTINEL_COLS if c not in renamed]
    assert not bad, f"SENTINEL_COLS use raw ACS codes (should use renamed): {bad}"



# Fetch tests - check that fetch_year is very verbose when an API key isn't provided

def test_missing_api_key_raises():
    # fetch_year should raise EnvironmentError when CENSUS_API_KEY is unset
    key = os.environ.pop("CENSUS_API_KEY", None)
    try:
        from utah_housing.fetch import fetch_year
        with pytest.raises(EnvironmentError, match="CENSUS_API_KEY"):
            fetch_year(2020)
    finally:
        if key:
            os.environ["CENSUS_API_KEY"] = key


# Model tests using made-up data (because real data takes a bit to load). 

def _make_synthetic(n_tracts=40, n_years=5, seed=42) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    years = list(range(2015, 2015 + n_years))
    geoids = [f"49{str(i).zfill(9)}" for i in range(n_tracts)]
    counties = [str(i % 5) for i in range(n_tracts)]

    rows = []
    for geoid, county in zip(geoids, counties):
        tract_effect = rng.normal(0, 200)
        for year in years:
            inc = rng.uniform(40_000, 120_000)
            rows.append({
                "GEOID":                        geoid,
                "county":                       county,
                "year":                         year,
                "median_owner_costs_with_mortgage": 1200 + tract_effect + 0.003 * inc + rng.normal(0, 100),
                "pct_sf_renter_occupied":       rng.uniform(0.05, 0.5),
                "median_household_income":      inc,
                "owner_renter_income_gap":      rng.uniform(5_000, 40_000),
                "pct_vacant":                   rng.uniform(0.01, 0.15),
                "pop_in_occupied_total":        rng.integers(500, 5000),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_df():
    return _make_synthetic()


def test_run_model_returns_coef_df(synthetic_df):
    _, coef = run_model(synthetic_df, verbose=False)
    assert isinstance(coef, pd.DataFrame)
    assert set(PREDICTORS).issubset(set(coef.index))



def test_models_drop_missing_gracefully(synthetic_df):
    # models should handle NaN rows without crashing
    df_with_nulls = synthetic_df.copy()
    df_with_nulls.loc[df_with_nulls.index[:10], "median_household_income"] = np.nan
    _, coef = run_model(df_with_nulls, verbose=False)
    assert len(coef) > 0
