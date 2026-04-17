from __future__ import annotations
import os
import warnings
import requests
import pandas as pd
from .variables import ALL_VARS, RENAME_MAP, SENTINEL_COLS

STATE = "49"  # Utah Federal Information Processing Standards (FIPS) code
_BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"
_CHUNK_SIZE = 49  # Census API limit is 50 variables. We leave one slot open for NAME

# begin helper functions 

def _get_api_key() -> str:
    key = os.environ.get("CENSUS_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("API key required! API key is read from the CENSUS_API_KEY environment variable. " \
        "Obtain a key (for free) at: https://api.census.gov/data/key_signup.html")
    return key


def _fetch_chunk(base_url: str, chunk: list[str], geo_params: dict,) -> pd.DataFrame | None:
    """Fetch one chunk of ≤49 variables for all Utah census tracts."""
    params = {**geo_params, "get": ",".join(["NAME"] + chunk)}
    try:
        response = requests.get(base_url, params=params, timeout=30)
    except requests.RequestException as exc:
        warnings.warn(f"Request error: {exc}")
        return None

    if not response.ok:
        warnings.warn(f"API error ({response.status_code}): {response.text[:300]}")
        return None

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        warnings.warn(f"Bad JSON response: {response.text[:300]}")
        return None

    df = pd.DataFrame(data[1:], columns=data[0])
    geo_cols = {"NAME", "state", "county", "tract"}
    num_cols = [c for c in df.columns if c not in geo_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    return df


def _add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    
    """calculates columns that are derived - single_family_units, pct_single_family, pct_vacant, 
    pct_occupied, pct_renter_occupied, pct_owner_occupied, renter_single_family, 
    pct_sf_renter_occupied, owner_renter_income_gap"""

    df = df.copy()

    df["single_family_units"] = df["units_1_detached"] + df["units_1_attached"]
    df["pct_single_family"]   = df["single_family_units"] / df["total_units_b25024"]

    df["pct_vacant"]          = df["vacant_units"]  / df["total_units_b25002"]
    df["pct_occupied"]        = df["occupied_units"] / df["total_units_b25002"]

    df["pct_renter_occupied"] = df["renter_occupied"] / df["tenure_total"]
    df["pct_owner_occupied"]  = df["owner_occupied"]  / df["tenure_total"]

    df["renter_single_family"]   = df["renter_1_detached"] + df["renter_1_attached"]
    df["pct_sf_renter_occupied"] = df["renter_single_family"] / df["single_family_units"]

    df["owner_renter_income_gap"] = (
        df["median_income_owner_occupied"] - df["median_income_renter_occupied"]
    )
    return df


def fetch_year(year: int, api_key: str | None = None, verbose: bool = True) -> pd.DataFrame | None:
    """ Fetch data for all Utah census tracts for a single year.

    Parameters
    year : int
        Survey year (2009–2023).
    api_key : str, optional
        Census API key. Defaults to the CENSUS_API_KEY environment variable.
    verbose : bool
        Print progress. Default True.

    Returns: pandas dataframe, or None if failure. 
    """
    key = api_key or _get_api_key()
    base_url = _BASE_URL.format(year=year)
    geo_params = {"for": "tract:*", "in": f"state:{STATE}", "key": key}

    if verbose:
        print(f"  Fetching {year}...", end=" ", flush=True)

    chunks = [ALL_VARS[i : i + _CHUNK_SIZE] for i in range(0, len(ALL_VARS), _CHUNK_SIZE)]
    chunk_dfs = []
    for chunk in chunks:
        cdf = _fetch_chunk(base_url, chunk, geo_params)
        if cdf is None:
            return None
        chunk_dfs.append(cdf)

    # Merge all chunks on GEOID
    drop_cols = ["NAME", "state", "county", "tract"]
    df = chunk_dfs[0]
    for cdf in chunk_dfs[1:]:
        df = df.merge(cdf.drop(columns=drop_cols), on="GEOID")

    # Rename raw codes → readable names
    df = df.rename(columns=RENAME_MAP)

    # Replace Census sentinel values with NaN
    for col in SENTINEL_COLS:
        if col in df.columns:
            df[col] = df[col].where(df[col] > 0)

    # Add derived variables
    df = _add_derived_variables(df)

    df["year"] = year

    # Reorder: geography first
    geo_first = ["year", "GEOID", "NAME", "state", "county", "tract"]
    other_cols = [c for c in df.columns if c not in geo_first]
    df = df[geo_first + other_cols]

    if verbose:
        print(f"{len(df)} tracts")

    return df


def fetch_all_years(years: range | list[int] = range(2009, 2024), api_key: str | None = None, verbose: bool = True,) -> pd.DataFrame:
    """
    Fetch data for all Utah census tracts across multiple years.

    Parameters
    years : range or list[int]
        Survey years to pull. Defaults to 2009–2023.
    api_key : str, optional
        Census API key. Defaults to the CENSUS_API_KEY environment variable.
    verbose : bool
        Print progress. Default True.

    Returns pandas dataframe, with one row per tract per year. 
    """
    if verbose:
        print(f"Pulling data for Utah ({min(years)}–{max(years)})...\n")

    frames = [fetch_year(y, api_key=api_key, verbose=verbose) for y in years]
    good_frames = [f for f in frames if f is not None]

    if not good_frames:
        raise RuntimeError("All year fetches failed. Check your API key and network connection.")

    df = pd.concat(good_frames, ignore_index=True)

    if verbose:
        n_years = df["year"].nunique()
        n_tracts = len(df) // n_years
        print(f"\nTotal rows: {len(df):,}  ({n_years} years × ~{n_tracts} tracts)\n")

    return df
