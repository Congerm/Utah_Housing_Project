"""
Variable definitions for Utah American Community Survey housing data pull.

All variable lists, the outcome, and predictor sets are here to facilitate importing
by fetch.py, models.py, and any future work. 
"""

# Variable Lists

B25024_VARS = [f"B25024_{str(i).zfill(3)}E" for i in range(1, 12)]
B25001_VARS = ["B25001_001E"]
B25002_VARS = [f"B25002_{str(i).zfill(3)}E" for i in range(1, 4)]
B25003_VARS = [f"B25003_{str(i).zfill(3)}E" for i in range(1, 4)]
B25008_VARS = [f"B25008_{str(i).zfill(3)}E" for i in range(1, 4)]
B25119_VARS = [f"B25119_{str(i).zfill(3)}E" for i in range(1, 4)]
B25032_VARS = [f"B25032_{str(i).zfill(3)}E" for i in range(1, 24)]
B25088_VARS = [f"B25088_{str(i).zfill(3)}E" for i in range(1, 4)]
B19013_VARS = ["B19013_001E"]

ALL_VARS: list[str] = (
    B25024_VARS + B25001_VARS + B25002_VARS + B25003_VARS
    + B25008_VARS + B25119_VARS + B25032_VARS + B25088_VARS + B19013_VARS
)

# Column Renaming

RENAME_MAP: dict[str, str] = {
    # B25024 - Units in Structure
    "B25024_001E": "total_units_b25024",
    "B25024_002E": "units_1_detached",
    "B25024_003E": "units_1_attached",
    "B25024_004E": "units_2",
    "B25024_005E": "units_3_4",
    "B25024_006E": "units_5_9",
    "B25024_007E": "units_10_19",
    "B25024_008E": "units_20_49",
    "B25024_009E": "units_50_plus",
    "B25024_010E": "mobile_home",
    "B25024_011E": "boat_rv_van",
    # B25001
    "B25001_001E": "total_housing_units",
    # B25002
    "B25002_001E": "total_units_b25002",
    "B25002_002E": "occupied_units",
    "B25002_003E": "vacant_units",
    # B25003
    "B25003_001E": "tenure_total",
    "B25003_002E": "owner_occupied",
    "B25003_003E": "renter_occupied",
    # B25008
    "B25008_001E": "pop_in_occupied_total",
    "B25008_002E": "pop_in_owner_occupied",
    "B25008_003E": "pop_in_renter_occupied",
    # B25119
    "B25119_001E": "median_income_all_occupied",
    "B25119_002E": "median_income_owner_occupied",
    "B25119_003E": "median_income_renter_occupied",
    # B25032
    "B25032_001E": "units_by_tenure_total",
    "B25032_002E": "owner_units_total",
    "B25032_003E": "owner_1_detached",
    "B25032_004E": "owner_1_attached",
    "B25032_005E": "owner_2",
    "B25032_006E": "owner_3_4",
    "B25032_007E": "owner_5_9",
    "B25032_008E": "owner_10_19",
    "B25032_009E": "owner_20_49",
    "B25032_010E": "owner_50_plus",
    "B25032_011E": "owner_mobile_home",
    "B25032_012E": "owner_boat_rv_van",
    "B25032_013E": "renter_units_total",
    "B25032_014E": "renter_1_detached",
    "B25032_015E": "renter_1_attached",
    "B25032_016E": "renter_2",
    "B25032_017E": "renter_3_4",
    "B25032_018E": "renter_5_9",
    "B25032_019E": "renter_10_19",
    "B25032_020E": "renter_20_49",
    "B25032_021E": "renter_50_plus",
    "B25032_022E": "renter_mobile_home",
    "B25032_023E": "renter_boat_rv_van",
    # B25088
    "B25088_001E": "median_owner_costs_total",
    "B25088_002E": "median_owner_costs_with_mortgage",
    "B25088_003E": "median_owner_costs_no_mortgage",
    # B19013
    "B19013_001E": "median_household_income",
}

# Sentinel columns that use -666666666 for suppressed values
SENTINEL_COLS: list[str] = [
    "median_income_all_occupied",
    "median_income_owner_occupied",
    "median_income_renter_occupied",
    "median_owner_costs_total",
    "median_owner_costs_with_mortgage",
    "median_owner_costs_no_mortgage",
    "median_household_income",
]

# Model Variable Sets (outcome variable, predictor variables)

OUTCOME = "median_owner_costs_with_mortgage"

PREDICTORS: list[str] = [    
    "pct_sf_renter_occupied",   # single-family homes rented out (investor proxy)
    "median_household_income",  # demand-side income
    "owner_renter_income_gap",  # income stratification
    "pct_vacant",               # market slack
    "pop_in_occupied_total",    # population demand pressure
]
