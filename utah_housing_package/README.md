# utah-housing

Census American Community Survey data fetcher and fixed effects model for Utah housing research.

## Setup

```bash
# Recommended, but not required: create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

pip install utah-housing
export CENSUS_API_KEY=your_key_here  # https://api.census.gov/data/key_signup.html
```

## Usage

### Pull data

```python
from utah_housing import fetch_all_years

df = fetch_all_years(years=range(2009, 2024))
df.to_csv("utah_housing_2009_2023.csv", index=False)
```

Pull a single year:

```python
from utah_housing import fetch_year

df_2022 = fetch_year(2022)
```

### Run models

```python
import pandas as pd
from utah_housing import run_model

df = pd.read_csv("utah_housing_2009_2023.csv")

results, coefs = run_model(df)
coefs.to_csv("results.csv")
```

### Run diagnostics

```python
from utah_housing.fixed_effect_model import run_diagnostics
from utah_housing import PREDICTORS

run_diagnostics(df, PREDICTORS)
```

## Model Overview 

We built a two-way fixed effects (or fixed panel) regression model to identify within-tract drivers of ownership costs. We initially wanted to use a simple multiple linear regression model, but found that we were explaining almost none of the variance in our data. A fixed effects model helps to remedy the inevitable unknown elements of variance in such an economic model, as it holds certain elements constant to account for inherent variance. 

Our model controls for tract-level fixed effects, meaning time-invariant characteristics of a neighborhood like location and county $\times$ year fixed effects, which account for unknown variables like employer expansion and COVID-era impacts. 

The model is as follows: 

$$
y_{it} = \beta X_{it} + \alpha_i + \lambda_{c(i), t} + u_{i,t}
$$

where $i =$ tract (Census GEOID), $t =$ year, and $c(i) =$ county of tract $i$.

$\alpha_i$ stands for tract fixed effect, which controls for anything constant within a tract over time, like location desirability, zoning baseline, and neighborhood quality. In other words, $\alpha$ lets us compare a tract to itself over time. 

$\lambda_{c(i), t}$ stands for the county $\times$ year fixed effect, controlling for anything affecting a county in a specific year; things like local economic shocks (ex: big employer entering or leaving), policy changes, or impacts from COVID. In other words, $\lambda_{c(i), t}$ allows us to remove shared shocks across nearby areas at the same time.

**Outcome:** `median_owner_costs_with_mortgage`

**Predictors:**
- `pct_sf_renter_occupied` — share of single-family homes that are renter-occupied (investor proxy)
- `median_household_income` — demand-side income
- `owner_renter_income_gap` — income stratification signal
- `pct_vacant` — market slack
- `pop_in_occupied_total` - population demand pressure


## Data tables pulled

| Table | Description |
|-------|-------------|
| B25024 | Units in structure |
| B25001 | Total housing units |
| B25002 | Occupancy status |
| B25003 | Tenure (owner vs. renter) |
| B25008 | Population in occupied housing by tenure |
| B25119 | Median household income by tenure |
| B25032 | Units in structure by tenure |
| B25088 | Median monthly owner costs |
| B19013 | Median household income |

## Package layout

```
utah_housing/
├── __init__.py      # public API
├── variables.py     # all ACS variable lists, rename map, model variable sets
├── fetch.py         # Census API fetcher
└── models.py        # fixed effects models + diagnostics
```
