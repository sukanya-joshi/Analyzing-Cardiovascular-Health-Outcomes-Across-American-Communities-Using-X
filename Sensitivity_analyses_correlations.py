import pandas as pd
from scipy.stats import pearsonr, spearmanr
import requests

# ============================================================
# FILE PATHS
# ============================================================
MERGED_FILE = "merged_fips_tweets_health.csv"

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(MERGED_FILE)
df.columns = [c.strip().lower() for c in df.columns]
df["fips"] = df["fips"].astype(str).str[:5].str.zfill(5)

# ============================================================
# GET COUNTY POPULATION FROM ACS
# ============================================================
url = "https://api.census.gov/data/2019/acs/acs5?get=B01003_001E&for=county:*"
res = requests.get(url)
res.raise_for_status()

pop_json = res.json()
pop_df = pd.DataFrame(pop_json[1:], columns=pop_json[0])
pop_df["fips"] = pop_df["state"] + pop_df["county"]
pop_df["population"] = pd.to_numeric(pop_df["B01003_001E"], errors="coerce")
pop_df = pop_df[["fips", "population"]]

# Merge population
df = df.merge(pop_df, on="fips", how="left")

print("Total rows:", len(df))
print("Missing population:", df["population"].isna().sum())

# Keep only rows with population for sensitivity analysis
df_sens = df.dropna(subset=["population"]).copy()

# ============================================================
# COMPUTE PER-100K TWEET METRICS
# ============================================================
tweet_cols = ["pa_tweets", "diet_tweets", "smoking_tweets", "alcohol_tweets", "sleep_tweets"]

for col in tweet_cols:
    if col in df_sens.columns:
        per100k_col = col.replace("_tweets", "_per100k")
        df_sens[per100k_col] = (df_sens[col] / df_sens["population"]) * 100000

# ============================================================
# CORRELATION FUNCTION
# ============================================================
def get_corr_table(data, pairs, level_name):
    rows = []

    for x, y in pairs.items():
        if x not in data.columns or y not in data.columns:
            continue

        sub = data[[x, y]].dropna()
        if len(sub) < 3:
            continue

        pear_r, pear_p = pearsonr(sub[x], sub[y])
        spear_r, spear_p = spearmanr(sub[x], sub[y])

        rows.append({
            "level": level_name,
            "tweet_metric": x,
            "prevalence_metric": y,
            "n": len(sub),
            "pearson_r": pear_r,
            "pearson_p": pear_p,
            "spearman_rho": spear_r,
            "spearman_p": spear_p
        })

    return pd.DataFrame(rows)

# ============================================================
# COUNTY-LEVEL SENSITIVITY
# ============================================================
pairs = {
    "pa_per100k": "leisure_prev",
    "smoking_per100k": "smoking_prev",
    "alcohol_per100k": "binge_prev",
    "sleep_per100k": "sleep_prev"
}

if "diet_prevalence" in df_sens.columns and "diet_per100k" in df_sens.columns:
    pairs["diet_per100k"] = "diet_prevalence"

county_corr_per100k = get_corr_table(df_sens, pairs, "county")
print("\nCounty-level per-100k correlations:")
print(county_corr_per100k)

# ============================================================
# STATE-LEVEL SENSITIVITY
# ============================================================
df_sens["state"] = df_sens["fips"].str[:2]

agg_dict = {
    "pa_tweets": "sum",
    "diet_tweets": "sum",
    "smoking_tweets": "sum",
    "alcohol_tweets": "sum",
    "sleep_tweets": "sum",
    "population": "sum",
    "leisure_prev": "mean",
    "smoking_prev": "mean",
    "binge_prev": "mean",
    "sleep_prev": "mean"
}

if "diet_prevalence" in df_sens.columns:
    agg_dict["diet_prevalence"] = "mean"

state_df = df_sens.groupby("state", as_index=False).agg(agg_dict)

for col in tweet_cols:
    if col in state_df.columns:
        per100k_col = col.replace("_tweets", "_per100k")
        state_df[per100k_col] = (state_df[col] / state_df["population"]) * 100000

state_corr_per100k = get_corr_table(state_df, pairs, "state")
print("\nState-level per-100k correlations:")
print(state_corr_per100k)

# ============================================================
# SAVE
# ============================================================
df_sens.to_csv("df_with_per100k.csv", index=False)
county_corr_per100k.to_csv("county_correlations_per100k.csv", index=False)
state_corr_per100k.to_csv("state_correlations_per100k.csv", index=False)

print("\nSaved:")
print("- df_with_per100k.csv")
print("- county_correlations_per100k.csv")
print("- state_correlations_per100k.csv")
