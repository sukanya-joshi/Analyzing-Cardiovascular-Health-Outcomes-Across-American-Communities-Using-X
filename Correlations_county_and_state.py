import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ============================================================
# FILE PATHS
# ============================================================
FIPS_COUNTS_FILE = "final_grouped_fips_2026.csv"
HEALTH_FILE = "health_info2.csv"
DIET_FILE = "county_diet_prevalence.csv"   

# ============================================================
# LOAD DATA
# ============================================================
fips_df = pd.read_csv(FIPS_COUNTS_FILE)
health_df = pd.read_csv(HEALTH_FILE)

fips_df.columns = [c.strip().lower() for c in fips_df.columns]
health_df.columns = [c.strip().lower() for c in health_df.columns]

fips_df["fips"] = fips_df["fips"].astype(str).str[:5].str.zfill(5)
health_df["fips"] = health_df["fips"].astype(str).str[:5].str.zfill(5)

# Merge county health outcomes
df = fips_df.merge(
    health_df[[
        "fips",
        "leisure",
        "smoking",
        "binge_drinking",
        "sleeping",
        "obesity"
    ]],
    on="fips",
    how="inner"
)

df = df.rename(columns={
    "leisure": "leisure_prev",
    "smoking": "smoking_prev",
    "binge_drinking": "binge_prev",
    "sleeping": "sleep_prev",
    "obesity": "obesity_prev"
})

# Optional diet merge
try:
    diet_df = pd.read_csv(DIET_FILE)
    diet_df.columns = [c.strip().lower() for c in diet_df.columns]
    diet_df["fips"] = diet_df["fips"].astype(str).str[:5].str.zfill(5)

    df = df.merge(
        diet_df[["fips", "diet_prevalence"]],
        on="fips",
        how="left"
    )
    print("Merged diet prevalence.")
except FileNotFoundError:
    print("Diet file not found; skipping diet correlations.")

# ============================================================
# CREATE RATE VARIABLES
# ============================================================
for behavior in ["pa", "diet", "smoking", "alcohol", "sleep"]:
    tweet_col = f"{behavior}_tweets"
    rate_col = f"{behavior}_rate"
    if tweet_col in df.columns and "total_tweets" in df.columns:
        df[rate_col] = df[tweet_col] / df["total_tweets"]

# ============================================================
# CORRELATION FUNCTION
# ============================================================
def get_corr_table(data, pairs, level_name):
    rows = []

    for x, y in pairs.items():
        if x not in data.columns or y not in data.columns:
            continue

        sub = data[[x, y]].dropna().copy()
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
# COUNTY-LEVEL CORRELATIONS
# ============================================================
pairs = {
    "pa_rate": "leisure_prev",
    "smoking_rate": "smoking_prev",
    "alcohol_rate": "binge_prev",
    "sleep_rate": "sleep_prev"
}

if "diet_rate" in df.columns and "diet_prevalence" in df.columns:
    pairs["diet_rate"] = "diet_prevalence"

county_corr_df = get_corr_table(df, pairs, "county")
print("\nCounty-level correlations:")
print(county_corr_df)

# ============================================================
# STATE-LEVEL CORRELATIONS
# ============================================================
df["state"] = df["fips"].str[:2]

agg_dict = {
    "pa_tweets": "sum",
    "diet_tweets": "sum",
    "smoking_tweets": "sum",
    "alcohol_tweets": "sum",
    "sleep_tweets": "sum",
    "total_tweets": "sum",
    "leisure_prev": "mean",
    "smoking_prev": "mean",
    "binge_prev": "mean",
    "sleep_prev": "mean",
}

if "diet_prevalence" in df.columns:
    agg_dict["diet_prevalence"] = "mean"

state_df = df.groupby("state", as_index=False).agg(agg_dict)

for behavior in ["pa", "diet", "smoking", "alcohol", "sleep"]:
    tweet_col = f"{behavior}_tweets"
    rate_col = f"{behavior}_rate"
    if tweet_col in state_df.columns and "total_tweets" in state_df.columns:
        state_df[rate_col] = state_df[tweet_col] / state_df["total_tweets"]

state_corr_df = get_corr_table(state_df, pairs, "state")
print("\nState-level correlations:")
print(state_corr_df)

# ============================================================
# SAVE
# ============================================================
county_corr_df.to_csv("county_correlations.csv", index=False)
state_corr_df.to_csv("state_correlations.csv", index=False)

print("\nSaved:")
print("- county_correlations.csv")
print("- state_correlations.csv")
