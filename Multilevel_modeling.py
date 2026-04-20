import pandas as pd
import statsmodels.formula.api as smf

# ============================================================
# FILE PATHS
# ============================================================
MERGED_FILE = "merged_fips_tweets_health.csv"
FIPS_ACP_FILE = "fips_acp.csv"
DIET_FILE = "county_diet_prevalence.csv"   # optional

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(MERGED_FILE)
fips_acp = pd.read_csv(FIPS_ACP_FILE)

df.columns = [c.strip().lower() for c in df.columns]
fips_acp.columns = [c.strip().lower() for c in fips_acp.columns]

df["fips"] = df["fips"].astype(str).str[:5].str.zfill(5)
fips_acp["fips"] = fips_acp["fips"].astype(str).str[:5].str.zfill(5)

# Merge ACP labels
df = df.merge(
    fips_acp[["fips", "acp_name"]],
    on="fips",
    how="left"
)

# Drop missing ACP labels
df = df.dropna(subset=["acp_name"]).copy()

# If diet prevalence is not already in the merged data, add it
if "diet_prevalence" not in df.columns:
    try:
        diet_df = pd.read_csv(DIET_FILE)
        diet_df.columns = [c.strip().lower() for c in diet_df.columns]
        diet_df["fips"] = diet_df["fips"].astype(str).str[:5].str.zfill(5)

        df = df.merge(
            diet_df[["fips", "diet_prevalence"]],
            on="fips",
            how="left"
        )
        print("Merged diet_prevalence.")
    except FileNotFoundError:
        print("Diet file not found; skipping diet model.")

# ============================================================
# RUN MIXED MODEL
# ============================================================
def run_mixed_model(data, outcome, predictor, group_var="acp_name"):
    sub = data[[outcome, predictor, group_var]].dropna().copy()

    print(f"\nRunning: {outcome} ~ {predictor}")
    print(f"N = {len(sub)}")
    print(f"Groups = {sub[group_var].nunique()}")

    model = smf.mixedlm(
        f"{outcome} ~ {predictor}",
        data=sub,
        groups=sub[group_var]
    )

    result = model.fit(reml=False)

    print(result.summary())

    return {
        "outcome": outcome,
        "predictor": predictor,
        "coef": result.params.get(predictor),
        "p_value": result.pvalues.get(predictor),
        "ci_lower": result.conf_int().loc[predictor, 0] if predictor in result.params.index else None,
        "ci_upper": result.conf_int().loc[predictor, 1] if predictor in result.params.index else None,
        "n_counties": len(sub),
        "n_groups": sub[group_var].nunique()
    }

# ============================================================
# MODEL SPECIFICATIONS
# ============================================================
models = [
    ("leisure_prev", "pa_rate"),
    ("smoking_prev", "smoking_rate"),
    ("binge_prev", "alcohol_rate"),
    ("sleep_prev", "sleep_rate"),
]

if "diet_prevalence" in df.columns and "diet_rate" in df.columns:
    models.append(("diet_prevalence", "diet_rate"))

# ============================================================
# RUN ALL MODELS
# ============================================================
results = []
for outcome, predictor in models:
    try:
        results.append(run_mixed_model(df, outcome, predictor))
    except Exception as e:
        print(f"Model failed for {outcome} ~ {predictor}: {e}")

summary_df = pd.DataFrame(results)
print("\nFinal summary:")
print(summary_df)

summary_df.to_csv("multilevel_model_summary.csv", index=False)
print("\nSaved: multilevel_model_summary.csv")
