import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# ============================================================
# FILES
# ============================================================
FEAT_FILE = "feat_table.csv"                 # columns: id, group_id, feat, value, group_norm
OUTCOME_FILE = "health_info2.csv"            # columns include FIPS + outcomes + controls
ACP_FILE = "fips_acp.csv"                    # columns: fips, acp_name
DIET_FILE = "county_diet_prevalence.csv"     # columns: FIPS, diet_prevalence

# ============================================================
# CONFIG
# ============================================================
GROUP_COL = "acp_name"
FIPS_COL_OUTCOME = "FIPS"

OUTCOMES = [
    "leisure",
    "smoking",
    "sleeping",
    "binge_drinking",
    "diet_prevalence"
]

CONTROL_COLS = ["LogIncome", "EducationIndex"]

OUTER_SPLITS = 10
INNER_MAX_SPLITS = 5

# ============================================================
# LOAD DATA
# ============================================================
feat_df = pd.read_csv(FEAT_FILE, low_memory=False)
outcome_df = pd.read_csv(OUTCOME_FILE, low_memory=False)
acp_df = pd.read_csv(ACP_FILE, low_memory=False)
diet_df = pd.read_csv(DIET_FILE, low_memory=False)

# Standardize column names
feat_df.columns = [c.strip() for c in feat_df.columns]
outcome_df.columns = [c.strip() for c in outcome_df.columns]
acp_df.columns = [c.strip().lower() for c in acp_df.columns]
diet_df.columns = [c.strip() for c in diet_df.columns]

# ============================================================
# CLEAN FIPS
# ============================================================
feat_df["group_id"] = feat_df["group_id"].astype(str).str[:5].str.zfill(5)
outcome_df[FIPS_COL_OUTCOME] = outcome_df[FIPS_COL_OUTCOME].astype(str).str[:5].str.zfill(5)
acp_df["fips"] = acp_df["fips"].astype(str).str[:5].str.zfill(5)
diet_df["FIPS"] = diet_df["FIPS"].astype(str).str[:5].str.zfill(5)

# ============================================================
# PIVOT FEATURES TO WIDE FORMAT
# Use group_norm as the feature value
# ============================================================
wide_df = feat_df.pivot_table(
    index="group_id",
    columns="feat",
    values="group_norm",
    aggfunc="first"
).reset_index()

wide_df.columns = ["fips"] + [f"topic_{c}" for c in wide_df.columns[1:]]

print("Wide feature table shape:", wide_df.shape)

# ============================================================
# MERGE OUTCOMES + DIET + ACP
# ============================================================
outcome_df = outcome_df.rename(columns={FIPS_COL_OUTCOME: "fips"})
diet_df = diet_df.rename(columns={"FIPS": "fips"})

outcome_df = outcome_df.merge(
    diet_df[["fips", "diet_prevalence"]],
    on="fips",
    how="left"
)

df = wide_df.merge(outcome_df, on="fips", how="inner")
df = df.merge(acp_df[["fips", "acp_name"]], on="fips", how="left")

print("Merged analytic table shape:", df.shape)
print("Missing ACP labels:", df["acp_name"].isna().sum())

# Drop rows missing ACP
df = df.dropna(subset=["acp_name"]).copy()

# ============================================================
# FEATURE COLUMNS
# ============================================================
topic_cols = [c for c in df.columns if c.startswith("topic_")]

if not topic_cols:
    raise ValueError("No topic columns found after pivot.")

# Fill missing topic values with 0
df[topic_cols] = df[topic_cols].fillna(0)

n_groups = df["acp_name"].nunique()
print("Distinct ACP groups:", n_groups)

if n_groups < OUTER_SPLITS:
    raise ValueError(
        f"Need at least {OUTER_SPLITS} ACP groups for GroupKFold, found {n_groups}."
    )
print("Total counties after merge:", len(df))

# ============================================================
# FUNCTION: RUN GROUPED RIDGE CV
# ============================================================
def run_grouped_ridge(
    df,
    outcome,
    topic_cols,
    control_cols,
    group_col="acp_name",
    outer_splits=10,
    inner_max_splits=5
):
    needed = [outcome, group_col] + topic_cols + control_cols
    sub = df[needed].dropna().copy()

    X_cols = topic_cols + control_cols
    X = sub[X_cols].to_numpy()
    y = sub[outcome].to_numpy()
    groups = sub[group_col].astype(str).to_numpy()

    outer_cv = GroupKFold(n_splits=outer_splits)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    param_grid = {
        "ridge__alpha": np.logspace(-3, 3, 25)
    }

    y_true_all = []
    y_pred_all = []
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups=groups),
        start=1
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        # Confirm no ACP leakage
        overlap = set(groups_train).intersection(set(groups_test))
        if overlap:
            raise ValueError(f"Leakage detected in fold {fold_idx}: {overlap}")

        inner_n_groups = pd.Series(groups_train).nunique()
        inner_splits = min(inner_max_splits, inner_n_groups)

        if inner_splits < 2:
            raise ValueError(
                f"Not enough ACP groups in training fold {fold_idx} for inner CV."
            )

        inner_cv = GroupKFold(n_splits=inner_splits)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_cv.split(X_train, y_train, groups=groups_train),
            scoring="r2",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        fold_rows.append({
            "outcome": outcome,
            "fold": fold_idx,
            "best_alpha": grid.best_params_["ridge__alpha"],
            "test_r2": r2_score(y_test, y_pred),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "n_test_counties": len(test_idx),
            "n_test_acps": pd.Series(groups_test).nunique()
        })

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    overall = {
        "outcome": outcome,
        "n_counties": len(sub),
        "n_acps": pd.Series(groups).nunique(),
        "overall_r2": r2_score(y_true_all, y_pred_all),
        "overall_mae": mean_absolute_error(y_true_all, y_pred_all),
        "pearson_r": pearsonr(y_true_all, y_pred_all)[0],
        "pearson_p": pearsonr(y_true_all, y_pred_all)[1],
        "spearman_rho": spearmanr(y_true_all, y_pred_all)[0],
        "spearman_p": spearmanr(y_true_all, y_pred_all)[1]
    }

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame({
        "outcome": outcome,
        "y_true": y_true_all,
        "y_pred": y_pred_all
    })

    return overall, fold_df, pred_df

# ============================================================
# RUN ALL OUTCOMES
# ============================================================
overall_rows = []
all_folds = []
all_preds = []

for outcome in OUTCOMES:
    print(f"\nRunning grouped CV for: {outcome}")
    overall, fold_df, pred_df = run_grouped_ridge(
        df=df,
        outcome=outcome,
        topic_cols=topic_cols,
        control_cols=CONTROL_COLS,
        group_col=GROUP_COL,
        outer_splits=OUTER_SPLITS,
        inner_max_splits=INNER_MAX_SPLITS
    )

    overall_rows.append(overall)
    all_folds.append(fold_df)
    all_preds.append(pred_df)

overall_df = pd.DataFrame(overall_rows)
folds_df = pd.concat(all_folds, ignore_index=True)
preds_df = pd.concat(all_preds, ignore_index=True)

print("\nOverall grouped CV results:")
print(overall_df)

overall_df.to_csv("grouped_cv_by_acp_overall.csv", index=False)
folds_df.to_csv("grouped_cv_by_acp_folds.csv", index=False)
preds_df.to_csv("grouped_cv_by_acp_predictions.csv", index=False)

print("\nSaved:")
print("- grouped_cv_by_acp_overall.csv")
print("- grouped_cv_by_acp_folds.csv")
print("- grouped_cv_by_acp_predictions.csv")
