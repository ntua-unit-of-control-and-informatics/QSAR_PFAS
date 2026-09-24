"""Seed-sensitivity check for the nested-CV result.

Fully self-contained: everything needed to run the outer leave-one-PFAS-out
nested-CV loop (dataset loading/encoding, preprocessing, forward selection,
per-fold fit/predict, macro/micro RMSE/MAE/R^2) is copied here identically
from Restructured/old/nested_CV_forced_halflife_type.py, so the recipe being
tested is exactly the same one - this script imports nothing from it and
does not modify or run it.

Left out on purpose, since none of it is needed to test seed sensitivity:
the reference-only 5-fold CV, the final deployed model fit on the full
dataset, the selection-frequency bookkeeping, and all plotting.

For 10 distinct seeds (0-9), RANDOM_STATE is set to that value and the full
outer LOPO loop (with per-fold forward selection) is rerun for both arms,
collecting the six nested-CV metrics per arm per seed. After all 10 seeds,
only one summary is printed: per metric, the mean and standard deviation
across seeds for each arm, and how many of the 10 seeds favored Arm B over
Arm A on that metric.
"""

import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42  # reassigned per seed in the sweep below
N_SELECT_STEPS = 6
N_INNER_SPLITS = 5
# Binary features forced into every feature set (both arms): apparent vs
# intrinsic half-life are different estimations of the outcome, so the model
# should always account for them. LogKa (Arm B) is forced too; see
# forced_features_for.
FORCED_BINARY_FEATURES = ["half_life_type"]
NON_FEATURE_COLS = {
    "PFAS",
    "log_half_life",
    "half_life",
    "SE",
    "SE_method",
    "CV",
    "weight",
}

dataset = pd.read_csv("Restructured/data/sum_featurized_dataset.csv")
dataset.drop(["Study"], axis=1, inplace=True)

# Binary encoding
dataset["half_life_type"] = dataset["half_life_type"].map(
    {"apparent": 0, "intrinsic": 1}
)
dataset["Occupational_exposure"] = dataset["Occupational_exposure"].map(
    {False: 0, True: 1}
)

rdkit_descriptor_names = {name for name, _ in Descriptors.descList}


def correlation_filter(X, threshold=0.99):
    corr_matrix = X.corr().abs()
    mean_corr = corr_matrix.mean()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        for corr_feat in upper.index[upper[col] > threshold].tolist():
            if corr_feat in to_drop:
                continue
            if mean_corr[col] < mean_corr[corr_feat]:
                to_drop.add(col)
                break
            else:
                to_drop.add(corr_feat)
    return X.drop(columns=list(to_drop))


def _numerical_and_binary_cols(columns, include_logka):
    numerical_cols = [
        c
        for c in columns
        if c in rdkit_descriptor_names or (include_logka and c == "LogKa")
    ]
    binary_cols = [
        c for c in columns if c not in numerical_cols and c not in NON_FEATURE_COLS
    ]
    return numerical_cols, binary_cols


def forced_features_for(include_logka):
    """Features always included, in order: LogKa first (Arm B only), then the
    forced binary features."""
    return (["LogKa"] if include_logka else []) + FORCED_BINARY_FEATURES


def _grouped_cv_rmse(X, y, groups, n_splits=N_INNER_SPLITS):
    # shuffle=True + fixed random_state gives a reproducible, but non-trivial,
    # grouped split; the same (X, groups) always yields the same folds, so
    # candidates within a forward-selection step are compared on equal footing.
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    squared_errors = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        # n_jobs=1 avoids per-fit threadpool startup overhead across the very
        # large number of tiny fits forward selection performs; it does not
        # change the fitted model, only wall-clock time.
        model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        squared_errors.append(mean_squared_error(y.iloc[val_idx], preds))
    return float(np.sqrt(np.mean(squared_errors)))


def forward_select_features(X, y, groups, n_steps=N_SELECT_STEPS, forced_features=None):
    """Greedy forward selection. `forced_features` (if given) are placed in
    `selected` up front, without competing for their slot, and count toward
    `n_steps` - the remaining slots are filled competitively as usual."""
    selected = list(forced_features) if forced_features else []
    remaining = [c for c in X.columns if c not in selected]
    history = [
        {"feature": feat, "inner_cv_rmse": None, "forced": True} for feat in selected
    ]
    for _ in range(n_steps - len(selected)):
        best_feature, best_rmse = None, np.inf
        for candidate in remaining:
            trial_features = selected + [candidate]
            rmse = _grouped_cv_rmse(X[trial_features], y, groups)
            if rmse < best_rmse:
                best_rmse, best_feature = rmse, candidate
        selected.append(best_feature)
        remaining.remove(best_feature)
        history.append(
            {"feature": best_feature, "inner_cv_rmse": best_rmse, "forced": False}
        )
    return selected, history


def fit_preprocessing(train_df, include_logka):
    """Fit scaling/variance/correlation filters on `train_df` only. Returns the
    fitted transforms plus the processed training feature matrix/target/groups
    (no feature selection)."""
    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    numerical_cols, binary_cols = _numerical_and_binary_cols(
        feature_cols, include_logka
    )

    X_num = train_df[numerical_cols].copy()
    scaler = MinMaxScaler().fit(X_num)
    X_num[:] = scaler.transform(X_num)

    # LogKa is forced as Arm B's first selected feature (see
    # forced_features_for), so it must survive preprocessing even
    # if it would otherwise be dropped by the variance/correlation filters -
    # exempt it from both and add it back afterward.
    if include_logka:
        logka_col = X_num[["LogKa"]]
        X_num = X_num.drop(columns=["LogKa"])

    num_var_filter = VarianceThreshold().fit(X_num)
    X_num = X_num.loc[:, num_var_filter.get_support()]
    X_num = correlation_filter(X_num)

    if include_logka:
        X_num = pd.concat([logka_col, X_num], axis=1)

    X_bin = train_df[binary_cols].copy()

    # Forced binary features (half_life_type) get the same protection as
    # LogKa above: pulled out before the binary variance/correlation filters
    # and added back afterward, so they can never be dropped.
    forced_bin_cols = [c for c in FORCED_BINARY_FEATURES if c in X_bin.columns]
    forced_bin = X_bin[forced_bin_cols]
    X_bin = X_bin.drop(columns=forced_bin_cols)

    bin_var_filter = VarianceThreshold(threshold=0).fit(X_bin)
    X_bin = X_bin.loc[:, bin_var_filter.get_support()]
    X_bin = correlation_filter(X_bin)

    X_bin = pd.concat([forced_bin, X_bin], axis=1)

    X_train = pd.concat([X_num, X_bin], axis=1)
    y_train = train_df["log_half_life"]
    groups_train = train_df["PFAS"]

    fit_result = {
        "scaler": scaler,
        "numerical_cols_all": numerical_cols,
        "numerical_cols_kept": X_num.columns.tolist(),
        "binary_cols_kept": X_bin.columns.tolist(),
    }
    return fit_result, X_train, y_train, groups_train


def fit_preprocessing_and_select_features(train_df, include_logka):
    """`fit_preprocessing`, then forward feature selection scored by inner
    GroupKFold RMSE (grouped by PFAS). Adds `selected_features` and
    `selection_history` to the returned fit_result."""
    fit_result, X_train, y_train, groups_train = fit_preprocessing(
        train_df, include_logka
    )
    forced_features = forced_features_for(include_logka)
    selected_features, selection_history = forward_select_features(
        X_train, y_train, groups_train, forced_features=forced_features
    )
    fit_result["selected_features"] = selected_features
    fit_result["selection_history"] = selection_history
    return fit_result, X_train, y_train, groups_train


def transform_with_fit(test_df, fit_result):
    num_all = fit_result["numerical_cols_all"]
    X_num_test = pd.DataFrame(
        fit_result["scaler"].transform(test_df[num_all]),
        columns=num_all,
        index=test_df.index,
    )
    X_num_test = X_num_test[fit_result["numerical_cols_kept"]]
    X_bin_test = test_df[fit_result["binary_cols_kept"]].copy()
    return pd.concat([X_num_test, X_bin_test], axis=1)


def run_outer_loop_metrics(df, include_logka):
    """Outer leave-one-PFAS-out loop with per-fold forward selection, for one
    arm, returning only the six nested-CV metrics (macro/micro RMSE/MAE/R^2).
    Copied from run_nested_cv's outer loop and metric computation, minus the
    printing, the final deployed model, the reference-only CV, and the
    selection-frequency bookkeeping - none of which this script needs.
    """
    df = df.copy()
    if include_logka:
        df = df[~df["LogKa"].isna()].reset_index(drop=True)
    else:
        df = df.drop(columns=["LogKa"]).reset_index(drop=True)

    groups_all = sorted(df["PFAS"].unique())
    fold_records = []
    per_fold_rmse = []
    per_fold_mae = []

    for held_out in groups_all:
        outer_test = df[df["PFAS"] == held_out].reset_index(drop=True)
        outer_train = df[df["PFAS"] != held_out].reset_index(drop=True)

        fit_result, X_train, y_train, _ = fit_preprocessing_and_select_features(
            outer_train, include_logka
        )
        selected = fit_result["selected_features"]

        model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1)
        model.fit(X_train[selected], y_train)

        X_test = transform_with_fit(outer_test, fit_result)[selected]
        y_test = outer_test["log_half_life"]
        preds = model.predict(X_test)

        per_fold_rmse.append(float(np.sqrt(mean_squared_error(y_test, preds))))
        per_fold_mae.append(float(mean_absolute_error(y_test, preds)))

        for true_val, pred_val in zip(y_test.tolist(), preds.tolist()):
            fold_records.append(
                {"PFAS": held_out, "true": true_val, "predicted": pred_val}
            )

    results_df = pd.DataFrame(fold_records)

    # Macro metrics: unweighted mean of each held-out PFAS group's own
    # RMSE/MAE, so a compound with many rows counts exactly the same as a
    # singleton compound.
    macro_rmse = float(np.mean(per_fold_rmse))
    macro_mae = float(np.mean(per_fold_mae))

    # Micro metrics: every row pooled together before computing one metric.
    micro_rmse = float(
        np.sqrt(mean_squared_error(results_df["true"], results_df["predicted"]))
    )
    micro_mae = float(mean_absolute_error(results_df["true"], results_df["predicted"]))
    micro_r2 = float(r2_score(results_df["true"], results_df["predicted"]))

    # Macro R^2: collapse each congener to its mean true / mean predicted
    # value, then compute one classical R^2 across congeners.
    collapsed = results_df.groupby("PFAS")[["true", "predicted"]].mean()
    macro_r2 = float(r2_score(collapsed["true"], collapsed["predicted"]))

    return {
        "macro_rmse": macro_rmse,
        "macro_mae": macro_mae,
        "macro_r2": macro_r2,
        "micro_rmse": micro_rmse,
        "micro_mae": micro_mae,
        "micro_r2": micro_r2,
    }


SEEDS = list(range(10))
ARMS = [
    ("A (no LogKa)", False),
    ("B (LogKa+half_life_type forced)", True),
]
METRICS = ["macro_rmse", "macro_mae", "macro_r2", "micro_rmse", "micro_mae", "micro_r2"]
LOWER_IS_BETTER = {"macro_rmse", "macro_mae", "micro_rmse", "micro_mae"}


if __name__ == "__main__":
    per_arm_runs = {arm_name: [] for arm_name, _ in ARMS}

    for seed in SEEDS:
        RANDOM_STATE = seed
        for arm_name, include_logka in ARMS:
            print(f"Running seed={seed}, arm={arm_name}...")
            per_arm_runs[arm_name].append(
                run_outer_loop_metrics(dataset, include_logka)
            )

    arm_a_name, arm_b_name = ARMS[0][0], ARMS[1][0]
    summary_rows = []
    for metric in METRICS:
        a_vals = np.array([run[metric] for run in per_arm_runs[arm_a_name]])
        b_vals = np.array([run[metric] for run in per_arm_runs[arm_b_name]])
        b_better = (
            int(np.sum(b_vals < a_vals))
            if metric in LOWER_IS_BETTER
            else int(np.sum(b_vals > a_vals))
        )
        summary_rows.append(
            {
                "metric": metric,
                f"{arm_a_name} mean": a_vals.mean(),
                f"{arm_a_name} std": a_vals.std(ddof=1),
                f"{arm_b_name} mean": b_vals.mean(),
                f"{arm_b_name} std": b_vals.std(ddof=1),
                f"B better ({len(SEEDS)} seeds)": b_better,
            }
        )

    summary_df = pd.DataFrame(summary_rows).set_index("metric")
    print(
        f"\nSeed sensitivity of the nested-CV outer loop across seeds {SEEDS} "
        "(RANDOM_STATE), mean +/- std across seeds per arm, and how many "
        "seeds favored Arm B over Arm A on each metric:"
    )
    print(summary_df.to_string())
