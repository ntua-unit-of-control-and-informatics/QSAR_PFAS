import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
N_SELECT_STEPS = 10
N_INNER_SPLITS = 5
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

# CV-based inverse-variance weighting: weight = 1 / CV**2, used identically for
# training and evaluation at both the inner and outer CV levels. Raw-scale
# SE (1/SE**2) was tried first but is scale-inconsistent with the log10
# target - short-half-life compounds get artificially tiny absolute SE and
# therefore enormous weight unrelated to how well-measured they actually are.
# CV = SE / half_life is scale-consistent instead. The featurized dataset
# only kept the log column, so half_life is reconstructed from it.
dataset["half_life"] = 10 ** dataset["log_half_life"]
dataset["CV"] = dataset["SE"] / dataset["half_life"]

# A zero or NaN CV would blow up (inf/NaN weight), so check for that up front.
print(
    f"CV sanity check: min={dataset['CV'].min():.6g}, max={dataset['CV'].max():.6g}, "
    f"zeros={int((dataset['CV'] == 0).sum())}, NaNs={int(dataset['CV'].isna().sum())}"
)
dataset["weight"] = 1.0 / (dataset["CV"] ** 2)
weight_median = dataset["weight"].median()
print(
    f"Weight sanity check: min={dataset['weight'].min():.6g}, "
    f"max={dataset['weight'].max():.6g}, median={weight_median:.6g}, "
    f"max/median ratio={dataset['weight'].max() / weight_median:.6g}"
)

dataset.drop(["Study"], axis=1, inplace=True)

# Binary encoding
dataset["half_life_type"] = dataset["half_life_type"].map(
    {"apparent": 0, "intrinsic": 1}
)
dataset["Occupational_exposure"] = dataset["Occupational_exposure"].map(
    {False: 0, True: 1}
)

rdkit_descriptor_names = {name for name, _ in Descriptors.descList}


def correlation_filter(X, threshold=0.95):
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


def _grouped_cv_rmse(X, y, groups, weights, n_splits=N_INNER_SPLITS):
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
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx], sample_weight=weights.iloc[train_idx]
        )
        preds = model.predict(X.iloc[val_idx])
        squared_errors.append(
            mean_squared_error(
                y.iloc[val_idx], preds, sample_weight=weights.iloc[val_idx]
            )
        )
    return float(np.sqrt(np.mean(squared_errors)))


def _macro_r2(true_vals, pred_vals, groups, weights):
    """Collapse each group's (true, predicted) pairs to one weighted-average
    representative point using that group's own weights, then compute
    classical (unweighted) R^2 across the resulting one-point-per-group set.
    Compound-equal-weighted, unlike a row-pooled R^2."""
    df = pd.DataFrame(
        {
            "group": list(groups),
            "true": list(true_vals),
            "pred": list(pred_vals),
            "weight": list(weights),
        }
    )
    collapsed = df.groupby("group").apply(
        lambda g: pd.Series(
            {
                "true": np.average(g["true"], weights=g["weight"]),
                "pred": np.average(g["pred"], weights=g["weight"]),
            }
        ),
        include_groups=False,
    )
    return float(r2_score(collapsed["true"], collapsed["pred"])), len(collapsed)


def _grouped_cv_metrics(X, y, groups, weights, n_splits=N_INNER_SPLITS):
    """Cross-validate a FIXED feature set with GroupKFold (grouped by PFAS),
    aggregating out-of-fold predictions across all folds before computing
    CV-weighted RMSE/MAE once - unlike nested CV, the feature set here is not
    re-selected per fold. R2 is compound-collapsed (see _macro_r2), so a
    compound with many rows in these folds doesn't dominate it."""
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_true_all, y_pred_all, w_all, groups_all = [], [], [], []
    for train_idx, val_idx in gkf.split(X, y, groups):
        model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx], sample_weight=weights.iloc[train_idx]
        )
        preds = model.predict(X.iloc[val_idx])
        y_true_all.extend(y.iloc[val_idx].tolist())
        y_pred_all.extend(preds.tolist())
        w_all.extend(weights.iloc[val_idx].tolist())
        groups_all.extend(groups.iloc[val_idx].tolist())
    rmse = float(
        np.sqrt(mean_squared_error(y_true_all, y_pred_all, sample_weight=w_all))
    )
    r2, _ = _macro_r2(y_true_all, y_pred_all, groups_all, w_all)
    mae = mean_absolute_error(y_true_all, y_pred_all, sample_weight=w_all)
    return rmse, r2, mae


def forward_select_features(
    X, y, groups, weights, n_steps=N_SELECT_STEPS, forced_features=None
):
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
            rmse = _grouped_cv_rmse(X[trial_features], y, groups, weights)
            if rmse < best_rmse:
                best_rmse, best_feature = rmse, candidate
        selected.append(best_feature)
        remaining.remove(best_feature)
        history.append(
            {"feature": best_feature, "inner_cv_rmse": best_rmse, "forced": False}
        )
    return selected, history


def fit_preprocessing_and_select_features(train_df, include_logka):
    """Fit scaling/variance/correlation filters on `train_df` only, then run
    forward feature selection scored by inner GroupKFold RMSE (grouped by
    PFAS). Returns the fitted transforms plus the processed training feature
    matrix/target/groups and the selected feature names."""
    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    numerical_cols, binary_cols = _numerical_and_binary_cols(
        feature_cols, include_logka
    )

    X_num = train_df[numerical_cols].copy()
    scaler = MinMaxScaler().fit(X_num)
    X_num[:] = scaler.transform(X_num)

    # LogKa is forced as Arm B's first selected feature (see
    # forward_select_features below), so it must survive preprocessing even
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
    bin_var_filter = VarianceThreshold(threshold=0).fit(X_bin)
    X_bin = X_bin.loc[:, bin_var_filter.get_support()]
    X_bin = correlation_filter(X_bin)

    X_train = pd.concat([X_num, X_bin], axis=1)
    y_train = train_df["log_half_life"]
    groups_train = train_df["PFAS"]
    weights_train = train_df["weight"]

    forced_features = ["LogKa"] if include_logka else None
    selected_features, selection_history = forward_select_features(
        X_train, y_train, groups_train, weights_train, forced_features=forced_features
    )

    fit_result = {
        "scaler": scaler,
        "numerical_cols_all": numerical_cols,
        "numerical_cols_kept": X_num.columns.tolist(),
        "binary_cols_kept": X_bin.columns.tolist(),
        "selected_features": selected_features,
        "selection_history": selection_history,
    }
    return fit_result, X_train, y_train, groups_train, weights_train


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


def run_nested_cv(df, include_logka, arm_name):
    print(f"\n{'=' * 20} Arm {arm_name} (include LogKa={include_logka}) {'=' * 20}")

    df = df.copy()
    if include_logka:
        missing_mask = df["LogKa"].isna()
        if missing_mask.any():
            excluded_pfas = df.loc[missing_mask, "PFAS"].tolist()
            print(
                f"Arm {arm_name}: excluding {missing_mask.sum()} row(s) with missing "
                f"LogKa (PFAS={excluded_pfas}) because LogKa is a candidate feature "
                f"in this arm."
            )
        df = df[~missing_mask].reset_index(drop=True)
    else:
        df = df.drop(columns=["LogKa"]).reset_index(drop=True)

    groups_all = sorted(df["PFAS"].unique())
    fold_records = []
    selection_counts = {}
    per_fold_rmse = []
    per_fold_mae = []

    for held_out in groups_all:
        outer_test = df[df["PFAS"] == held_out].reset_index(drop=True)
        outer_train = df[df["PFAS"] != held_out].reset_index(drop=True)

        fit_result, X_train, y_train, _, weights_train = (
            fit_preprocessing_and_select_features(outer_train, include_logka)
        )
        selected = fit_result["selected_features"]
        for feat in selected:
            selection_counts[feat] = selection_counts.get(feat, 0) + 1

        model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1)
        model.fit(X_train[selected], y_train, sample_weight=weights_train)

        X_test = transform_with_fit(outer_test, fit_result)[selected]
        y_test = outer_test["log_half_life"]
        outer_test_weights = outer_test["weight"]
        preds = model.predict(X_test)

        fold_rmse = float(
            np.sqrt(mean_squared_error(y_test, preds, sample_weight=outer_test_weights))
        )
        fold_mae = float(
            mean_absolute_error(y_test, preds, sample_weight=outer_test_weights)
        )
        per_fold_rmse.append(fold_rmse)
        per_fold_mae.append(fold_mae)
        print(
            f"[{arm_name}] held out {held_out:>15s} | features: {selected} | "
            f"CV-weighted fold RMSE: {fold_rmse:.4f} | CV-weighted fold MAE: {fold_mae:.4f}"
        )

        for true_val, pred_val, w_val in zip(
            y_test.tolist(), preds.tolist(), outer_test_weights.tolist()
        ):
            fold_records.append(
                {
                    "PFAS": held_out,
                    "true": true_val,
                    "predicted": pred_val,
                    "residual": true_val - pred_val,
                    "weight": w_val,
                }
            )

    results_df = pd.DataFrame(fold_records)

    # Macro metrics: unweighted mean of each held-out PFAS group's own
    # (internally CV-weighted) RMSE/MAE, so a compound with many rows (e.g.
    # PFOA=13, PFOS=10) counts exactly the same as a singleton compound. This
    # is the primary generalization estimate, since the outer loop's unit is
    # "one compound", not "one row". Only what happens inside each fold is
    # CV-weighted now; this aggregation step across folds is unchanged.
    macro_rmse = float(np.mean(per_fold_rmse))
    macro_mae = float(np.mean(per_fold_mae))

    # Macro R^2: collapse each held-out compound's rows to one weighted-avg
    # point (see _macro_r2) before scoring, so a compound with many rows
    # (e.g. PFHxS=6) doesn't dominate R^2 the way a row-pooled R^2 would -
    # same compound-equal-weighting principle as macro RMSE/MAE above.
    macro_r2, _n_compounds_r2 = _macro_r2(
        results_df["true"],
        results_df["predicted"],
        results_df["PFAS"],
        results_df["weight"],
    )

    # Micro metrics: every row pooled together before computing one metric,
    # so compounds with more rows implicitly get more weight. RMSE/MAE are
    # additionally CV-weighted (weight = 1/CV**2) within that pool. Micro R^2
    # uses the classical (unweighted), row-pooled formula - kept alongside
    # macro R^2 above purely as a secondary/diagnostic comparison, not the
    # headline.
    micro_rmse = float(
        np.sqrt(
            mean_squared_error(
                results_df["true"],
                results_df["predicted"],
                sample_weight=results_df["weight"],
            )
        )
    )
    micro_r2 = r2_score(results_df["true"], results_df["predicted"])
    micro_mae = float(
        mean_absolute_error(
            results_df["true"],
            results_df["predicted"],
            sample_weight=results_df["weight"],
        )
    )

    print(
        f"\nArm {arm_name} nested-CV generalization estimate ({len(groups_all)} outer folds, "
        "CV-weighted fits/RMSE/MAE throughout; R^2 uses the classical unweighted formula):"
    )
    print(
        f"  Macro RMSE = {macro_rmse:.4f} (primary, compound-equal-weighted, CV-weighted per fold)"
    )
    print(
        f"  Macro MAE  = {macro_mae:.4f} (primary, compound-equal-weighted, CV-weighted per fold)"
    )
    print(
        f"  Macro R^2  = {macro_r2:.4f} (primary, compound-equal-weighted, collapses "
        "each held-out compound's rows to one weighted-avg point)"
    )
    print(f"  Micro RMSE = {micro_rmse:.4f} (secondary, CV-weighted, row-pooled)")
    print(
        f"  Micro R^2  = {micro_r2:.4f} (secondary, classical/unweighted, row-pooled)"
    )
    print(f"  Micro MAE  = {micro_mae:.4f} (secondary, CV-weighted, row-pooled)")

    selection_frequency = (
        pd.Series(selection_counts, name="times_selected")
        .rename_axis("feature")
        .reset_index()
        .sort_values("times_selected", ascending=False, ignore_index=True)
    )
    selection_frequency["out_of_folds"] = len(groups_all)

    # Final deployed feature set: refit preprocessing + forward selection on
    # ALL of this arm's data (no held-out compound). Kept separate from the
    # nested-CV estimate above, which is the number that reflects
    # generalization to an unseen compound.
    final_fit, X_full, y_full, groups_full, weights_full = (
        fit_preprocessing_and_select_features(df, include_logka)
    )
    final_features = final_fit["selected_features"]
    final_model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1)
    final_model.fit(X_full[final_features], y_full, sample_weight=weights_full)

    # 5-fold GroupKFold CV (grouped by PFAS) of the final, FIXED feature set,
    # with CV-weighted fits and RMSE/MAE scoring (R^2 uses the classical
    # unweighted formula). This is NOT the nested-CV estimate above (which
    # re-selects features per outer fold) - it scores the actual feature set
    # being deployed, so it's a more honest generalization estimate than
    # in-sample fit.
    final_cv_rmse, final_cv_r2, final_cv_mae = _grouped_cv_metrics(
        X_full[final_features], y_full, groups_full, weights_full
    )

    print(
        f"\nArm {arm_name} final deployed feature set (fit on all {len(df)} rows; "
        "NOT the nested-CV RMSE above):"
    )
    print(f"  {final_features}")
    print(
        f"Arm {arm_name} final model 5-fold GroupKFold CV (grouped by PFAS, "
        "CV-weighted, fixed feature set):"
    )
    print(f"  CV-weighted RMSE = {final_cv_rmse:.4f}")
    print(f"  Macro R^2 (compound-collapsed) = {final_cv_r2:.4f}")
    print(f"  CV-weighted MAE  = {final_cv_mae:.4f}")

    return {
        "arm_name": arm_name,
        "n_outer_folds": len(groups_all),
        "results_df": results_df,
        "macro_rmse": macro_rmse,
        "macro_mae": macro_mae,
        "macro_r2": macro_r2,
        "micro_rmse": micro_rmse,
        "micro_r2": micro_r2,
        "micro_mae": micro_mae,
        "selection_frequency": selection_frequency,
        "final_features": final_features,
        "final_model": final_model,
        "final_cv_rmse": final_cv_rmse,
        "final_cv_r2": final_cv_r2,
        "final_cv_mae": final_cv_mae,
    }


if __name__ == "__main__":
    arm_a = run_nested_cv(dataset, include_logka=False, arm_name="A (no LogKa)")
    arm_b = run_nested_cv(dataset, include_logka=True, arm_name="B (with LogKa)")

    print(f"\n{'=' * 25} Summary: Arm A vs Arm B {'=' * 25}")
    summary_df = pd.DataFrame(
        {
            "Arm A (no LogKa)": {
                "outer_folds": arm_a["n_outer_folds"],
                "Nested-CV Macro RMSE (CV-weighted per fold)": arm_a["macro_rmse"],
                "Nested-CV Macro MAE (CV-weighted per fold)": arm_a["macro_mae"],
                "Nested-CV Macro R2 (compound-collapsed)": arm_a["macro_r2"],
                "Nested-CV Micro RMSE (CV-weighted, row-pooled)": arm_a["micro_rmse"],
                "Nested-CV Micro R2 (classical/unweighted, row-pooled)": arm_a[
                    "micro_r2"
                ],
                "Nested-CV Micro MAE (CV-weighted, row-pooled)": arm_a["micro_mae"],
                "Final model 5-fold CV RMSE (CV-weighted)": arm_a["final_cv_rmse"],
                "Final model 5-fold CV R2 (compound-collapsed)": arm_a["final_cv_r2"],
                "Final model 5-fold CV MAE (CV-weighted)": arm_a["final_cv_mae"],
            },
            "Arm B (with LogKa)": {
                "outer_folds": arm_b["n_outer_folds"],
                "Nested-CV Macro RMSE (CV-weighted per fold)": arm_b["macro_rmse"],
                "Nested-CV Macro MAE (CV-weighted per fold)": arm_b["macro_mae"],
                "Nested-CV Macro R2 (compound-collapsed)": arm_b["macro_r2"],
                "Nested-CV Micro RMSE (CV-weighted, row-pooled)": arm_b["micro_rmse"],
                "Nested-CV Micro R2 (classical/unweighted, row-pooled)": arm_b[
                    "micro_r2"
                ],
                "Nested-CV Micro MAE (CV-weighted, row-pooled)": arm_b["micro_mae"],
                "Final model 5-fold CV RMSE (CV-weighted)": arm_b["final_cv_rmse"],
                "Final model 5-fold CV R2 (compound-collapsed)": arm_b["final_cv_r2"],
                "Final model 5-fold CV MAE (CV-weighted)": arm_b["final_cv_mae"],
            },
        }
    )
    print(summary_df)

    print("\nFinal deployed feature sets (fit on all data for that arm):")
    print(f"  Arm A: {arm_a['final_features']}")
    print(f"  Arm B: {arm_b['final_features']}")

    print(
        "\nArm B forces LogKa as its first selected feature in every fold "
        f"(see forward_select_features); it is always among Arm B's "
        f"{N_SELECT_STEPS} final features by construction."
    )

    print("\nPer-compound predictions (Arm A):")
    print(arm_a["results_df"].to_string(index=False))
    print(
        f"\nFeature-selection frequency (Arm A, out of {arm_a['n_outer_folds']} outer folds):"
    )
    print(arm_a["selection_frequency"].to_string(index=False))

    print("\nPer-compound predictions (Arm B):")
    print(arm_b["results_df"].to_string(index=False))
    print(
        f"\nFeature-selection frequency (Arm B, out of {arm_b['n_outer_folds']} outer folds):"
    )
    print(arm_b["selection_frequency"].to_string(index=False))
