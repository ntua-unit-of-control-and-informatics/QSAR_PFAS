import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
N_SELECT_STEPS = 6
N_INNER_SPLITS = 5
# The reference-only 5-fold check of the fixed final feature set (see
# reference_cv_fixed_features) is repeated across this many distinct seeds
# and averaged, rather than run once, so it isn't sensitive to the luck of a
# single GroupKFold split. All of them are deliberately different from
# RANDOM_STATE, which the forward-selection scoring uses internally, so this
# check isn't just re-reporting the exact fold split the search already
# optimized against.
REFERENCE_CV_N_REPEATS = 10
REFERENCE_CV_SEED = 20260922
_ref_seed_rng = np.random.RandomState(REFERENCE_CV_SEED)
REFERENCE_CV_SEEDS = []
while len(REFERENCE_CV_SEEDS) < REFERENCE_CV_N_REPEATS:
    candidate = int(_ref_seed_rng.randint(0, 2**31 - 1))
    if candidate != RANDOM_STATE and candidate not in REFERENCE_CV_SEEDS:
        REFERENCE_CV_SEEDS.append(candidate)
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
}  # {"PFAS", "log_half_life"}

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


def _reference_cv_single_seed(df, features, include_logka, seed):
    """One run of the reference-only 5-fold GroupKFold (grouped by PFAS)
    check of the fixed final feature set on the entire dataset, using the
    given `seed` for the split - fit on 4/5 of the compounds, predict the
    held-out 1/5, pool all 5 folds' out-of-fold predictions, then compute
    macro (compound-equal-weighted) and micro (row-pooled) RMSE/MAE/R^2,
    mirroring how the nested-CV metrics above are computed.

    No variance/correlation filtering is (re-)applied here: `features` is
    already fixed, so folds only need scaling for the numerical members of
    that set - there is nothing left to select or to filter out.

    Called once per seed in REFERENCE_CV_SEEDS by reference_cv_fixed_features,
    which averages the results across seeds; see that function for why this
    is not an independent generalization estimate.
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    numerical_cols, _ = _numerical_and_binary_cols(feature_cols, include_logka)
    numerical_features = [c for c in features if c in numerical_cols]

    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    groups = df["PFAS"]
    records = []
    for train_idx, val_idx in gkf.split(df, df["log_half_life"], groups):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        scaler = MinMaxScaler().fit(train_df[numerical_cols])

        def _prep(frame):
            X = frame[features].copy()
            if numerical_features:
                scaled_all = pd.DataFrame(
                    scaler.transform(frame[numerical_cols]),
                    columns=numerical_cols,
                    index=frame.index,
                )
                X[numerical_features] = scaled_all[numerical_features]
            return X

        X_train, y_train = _prep(train_df), train_df["log_half_life"]
        X_val, y_val = _prep(val_df), val_df["log_half_life"]

        model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        for pfas, true_val, pred_val in zip(
            val_df["PFAS"].tolist(), y_val.tolist(), preds.tolist()
        ):
            records.append({"PFAS": pfas, "true": true_val, "predicted": pred_val})

    results = pd.DataFrame(records)

    per_compound_rmse, per_compound_mae = [], []
    for _, group in results.groupby("PFAS"):
        per_compound_rmse.append(
            np.sqrt(mean_squared_error(group["true"], group["predicted"]))
        )
        per_compound_mae.append(mean_absolute_error(group["true"], group["predicted"]))
    macro_rmse = float(np.mean(per_compound_rmse))
    macro_mae = float(np.mean(per_compound_mae))

    collapsed = results.groupby("PFAS")[["true", "predicted"]].mean()
    macro_r2 = float(r2_score(collapsed["true"], collapsed["predicted"]))

    micro_rmse = float(
        np.sqrt(mean_squared_error(results["true"], results["predicted"]))
    )
    micro_mae = float(mean_absolute_error(results["true"], results["predicted"]))
    micro_r2 = float(r2_score(results["true"], results["predicted"]))

    return {
        "macro_rmse": macro_rmse,
        "macro_mae": macro_mae,
        "macro_r2": macro_r2,
        "micro_rmse": micro_rmse,
        "micro_mae": micro_mae,
        "micro_r2": micro_r2,
    }


def reference_cv_fixed_features(df, features, include_logka, arm_name):
    """REFERENCE ONLY: runs _reference_cv_single_seed REFERENCE_CV_N_REPEATS
    times, once per seed in REFERENCE_CV_SEEDS, and reports the mean of each
    metric across those runs - so the reported number isn't sensitive to the
    luck of any single 5-fold GroupKFold split.

    This is NOT an independent generalization estimate: `features` was
    chosen using all of this same data (the single all-data forward-selection
    run that built the final model), so scoring it again on this same data is
    optimistically biased. Every seed used here is distinct from
    RANDOM_STATE, the one the forward-selection scoring uses internally, so
    this is at least not simply replaying the exact split that search already
    optimized against - but it is still not a substitute for the honest
    nested-CV numbers.
    """
    per_run = [
        _reference_cv_single_seed(df, features, include_logka, seed)
        for seed in REFERENCE_CV_SEEDS
    ]
    metrics = {
        key: float(np.mean([run[key] for run in per_run]))
        for key in (
            "macro_rmse",
            "macro_mae",
            "macro_r2",
            "micro_rmse",
            "micro_mae",
            "micro_r2",
        )
    }

    print(
        f"\nArm {arm_name} REFERENCE ONLY 5-fold GroupKFold check of the fixed "
        f"final feature set, averaged over {REFERENCE_CV_N_REPEATS} seeds "
        f"({REFERENCE_CV_SEEDS}, each distinct from the forward-selection "
        "scoring seed) - optimistically biased: this feature set was chosen "
        "using all of the same data being scored here, so this is NOT an "
        "independent generalization estimate."
    )
    print(f"  Macro RMSE = {metrics['macro_rmse']:.4f}")
    print(f"  Macro MAE  = {metrics['macro_mae']:.4f}")
    print(f"  Macro R^2  = {metrics['macro_r2']:.4f}")
    print(f"  Micro RMSE = {metrics['micro_rmse']:.4f}")
    print(f"  Micro MAE  = {metrics['micro_mae']:.4f}")
    print(f"  Micro R^2  = {metrics['micro_r2']:.4f}")

    return metrics


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

        fit_result, X_train, y_train, _ = fit_preprocessing_and_select_features(
            outer_train, include_logka
        )
        selected = fit_result["selected_features"]
        for feat in selected:
            selection_counts[feat] = selection_counts.get(feat, 0) + 1

        model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1)
        model.fit(X_train[selected], y_train)

        X_test = transform_with_fit(outer_test, fit_result)[selected]
        y_test = outer_test["log_half_life"]
        preds = model.predict(X_test)

        fold_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        fold_mae = float(mean_absolute_error(y_test, preds))
        per_fold_rmse.append(fold_rmse)
        per_fold_mae.append(fold_mae)
        print(
            f"[{arm_name}] held out {held_out:>15s} | features: {selected} | "
            f"fold RMSE: {fold_rmse:.4f} | fold MAE: {fold_mae:.4f}"
        )

        for true_val, pred_val in zip(y_test.tolist(), preds.tolist()):
            fold_records.append(
                {
                    "PFAS": held_out,
                    "true": true_val,
                    "predicted": pred_val,
                    "residual": true_val - pred_val,
                }
            )

    results_df = pd.DataFrame(fold_records)

    # Macro metrics: unweighted mean of each held-out PFAS group's own
    # RMSE/MAE, so a compound with many rows (e.g. PFOA=13, PFOS=10) counts
    # exactly the same as a singleton compound. This is the primary
    # generalization estimate, since the outer loop's unit is "one compound",
    # not "one row".
    macro_rmse = float(np.mean(per_fold_rmse))
    macro_mae = float(np.mean(per_fold_mae))

    # Micro metrics: every row pooled together before computing one metric,
    # so compounds with more rows implicitly get more weight. Kept as a
    # secondary number, not the headline. R^2 has no clean macro equivalent:
    # many held-out groups have exactly one row, and R^2 is undefined when
    # the true values in a fold have zero variance - so R^2 is reported only
    # in this pooled/micro form.
    micro_rmse = float(
        np.sqrt(mean_squared_error(results_df["true"], results_df["predicted"]))
    )
    micro_r2 = r2_score(results_df["true"], results_df["predicted"])

    # Macro R^2: collapse each congener to its mean true / mean predicted
    # value (unweighted mean over its rows), then compute one classical R^2
    # across congeners, so every compound contributes equally.
    collapsed = results_df.groupby("PFAS")[["true", "predicted"]].mean()
    macro_r2 = float(r2_score(collapsed["true"], collapsed["predicted"]))
    micro_mae = float(mean_absolute_error(results_df["true"], results_df["predicted"]))

    print(
        f"\nArm {arm_name} nested-CV generalization estimate ({len(groups_all)} outer folds):"
    )
    print(f"  Macro RMSE = {macro_rmse:.4f} (primary, compound-equal-weighted)")
    print(f"  Macro MAE  = {macro_mae:.4f} (primary, compound-equal-weighted)")
    print(
        f"  Macro R^2  = {macro_r2:.4f} (primary, congener-averaged, "
        f"{len(collapsed)} congeners)"
    )
    print(f"  Micro RMSE = {micro_rmse:.4f} (secondary, row-pooled)")
    print(f"  Micro R^2  = {micro_r2:.4f} (secondary, row-pooled)")
    print(f"  Micro MAE  = {micro_mae:.4f} (secondary, row-pooled)")

    # DIAGNOSTIC ONLY: how often each feature was picked across the outer
    # folds (a view of selection stability). It is NOT used to build the
    # final feature set.
    selection_frequency = (
        pd.Series(selection_counts, name="times_selected")
        .rename_axis("feature")
        .reset_index()
        .sort_values("times_selected", ascending=False, ignore_index=True)
    )
    selection_frequency["out_of_folds"] = len(groups_all)

    # Final deployed model: run the same preprocessing + forward selection
    # (same forced features: half_life_type, plus LogKa in Arm B) exactly once
    # on ALL of this arm's rows, and fit one model on the result. Nothing is
    # evaluated on it; the nested-CV metrics above are the generalization
    # estimate for this whole recipe.
    final_fit, X_full, y_full, _ = fit_preprocessing_and_select_features(
        df, include_logka
    )
    final_features = final_fit["selected_features"]
    final_model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1)
    final_model.fit(X_full[final_features], y_full)

    print(
        f"\nArm {arm_name} final feature set (single forward-selection run on "
        f"all {len(df)} rows; deployed model fit on the same rows):"
    )
    print(f"  {final_features}")

    # Secondary, REFERENCE-ONLY check of that fixed final feature set (see
    # reference_cv_fixed_features's docstring for why it is not an
    # independent generalization estimate). The nested-CV metrics above
    # remain the headline numbers.
    reference_cv = reference_cv_fixed_features(
        df, final_features, include_logka, arm_name
    )

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
        "final_fit": final_fit,
        "reference_cv": reference_cv,
    }


if __name__ == "__main__":
    arm_a = run_nested_cv(dataset, include_logka=False, arm_name="A (no LogKa)")
    arm_b = run_nested_cv(dataset, include_logka=True, arm_name="B (with LogKa)")

    print(f"\n{'=' * 25} Summary: Arm A vs Arm B {'=' * 25}")
    summary_df = pd.DataFrame(
        {
            "Arm A (no LogKa)": {
                "outer_folds": arm_a["n_outer_folds"],
                "Nested-CV Macro RMSE": arm_a["macro_rmse"],
                "Nested-CV Macro MAE": arm_a["macro_mae"],
                "Nested-CV Macro R2 (congener-averaged)": arm_a["macro_r2"],
                "Nested-CV Micro RMSE": arm_a["micro_rmse"],
                "Nested-CV Micro R2": arm_a["micro_r2"],
                "Nested-CV Micro MAE": arm_a["micro_mae"],
            },
            "Arm B (with LogKa)": {
                "outer_folds": arm_b["n_outer_folds"],
                "Nested-CV Macro RMSE": arm_b["macro_rmse"],
                "Nested-CV Macro MAE": arm_b["macro_mae"],
                "Nested-CV Macro R2 (congener-averaged)": arm_b["macro_r2"],
                "Nested-CV Micro RMSE": arm_b["micro_rmse"],
                "Nested-CV Micro R2": arm_b["micro_r2"],
                "Nested-CV Micro MAE": arm_b["micro_mae"],
            },
        }
    )
    print(summary_df)

    print("\nFinal feature sets (single forward-selection run on all data per arm):")
    print(f"  Arm A: {arm_a['final_features']}")
    print(f"  Arm B: {arm_b['final_features']}")

    print(
        "\nForced in every fold and in the final run (see forced_features_for): "
        "half_life_type in both arms, plus LogKa in Arm B. Each final set has "
        f"{N_SELECT_STEPS} features."
    )

    print(
        f"\n{'=' * 15} REFERENCE ONLY - optimistically biased, NOT an "
        f"independent generalization estimate {'=' * 15}"
    )
    print(
        "Each arm's final feature set was chosen using all of the same data "
        "being scored below (plain 5-fold GroupKFold, averaged over "
        f"{REFERENCE_CV_N_REPEATS} seeds - {REFERENCE_CV_SEEDS} - each "
        "distinct from the forward-selection scoring seed). Treat these "
        "numbers only as a sanity check on the final feature set, never in "
        "place of the honest nested-CV numbers above."
    )
    reference_df = pd.DataFrame(
        {
            "Arm A (no LogKa)": {
                "Reference-only CV Macro RMSE": arm_a["reference_cv"]["macro_rmse"],
                "Reference-only CV Macro MAE": arm_a["reference_cv"]["macro_mae"],
                "Reference-only CV Macro R2": arm_a["reference_cv"]["macro_r2"],
                "Reference-only CV Micro RMSE": arm_a["reference_cv"]["micro_rmse"],
                "Reference-only CV Micro MAE": arm_a["reference_cv"]["micro_mae"],
                "Reference-only CV Micro R2": arm_a["reference_cv"]["micro_r2"],
            },
            "Arm B (with LogKa)": {
                "Reference-only CV Macro RMSE": arm_b["reference_cv"]["macro_rmse"],
                "Reference-only CV Macro MAE": arm_b["reference_cv"]["macro_mae"],
                "Reference-only CV Macro R2": arm_b["reference_cv"]["macro_r2"],
                "Reference-only CV Micro RMSE": arm_b["reference_cv"]["micro_rmse"],
                "Reference-only CV Micro MAE": arm_b["reference_cv"]["micro_mae"],
                "Reference-only CV Micro R2": arm_b["reference_cv"]["micro_r2"],
            },
        }
    )
    print(reference_df)

    print("\nPer-compound predictions (Arm A):")
    print(arm_a["results_df"].to_string(index=False))
    print(
        f"\nFeature-selection frequency (Arm A, out of {arm_a['n_outer_folds']} outer folds) "
        "- DIAGNOSTIC ONLY, not used to build the final feature set:"
    )
    print(arm_a["selection_frequency"].to_string(index=False))

    print("\nPer-compound predictions (Arm B):")
    print(arm_b["results_df"].to_string(index=False))
    print(
        f"\nFeature-selection frequency (Arm B, out of {arm_b['n_outer_folds']} outer folds) "
        "- DIAGNOSTIC ONLY, not used to build the final feature set:"
    )
    print(arm_b["selection_frequency"].to_string(index=False))

    # Predicted vs true log10(half-life): one point per PFAS (mean over its
    # rows). Same color per PFAS group across both panels.
    all_pfas = sorted(
        set(arm_a["results_df"]["PFAS"]) | set(arm_b["results_df"]["PFAS"])
    )
    # tab20/tab20b/tab20c are qualitative palettes (60 distinct colors total)
    # designed for categorical data, so nothing washes out near-white the way
    # a continuous colormap can when sampled at many points.
    tab_colors = (
        list(plt.get_cmap("tab20").colors)
        + list(plt.get_cmap("tab20b").colors)
        + list(plt.get_cmap("tab20c").colors)
    )
    color_map = {
        pfas: tab_colors[i % len(tab_colors)] for i, pfas in enumerate(all_pfas)
    }

    def plot_pred_vs_true(results_key, r2_key, title_suffix, save_path):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        for ax, arm_result, title in zip(
            axes, [arm_a, arm_b], ["Arm A (no LogKa)", "Arm B (with LogKa)"]
        ):
            rdf = (
                arm_result[results_key]
                .groupby("PFAS", as_index=False)[["true", "predicted"]]
                .mean()
            )
            for _, row in rdf.iterrows():
                ax.scatter(
                    row["true"],
                    row["predicted"],
                    color=color_map[row["PFAS"]],
                    label=row["PFAS"],
                    s=40,
                )
            lims = [
                min(rdf["true"].min(), rdf["predicted"].min()),
                max(rdf["true"].max(), rdf["predicted"].max()),
            ]
            ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
            ax.set_xlabel("True log10(half-life)")
            ax.set_ylabel("Predicted log10(half-life)")
            ax.set_title(
                f"{title}: {title_suffix}\nmacro R2 = {arm_result[r2_key]:.3f}"
            )

        combined_handles = {}
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                combined_handles[label] = handle
        fig.legend(
            combined_handles.values(),
            combined_handles.keys(),
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Nested-CV outer-loop predictions (features re-selected per fold) - the
    # honest generalization estimate.
    plot_pred_vs_true(
        "results_df",
        "macro_r2",
        "per-PFAS mean nested-CV predictions",
        "Restructured/old/nested_CV_forced_halflife_type_pred_vs_true.png",
    )
    plt.show()
