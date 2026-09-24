"""Three-arm comparison of how LogKa is handled, using only a single 5-fold
GroupKFold structure - no nested CV, no leave-one-PFAS-out anywhere.

Fully self-contained: no import from any other script in this repo. The
preprocessing/forward-selection/reference-CV code below is copied in from
Restructured/old/nested_CV_forced_halflife_type.py (generalized slightly so
"which features are forced" can vary by arm), so the recipe being tested is
the same one, just applied to three arms instead of two.

Arm A: LogKa excluded entirely - not a candidate, not forced, not present in
       the feature space at all. Physicochemical descriptors compete for the
       remaining slots. half_life_type is forced.
Arm B: LogKa included as an ordinary free candidate - not forced, so it goes
       through the same variance/correlation filtering as any other
       numerical feature and only ends up in the final set if forward
       selection picks it. half_life_type is forced.
Arm C: LogKa forced - exempted from the variance/correlation filters and
       guaranteed a slot, exactly like the existing forced-feature handling.
       half_life_type is forced too.

Total feature budget is N_SELECT_STEPS=6 per arm (matching elsewhere in this
repo): Arm A forces 1 (half_life_type) with 5 competing; Arm B forces 1
(half_life_type) with LogKa competing among the other 5; Arm C forces 2
(half_life_type + LogKa) with 4 competing.

For each arm: forward selection is scored by the existing 5-fold GroupKFold
CV mechanism (micro/row-pooled RMSE, grouped by PFAS, random_state=
RANDOM_STATE) and run exactly once on the entire dataset, giving that arm's
fixed final feature set - the same mechanism used for the final-model
derivation in nested_CV_forced_halflife_type.py, just with three arms and no
outer LOPO loop around it.

Then, for each arm's fixed final feature set, a fresh 5-fold GroupKFold CV
(macro + micro RMSE/MAE/R^2) is run and averaged over several seeds distinct
from RANDOM_STATE - the same approach as _reference_cv_single_seed/
reference_cv_fixed_features in nested_CV_forced_halflife_type.py, copied in
here rather than imported. This is REFERENCE ONLY and optimistically biased:
each arm's feature set was chosen using all of the same data being scored.

Prints, per arm: the final feature set, and the six averaged reference
metrics with the bias caveat. No per-seed breakdown, no plots, no nested-CV
output.
"""

import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
N_SELECT_STEPS = 6
N_INNER_SPLITS = 5
# The reference-only 5-fold check of each arm's fixed final feature set is
# repeated across this many distinct seeds and averaged, rather than run
# once, so it isn't sensitive to the luck of a single GroupKFold split. All
# of them are deliberately different from RANDOM_STATE, which the
# forward-selection scoring uses internally.
REFERENCE_CV_N_REPEATS = 10
REFERENCE_CV_SEED = 20260922
_ref_seed_rng = np.random.RandomState(REFERENCE_CV_SEED)
REFERENCE_CV_SEEDS = []
while len(REFERENCE_CV_SEEDS) < REFERENCE_CV_N_REPEATS:
    candidate = int(_ref_seed_rng.randint(0, 2**31 - 1))
    if candidate != RANDOM_STATE and candidate not in REFERENCE_CV_SEEDS:
        REFERENCE_CV_SEEDS.append(candidate)

# Forced in every arm: apparent vs intrinsic half-life are fundamentally
# different estimations of the outcome, so the model should always account
# for them. Whether/how LogKa is forced is what varies per arm below.
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


def fit_preprocessing(
    train_df, include_logka, forced_numerical_features=None, forced_binary_features=None
):
    """Fit scaling/variance/correlation filters on `train_df` only. Returns
    the fitted transforms plus the processed training feature matrix/target/
    groups (no feature selection).

    Generalized from nested_CV_forced_halflife_type.py's fit_preprocessing:
    there, LogKa was always the one forced numerical feature (Arm B only).
    Here, `forced_numerical_features` is passed in per arm, so LogKa can be
    absent (Arm A), a free/filterable candidate (Arm B, empty list here), or
    forced (Arm C, ["LogKa"]) - the exemption mechanism itself (pull out
    before the variance/correlation filters, add back afterward) is
    unchanged, it just now applies to whichever features are named.
    """
    forced_numerical_features = forced_numerical_features or []
    forced_binary_features = forced_binary_features or []

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    numerical_cols, binary_cols = _numerical_and_binary_cols(
        feature_cols, include_logka
    )

    X_num = train_df[numerical_cols].copy()
    scaler = MinMaxScaler().fit(X_num)
    X_num[:] = scaler.transform(X_num)

    forced_num_cols = [c for c in forced_numerical_features if c in X_num.columns]
    forced_num = X_num[forced_num_cols]
    X_num = X_num.drop(columns=forced_num_cols)

    num_var_filter = VarianceThreshold().fit(X_num)
    X_num = X_num.loc[:, num_var_filter.get_support()]
    X_num = correlation_filter(X_num)

    X_num = pd.concat([forced_num, X_num], axis=1)

    X_bin = train_df[binary_cols].copy()

    forced_bin_cols = [c for c in forced_binary_features if c in X_bin.columns]
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


def fit_preprocessing_and_select_features(
    train_df,
    include_logka,
    forced_numerical_features=None,
    forced_binary_features=None,
    n_steps=N_SELECT_STEPS,
):
    """`fit_preprocessing`, then forward feature selection scored by the
    5-fold GroupKFold RMSE (grouped by PFAS, `_grouped_cv_rmse`). Adds
    `selected_features` and `selection_history` to the returned fit_result.
    """
    forced_numerical_features = forced_numerical_features or []
    forced_binary_features = forced_binary_features or []

    fit_result, X_train, y_train, groups_train = fit_preprocessing(
        train_df, include_logka, forced_numerical_features, forced_binary_features
    )
    forced_features = forced_numerical_features + forced_binary_features
    selected_features, selection_history = forward_select_features(
        X_train, y_train, groups_train, n_steps=n_steps, forced_features=forced_features
    )
    fit_result["selected_features"] = selected_features
    fit_result["selection_history"] = selection_history
    return fit_result, X_train, y_train, groups_train


def _reference_cv_single_seed(df, features, include_logka, seed):
    """One run of the reference-only 5-fold GroupKFold (grouped by PFAS)
    check of a fixed final feature set, using the given `seed` for the
    split. See reference_cv_fixed_features for why this is not an
    independent generalization estimate. Copied from
    nested_CV_forced_halflife_type.py's _reference_cv_single_seed.
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

    micro_rmse = float(np.sqrt(mean_squared_error(results["true"], results["predicted"])))
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
    chosen using all of this same data (the single all-data forward-
    selection run for this arm), so scoring it again on this same data is
    optimistically biased. Every seed used here is distinct from
    RANDOM_STATE, the one the forward-selection scoring uses internally.
    Copied from nested_CV_forced_halflife_type.py's reference_cv_fixed_
    features.
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


# Arm A: LogKa excluded entirely (not a candidate). Arm B: LogKa a free,
# competable candidate. Arm C: LogKa forced. half_life_type is forced in all
# three, per FORCED_BINARY_FEATURES.
ARM_CONFIGS = [
    {
        "name": "A (LogKa excluded)",
        "include_logka": False,
        "drop_logka_column": True,
        "forced_numerical_features": [],
    },
    {
        "name": "B (LogKa free candidate)",
        "include_logka": True,
        "drop_logka_column": False,
        "forced_numerical_features": [],
    },
    {
        "name": "C (LogKa forced)",
        "include_logka": True,
        "drop_logka_column": False,
        "forced_numerical_features": ["LogKa"],
    },
]


if __name__ == "__main__":
    for cfg in ARM_CONFIGS:
        arm_df = dataset.copy()
        if cfg["drop_logka_column"]:
            arm_df = arm_df.drop(columns=["LogKa"]).reset_index(drop=True)
        else:
            arm_df = arm_df[~arm_df["LogKa"].isna()].reset_index(drop=True)

        fit_result, _, _, _ = fit_preprocessing_and_select_features(
            arm_df,
            cfg["include_logka"],
            forced_numerical_features=cfg["forced_numerical_features"],
            forced_binary_features=FORCED_BINARY_FEATURES,
        )
        final_features = fit_result["selected_features"]

        print(f"\n{'=' * 20} Arm {cfg['name']} {'=' * 20}")
        print(
            f"Final feature set ({len(final_features)} features, single "
            f"5-fold-GroupKFold-scored forward selection on all "
            f"{len(arm_df)} rows): {final_features}"
        )

        reference_cv_fixed_features(
            arm_df, final_features, cfg["include_logka"], cfg["name"]
        )
