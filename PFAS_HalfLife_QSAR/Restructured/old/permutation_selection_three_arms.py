"""Three-arm LogKa comparison using permutation-importance feature selection
(Kvasnicka et al. 2024, SI "Model Training Steps") under nested
leave-one-PFAS-out CV, swept across 10 seeds.

Fully self-contained: nothing is imported from any other project script. The
loading/encoding/scaling/filtering code below is copied from
Restructured/old/nested_CV_forced_halflife_type.py (which is left untouched),
with two deliberate changes:

  * the correlation threshold is 0.95 instead of 0.99, for both the numerical
    and the binary block;
  * greedy forward selection is replaced everywhere by the repeated grouped
    K-fold permutation-importance ranking in `permutation_select_features`.

Arms (half_life_type is forced in all three):
  A  LogKa excluded entirely - the column is dropped, so it is neither a
     candidate nor forced.
  B  LogKa is an ordinary candidate - it passes through the variance and
     correlation filters and the permutation ranking like any other feature.
  C  LogKa is forced - exempt from both filters and always selected, exactly
     the way LogKa is handled in nested_CV_forced_halflife_type.py.
Arms B and C drop rows with missing LogKa; Arm A drops the LogKa column.

Budget is N_SELECT_STEPS=6 features per arm: A and B use 1 forced + 5 ranked,
C uses 2 forced + 4 ranked.

Output is summary only (no per-seed table): per-arm candidate counts after
filtering, nested-CV mean +/- std per metric across the 10 seeds, pairwise
arm win counts, ranked-feature Jaccard stability, then the three final
models' feature sets with in-sample (NOT generalization) metrics, and a
1 x 3 SHAP beeswarm figure.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost
from rdkit.Chem import Descriptors
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42  # rebound per seed in the sweep, restored to 42 for the final models
N_SELECT_STEPS = 4  # total features per arm, forced included
N_INNER_SPLITS = 5  # K in the permutation selector's grouped K-fold
PERM_REPEATS = 10  # R repetitions of the grouped K-fold
N_PERMUTATIONS = 5  # P shuffles per candidate per held-out fold
# Lowered from the 0.99 used in nested_CV_forced_halflife_type.py: with both
# RDKit descriptors and ECFP bits kept as candidates, 0.95 prunes more of the
# near-duplicate columns before the ranking sees them.
CORR_THRESHOLD = 0.99
# n_jobs for every XGBRegressor here. The selector fits R*K=50 models per
# training set on the full (wide) candidate matrix, so this is the main
# wall-clock knob in this script.
N_JOBS = 1
SEEDS = list(range(5))

# Forced in every arm: apparent vs intrinsic half-life are fundamentally
# different estimations of the outcome, so the model should always account
# for them. How LogKa is treated is what varies per arm (see ARM_CONFIGS).
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

ARM_CONFIGS = [
    {
        "name": "A",
        "label": "A (LogKa excluded)",
        "include_logka": False,
        "drop_logka_column": True,
        "forced_numerical_features": [],
    },
    {
        "name": "B",
        "label": "B (LogKa ordinary candidate)",
        "include_logka": True,
        "drop_logka_column": False,
        "forced_numerical_features": [],
    },
    {
        "name": "C",
        "label": "C (LogKa forced)",
        "include_logka": True,
        "drop_logka_column": False,
        "forced_numerical_features": ["LogKa"],
    },
]
METRICS = ["macro_rmse", "macro_mae", "macro_r2", "micro_rmse", "micro_mae", "micro_r2"]
LOWER_IS_BETTER = {"macro_rmse", "macro_mae", "micro_rmse", "micro_mae"}


def arm_dataframe(df, cfg):
    """This arm's rows/columns: Arm A drops the LogKa column outright, Arms B
    and C drop rows with missing LogKa (it is a real candidate for them)."""
    df = df.copy()
    if cfg["drop_logka_column"]:
        return df.drop(columns=["LogKa"]).reset_index(drop=True)
    return df[~df["LogKa"].isna()].reset_index(drop=True)


def forced_features_for(cfg):
    """Features always included, in order: forced numerical (LogKa in Arm C)
    first, then the forced binary features."""
    return list(cfg["forced_numerical_features"]) + list(FORCED_BINARY_FEATURES)


def correlation_filter(X, threshold=CORR_THRESHOLD):
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


def fit_preprocessing(train_df, cfg):
    """Fit MinMax scaling plus the variance and correlation filters on
    `train_df` only. Both RDKit descriptors and ECFP bits stay as candidates.

    Forced features are exempt from both filters: they are pulled out of
    their block before the variance/correlation filters run and concatenated
    back afterward, so nothing can drop them. That is the same mechanism
    nested_CV_forced_halflife_type.py uses for LogKa and half_life_type, just
    parameterized per arm (Arm A/B force no numerical feature, Arm C forces
    LogKa).
    """
    include_logka = cfg["include_logka"]
    forced_numerical = cfg["forced_numerical_features"]

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    numerical_cols, binary_cols = _numerical_and_binary_cols(
        feature_cols, include_logka
    )

    X_num = train_df[numerical_cols].copy()
    scaler = MinMaxScaler().fit(X_num)
    X_num[:] = scaler.transform(X_num)

    forced_num_cols = [c for c in forced_numerical if c in X_num.columns]
    forced_num = X_num[forced_num_cols]
    X_num = X_num.drop(columns=forced_num_cols)

    num_var_filter = VarianceThreshold().fit(X_num)
    X_num = X_num.loc[:, num_var_filter.get_support()]
    X_num = correlation_filter(X_num, threshold=CORR_THRESHOLD)

    X_num = pd.concat([forced_num, X_num], axis=1)

    X_bin = train_df[binary_cols].copy()

    forced_bin_cols = [c for c in FORCED_BINARY_FEATURES if c in X_bin.columns]
    forced_bin = X_bin[forced_bin_cols]
    X_bin = X_bin.drop(columns=forced_bin_cols)

    bin_var_filter = VarianceThreshold(threshold=0).fit(X_bin)
    X_bin = X_bin.loc[:, bin_var_filter.get_support()]
    X_bin = correlation_filter(X_bin, threshold=CORR_THRESHOLD)

    X_bin = pd.concat([forced_bin, X_bin], axis=1)

    X_train = pd.concat([X_num, X_bin], axis=1)
    y_train = train_df["log_half_life"]
    groups_train = train_df["PFAS"]

    fit_result = {
        "scaler": scaler,
        "numerical_cols_all": numerical_cols,
        "numerical_cols_kept": X_num.columns.tolist(),
        "binary_cols_kept": X_bin.columns.tolist(),
        "n_candidates_after_filtering": X_train.shape[1],
    }
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


def _fold_permutation_importances(
    model, X_val, y_val, cand_idx, rng, baseline_rmse, chunk_rows=4000
):
    """Permutation importances for one held-out fold, as the increase in RMSE.

    For every non-forced candidate the fold's values of that one column are
    shuffled N_PERMUTATIONS times and re-predicted with the already-trained
    `model` (no retraining); the importance recorded for each shuffle is
    `permuted_rmse - baseline_rmse`. Kvasnicka et al. write the difference the
    other way round (s - s_perm), which is negative for an informative
    feature; the increase in error is used here, so larger is better.

    The shuffled copies are stacked and predicted in batches of roughly
    `chunk_rows` rows purely for speed - it is arithmetically identical to
    predicting each shuffled copy on its own.
    """
    n_val = X_val.shape[0]
    rows_per_candidate = n_val * N_PERMUTATIONS
    candidates_per_chunk = max(1, chunk_rows // rows_per_candidate)
    importances = {}

    for start in range(0, len(cand_idx), candidates_per_chunk):
        chunk = cand_idx[start : start + candidates_per_chunk]
        blocks = []
        for col in chunk:
            tiled = np.tile(X_val, (N_PERMUTATIONS, 1))
            for p in range(N_PERMUTATIONS):
                shuffled = X_val[rng.permutation(n_val), col]
                tiled[p * n_val : (p + 1) * n_val, col] = shuffled
            blocks.append(tiled)
        preds = model.predict(np.vstack(blocks))

        for block_i, col in enumerate(chunk):
            offset = block_i * rows_per_candidate
            col_importances = []
            for p in range(N_PERMUTATIONS):
                start_row = offset + p * n_val
                fold_preds = preds[start_row : start_row + n_val]
                permuted_rmse = float(np.sqrt(mean_squared_error(y_val, fold_preds)))
                col_importances.append(permuted_rmse - baseline_rmse)
            importances[col] = col_importances

    return importances


def permutation_select_features(X, y, groups, forced_features, n_total=N_SELECT_STEPS):
    """Repeated grouped K-fold permutation-importance selection, following
    Kvasnicka et al. 2024 (SI, "Model Training Steps"). Replaces the greedy
    forward selection used in nested_CV_forced_halflife_type.py.

    R = PERM_REPEATS repetitions, each with its own seed derived from
    RANDOM_STATE, each splitting `X` with a 5-fold GroupKFold grouped by PFAS
    so every row of a compound stays in the same fold. Per fold: fit one
    baseline model on the other four folds using every remaining candidate
    plus the forced features, score its RMSE on the held-out fold, then
    permute each non-forced candidate P = N_PERMUTATIONS times and record the
    RMSE increase.

    Each candidate's R * K * P importances are pooled and reduced to their
    median; the selected set is the forced features plus the highest-median
    candidates, up to `n_total` features in total.
    """
    forced = [f for f in forced_features if f in X.columns]
    candidates = [c for c in X.columns if c not in forced]
    n_ranked = max(0, min(n_total - len(forced), len(candidates)))

    columns = X.columns.tolist()
    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    cand_idx = [columns.index(c) for c in candidates]
    pooled = {col: [] for col in cand_idx}

    for repeat in range(PERM_REPEATS):
        # Distinct per (RANDOM_STATE, repeat) and derived from RANDOM_STATE,
        # so the whole selector moves with the seed being swept.
        repeat_seed = int(RANDOM_STATE) * 100 + repeat
        gkf = GroupKFold(
            n_splits=N_INNER_SPLITS, shuffle=True, random_state=repeat_seed
        )
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_arr, y_arr, groups)):
            model = XGBRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=2,
                min_child_weight=3,
                reg_lambda=5.0,
                subsample=1.0,
                colsample_bytree=1.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            model.fit(X_arr[train_idx], y_arr[train_idx])

            X_val, y_val = X_arr[val_idx], y_arr[val_idx]
            baseline_rmse = float(
                np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
            )

            rng = np.random.RandomState(repeat_seed * 100 + fold)
            fold_importances = _fold_permutation_importances(
                model, X_val, y_val, cand_idx, rng, baseline_rmse
            )
            for col, values in fold_importances.items():
                pooled[col].extend(values)

    median_importance = {columns[col]: float(np.median(v)) for col, v in pooled.items()}
    ranked = sorted(candidates, key=lambda c: (-median_importance[c], c))[:n_ranked]
    return forced + ranked, median_importance


def _macro_micro_metrics(results_df, per_group_rmse=None, per_group_mae=None):
    """Macro (compound-equal-weighted) and micro (row-pooled) RMSE/MAE/R^2,
    computed exactly as in nested_CV_forced_halflife_type.py: macro RMSE/MAE
    are the unweighted mean of each compound's own RMSE/MAE, macro R^2
    collapses each congener to its mean true/mean predicted value before one
    classical R^2, and the micro metrics pool every row first."""
    if per_group_rmse is None or per_group_mae is None:
        per_group_rmse, per_group_mae = [], []
        for _, group in results_df.groupby("PFAS"):
            per_group_rmse.append(
                np.sqrt(mean_squared_error(group["true"], group["predicted"]))
            )
            per_group_mae.append(mean_absolute_error(group["true"], group["predicted"]))

    collapsed = results_df.groupby("PFAS")[["true", "predicted"]].mean()
    return {
        "macro_rmse": float(np.mean(per_group_rmse)),
        "macro_mae": float(np.mean(per_group_mae)),
        "macro_r2": float(r2_score(collapsed["true"], collapsed["predicted"])),
        "micro_rmse": float(
            np.sqrt(mean_squared_error(results_df["true"], results_df["predicted"]))
        ),
        "micro_mae": float(
            mean_absolute_error(results_df["true"], results_df["predicted"])
        ),
        "micro_r2": float(r2_score(results_df["true"], results_df["predicted"])),
    }


def run_outer_lopo(df, cfg):
    """Outer leave-one-PFAS-out loop for one arm: preprocessing and
    permutation selection are redone inside every outer training set, then a
    model is fit on the selected features and the held-out compound is
    predicted. Returns the six metrics plus, for the summary, the per-fold
    candidate counts and the per-fold ranked (non-forced) feature sets."""
    forced = forced_features_for(cfg)
    groups_all = sorted(df["PFAS"].unique())

    fold_records = []
    per_fold_rmse, per_fold_mae = [], []
    candidate_counts, ranked_sets = [], []

    for held_out in groups_all:
        outer_test = df[df["PFAS"] == held_out].reset_index(drop=True)
        outer_train = df[df["PFAS"] != held_out].reset_index(drop=True)

        fit_result, X_train, y_train, groups_train = fit_preprocessing(outer_train, cfg)
        candidate_counts.append(fit_result["n_candidates_after_filtering"])

        selected, _ = permutation_select_features(
            X_train, y_train, groups_train, forced
        )
        ranked_sets.append(frozenset(f for f in selected if f not in forced))

        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=3,
            reg_lambda=5.0,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
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
    metrics = _macro_micro_metrics(results_df, per_fold_rmse, per_fold_mae)
    return {
        "metrics": metrics,
        "candidate_counts": candidate_counts,
        "ranked_sets": ranked_sets,
    }


def mean_pairwise_jaccard(sets):
    """Mean Jaccard overlap over every pair of selected ranked-feature sets.
    Forced features are excluded by the caller, so this measures stability
    only among the features the ranking actually chose."""
    overlaps = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            overlaps.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return float(np.mean(overlaps)) if overlaps else float("nan")


def fit_final_model(df, cfg):
    """Preprocessing plus permutation selection run once on all of this arm's
    rows, and one model fit on the selected features. Deployed model: nothing
    is held out, so it gets no generalization estimate."""
    fit_result, X_full, y_full, groups_full = fit_preprocessing(df, cfg)
    selected, _ = permutation_select_features(
        X_full, y_full, groups_full, forced_features_for(cfg)
    )

    X_selected = X_full[selected]
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=3,
        reg_lambda=5.0,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X_selected, y_full)

    in_sample = pd.DataFrame(
        {
            "PFAS": groups_full.tolist(),
            "true": y_full.tolist(),
            "predicted": model.predict(X_selected).tolist(),
        }
    )
    return {
        "cfg": cfg,
        "model": model,
        "selected": selected,
        "X_selected": X_selected,
        "n_candidates_after_filtering": fit_result["n_candidates_after_filtering"],
        "in_sample_metrics": _macro_micro_metrics(in_sample),
    }


if __name__ == "__main__":
    arm_dfs = {cfg["name"]: arm_dataframe(dataset, cfg) for cfg in ARM_CONFIGS}
    sweep = {
        cfg["name"]: {"metrics": [], "candidate_counts": [], "ranked_sets": []}
        for cfg in ARM_CONFIGS
    }

    for seed in SEEDS:
        RANDOM_STATE = seed
        for cfg in ARM_CONFIGS:
            print(f"Running seed={seed}, arm={cfg['label']}...", flush=True)
            result = run_outer_lopo(arm_dfs[cfg["name"]], cfg)
            sweep[cfg["name"]]["metrics"].append(result["metrics"])
            sweep[cfg["name"]]["candidate_counts"].extend(result["candidate_counts"])
            sweep[cfg["name"]]["ranked_sets"].extend(result["ranked_sets"])

    print(f"\n{'=' * 30} Summary {'=' * 30}")
    print(
        f"Permutation selection: R={PERM_REPEATS} repeats x K={N_INNER_SPLITS} "
        f"grouped folds x P={N_PERMUTATIONS} shuffles = "
        f"{PERM_REPEATS * N_INNER_SPLITS * N_PERMUTATIONS} importances per "
        f"candidate, median-pooled; {N_SELECT_STEPS} features per arm "
        f"(forced included); correlation threshold {CORR_THRESHOLD}."
    )

    print("\nCandidate features remaining after filtering (all outer folds x seeds):")
    for cfg in ARM_CONFIGS:
        counts = np.array(sweep[cfg["name"]]["candidate_counts"])
        print(
            f"  Arm {cfg['label']}: min={counts.min()}, "
            f"mean={counts.mean():.1f}, max={counts.max()}"
        )

    print(
        f"\nNested LOPO-CV across {len(SEEDS)} seeds {SEEDS} "
        "(mean +/- std of each metric):"
    )
    metric_table = pd.DataFrame(
        {
            f"Arm {cfg['name']}": {
                metric: (
                    f"{np.mean([m[metric] for m in sweep[cfg['name']]['metrics']]):.4f}"
                    " +/- "
                    f"{np.std([m[metric] for m in sweep[cfg['name']]['metrics']], ddof=1):.4f}"
                )
                for metric in METRICS
            }
            for cfg in ARM_CONFIGS
        }
    )
    print(metric_table.to_string())

    print(f"\nHead-to-head wins (out of {len(SEEDS)} seeds, same seed compared):")
    win_table = pd.DataFrame(
        {
            f"{first} vs {second}": {
                metric: int(
                    np.sum(
                        np.array([m[metric] for m in sweep[first]["metrics"]])
                        < np.array([m[metric] for m in sweep[second]["metrics"]])
                    )
                    if metric in LOWER_IS_BETTER
                    else np.sum(
                        np.array([m[metric] for m in sweep[first]["metrics"]])
                        > np.array([m[metric] for m in sweep[second]["metrics"]])
                    )
                )
                for metric in METRICS
            }
            for first, second in [("B", "A"), ("C", "A"), ("C", "B")]
        }
    )
    print(win_table.to_string())

    print(
        "\nRanked-feature stability (mean pairwise Jaccard across all outer "
        "folds x seeds, forced features excluded):"
    )
    for cfg in ARM_CONFIGS:
        sets = sweep[cfg["name"]]["ranked_sets"]
        print(
            f"  Arm {cfg['label']}: {mean_pairwise_jaccard(sets):.4f} "
            f"({len(sets)} selected sets)"
        )

    # Final deployed models, back on the script's own seed.
    RANDOM_STATE = 42
    finals = []
    for cfg in ARM_CONFIGS:
        print(f"\nFitting final model, arm={cfg['label']}...", flush=True)
        finals.append(fit_final_model(arm_dfs[cfg["name"]], cfg))

    print(
        f"\n{'=' * 25} Final models (RANDOM_STATE={RANDOM_STATE}, "
        f"selection run once on all rows) {'=' * 25}"
    )
    for final in finals:
        print(
            f"\nArm {final['cfg']['label']} final feature set "
            f"({len(final['selected'])} features, from "
            f"{final['n_candidates_after_filtering']} candidates after "
            f"filtering):"
        )
        print(f"  {final['selected']}")

    print(
        "\nFinal models on their own training data - fitted (in-sample), NOT a "
        "generalization estimate. The nested-CV mean +/- std above is the "
        "performance estimate:"
    )
    in_sample_table = pd.DataFrame(
        {f"Arm {final['cfg']['name']}": final["in_sample_metrics"] for final in finals}
    )
    print(in_sample_table.to_string())

    # One more honest LOPO pass per arm, at the final models' own seed
    # (RANDOM_STATE is 42 here, outside the swept 0-9 range, so this number
    # does not appear in the sweep above). Preprocessing and permutation
    # selection are redone inside every outer fold exactly as in the sweep,
    # so this is a genuine generalization estimate - just from a single seed
    # rather than averaged over ten.
    final_lopo = {}
    for cfg in ARM_CONFIGS:
        print(f"\nRunning final LOPO pass, arm={cfg['label']}...", flush=True)
        final_lopo[cfg["name"]] = run_outer_lopo(arm_dfs[cfg["name"]], cfg)["metrics"]

    print(
        f"\nNested LOPO-CV at RANDOM_STATE={RANDOM_STATE}, the seed the final "
        "models above were built with (selection redone inside every outer "
        "fold; single-seed estimate, so it will not match the 10-seed mean "
        "exactly):"
    )
    final_lopo_table = pd.DataFrame(
        {f"Arm {cfg['name']}": final_lopo[cfg["name"]] for cfg in ARM_CONFIGS}
    )
    print(final_lopo_table.to_string())

    # SHAP beeswarms, one panel per arm. Plotted feature values are the
    # MinMax-scaled values the models were fit on, not raw descriptor units.
    #
    # The SHAP values are tree-path-dependent TreeSHAP taken from XGBoost's
    # own `pred_contribs=True` rather than from shap.TreeExplainer: shap
    # 0.49.1 (the latest release) cannot parse the `base_score` that xgboost
    # 3.2.0 writes into its model JSON, so constructing a TreeExplainer on
    # this booster raises ValueError. XGBoost computes the same TreeSHAP
    # values internally; the trailing column it appends is the bias/base
    # value, which is dropped here. shap is still used for the beeswarm.
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, final in zip(axes, finals):
        booster = final["model"].get_booster()
        contributions = booster.predict(
            xgboost.DMatrix(final["X_selected"]), pred_contribs=True
        )
        shap_values = contributions[:, :-1]
        plt.sca(ax)
        shap.summary_plot(
            shap_values,
            final["X_selected"],
            show=False,
            plot_size=None,
            color_bar=False,
        )
        ax.set_title(f"Arm {final['cfg']['name']}")
    fig.suptitle(
        "SHAP beeswarm per arm - point color is the MinMax-scaled feature "
        "value (low to high), not the raw descriptor value"
    )
    fig.tight_layout()
    fig.savefig(
        "Restructured/old/permutation_selection_three_arms_shap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
