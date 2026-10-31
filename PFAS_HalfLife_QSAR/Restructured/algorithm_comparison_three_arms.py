"""Four-algorithm x three-arm comparison, on the macro-scored permutation
selection pipeline.

Extends Restructured/old/permutation_selection_three_arms_macro.py along one
new dimension - the estimator - and changes nothing else. The preprocessing,
the repeated grouped K-fold permutation-importance ranking with its
compound-averaged (macro) RMSE score, the three arms and their forced
features, the feature budget, the correlation threshold, R/K/P and the nested
leave-one-PFAS-out outer loop with its macro/micro metrics are all as in that
script. Permutation importance is model-agnostic, so the selection machinery
needed no change beyond which estimator gets fitted: every pipeline function
below takes a `make_estimator` callable.

Fully self-contained: nothing is imported from any other project script, and
none of them are modified.

The four estimators, all with hyperparameters fixed a priori and never tuned
inside any loop (see the `make_*` factories): Ridge, random forest, the
constrained XGBoost configuration inherited from the macro script, and
Gaussian process regression.

Arms (half_life_type is forced in all three):
  A  LogKa excluded entirely - the column is dropped, so it is neither a
     candidate nor forced.
  B  LogKa is an ordinary candidate - it passes through the variance and
     correlation filters and the permutation ranking like any other feature.
  C  LogKa is forced - exempt from both filters and always selected.
Arms B and C drop rows with missing LogKa; Arm A drops the LogKa column.

The grid is 4 algorithms x 3 arms x len(SEEDS) seeds, i.e. twelve full sweeps
rather than the macro script's three, and both GPR and the random forest are
far slower per fit than a depth-2 tree - so this runs considerably longer than
anything else here. A failure in one cell of the grid is caught, reported and
skipped rather than aborting the run.

Reading the spread: the random forest (bootstrapping, per-split feature
subsampling) and GPR (optimizer restarts) are stochastic, while Ridge and this
XGBoost configuration are deterministic given their inputs. For the latter two
the only thing the swept seed changes is the GroupKFold partitioning inside the
selector, so their seed-to-seed standard deviation is not measuring the same
quantity as the former two's.

Output: the candidate counts after filtering (algorithm-independent, so
reported once), then per algorithm the three-arm summary - mean +/- std per
metric across seeds, head-to-head win counts for B vs A, C vs A and C vs B,
and ranked-feature Jaccard stability per arm - then one cross-algorithm table
of macro RMSE and macro R^2 for all three arms side by side, then the final
models at seed 42 with each algorithm's selected feature set per arm. No
in-sample metrics, no seed-42 LOPO pass, no SHAP.
"""

import warnings

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42  # rebound per seed in the sweep, restored to 42 for the final models
N_SELECT_STEPS = 7  # total features per arm, forced included
N_INNER_SPLITS = 5  # K in the permutation selector's grouped K-fold
PERM_REPEATS = 10  # R repetitions of the grouped K-fold
N_PERMUTATIONS = 5  # P shuffles per candidate per held-out fold
CORR_THRESHOLD = 0.99
SEEDS = list(range(5))

# Precautionary only. GPR's optimizer warns when a kernel hyperparameter
# converges near one of its bounds, and because there are R*K = 50 fits per
# selection run and ~29 selection runs per arm-seed, even an occasional warning
# would repeat enough to bury the progress lines and the summary. Measured on
# this data the bounds below are wide enough that no such warning fires, so
# these filters currently suppress nothing; they are here so that a future
# change to the kernel, the bounds or the feature count cannot flood the
# output. Silencing a warning never changes a fit, and genuine failures still
# surface through the try/except around each cell of the grid.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore", message=".*close to the specified.*bound.*", category=UserWarning
)

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


def make_ridge():
    """Ridge, alpha=1.0, everything else at scikit-learn's defaults.

    The default solver is deterministic, so Ridge contributes no seed variance
    of its own: across the swept seeds the only thing that changes for Ridge is
    the GroupKFold partitioning inside the selector, never the fit itself.
    """
    return Ridge(alpha=1.0)


def make_random_forest():
    """Random forest.

    max_features is set explicitly because scikit-learn's default for
    regression is 1.0 - every feature considered at every split - which makes
    the ensemble bagged trees rather than a random forest. 0.33 restores the
    per-split feature subsampling that decorrelates the trees.
    """
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.33,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def make_xgboost():
    """The constrained XGBoost configuration from
    permutation_selection_three_arms_macro.py, unchanged."""
    return XGBRegressor(
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


def make_gpr():
    """Gaussian process regression with a constant * RBF + white-noise kernel.

    The RBF is isotropic - one scalar length scale - on purpose. Per-feature
    (ARD) length scales would mean roughly 91 hyperparameters fitted on about
    55 training rows inside the selector, far more than this data can identify.

    The length-scale bounds are deliberately generous (1e-2 to 1e3) because the
    same kernel has to work in two quite different regimes: the ~91-feature
    baseline fits inside the permutation selector, where pairwise distances in
    the MinMax-scaled space are comparatively large, and the 7-feature final
    model, where they are much smaller.

    A fresh kernel is built on every call, so the starting hyperparameters
    never drift between fits (a fitted GPR keeps its optimized copy in
    `kernel_` and leaves `kernel` alone, but this keeps it unambiguous).
    """
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e3)) + WhiteKernel(
        0.1, (1e-4, 1e1)
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=RANDOM_STATE,
    )


ALGORITHMS = [
    {"name": "Ridge", "factory": make_ridge, "stochastic": False},
    {"name": "RandomForest", "factory": make_random_forest, "stochastic": True},
    {"name": "XGBoost", "factory": make_xgboost, "stochastic": False},
    {"name": "GPR", "factory": make_gpr, "stochastic": True},
]


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
    back afterward, so nothing can drop them.
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


def _group_positions(groups_val):
    """Row positions within one validation fold, grouped by PFAS. Computed
    once per fold: the fold's rows and their labels never change, only the
    values inside a single permuted column do."""
    positions = {}
    for row, label in enumerate(groups_val):
        positions.setdefault(label, []).append(row)
    return [np.asarray(rows) for rows in positions.values()]


def _macro_rmse(y_true, y_pred, group_positions):
    """Compound-averaged (macro) RMSE: every PFAS contributes its own RMSE,
    computed from its own rows, and those per-compound values are averaged
    unweighted, so a compound with many rows counts no more than a singleton.

    Per-compound RMSE is well defined even when a compound contributes a
    single row - it is just that row's absolute error - so no fold needs
    special handling here, unlike R^2, which is undefined for a group whose
    true values have zero variance.
    """
    per_compound = [
        np.sqrt(np.mean((y_true[rows] - y_pred[rows]) ** 2)) for rows in group_positions
    ]
    return float(np.mean(per_compound))


def _fold_permutation_importances(
    model, X_val, y_val, groups_val, cand_idx, rng, baseline_rmse, chunk_rows=4000
):
    """Permutation importances for one held-out fold, as the increase in the
    compound-averaged (macro) RMSE.

    For every non-forced candidate the fold's values of that one column are
    shuffled N_PERMUTATIONS times and re-predicted with the already-trained
    `model` (no retraining); the importance recorded for each shuffle is
    `permuted_rmse - baseline_rmse`. Kvasnicka et al. write the difference the
    other way round (s - s_perm), which is negative for an informative
    feature; the increase in error is used here, so larger is better.

    Both scores are macro RMSE (see `_macro_rmse`): `baseline_rmse` is passed
    in already computed that way, so the importance stays the difference
    between two comparably computed numbers.

    Nothing here is specific to any estimator - `model` is only ever asked to
    predict - which is what makes the four-algorithm comparison possible
    without touching the selection machinery.

    The shuffled copies are stacked and predicted in batches of roughly
    `chunk_rows` rows purely for speed - it is arithmetically identical to
    predicting each shuffled copy on its own.
    """
    group_positions = _group_positions(groups_val)
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
                permuted_rmse = _macro_rmse(y_val, fold_preds, group_positions)
                col_importances.append(permuted_rmse - baseline_rmse)
            importances[col] = col_importances

    return importances


def permutation_select_features(
    X, y, groups, forced_features, make_estimator, n_total=N_SELECT_STEPS
):
    """Repeated grouped K-fold permutation-importance selection, following
    Kvasnicka et al. 2024 (SI, "Model Training Steps").

    R = PERM_REPEATS repetitions, each with its own seed derived from
    RANDOM_STATE, each splitting `X` with a 5-fold GroupKFold grouped by PFAS
    so every row of a compound stays in the same fold. Per fold: fit one
    baseline model from `make_estimator` on the other four folds using every
    remaining candidate plus the forced features, score its compound-averaged
    RMSE on the held-out fold, then permute each non-forced candidate
    P = N_PERMUTATIONS times and record the increase in that same score.

    `make_estimator` is a zero-argument factory returning an unfitted
    estimator; it is the only estimator-specific thing in this function, and
    permutation importance needs nothing from the model but `fit` and
    `predict`.

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
    groups_arr = np.asarray(groups)
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
            model = make_estimator()
            model.fit(X_arr[train_idx], y_arr[train_idx])

            X_val, y_val = X_arr[val_idx], y_arr[val_idx]
            groups_val = groups_arr[val_idx]
            # Baseline scored the same way as every permuted score below:
            # compound-averaged, not row-pooled.
            baseline_rmse = _macro_rmse(
                y_val, model.predict(X_val), _group_positions(groups_val)
            )

            rng = np.random.RandomState(repeat_seed * 100 + fold)
            fold_importances = _fold_permutation_importances(
                model, X_val, y_val, groups_val, cand_idx, rng, baseline_rmse
            )
            for col, values in fold_importances.items():
                pooled[col].extend(values)

    median_importance = {columns[col]: float(np.median(v)) for col, v in pooled.items()}
    ranked = sorted(candidates, key=lambda c: (-median_importance[c], c))[:n_ranked]
    return forced + ranked, median_importance


def _macro_micro_metrics(results_df, per_group_rmse=None, per_group_mae=None):
    """Macro (compound-equal-weighted) and micro (row-pooled) RMSE/MAE/R^2:
    macro RMSE/MAE are the unweighted mean of each compound's own RMSE/MAE,
    macro R^2 collapses each congener to its mean true/mean predicted value
    before one classical R^2, and the micro metrics pool every row first."""
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


def run_outer_lopo(df, cfg, make_estimator):
    """Outer leave-one-PFAS-out loop for one arm and one estimator:
    preprocessing and permutation selection are redone inside every outer
    training set, then a model is fit on the selected features and the
    held-out compound is predicted. Returns the six metrics plus, for the
    summary, the per-fold candidate counts and the per-fold ranked
    (non-forced) feature sets."""
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
            X_train, y_train, groups_train, forced, make_estimator
        )
        ranked_sets.append(frozenset(f for f in selected if f not in forced))

        model = make_estimator()
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


def fit_final_model(df, cfg, make_estimator):
    """Preprocessing plus permutation selection run once on all of this arm's
    rows, and one model fit on the selected features. Deployed model: nothing
    is held out, so it gets no metrics here."""
    fit_result, X_full, y_full, groups_full = fit_preprocessing(df, cfg)
    selected, _ = permutation_select_features(
        X_full, y_full, groups_full, forced_features_for(cfg), make_estimator
    )

    model = make_estimator()
    model.fit(X_full[selected], y_full)
    return {
        "cfg": cfg,
        "model": model,
        "selected": selected,
        "n_candidates_after_filtering": fit_result["n_candidates_after_filtering"],
    }


def _mean_std(values):
    """Mean and sample std of the successful seeds. Std needs at least two
    values, so a single surviving seed reports NaN rather than a spurious 0."""
    if not values:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return mean, std


if __name__ == "__main__":
    arm_dfs = {cfg["name"]: arm_dataframe(dataset, cfg) for cfg in ARM_CONFIGS}

    # sweep[algorithm][arm]["metrics"][seed] -> the six metrics for that cell,
    # so a failed cell simply leaves its seed out instead of shifting others.
    sweep = {
        algo["name"]: {
            cfg["name"]: {"metrics": {}, "ranked_sets": []} for cfg in ARM_CONFIGS
        }
        for algo in ALGORITHMS
    }
    candidate_counts = {cfg["name"]: [] for cfg in ARM_CONFIGS}
    failures = []

    for algo in ALGORITHMS:
        for seed in SEEDS:
            RANDOM_STATE = seed
            for cfg in ARM_CONFIGS:
                print(
                    f"Running algorithm={algo['name']}, seed={seed}, "
                    f"arm={cfg['label']}...",
                    flush=True,
                )
                try:
                    result = run_outer_lopo(
                        arm_dfs[cfg["name"]], cfg, algo["factory"]
                    )
                except Exception as exc:  # noqa: BLE001 - keep the grid going
                    failures.append(
                        {
                            "algorithm": algo["name"],
                            "arm": cfg["name"],
                            "seed": seed,
                            "stage": "sweep",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(
                        f"  FAILED (skipping this cell): algorithm={algo['name']}, "
                        f"arm={cfg['name']}, seed={seed} -> "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue

                cell = sweep[algo["name"]][cfg["name"]]
                cell["metrics"][seed] = result["metrics"]
                cell["ranked_sets"].extend(result["ranked_sets"])
                candidate_counts[cfg["name"]].extend(result["candidate_counts"])

    print(f"\n{'=' * 30} Summary {'=' * 30}")
    print(
        f"Grid: {len(ALGORITHMS)} algorithms x {len(ARM_CONFIGS)} arms x "
        f"{len(SEEDS)} seeds {SEEDS}. Permutation selection: R={PERM_REPEATS} "
        f"repeats x K={N_INNER_SPLITS} grouped folds x P={N_PERMUTATIONS} "
        f"shuffles, median-pooled, scored by compound-averaged (macro) RMSE; "
        f"{N_SELECT_STEPS} features per arm (forced included); correlation "
        f"threshold {CORR_THRESHOLD}."
    )
    print(
        "Ridge and this XGBoost configuration are deterministic, so their "
        "seed-to-seed spread reflects only the selector's changing GroupKFold "
        "partitions; RandomForest and GPR are stochastic on top of that."
    )

    print(
        "\nCandidate features remaining after filtering (algorithm-independent, "
        "so reported once; all outer folds x seeds x algorithms):"
    )
    for cfg in ARM_CONFIGS:
        counts = np.array(candidate_counts[cfg["name"]])
        if counts.size:
            print(
                f"  Arm {cfg['label']}: min={counts.min()}, "
                f"mean={counts.mean():.1f}, max={counts.max()}"
            )
        else:
            print(f"  Arm {cfg['label']}: no successful folds")

    for algo in ALGORITHMS:
        tag = "stochastic" if algo["stochastic"] else "deterministic"
        print(f"\n{'=' * 25} {algo['name']} ({tag}) {'=' * 25}")

        seeds_done = {
            cfg["name"]: sorted(sweep[algo["name"]][cfg["name"]]["metrics"])
            for cfg in ARM_CONFIGS
        }
        if not any(seeds_done.values()):
            print("  every cell failed for this algorithm; nothing to report")
            continue

        print("Nested LOPO-CV, mean +/- std across seeds:")
        metric_table = pd.DataFrame(
            {
                f"Arm {cfg['name']}": {
                    metric: "%.4f +/- %.4f"
                    % _mean_std(
                        [
                            sweep[algo["name"]][cfg["name"]]["metrics"][s][metric]
                            for s in seeds_done[cfg["name"]]
                        ]
                    )
                    for metric in METRICS
                }
                for cfg in ARM_CONFIGS
            }
        )
        print(metric_table.to_string())
        for cfg in ARM_CONFIGS:
            if len(seeds_done[cfg["name"]]) != len(SEEDS):
                print(
                    f"  note: Arm {cfg['name']} averaged over "
                    f"{len(seeds_done[cfg['name']])}/{len(SEEDS)} seeds "
                    f"{seeds_done[cfg['name']]} (the rest failed)"
                )

        print("\nHead-to-head wins (seeds where both arms succeeded):")
        win_columns = {}
        for first, second in [("B", "A"), ("C", "A"), ("C", "B")]:
            shared = sorted(set(seeds_done[first]) & set(seeds_done[second]))
            column = {}
            for metric in METRICS:
                first_vals = np.array(
                    [sweep[algo["name"]][first]["metrics"][s][metric] for s in shared]
                )
                second_vals = np.array(
                    [sweep[algo["name"]][second]["metrics"][s][metric] for s in shared]
                )
                if metric in LOWER_IS_BETTER:
                    column[metric] = int(np.sum(first_vals < second_vals))
                else:
                    column[metric] = int(np.sum(first_vals > second_vals))
            win_columns[f"{first} vs {second} (n={len(shared)})"] = column
        print(pd.DataFrame(win_columns).to_string())

        print(
            "\nRanked-feature stability (mean pairwise Jaccard across all outer "
            "folds x seeds, forced features excluded):"
        )
        for cfg in ARM_CONFIGS:
            sets = sweep[algo["name"]][cfg["name"]]["ranked_sets"]
            print(
                f"  Arm {cfg['label']}: {mean_pairwise_jaccard(sets):.4f} "
                f"({len(sets)} selected sets)"
            )

    print(f"\n{'=' * 22} Cross-algorithm comparison {'=' * 22}")
    print("Mean across seeds of the nested LOPO-CV macro metrics:")
    cross_rows = {}
    for algo in ALGORITHMS:
        row = {}
        for cfg in ARM_CONFIGS:
            done = sorted(sweep[algo["name"]][cfg["name"]]["metrics"])
            for metric, label in [("macro_rmse", "macro RMSE"), ("macro_r2", "macro R2")]:
                mean, _ = _mean_std(
                    [
                        sweep[algo["name"]][cfg["name"]]["metrics"][s][metric]
                        for s in done
                    ]
                )
                row[f"Arm {cfg['name']} {label}"] = mean
        cross_rows[algo["name"]] = row
    print(pd.DataFrame(cross_rows).T.to_string(float_format=lambda v: f"{v:.4f}"))

    # Final models, back on the script's own seed. Selection is run once on all
    # of each arm's rows; nothing is held out, so no metrics are reported.
    RANDOM_STATE = 42
    print(
        f"\n{'=' * 20} Final models (RANDOM_STATE={RANDOM_STATE}, selection run "
        f"once on all rows) {'=' * 20}"
    )
    for algo in ALGORITHMS:
        print(f"\n{algo['name']}:")
        for cfg in ARM_CONFIGS:
            try:
                final = fit_final_model(arm_dfs[cfg["name"]], cfg, algo["factory"])
            except Exception as exc:  # noqa: BLE001 - keep the grid going
                failures.append(
                    {
                        "algorithm": algo["name"],
                        "arm": cfg["name"],
                        "seed": RANDOM_STATE,
                        "stage": "final model",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"  Arm {cfg['label']}: FAILED -> "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            print(f"  Arm {cfg['label']}: {final['selected']}")

    if failures:
        print(f"\n{'=' * 25} Failures ({len(failures)}) {'=' * 25}")
        print(pd.DataFrame(failures).to_string(index=False))
    else:
        print("\nNo failures: every cell of the grid completed.")
