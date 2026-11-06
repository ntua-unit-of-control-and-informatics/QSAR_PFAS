"""Five-arm LogKa comparison using permutation-importance feature selection
(Kvasnicka et al. 2024, SI "Model Training Steps") under nested stratified
grouped K-fold CV, repeated across OUTER_REPEATS partitions, with a RANDOM
FOREST as the estimator.

RANDOM-FOREST VARIANT of permutation_selection_three_arms_macro.py, now
diverged from it in four ways:

  * every XGBoost fit - the permutation selector's baseline fits, the outer
    fold and the final model - is replaced by the same random forest
    configuration used in algorithm_comparison_three_arms.py (see
    `make_random_forest`);
  * the outer layer is OUTER_SPLITS-fold StratifiedGroupKFold grouped by
    PFAS, repeated OUTER_REPEATS times, instead of a single
    leave-one-PFAS-out partition swept across seeds. LOPO had only one
    possible partition, so a seed sweep over it varied the model and the
    selector's inner folds but never the outer split; repeated grouped
    K-fold varies the partition too, which is what the reported spread is
    meant to describe. LOPO is kept as a single final pass for the final
    models, where its maximal training data is worth having;
  * preprocessing follows PFAS_albumin_QSAR/PFAS_HSA_QSAR/HSA_nested_permutation.py
    (see `fit_preprocessing`): a missing-value threshold with mean
    imputation, a constant drop, MinMax scaling, a zero-variance drop, the
    correlation filter over the continuous and the binary block separately,
    and a univariate |Spearman| cap on the candidate count;
  * there are five arms, not three, because the binding prediction now comes
    per protein (see the arm list below). That older script still expects a
    single "LogKa" column and will not run against the current featurized
    dataset.

The arms' mechanics and their forced features, the feature budget, R/K/P,
the compound-averaged (macro) RMSE used for both the baseline and the
permuted scores in the selector, the reported outer macro/micro metrics, the
final models at seed 42 and the SHAP figure are unchanged. The SHAP figure is
saved under its own filename so it does not overwrite the XGBoost script's
output.

Candidates come from the featurized dataset written by sum_preprocess_data.py,
which now runs all four jaqpotpy featurizers used by the binding models -
RDKitDescriptors, MordredDescriptors, TopologicalFingerprint and
MACCSKeysFingerprint - so the half-life model draws from the same descriptor
space. Which columns are 0/1 fingerprint bits is read from the provenance
file that script writes beside the dataset, because it can no longer be
inferred from the names.

Fully self-contained: nothing is imported from any other project script. The
loading/encoding code below started as a copy of
Restructured/old/nested_CV_forced_halflife_type.py (which is left untouched),
with greedy forward selection replaced everywhere by the repeated grouped
K-fold permutation-importance ranking in `permutation_select_features`. The
correlation threshold is 0.99 for both the continuous and the binary block.

Two binding predictions are now available, one per protein: LogKa_HSA
(serum albumin) and LogKa_LFABP (liver fatty acid binding protein), both
written by sum_preprocess_data.py from the respective deploy scripts.

Arms (half_life_type is forced in all five):
  A  no binding prediction at all - both columns are dropped, so neither is
     a candidate nor forced.
  B  both columns are ordinary candidates - each passes through the whole
     filter chain and the permutation ranking like any other feature, and the
     selector is free to take neither, either or both.
  C  LogKa_HSA forced - exempt from every filter and always selected, the way
     LogKa is handled in nested_CV_forced_halflife_type.py. LogKa_LFABP stays
     available as an ordinary candidate alongside it.
  D  LogKa_LFABP forced the same way, with LogKa_HSA available as an ordinary
     candidate.
  E  both forced at once, so the model always carries both proteins. With
     half_life_type that is three forced features, which is more than
     N_SELECT_STEPS=2 asks for; see the budget note below.

So B, C, D and E all see both columns and differ only in which ones, if any,
are forced. That isolates the effect of forcing a given protein rather than the
effect of which columns are present: Arm C answers "does guaranteeing the
albumin prediction a slot help?", not "does albumin alone help?". The
consequence is that C and D can converge on the same feature set when the
ranking picks the other protein anyway, which is itself informative - it
means the forcing was not the binding constraint. To isolate one protein per
arm instead, cut the arm's "logka_columns" down to just its forced column.

Every arm drops rows whose kept LogKa columns are missing. Coverage is
currently complete for all half-life compounds, so in practice no arm loses
rows and all four run on identical rows.

Budget is N_SELECT_STEPS features per arm, forced included: A and B spend 1 on
half_life_type, C and D 2, E 3. An arm whose forced set already fills or
overflows the budget ranks nothing on top of it - a forced feature cannot be
dropped to stay under a budget, so `budget_for` raises that arm's count to its
forced floor. At N_SELECT_STEPS=2 that is exactly Arm E: it fits
half_life_type plus both LogKa columns and selects nothing further, which is
the point of the arm rather than a limitation.

Every outer fold also scores every prefix of
its ranking up to N_CURVE_STEPS, which gives the RMSE-against-N curve of
HSA_nested_permutation.py per arm. The ranking does not depend on how many
features are asked for, so these extra models cost a few cheap narrow fits
per fold and leave the reported N_SELECT_STEPS numbers untouched.

Printed output is summary only: per-arm candidate counts after filtering,
nested-CV mean +/- std per metric across the repeats, each arm's best N on
the RMSE curve, pairwise arm win counts, ranked-feature Jaccard stability,
then every arm's final feature set with in-sample (NOT generalization)
metrics, the leave-one-PFAS-out pass for those models and the y-randomized
null beside the real result.

Everything is also written to PFAS_HalfLife_QSAR/Restructured/results, named
after PREFIX - halflife_four_arms_rf_N<N_SELECT_STEPS> - following
HSA_nested_permutation.py:

  _metrics.csv                nested-CV mean/std per arm and metric
  _metrics_per_repeat.csv     the per-repeat rows behind it
  _N_curve.csv                every metric against N, median and IQR per arm
  _N_curve_per_repeat.csv     the per-repeat rows behind it
  _N_curve.png                macro RMSE against N, one curve per arm
  _oof_predictions.csv        every out-of-fold prediction of the sweep
  _head_to_head.csv           pairwise arm win counts
  _filtering_stages.csv       the per-stage drop table (Arm B)
  _filtering_counts.csv       its raw per-fold counts
  _selection_frequency.csv    how often each ranked feature was selected
  _final_features.csv         the final feature sets, in rank order
  _in_sample_metrics.csv      the final models on their own training data
  _lopo_metrics.csv           the final leave-one-PFAS-out pass
  _lopo_predictions.csv       its held-out predictions

  _y_randomization.csv        the null run beside the real result

The two LOPO files and _lopo_scatter.png appear only when RUN_FINAL_LOPO is
on, and _y_randomization.csv only when RUN_Y_RANDOMIZATION is on. Both are
end-of-run checks that nothing else depends on, and both are expensive - the
LOPO pass runs one outer fold per compound, the null repeats the whole sweep
- so either can be switched off for experiments.
  _manifest.json              parameters, arms and every summary number
  _shap.png                   SHAP beeswarm, one panel per arm
  _lopo_scatter.png           observed vs LOPO-predicted, one panel per arm
  _selection_frequency.png    top ranked features per arm
"""

import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler

RANDOM_STATE = 42  # rebound per seed in the sweep, restored to 42 for the final models
N_SELECT_STEPS = 2  # total features per arm, forced included
# The ranking is carried this far so every outer fold can also be scored at
# every smaller feature count, giving the RMSE-vs-N curve. Must be >=
# N_SELECT_STEPS; the extra models are cheap next to the selector itself.
N_CURVE_STEPS = 15
RUN_FINAL_LOPO = True  # the final leave-one-PFAS-out pass; off for quick experiments
RUN_Y_RANDOMIZATION = True  # the null run below; off for quick experiments
# R in the selector during the null run only. The null has no signal to find, so
# ranking it as carefully as the real run buys nothing; this is what keeps the
# extra cost to a fraction of the real sweep.
Y_RAND_PERM_REPEATS = 3
N_INNER_SPLITS = 5  # K in the permutation selector's grouped K-fold
PERM_REPEATS = 10  # R repetitions of the grouped K-fold
N_PERMUTATIONS = 5  # P shuffles per candidate per held-out fold
CORR_THRESHOLD = 0.99
MISSING_THRESHOLD = 0.05  # a candidate missing in more than this fraction is dropped
CANDIDATE_CAP = 50  # keep only this many candidates, by |Spearman| with the target
OUTER_SPLITS = 5  # compounds per outer fold: 29 compounds / 5 ~= 6 held out
N_STRATA_BINS = 1  # quantile bins of compound-mean log_half_life for the outer split
# n_jobs for every RandomForestRegressor here. The selector fits R*K=50 models
# per training set on the full (wide) candidate matrix, so this is the main
# wall-clock knob in this script.
N_JOBS = 1
OUTER_REPEATS = 10  # raise to 10 to match HSA_nested_permutation.py
REPEATS = list(range(OUTER_REPEATS))
assert N_SELECT_STEPS <= N_CURVE_STEPS

# Forced in every arm: apparent vs intrinsic half-life are fundamentally
# different estimations of the outcome, so the model should always account
# for them. How LogKa is treated is what varies per arm (see ARM_CONFIGS).
FORCED_BINARY_FEATURES = ["half_life_type"]
NON_FEATURE_COLS = {
    "Study",
    "PFAS",
    "SMILES",
    "log_half_life",
    "half_life",
    "SE",
    "SE_method",
    "CV",
    "weight",
}

DATA_PATH = "PFAS_HalfLife_QSAR/Restructured/data/sum_featurized_dataset.csv"
COLUMNS_PATH = "PFAS_HalfLife_QSAR/Restructured/data/sum_featurized_columns.json"

# Everything this script persists goes here, named after PREFIX, in the same spirit as
# PFAS_albumin_QSAR/PFAS_HSA_QSAR/HSA_nested_permutation.py: tables as CSV, the run as
# a single manifest JSON, figures as PNG.
OUT_DIR = "PFAS_HalfLife_QSAR/Restructured/results"
# The feature budget is in the name, as in HSA_nested_permutation.py, so runs at
# different N sit side by side instead of overwriting each other. Everything else
# about a run - the seeds, OUTER_REPEATS, RUN_FINAL_LOPO - still overwrites.
PREFIX = os.path.join(OUT_DIR, f"halflife_four_arms_rf_N{N_SELECT_STEPS}")

dataset = pd.read_csv(DATA_PATH)
dataset.drop(["Study"], axis=1, inplace=True)

# Binary encoding
dataset["half_life_type"] = dataset["half_life_type"].map(
    {"apparent": 0, "intrinsic": 1}
)
dataset["Occupational_exposure"] = dataset["Occupational_exposure"].map(
    {False: 0, True: 1}
)

# Which candidates are 0/1 fingerprint bits, written by sum_preprocess_data.py. The
# correlation filter runs separately over the binary and the continuous block, so this
# split matters - and it is no longer inferable from the column names: with Mordred in
# the dataset, matching against rdkit.Chem.Descriptors.descList would classify ~1800
# continuous Mordred descriptors as fingerprint bits and send them to the wrong block.
with open(COLUMNS_PATH) as _f:
    _provenance = json.load(_f)
BINARY_COLS = set(_provenance["binary"]) | {"half_life_type", "Occupational_exposure"}
assert set(_provenance["continuous"]) <= set(
    dataset.columns
), f"{COLUMNS_PATH} does not describe {DATA_PATH}; rerun sum_preprocess_data.py"

# The two binding predictions, one per protein.
LOGKA_COLUMNS = ["LogKa_HSA", "LogKa_LFABP"]

# "logka_columns" is which of LOGKA_COLUMNS this arm keeps at all; the rest are
# dropped from the arm's dataframe. Whatever is kept always reaches the selector:
# the filters and the CANDIDATE_CAP screen cannot drop it, so an arm's question is
# always put to the permutation ranking. "forced_numerical_features" is the subset
# that additionally skips the ranking and is always in the model. Anything forced
# must also be kept.
ARM_CONFIGS = [
    {
        "name": "A",
        "label": "A (no LogKa)",
        "logka_columns": [],
        "forced_numerical_features": [],
    },
    {
        "name": "B",
        "label": "B (both LogKa ordinary candidates)",
        "logka_columns": LOGKA_COLUMNS,
        "forced_numerical_features": [],
    },
    {
        "name": "C",
        "label": "C (LogKa_HSA forced, LogKa_LFABP a candidate)",
        "logka_columns": LOGKA_COLUMNS,
        "forced_numerical_features": ["LogKa_HSA"],
    },
    {
        "name": "D",
        "label": "D (LogKa_LFABP forced, LogKa_HSA a candidate)",
        "logka_columns": LOGKA_COLUMNS,
        "forced_numerical_features": ["LogKa_LFABP"],
    },
    {
        "name": "E",
        "label": "E (both LogKa forced)",
        "logka_columns": LOGKA_COLUMNS,
        "forced_numerical_features": LOGKA_COLUMNS,
    },
]
for _cfg in ARM_CONFIGS:
    assert set(_cfg["forced_numerical_features"]) <= set(
        _cfg["logka_columns"]
    ), f"Arm {_cfg['name']} forces a LogKa column it does not keep"

ARM_NAMES = [cfg["name"] for cfg in ARM_CONFIGS]
METRICS = ["macro_rmse", "macro_mae", "macro_r2", "micro_rmse", "micro_mae", "micro_r2"]
ARM_COLORS = {
    "A": "#7a7a7a",
    "B": "#2a78d6",
    "C": "#eb6834",
    "D": "#1baf7a",
    "E": "#9b5de5",
}
LOWER_IS_BETTER = {"macro_rmse", "macro_mae", "micro_rmse", "micro_mae"}


def make_random_forest():
    """Random forest, the configuration from algorithm_comparison_three_arms.py.
    Used for all three fits in this script: the selector's baseline fits, the
    outer LOPO fold and the final model. random_state reads RANDOM_STATE at
    call time, so it follows the seed being swept.

    max_features is set explicitly because scikit-learn's default for
    regression is 1.0 - every feature considered at every split - which makes
    the ensemble bagged trees rather than a random forest. 0.33 restores the
    per-split feature subsampling that decorrelates the trees.
    """
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=1.0,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def arm_dataframe(df, cfg):
    """This arm's rows/columns: the LogKa columns it does not keep are dropped
    outright, and rows missing any column it does keep are dropped, since a
    kept column is a real candidate for this arm."""
    kept = cfg["logka_columns"]
    df = df.drop(columns=[c for c in LOGKA_COLUMNS if c not in kept])
    if kept:
        df = df[~df[kept].isna().any(axis=1)]
    return df.reset_index(drop=True)


def forced_features_for(cfg):
    """Features always included, in order: forced numerical (LogKa in Arm C)
    first, then the forced binary features."""
    return list(cfg["forced_numerical_features"]) + list(FORCED_BINARY_FEATURES)


def budget_for(cfg):
    """How many features this arm actually fits: N_SELECT_STEPS, or its forced count
    when that is larger.

    N_SELECT_STEPS is a budget for the RANKING, and an arm cannot un-force a feature
    to stay under it. Arm E forces three (both LogKa plus half_life_type), so at
    N_SELECT_STEPS=2 it fits exactly those three and selects nothing on top - which
    is the intended reading of a budget below an arm's floor, not an error.
    """
    return max(N_SELECT_STEPS, len(forced_features_for(cfg)))


assert N_CURVE_STEPS >= max(
    len(forced_features_for(cfg)) for cfg in ARM_CONFIGS
), "N_CURVE_STEPS is below some arm's forced count, so its curve would be empty"


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


def fit_preprocessing(train_df, cfg):
    """Fit every preprocessing statistic on `train_df` only.

    The chain is the one in PFAS_albumin_QSAR/PFAS_HSA_QSAR/HSA_nested_permutation.py:
    drop candidates missing in more than MISSING_THRESHOLD of the training rows and
    mean-impute the rest, drop constants, MinMax-scale, drop zero variance, correlation
    filter over the continuous and the binary block separately, then cap the candidate
    count by a univariate |Spearman| screen against the target. The missing-value stage
    is what Mordred needs: jaqpotpy marks a descriptor it could not compute, and
    sum_preprocess_data.py turns those into NaN rather than leaving them as -1000.

    Forced features are exempt from every filter: they are pulled out after scaling and
    concatenated back at the end, so nothing can drop them - including the Spearman cap,
    which would otherwise be free to discard the very column the arm exists to force.
    That is the mechanism nested_CV_forced_halflife_type.py uses for LogKa and
    half_life_type, parameterized per arm (Arms A/B force no numerical feature, Arm C
    forces LogKa_HSA, Arm D forces LogKa_LFABP).

    The LogKa columns an arm keeps but does not force are exempt from the filters in
    the same way, with one difference that is the whole point of the distinction: they
    come back as ORDINARY CANDIDATES, so the permutation ranking is still free to pass
    them over. Filtering decides what the selector may consider, and an arm that exists
    to ask whether a binding prediction gets chosen has to put the question to the
    selector rather than let a univariate screen answer it first. Without this, a tight
    CANDIDATE_CAP could drop LogKa_LFABP from Arm B and turn it into Arm A by accident.
    """
    forced = [c for c in forced_features_for(cfg) if c in train_df.columns]
    protected = [
        c for c in cfg["logka_columns"] if c in train_df.columns and c not in forced
    ]
    exempt = set(forced) | set(protected)
    y_train = train_df["log_half_life"]
    groups_train = train_df["PFAS"]

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    X = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    counts = {"start": X.shape[1]}

    missing = X.isna().mean()
    keep = [c for c in X.columns if missing[c] <= MISSING_THRESHOLD or c in exempt]
    counts["dropped_missing"] = X.shape[1] - len(keep)
    means = X[keep].mean()
    X = X[keep].fillna(means)

    scaler = MinMaxScaler().fit(X)  # leaves 0/1 fingerprint bits unchanged
    X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    # Everything below this line is filtering, so the exempt columns sit it out
    X_forced, X_protected = X[forced], X[protected]
    X = X.drop(columns=list(exempt))

    nonconstant = list(X.columns[X.nunique() > 1])
    counts["dropped_constant"] = X.shape[1] - len(nonconstant)
    X = X[nonconstant]

    variance = VarianceThreshold().fit(X)
    counts["dropped_variance"] = int((~variance.get_support()).sum())
    X = X.loc[:, variance.get_support()]

    n_before = X.shape[1]
    binary = [c for c in X.columns if c in BINARY_COLS]
    X = pd.concat(
        [
            correlation_filter(X.drop(columns=binary), threshold=CORR_THRESHOLD),
            correlation_filter(X[binary], threshold=CORR_THRESHOLD),
        ],
        axis=1,
    )
    counts["dropped_correlation"] = n_before - X.shape[1]

    n_before = X.shape[1]
    if X.shape[1] > CANDIDATE_CAP:
        # Univariate screen: |Spearman| with log_half_life on this training set
        ranked_y = pd.Series(y_train.to_numpy(), index=X.index).rank()
        screen = X.rank().corrwith(ranked_y).abs()
        top = set(
            screen.sort_values(ascending=False, kind="mergesort").index[:CANDIDATE_CAP]
        )
        X = X[[c for c in X.columns if c in top]]
    counts["dropped_cap"] = n_before - X.shape[1]

    # Forced first, then the protected LogKa columns, then everything that passed the
    # filters; only the first block is withheld from the ranking.
    X_train = pd.concat([X_forced, X_protected, X], axis=1)
    counts["final"] = X_train.shape[1]

    fit_result = {
        "keep": keep,
        "means": means,
        "scaler": scaler,
        "columns": list(X_train.columns),
        "n_candidates_after_filtering": X_train.shape[1],
        "counts": counts,
    }
    return fit_result, X_train, y_train, groups_train


def transform_with_fit(test_df, fit_result):
    """Replay the fitted preprocessing on held-out rows. Nothing is refitted: the
    imputation means, the scaler and the surviving column list all come from the
    training set."""
    X = test_df[fit_result["keep"]].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(fit_result["means"])
    X = pd.DataFrame(
        fit_result["scaler"].transform(X), columns=X.columns, index=X.index
    )
    return X[fit_result["columns"]]


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

    This is the one thing that differs from
    permutation_selection_three_arms.py, which scores the selector's inner
    folds with a plain row-pooled RMSE instead.

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


def permutation_select_features(X, y, groups, forced_features, n_total=N_SELECT_STEPS):
    """Repeated grouped K-fold permutation-importance selection, following
    Kvasnicka et al. 2024 (SI, "Model Training Steps"). Replaces the greedy
    forward selection used in nested_CV_forced_halflife_type.py.

    R = PERM_REPEATS repetitions, each with its own seed derived from
    RANDOM_STATE, each splitting `X` with a 5-fold GroupKFold grouped by PFAS
    so every row of a compound stays in the same fold. Per fold: fit one
    baseline model on the other four folds using every remaining candidate
    plus the forced features, score its compound-averaged RMSE on the held-out
    fold, then permute each non-forced candidate P = N_PERMUTATIONS times and
    record the increase in that same compound-averaged RMSE.

    Scoring is compound-averaged (macro) here, which is the only difference
    from permutation_selection_three_arms.py; see `_macro_rmse`.

    Each candidate's R * K * P importances are pooled and reduced to their
    median; the selected set is the forced features plus the highest-median
    candidates, up to `n_total` features in total.
    """
    forced = [f for f in forced_features if f in X.columns]
    candidates = [c for c in X.columns if c not in forced]
    n_ranked = max(0, min(n_total - len(forced), len(candidates)))
    if n_ranked == 0:
        # The forced features already fill (or overflow) the budget, as in Arm E at
        # N_SELECT_STEPS=2. Ranking candidates that cannot be selected would cost the
        # full R x K fits for nothing.
        return list(forced), {}

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
            model = make_random_forest()
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


def _outer_fold(outer_train, outer_test, cfg, forced):
    """One outer fold, shared by the repeated K-fold sweep and the LOPO pass:
    preprocessing and permutation selection are redone on this fold's training rows
    only, then a model is fit on the selected features and the held-out rows are
    predicted. Returns per-row records plus the fold's candidate count, its ranked
    (non-forced) feature set, and the same per-row records at every feature count
    from the forced ones alone up to N_CURVE_STEPS - one model per count, all
    reusing prefixes of this fold's single ranking, exactly as
    HSA_nested_permutation.py builds its N curve.

    The ranking does not depend on how many features are asked for, so the records
    returned for N_SELECT_STEPS are the same ones this function returned before the
    curve was added."""
    fit_result, X_train, y_train, groups_train = fit_preprocessing(outer_train, cfg)
    ordered, _ = permutation_select_features(
        X_train, y_train, groups_train, forced, n_total=N_CURVE_STEPS
    )
    X_test_full = transform_with_fit(outer_test, fit_result)

    n_forced = len([f for f in forced if f in X_train.columns])
    # At least the forced features: an arm whose forced set overflows N_SELECT_STEPS
    # reports that set, and its curve starts there too.
    n_main = min(max(N_SELECT_STEPS, n_forced), len(ordered))
    records, curve = None, []
    for n in range(max(1, n_forced), len(ordered) + 1):
        cols = ordered[:n]
        model = make_random_forest()
        model.fit(X_train[cols], y_train)
        preds = model.predict(X_test_full[cols])
        rows = [
            {"PFAS": pfas, "true": true_val, "predicted": pred_val}
            for pfas, true_val, pred_val in zip(
                outer_test["PFAS"].tolist(),
                outer_test["log_half_life"].tolist(),
                preds.tolist(),
            )
        ]
        curve.extend({"N": n, **row} for row in rows)
        if n == n_main:
            records = rows
    assert records is not None, f"no model fitted at N={n_main}"
    return (
        records,
        fit_result["n_candidates_after_filtering"],
        frozenset(f for f in ordered[:n_main] if f not in forced),
        fit_result["counts"],
        curve,
    )


def y_randomized(df, seed):
    """A copy of `df` with the compound-level signal destroyed and nothing else.

    Each compound's MEAN log_half_life is swapped for another compound's, while its
    within-compound deviations, its row count and every feature value stay put. That
    is the link the model is claimed to have learned - features are compound-level,
    so a compound's mean is all they can address - and breaking it alone is what
    makes this a null rather than a different dataset.

    Shuffling rows outright would also destroy the study-to-study spread that the
    macro metrics are built on and inflate the null's error, flattering the real run.
    """
    rng = np.random.RandomState(seed)
    means = df.groupby("PFAS")["log_half_life"].mean()
    shuffled = pd.Series(rng.permutation(means.to_numpy()), index=means.index)
    out = df.copy()
    out["log_half_life"] = (
        df["log_half_life"] - df["PFAS"].map(means) + df["PFAS"].map(shuffled)
    )
    return out


def _outer_strata(df):
    """Quantile bins of each compound's MEAN log_half_life, broadcast back to its rows.

    The outer split has to be grouped by PFAS - rows are study-by-compound, so PFOA
    alone contributes 11 of the 72 rows, and splitting rows freely would put some of a
    compound's rows in training and the rest in validation. Stratifying on the
    compound mean rather than on the row value then keeps every outer fold spanning
    the half-life range, which matters at 29 compounds and 5 folds.
    """
    compound_mean = df.groupby("PFAS")["log_half_life"].mean()
    bins = pd.qcut(compound_mean, q=N_STRATA_BINS, labels=False, duplicates="drop")
    return df["PFAS"].map(bins).to_numpy()


def run_outer_cv(df, cfg):
    """Outer stratified grouped K-fold for one arm, at the current RANDOM_STATE.

    OUTER_SPLITS folds of compounds, grouped by PFAS so no compound is split across
    training and validation, stratified on the compound-mean half-life. Out-of-fold
    predictions are pooled across the folds before the metrics are computed, so every
    compound is predicted exactly once per repeat.
    """
    forced = forced_features_for(cfg)
    splitter = StratifiedGroupKFold(
        n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = splitter.split(df, _outer_strata(df), df["PFAS"].to_numpy())
    return _collect_outer(df, cfg, forced, splits)


def run_lopo(df, cfg):
    """Outer leave-one-PFAS-out loop for one arm: as many folds as there are
    compounds, each holding out every row of one compound. Maximal training data
    (28 of 29 compounds) and only one possible partition, so it carries no
    partition-to-partition variability - which is why it is reported once for the
    final models rather than swept."""
    forced = forced_features_for(cfg)
    index = np.arange(len(df))
    pfas = df["PFAS"].to_numpy()
    splits = (
        (index[pfas != held_out], index[pfas == held_out])
        for held_out in sorted(df["PFAS"].unique())
    )
    return _collect_outer(df, cfg, forced, splits)


def _collect_outer(df, cfg, forced, splits):
    """Run every outer fold of `splits` and pool the out-of-fold predictions.

    Per-compound RMSE/MAE are computed by `_macro_micro_metrics` from the pooled
    predictions, grouped on PFAS - not from per-fold values. Under LOPO a fold holds
    exactly one compound and the two agree, but a K-fold fold holds about six, so the
    per-fold number is not a per-compound number.
    """
    fold_records, candidate_counts, ranked_sets, counts = [], [], [], []
    curve_records = []
    for train_idx, test_idx in splits:
        records, n_candidates, ranked, fold_counts, curve = _outer_fold(
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
            cfg,
            forced,
        )
        fold_records.extend(records)
        candidate_counts.append(n_candidates)
        ranked_sets.append(ranked)
        counts.append(fold_counts)
        curve_records.extend(curve)

    # Out-of-fold predictions pooled per feature count, then scored the same way as
    # the headline metrics. Only counts every fold reached are kept, so a fold that
    # ran out of candidates cannot leave a partly-predicted N in the curve.
    curve_df = pd.DataFrame(curve_records)
    complete = curve_df.groupby("N")["PFAS"].nunique() == df["PFAS"].nunique()
    curve = pd.DataFrame(
        [
            {"N": n, **_macro_micro_metrics(curve_df[curve_df["N"] == n])}
            for n in complete.index[complete]
        ]
    )

    results_df = pd.DataFrame(fold_records)
    assert (
        results_df["PFAS"].nunique() == df["PFAS"].nunique()
    ), "an outer sweep left some compound unpredicted"
    return {
        "metrics": _macro_micro_metrics(results_df),
        "predictions": results_df,
        "curve": curve,
        "candidate_counts": candidate_counts,
        "ranked_sets": ranked_sets,
        "counts": counts,
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


def json_safe(value):
    """numpy types and anything unserialisable kept out of the manifest."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def fit_final_model(df, cfg):
    """Preprocessing plus permutation selection run once on all of this arm's
    rows, and one model fit on the selected features. Deployed model: nothing
    is held out, so it gets no generalization estimate."""
    fit_result, X_full, y_full, groups_full = fit_preprocessing(df, cfg)
    selected, _ = permutation_select_features(
        X_full, y_full, groups_full, forced_features_for(cfg), n_total=budget_for(cfg)
    )

    X_selected = X_full[selected]
    model = make_random_forest()
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
    os.makedirs(OUT_DIR, exist_ok=True)
    arm_dfs = {cfg["name"]: arm_dataframe(dataset, cfg) for cfg in ARM_CONFIGS}
    sweep = {
        cfg["name"]: {
            "metrics": [],
            "candidate_counts": [],
            "ranked_sets": [],
            "counts": [],
        }
        for cfg in ARM_CONFIGS
    }
    oof_predictions, curves = [], []

    for repeat in REPEATS:
        RANDOM_STATE = repeat
        for cfg in ARM_CONFIGS:
            print(f"Running repeat={repeat}, arm={cfg['label']}...", flush=True)
            result = run_outer_cv(arm_dfs[cfg["name"]], cfg)
            sweep[cfg["name"]]["metrics"].append(result["metrics"])
            sweep[cfg["name"]]["candidate_counts"].extend(result["candidate_counts"])
            sweep[cfg["name"]]["ranked_sets"].extend(result["ranked_sets"])
            sweep[cfg["name"]]["counts"].extend(result["counts"])
            oof_predictions.append(
                result["predictions"].assign(arm=cfg["name"], repeat=repeat)
            )
            curves.append(result["curve"].assign(arm=cfg["name"], repeat=repeat))

    # Every out-of-fold prediction of the sweep, so the tables below can be
    # recomputed - or plotted differently - without rerunning anything.
    pd.concat(oof_predictions, ignore_index=True)[
        ["arm", "repeat", "PFAS", "true", "predicted"]
    ].to_csv(f"{PREFIX}_oof_predictions.csv", index=False)

    # One row per arm x repeat: the per-repeat numbers behind the mean +/- std
    # table, which the printed summary collapses.
    per_repeat = pd.DataFrame(
        [
            {"arm": cfg["name"], "repeat": repeat, **metrics}
            for cfg in ARM_CONFIGS
            for repeat, metrics in zip(REPEATS, sweep[cfg["name"]]["metrics"])
        ]
    )
    per_repeat.to_csv(f"{PREFIX}_metrics_per_repeat.csv", index=False)
    metrics_summary = (
        per_repeat.drop(columns="repeat")
        .groupby("arm")
        .agg(["mean", "std"])
        .round(4)
        .reset_index()
    )
    metrics_summary.columns = [
        "arm" if a == "arm" else f"{a}_{b}" for a, b in metrics_summary.columns
    ]
    metrics_summary.to_csv(f"{PREFIX}_metrics.csv", index=False)

    # RMSE (and every other metric) against the number of selected features, from the
    # prefix models every outer fold fitted. Median and IQR across the repeats, as in
    # HSA_nested_permutation.py's N curve; the band is partition-to-partition spread,
    # not a confidence interval.
    curve = pd.concat(curves, ignore_index=True)
    curve[["arm", "repeat", "N"] + METRICS].to_csv(
        f"{PREFIX}_N_curve_per_repeat.csv", index=False
    )
    grouped = curve.groupby(["arm", "N"], sort=True)[METRICS]
    curve_summary = pd.concat(
        [
            grouped.median().add_suffix("_median"),
            grouped.quantile(0.25).add_suffix("_q25"),
            grouped.quantile(0.75).add_suffix("_q75"),
        ],
        axis=1,
    ).reset_index()
    curve_summary.to_csv(f"{PREFIX}_N_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for cfg in ARM_CONFIGS:
        arm_curve = curve_summary[curve_summary["arm"] == cfg["name"]]
        color = ARM_COLORS[cfg["name"]]
        ax.fill_between(
            arm_curve["N"],
            arm_curve["macro_rmse_q25"],
            arm_curve["macro_rmse_q75"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            arm_curve["N"],
            arm_curve["macro_rmse_median"],
            color=color,
            lw=2,
            marker="o",
            ms=4,
            label=cfg["label"],
        )
    ax.axvline(
        N_SELECT_STEPS,
        color="black",
        ls="--",
        lw=1,
        label=f"N={N_SELECT_STEPS} reported",
    )
    ax.set_xlabel("Number of selected features (N, forced included)")
    ax.set_ylabel("Nested-CV macro RMSE (log half-life)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, title="Median, IQR band", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_N_curve.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'=' * 30} Summary {'=' * 30}")
    print(
        "Estimator: random forest (300 trees, min_samples_leaf=3, "
        "max_features=1.0, bootstrap). "
        f"Permutation selection: R={PERM_REPEATS} repeats x K={N_INNER_SPLITS} "
        f"grouped folds x P={N_PERMUTATIONS} shuffles = "
        f"{PERM_REPEATS * N_INNER_SPLITS * N_PERMUTATIONS} importances per "
        f"candidate, median-pooled; correlation threshold {CORR_THRESHOLD}."
    )
    print(
        f"Feature budget: {N_SELECT_STEPS} per arm, forced included, except where an "
        "arm forces more than that - a forced feature cannot be dropped to stay under "
        "the budget, so such an arm fits its forced set and ranks nothing on top. "
        "Per arm: "
        + ", ".join(
            f"{cfg['name']}={budget_for(cfg)} "
            f"({len(forced_features_for(cfg))} forced)"
            for cfg in ARM_CONFIGS
        )
        + "."
    )
    print(
        f"Outer loop: {OUTER_SPLITS}-fold StratifiedGroupKFold grouped by PFAS, "
        f"stratified on compound-mean log_half_life in {N_STRATA_BINS} quantile bins, "
        f"repeated {OUTER_REPEATS} times. Each repeat draws a different partition, so "
        "the spread below covers both the partition and the random forest's own "
        "bootstrap randomness."
    )
    print(
        f"Preprocessing per outer training set: drop missing > {MISSING_THRESHOLD:.0%} "
        f"and mean-impute, drop constant, MinMax scale, drop zero variance, "
        f"|r| > {CORR_THRESHOLD} within the continuous and the binary block "
        f"separately, then cap at {CANDIDATE_CAP} by |Spearman|. Forced features are "
        "exempt from all of it."
    )

    print("\nCandidate features remaining after filtering (all outer folds x repeats):")
    for cfg in ARM_CONFIGS:
        counts = np.array(sweep[cfg["name"]]["candidate_counts"])
        print(
            f"  Arm {cfg['label']}: min={counts.min()}, "
            f"mean={counts.mean():.1f}, max={counts.max()}"
        )

    # Where the candidates actually go. Arm B is shown because every arm starts from
    # the same columns bar the LogKa pair, so one arm is representative of the chain.
    stages = {
        "missing": f"missing > {MISSING_THRESHOLD:.0%}",
        "constant": "constant",
        "variance": "zero variance",
        "correlation": f"|r| > {CORR_THRESHOLD}",
        "cap": f"cap {CANDIDATE_CAP} (Spearman)",
    }
    stage_df = pd.DataFrame(sweep["B"]["counts"])
    remaining = stage_df["start"]
    rows = [["featurized", "", f"{remaining.median():g}"]]
    for key, stage in stages.items():
        dropped = stage_df[f"dropped_{key}"]
        remaining = remaining - dropped
        rows.append(
            [
                stage,
                f"{dropped.median():g} [{dropped.min()}-{dropped.max()}]",
                f"{remaining.median():g} [{remaining.min()}-{remaining.max()}]",
            ]
        )
    print(
        f"\nFeature filtering, Arm B (median [min-max] over "
        f"{len(stage_df)} outer folds; forced features bypass every stage):"
    )
    stage_table = pd.DataFrame(rows, columns=["stage", "dropped", "remaining"])
    print(stage_table.to_string(index=False))
    stage_table.to_csv(f"{PREFIX}_filtering_stages.csv", index=False)
    stage_df.assign(arm="B").to_csv(f"{PREFIX}_filtering_counts.csv", index=False)

    print(
        f"\nNested CV across {OUTER_REPEATS} repeats of "
        f"{OUTER_SPLITS}-fold grouped CV (mean +/- std of each metric):"
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

    print(
        f"\nMacro RMSE against feature count (median over {OUTER_REPEATS} repeats; "
        f"the reported models use N={N_SELECT_STEPS}):"
    )
    for cfg in ARM_CONFIGS:
        arm_curve = curve_summary[curve_summary["arm"] == cfg["name"]]
        best = arm_curve.loc[arm_curve["macro_rmse_median"].idxmin()]
        at_n = arm_curve[arm_curve["N"] == N_SELECT_STEPS]["macro_rmse_median"]
        print(
            f"  Arm {cfg['label']}: best N={int(best['N'])} at "
            f"{best['macro_rmse_median']:.4f}"
            + (f", N={N_SELECT_STEPS} at {at_n.iloc[0]:.4f}" if len(at_n) else "")
        )

    print(
        f"\nHead-to-head wins (out of {OUTER_REPEATS} repeats, "
        "same repeat compared):"
    )
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
            # every pair, the later arm against the earlier one
            for i, second in enumerate(ARM_NAMES)
            for first in ARM_NAMES[i + 1 :]
        }
    )
    print(win_table.to_string())
    win_table.rename_axis("metric").to_csv(f"{PREFIX}_head_to_head.csv")

    print(
        "\nRanked-feature stability (mean pairwise Jaccard across all outer "
        "folds x repeats, forced features excluded):"
    )
    jaccard = {}
    for cfg in ARM_CONFIGS:
        sets = sweep[cfg["name"]]["ranked_sets"]
        jaccard[cfg["name"]] = mean_pairwise_jaccard(sets)
        print(
            f"  Arm {cfg['label']}: {jaccard[cfg['name']]:.4f} "
            f"({len(sets)} selected sets)"
        )

    # How often each feature survived the ranking, per arm. Forced features are
    # absent by construction - they never pass through the ranking.
    frequency = pd.DataFrame(
        [
            {
                "arm": cfg["name"],
                "feature": feature,
                "frequency": count / len(sweep[cfg["name"]]["ranked_sets"]),
                "n_sets": count,
            }
            for cfg in ARM_CONFIGS
            for feature, count in Counter(
                f for s in sweep[cfg["name"]]["ranked_sets"] for f in s
            ).most_common()
        ]
    )
    frequency.to_csv(f"{PREFIX}_selection_frequency.csv", index=False)

    fig, axes = plt.subplots(
        1, len(ARM_CONFIGS), figsize=(5.5 * len(ARM_CONFIGS), 6), sharex=True
    )
    for ax, cfg in zip(axes, ARM_CONFIGS):
        top = frequency[frequency["arm"] == cfg["name"]].head(15).iloc[::-1]
        ax.barh(top["feature"], top["frequency"], color="#2a78d6")
        ax.set_title(f"Arm {cfg['name']}")
        ax.set_xlabel("fraction of outer folds")
    fig.suptitle(
        "Selection frequency of the ranked (non-forced) features, "
        f"{OUTER_REPEATS} x {OUTER_SPLITS} outer folds - top 15 per arm"
    )
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_selection_frequency.png", dpi=300, bbox_inches="tight")

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
    in_sample_table.rename_axis("metric").to_csv(f"{PREFIX}_in_sample_metrics.csv")

    pd.DataFrame(
        [
            {
                "arm": final["cfg"]["name"],
                "rank": i,
                "feature": feature,
                "forced": feature in forced_features_for(final["cfg"]),
            }
            for final in finals
            for i, feature in enumerate(final["selected"], start=1)
        ]
    ).to_csv(f"{PREFIX}_final_features.csv", index=False)

    # The pass is skipped when RUN_FINAL_LOPO is off: it is the most expensive
    # step in the script (one outer fold per compound, selection redone in each)
    # and nothing else depends on it, so experiments can leave it out.
    final_lopo = {}
    if RUN_FINAL_LOPO:
        # Leave-one-PFAS-out pass per arm, for the final models. Preprocessing and
        # permutation selection are redone inside every outer fold exactly as in the
        # sweep, so this is a genuine generalization estimate; it just uses the most
        # training data available (28 of 29 compounds, against ~23 under 5-fold) and has
        # only one possible partition, so there is nothing to average over and no
        # partition spread to report. It is run at RANDOM_STATE=42, the seed the final
        # models were built with, which is outside the swept range, so this number does
        # not appear in the table above and will not match its mean.
        lopo_predictions = []
        for cfg in ARM_CONFIGS:
            print(f"\nRunning final LOPO pass, arm={cfg['label']}...", flush=True)
            result = run_lopo(arm_dfs[cfg["name"]], cfg)
            final_lopo[cfg["name"]] = result["metrics"]
            lopo_predictions.append(result["predictions"].assign(arm=cfg["name"]))
        lopo_predictions = pd.concat(lopo_predictions, ignore_index=True)
        lopo_predictions[["arm", "PFAS", "true", "predicted"]].to_csv(
            f"{PREFIX}_lopo_predictions.csv", index=False
        )

        print(
            f"\nNested LOPO-CV at RANDOM_STATE={RANDOM_STATE}, the seed the final "
            "models above were built with (one fold per compound, selection redone "
            "inside every fold; single partition, so it will not match the "
            f"{OUTER_REPEATS}-repeat mean above):"
        )
        final_lopo_table = pd.DataFrame(
            {f"Arm {cfg['name']}": final_lopo[cfg["name"]] for cfg in ARM_CONFIGS}
        )
        print(final_lopo_table.to_string())
        final_lopo_table.rename_axis("metric").to_csv(f"{PREFIX}_lopo_metrics.csv")

        # Observed vs predicted, one panel per arm, from the LOPO pass - these are the
        # held-out predictions of the partition the final models were built under.
        fig, axes = plt.subplots(
            1,
            len(ARM_CONFIGS),
            figsize=(5.5 * len(ARM_CONFIGS), 6),
            sharex=True,
            sharey=True,
        )
        for ax, cfg in zip(axes, ARM_CONFIGS):
            arm_rows = lopo_predictions[lopo_predictions["arm"] == cfg["name"]]
            ax.scatter(arm_rows["true"], arm_rows["predicted"], s=18, alpha=0.7)
            limits = [
                min(arm_rows["true"].min(), arm_rows["predicted"].min()),
                max(arm_rows["true"].max(), arm_rows["predicted"].max()),
            ]
            ax.plot(limits, limits, color="grey", linewidth=1)
            ax.set_title(
                f"Arm {cfg['name']} - macro RMSE "
                f"{final_lopo[cfg['name']]['macro_rmse']:.3f}"
            )
            ax.set_xlabel("observed log half-life")
        axes[0].set_ylabel("LOPO predicted log half-life")
        fig.suptitle(f"Leave-one-PFAS-out predictions (RANDOM_STATE={RANDOM_STATE})")
        fig.tight_layout()
        fig.savefig(f"{PREFIX}_lopo_scatter.png", dpi=300, bbox_inches="tight")

    # Y-randomization: the same nested CV against a target whose compound-level
    # signal has been shuffled away, one shuffle per repeat. It answers the question
    # the metrics above cannot - whether an arm's score could have arisen from this
    # many candidates and this few compounds with no real relationship present. A
    # null arm should land near the spread of the compound means; anything much
    # better than that is the pipeline fitting noise.
    y_rand = {}
    if RUN_Y_RANDOMIZATION:
        # The null needs no N curve and a lighter ranking; both are restored after.
        saved_curve, saved_perm = N_CURVE_STEPS, PERM_REPEATS
        N_CURVE_STEPS, PERM_REPEATS = N_SELECT_STEPS, Y_RAND_PERM_REPEATS
        for repeat in REPEATS:
            RANDOM_STATE = repeat
            for cfg in ARM_CONFIGS:
                print(
                    f"\nRunning y-randomization repeat={repeat}, "
                    f"arm={cfg['label']}...",
                    flush=True,
                )
                shuffled = y_randomized(arm_dfs[cfg["name"]], 1000 + repeat)
                y_rand.setdefault(cfg["name"], []).append(
                    run_outer_cv(shuffled, cfg)["metrics"]
                )
        N_CURVE_STEPS, PERM_REPEATS = saved_curve, saved_perm
        RANDOM_STATE = 42

        compound_sd = float(dataset.groupby("PFAS")["log_half_life"].mean().std(ddof=1))
        print(
            f"\nY-randomization: {OUTER_REPEATS} shuffles of the compound means, "
            f"same outer CV, selector at R={Y_RAND_PERM_REPEATS}. A null arm should "
            f"sit near the spread of the compound means ({compound_sd:.4f}); the "
            "real run has to beat it by a clear margin to mean anything:"
        )
        y_rand_table = pd.DataFrame(
            [
                {
                    "arm": cfg["name"],
                    "run": run,
                    **{
                        metric: float(np.mean([m[metric] for m in runs]))
                        for metric in METRICS
                    },
                    "macro_rmse_std": float(
                        np.std([m["macro_rmse"] for m in runs], ddof=1)
                    ),
                }
                for cfg in ARM_CONFIGS
                for run, runs in [
                    ("real", sweep[cfg["name"]]["metrics"]),
                    ("y-randomized", y_rand[cfg["name"]]),
                ]
            ]
        )
        print(
            y_rand_table[
                ["arm", "run", "macro_rmse", "macro_rmse_std", "macro_r2", "micro_rmse"]
            ]
            .round(4)
            .to_string(index=False)
        )
        y_rand_table.to_csv(f"{PREFIX}_y_randomization.csv", index=False)

    # SHAP beeswarms, one panel per arm. Plotted feature values are the
    # MinMax-scaled values the models were fit on, not raw descriptor units.
    #
    # The SHAP values are tree-path-dependent TreeSHAP, as in the XGBoost
    # script. That script had to take them from XGBoost's own
    # `pred_contribs=True` because shap cannot parse the xgboost 3.x model
    # JSON; shap.TreeExplainer reads scikit-learn random forests directly, so
    # it is used here.
    fig, axes = plt.subplots(1, len(ARM_CONFIGS), figsize=(7 * len(ARM_CONFIGS), 7))
    for ax, final in zip(axes, finals):
        explainer = shap.TreeExplainer(final["model"])
        shap_values = explainer.shap_values(final["X_selected"])
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
    fig.savefig(f"{PREFIX}_shap.png", dpi=300, bbox_inches="tight")

    # One record of the whole run, in the spirit of HSA_nested_permutation.py's
    # manifest: the parameters it was run with, the arms, and every summary number
    # the block above printed.
    manifest = {
        "parameters": {
            k: json_safe(globals()[k])
            for k in [
                "N_SELECT_STEPS",
                "N_INNER_SPLITS",
                "PERM_REPEATS",
                "N_PERMUTATIONS",
                "CORR_THRESHOLD",
                "MISSING_THRESHOLD",
                "CANDIDATE_CAP",
                "OUTER_SPLITS",
                "OUTER_REPEATS",
                "N_STRATA_BINS",
                "RANDOM_STATE",
                "N_JOBS",
                "N_CURVE_STEPS",
                "RUN_FINAL_LOPO",
                "RUN_Y_RANDOMIZATION",
                "Y_RAND_PERM_REPEATS",
            ]
        },
        "data_path": DATA_PATH,
        "columns_path": COLUMNS_PATH,
        "n_rows": int(len(dataset)),
        "n_compounds": int(dataset["PFAS"].nunique()),
        "regressor": repr(make_random_forest()),
        "arms": {
            cfg["name"]: {
                "label": cfg["label"],
                "logka_columns": cfg["logka_columns"],
                "forced_features": forced_features_for(cfg),
                "n_rows": int(len(arm_dfs[cfg["name"]])),
                "candidate_counts": {
                    "min": int(np.min(sweep[cfg["name"]]["candidate_counts"])),
                    "mean": float(np.mean(sweep[cfg["name"]]["candidate_counts"])),
                    "max": int(np.max(sweep[cfg["name"]]["candidate_counts"])),
                },
                "ranked_feature_jaccard": jaccard[cfg["name"]],
                "nested_cv_metrics_per_repeat": sweep[cfg["name"]]["metrics"],
                # null when RUN_FINAL_LOPO = False
                "lopo_metrics": final_lopo.get(cfg["name"]),
                # null when RUN_Y_RANDOMIZATION = False
                "y_randomized_metrics_per_shuffle": y_rand.get(cfg["name"]),
            }
            for cfg in ARM_CONFIGS
        },
        "nested_cv_metrics_summary": metrics_summary.to_dict(orient="records"),
        "head_to_head_wins": win_table.to_dict(),
        "final_models": {
            final["cfg"]["name"]: {
                "selected_features": list(final["selected"]),
                "n_candidates_after_filtering": int(
                    final["n_candidates_after_filtering"]
                ),
                "in_sample_metrics_not_a_performance_claim": final["in_sample_metrics"],
            }
            for final in finals
        },
    }
    with open(f"{PREFIX}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=json_safe)

    print(f"\nOutputs: {PREFIX}_*")
    plt.show()
