"""HSA Ka QSAR (Starnes et al. data): nested CV with permutation-importance feature selection, all compounds."""

# Adapted from PFAS_albumin_QSAR/PFAS_HSA_QSAR/HSA_nested_permutation.py (paths and labels only);
# the input CSV is built by prepare_starnes_data.py (binders only, PubChem SMILES, Ka in M^-1).
# Ported (copied, not imported): permutation selector, correlation filter and the RF/XGBoost
# configs from PFAS_HalfLife_QSAR/Restructured/old/permutation_selection_three_arms_macro_rf.py
# and algorithm_comparison_three_arms.py; SHAP as in
# PFAS_albumin_QSAR/Equilibrium_dialysis/SHAP_Analysis.py.

import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost
from jaqpot_api_client.models.model_task import ModelTask
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import (
    MordredDescriptors,
    RDKitDescriptors,
    TopologicalFingerprint,
)
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

FEATURIZERS = [
    RDKitDescriptors(),
    MordredDescriptors(),
    TopologicalFingerprint(),
]
N_FEATURES = 6
N_FEATURES_CURVE = 10
OUTER_SPLITS = 3
OUTER_REPEATS = 10
PERM_REPEATS = 10  # drop to 3 for exploratory runs
N_INNER_SPLITS = 3
N_PERMUTATIONS = 5
MISSING_THRESHOLD = 0.05
CORR_THRESHOLD = 0.99
CANDIDATE_CAP = 400
RANDOM_STATE = 42
N_JOBS = -1
N_STRATA_BINS = 5  # quantile bins of the target for every StratifiedKFold
Y_RAND_PERM_REPEATS = 3  # reduced selector repeats for y-randomization
LOG_TARGET = True  # False: raw Ka in mM^-1 (Ka_mM column); True: log10(Ka in M^-1)
RUN_Q2_LOO = False  # nested leave-one-out Q2; slow, set False for exploratory runs
EXCLUDE_CONGENERS = [
    "Octanoic acid",
    "Decanoic acid",
    "Hexadecanoic acid",
    # "1,1,2,3,3,3-hexafluoropropyl 1H,1H-heptafluorobutyl ether",
    # "EtFOSA",
    # "4,4,4-trifluoro-3-(trifluoromethyl)crotonic acid",
]  # congeners removed before modelling, e.g. ["5:3 FTB"]

DATA_PATH = "PFAS_albumin_QSAR/PFAS_HSA_QSAR/Starnes_HSA/HSA_Starnes_smiles.csv"
OUT_DIR = "PFAS_albumin_QSAR/PFAS_HSA_QSAR/Starnes_HSA/results"


# Settings chosen for this dataset: ~29 compounds per outer training fold, ~19 in the
# selector's inner fits (with all ~250 candidates), 1-10 features in the evaluated models


def make_ridge(seed):
    # alpha picked by efficient leave-one-out inside each fit (training data only): the
    # selector's p >> n fits need strong shrinkage, the few-feature models much less
    return RidgeCV(alphas=np.logspace(-2, 3, 11))


def make_random_forest(seed):
    # >= 2 compounds per leaf limits fitting single noisy values; 0.5 gives 3 of 6 features
    # per split in the final model (0.33 gave 1, i.e. random splits), ~125 of 250 in the selector
    return RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.5,
        bootstrap=True,
        random_state=seed,
        n_jobs=1,
    )


def make_xgboost(seed):
    # Shallow trees as before, but lighter regularization for ~19-29 training compounds;
    # row/column subsampling reduces the variance of boosting on so few samples
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=2,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=1,
    )


# Simplest first: a tie in median RMSE goes to the earlier (simpler) regressor
REGRESSORS = {
    "Ridge": make_ridge,
    "RandomForest": make_random_forest,
    "XGBoost": make_xgboost,
}

FEATURIZER_LABELS = {
    "RDKitDescriptors": "rdkit",
    "TopologicalFingerprint": "topfp",
    "MordredDescriptors": "mordred",
}
LABEL = "-".join(
    FEATURIZER_LABELS.get(type(f).__name__, type(f).__name__.lower())
    for f in FEATURIZERS
)
TARGET_LABEL = "Log10Ka" if LOG_TARGET else "Ka (mM$^{-1}$)"
PREFIX = os.path.join(
    OUT_DIR, f"Starnes_nested_{LABEL}_N{N_FEATURES}" + ("" if LOG_TARGET else "_rawKa")
)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    assert set(EXCLUDE_CONGENERS) <= set(df["Congener"]), (
        "EXCLUDE_CONGENERS names not found in the data"
    )
    df = df[~df["Congener"].isin(EXCLUDE_CONGENERS)].reset_index(drop=True)
    df["target"] = (
        np.log10(df["Ka"]) if LOG_TARGET else df["Ka_mM"]
    )  # Ka in M^-1, Ka_mM in mM^-1
    return df


def featurize(df):
    # Per-compound descriptors with nothing fitted, so computing them for all compounds leaks nothing
    blocks, binary_cols, seen = [], [], set()
    for featurizer in FEATURIZERS:
        X = JaqpotTabularDataset(
            df=df[["SMILES", "target"]],
            y_cols="target",
            x_cols=[],
            smiles_cols="SMILES",
            featurizers=[featurizer],
            task=ModelTask.REGRESSION,
            verbose=False,
        ).X
        name = type(featurizer).__name__
        if X.shape[1] == 0:
            raise RuntimeError(
                f"{name} returned no descriptors; is its package installed?"
            )
        if isinstance(featurizer, MordredDescriptors):
            X = X.replace(
                -1000.0, np.nan
            )  # jaqpotpy encodes failed Mordred values as -1000
        X = X.rename(
            columns=lambda c: (
                f"{c}_{FEATURIZER_LABELS.get(name, name)}" if c in seen else c
            )
        )
        seen.update(X.columns)
        if isinstance(featurizer, TopologicalFingerprint):
            binary_cols += list(X.columns)
        blocks.append(X)
    return pd.concat(blocks, axis=1).astype(float), binary_cols


def strata(y):
    return pd.qcut(y, q=N_STRATA_BINS, labels=False, duplicates="drop")


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def metrics(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


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


def fit_preprocessing(X, y, binary_cols):
    """Fit every preprocessing statistic on X (a training set) only."""
    counts = {"start": X.shape[1]}
    X = X.replace([np.inf, -np.inf], np.nan)
    missing = X.isna().mean()
    keep = list(missing.index[missing <= MISSING_THRESHOLD])
    counts["dropped_missing"] = X.shape[1] - len(keep)
    means = X[keep].mean()
    X = X[keep].fillna(means)

    nonconstant = list(X.columns[X.nunique() > 1])
    counts["dropped_constant"] = X.shape[1] - len(nonconstant)
    X = X[nonconstant]

    scaler = MinMaxScaler().fit(X)  # leaves 0/1 fingerprint bits unchanged
    X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    variance = VarianceThreshold().fit(X)
    counts["dropped_variance"] = int((~variance.get_support()).sum())
    X = X.loc[:, variance.get_support()]

    n_before = X.shape[1]
    binary = [c for c in X.columns if c in set(binary_cols)]
    X = pd.concat(
        [correlation_filter(X.drop(columns=binary)), correlation_filter(X[binary])],
        axis=1,
    )
    counts["dropped_correlation"] = n_before - X.shape[1]

    n_before = X.shape[1]
    if X.shape[1] > CANDIDATE_CAP:
        # Univariate screen: |Spearman| with the target on this training set
        screen = X.rank().corrwith(pd.Series(y, index=X.index).rank()).abs()
        top = set(
            screen.sort_values(ascending=False, kind="mergesort").index[:CANDIDATE_CAP]
        )
        X = X[[c for c in X.columns if c in top]]
    counts["dropped_cap"] = n_before - X.shape[1]
    counts["final"] = X.shape[1]

    prep = {
        "keep": keep,
        "means": means,
        "nonconstant": nonconstant,
        "scaler": scaler,
        "columns": list(X.columns),
    }
    return prep, X, counts


def transform(prep, X):
    X = X.replace([np.inf, -np.inf], np.nan)[prep["keep"]].fillna(prep["means"])
    X = X[prep["nonconstant"]]
    X = pd.DataFrame(prep["scaler"].transform(X), columns=X.columns, index=X.index)
    return X[prep["columns"]]


def fold_permutation_importances(
    model, X_val, y_val, rng, baseline_rmse, chunk_rows=4000
):
    # Shuffled copies are stacked and predicted in batches for speed; same result as one by one
    n_val = X_val.shape[0]
    rows_per_candidate = n_val * N_PERMUTATIONS
    candidates_per_chunk = max(1, chunk_rows // rows_per_candidate)
    importances = {}
    for start in range(0, X_val.shape[1], candidates_per_chunk):
        chunk = range(start, min(start + candidates_per_chunk, X_val.shape[1]))
        blocks = []
        for col in chunk:
            tiled = np.tile(X_val, (N_PERMUTATIONS, 1))
            for p in range(N_PERMUTATIONS):
                tiled[p * n_val : (p + 1) * n_val, col] = X_val[
                    rng.permutation(n_val), col
                ]
            blocks.append(tiled)
        preds = model.predict(np.vstack(blocks))
        for block_i, col in enumerate(chunk):
            offset = block_i * rows_per_candidate
            importances[col] = [
                rmse(y_val, preds[offset + p * n_val : offset + (p + 1) * n_val])
                - baseline_rmse
                for p in range(N_PERMUTATIONS)
            ]
    return importances


def permutation_select_features(X, y, make_model, perm_repeats):
    # Kvasnicka et al. 2024 (SI, "Model Training Steps"): repeated K-fold permutation
    # importance (increase in RMSE), pooled over R x K x P and reduced to the median
    columns = list(X.columns)
    X_arr, y_arr = X.to_numpy(dtype=float), np.asarray(y, dtype=float)
    pooled = {col: [] for col in range(len(columns))}
    for repeat in range(perm_repeats):
        repeat_seed = RANDOM_STATE * 100 + repeat
        skf = StratifiedKFold(
            n_splits=N_INNER_SPLITS, shuffle=True, random_state=repeat_seed
        )
        for fold, (tr, va) in enumerate(skf.split(X_arr, strata(y_arr))):
            model = make_model(RANDOM_STATE).fit(X_arr[tr], y_arr[tr])
            baseline = rmse(y_arr[va], model.predict(X_arr[va]))
            rng = np.random.RandomState(repeat_seed * 100 + fold)
            for col, values in fold_permutation_importances(
                model, X_arr[va], y_arr[va], rng, baseline
            ).items():
                pooled[col].extend(values)
    median = {columns[col]: float(np.median(v)) for col, v in pooled.items()}
    return sorted(columns, key=lambda c: (-median[c], c))


def run_repeat(X, binary_cols, y, repeat, regressor_names, perm_repeats):
    """One outer repeat: preprocessing and ranking redone inside every outer training fold."""
    skf = StratifiedKFold(
        n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat
    )
    oof = {
        name: np.full((N_FEATURES_CURVE, len(y)), np.nan) for name in regressor_names
    }
    rankings, counts = {}, []
    for fold, (tr, va) in enumerate(skf.split(X, strata(y))):
        prep, X_tr, fold_counts = fit_preprocessing(X.iloc[tr], y[tr], binary_cols)
        X_va = transform(prep, X.iloc[va])
        counts.append(fold_counts)
        for name in regressor_names:
            ranking = permutation_select_features(
                X_tr, y[tr], REGRESSORS[name], perm_repeats
            )
            assert (repeat, fold, name) not in rankings and len(
                ranking
            ) >= N_FEATURES_CURVE
            rankings[(repeat, fold, name)] = ranking[:N_FEATURES_CURVE]
            # Every N reuses top-N prefixes of this one ranking
            for n in range(1, N_FEATURES_CURVE + 1):
                cols = ranking[:n]
                model = REGRESSORS[name](RANDOM_STATE).fit(X_tr[cols].to_numpy(), y[tr])
                oof[name][n - 1, va] = model.predict(X_va[cols].to_numpy())
    return oof, rankings, counts


def nested_cv(X, binary_cols, y_by_repeat, regressor_names, perm_repeats):
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_repeat)(
            X, binary_cols, y_by_repeat[r], r, regressor_names, perm_repeats
        )
        for r in range(OUTER_REPEATS)
    )
    rankings = {k: v for _, rk, _ in results for k, v in rk.items()}
    assert len(rankings) == OUTER_REPEATS * OUTER_SPLITS * len(regressor_names)
    rows = [
        {
            "repeat": r,
            "regressor": name,
            "N": n + 1,
            **metrics(y_by_repeat[r], preds[n]),
        }
        for r, (oof, _, _) in enumerate(results)
        for name, preds in oof.items()
        for n in range(N_FEATURES_CURVE)
    ]
    oof_by_regressor = {
        name: np.stack([res[0][name] for res in results]) for name in regressor_names
    }
    counts = [c for _, _, fold_counts in results for c in fold_counts]
    return pd.DataFrame(rows), oof_by_regressor, rankings, counts


def summarize(scores):
    """Median and IQR across repeats of the per-repeat (pooled out-of-fold) metrics."""
    g = scores.groupby(["regressor", "N"], sort=False)[["RMSE", "MAE", "R2"]]
    q25, q50, q75 = g.quantile(0.25), g.median(), g.quantile(0.75)
    out = pd.DataFrame(index=q50.index)
    for m in ["RMSE", "MAE", "R2"]:
        out[f"{m}_median"], out[f"{m}_q25"], out[f"{m}_q75"] = q50[m], q25[m], q75[m]
        out[f"{m}_IQR"] = q75[m] - q25[m]
    return out.reset_index()


def win_matrix(scores, names):
    rmse_by_repeat = scores[scores["N"] == N_FEATURES].pivot(
        index="repeat", columns="regressor", values="RMSE"
    )
    return {
        a: {
            b: int((rmse_by_repeat[a] < rmse_by_repeat[b]).sum())
            for b in names
            if b != a
        }
        for a in names
    }


def choose_regressor(at_n):
    # Lowest nested-CV median RMSE at N_FEATURES; idxmin keeps the first (simpler) on ties
    chosen = at_n.set_index("regressor")["RMSE_median"].idxmin()
    return (
        chosen,
        f"{chosen} has the lowest nested-CV median RMSE at N={N_FEATURES}: deploy it",
    )


def loo_prediction(X, binary_cols, y, i, name):
    tr = np.delete(np.arange(len(y)), i)
    prep, X_tr, _ = fit_preprocessing(X.iloc[tr], y[tr], binary_cols)
    cols = permutation_select_features(X_tr, y[tr], REGRESSORS[name], PERM_REPEATS)[
        :N_FEATURES
    ]
    model = REGRESSORS[name](RANDOM_STATE).fit(X_tr[cols].to_numpy(), y[tr])
    return float(model.predict(transform(prep, X.iloc[[i]])[cols].to_numpy())[0])


def shap_values(model, X, name):
    if name == "XGBoost":
        # shap cannot read xgboost 3.x models; use xgboost's own TreeSHAP, as in the reference
        contribs = model.get_booster().predict(
            xgboost.DMatrix(X.to_numpy()), pred_contribs=True
        )
        return shap.Explanation(
            values=contribs[:, :-1],
            base_values=contribs[:, -1],
            data=X.to_numpy(),
            feature_names=list(X.columns),
        )
    return shap.Explainer(model, X)(X)


def plot_n_curve(summary, names):
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
    for name, color in zip(names, COLORS):
        s = summary[summary["regressor"] == name]
        ax.fill_between(
            s["N"], s["RMSE_q25"], s["RMSE_q75"], color=color, alpha=0.2, linewidth=0
        )
        ax.plot(
            s["N"], s["RMSE_median"], color=color, lw=2, marker="o", ms=4, label=name
        )
    ax.set_xlabel("Number of features (N)")
    ax.set_ylabel(f"Nested-CV RMSE ({TARGET_LABEL})")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, title="Median, IQR band", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_N_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_oof(df, preds, headline, name, n_label=5):
    y = df["target"].to_numpy()
    mean, low, high = preds.mean(axis=0), preds.min(axis=0), preds.max(axis=0)
    values = np.concatenate([y, low, high])
    pad = 0.05 * (values.max() - values.min())
    lims = [values.min() - pad, values.max() + pad]
    fig, ax = plt.subplots(figsize=(3.8, 3.8), dpi=300)
    ax.plot(lims, lims, "k--", lw=1, zorder=1)
    ax.errorbar(
        y,
        mean,
        yerr=[mean - low, high - mean],
        fmt="none",
        ecolor="#9a9a95",
        elinewidth=0.8,
        zorder=2,
    )
    ax.scatter(
        y, mean, s=24, color=COLORS[0], edgecolor="white", linewidth=0.5, zorder=3
    )
    # Labels point inward from the nearer edge and alternate above/below to avoid overlaps
    labelled = sorted(np.argsort(-np.abs(mean - y))[:n_label], key=lambda i: y[i])
    for k, i in enumerate(labelled):
        right = y[i] > np.mean(lims)
        ax.annotate(
            df["Congener"].iloc[i],
            (y[i], mean[i]),
            xytext=(-4 if right else 4, 4 if k % 2 == 0 else -9),
            textcoords="offset points",
            ha="right" if right else "left",
            fontsize=6,
        )
    ax.text(
        0.04,
        0.96,
        f"{name}, N = {N_FEATURES}\nRMSE = {headline['RMSE_median']:.2f}\n$R^2$ = {headline['R2_median']:.2f}\n(nested CV, median of {OUTER_REPEATS} repeats)",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Observed {TARGET_LABEL} (HSA, Starnes)", fontsize=9)
    ax.set_ylabel(f"Predicted {TARGET_LABEL} (HSA, Starnes)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_oof_scatter.png", bbox_inches="tight")
    plt.close(fig)


def plot_selection_frequency(freq, name, top=20):
    s = freq.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5, 0.25 * len(s) + 1), dpi=300)
    ax.barh(s["descriptor"], s["frequency"], color=COLORS[0], height=0.7)
    ax.set_xlabel(f"Fraction of outer folds in top {N_FEATURES} ({name})")
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="x", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_selection_frequency.png", bbox_inches="tight")
    plt.close(fig)


def plot_shap(sv, X):
    for kind, suffix in [("dot", "SHAP_beeswarm"), ("bar", "SHAP_bar")]:
        plt.figure()
        shap.summary_plot(
            sv, X, feature_names=list(X.columns), plot_type=kind, show=False
        )
        plt.savefig(f"{PREFIX}_{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close()


def spread(values):
    return f"{values.median():g} [{values.min()}-{values.max()}]"


def count_summary(counts):
    df = pd.DataFrame(counts)
    return {
        k: {
            "min": int(df[k].min()),
            "median": float(df[k].median()),
            "max": int(df[k].max()),
        }
        for k in df.columns
    }


def main():
    assert N_FEATURES <= N_FEATURES_CURVE
    os.makedirs(OUT_DIR, exist_ok=True)
    names = list(REGRESSORS)
    df = load_data()
    X, binary_cols = featurize(df)
    y = df["target"].to_numpy()

    # Nested CV, all three regressors
    scores, oof, rankings, counts = nested_cv(
        X, binary_cols, [y] * OUTER_REPEATS, names, PERM_REPEATS
    )
    summary = summarize(scores)
    summary.to_csv(f"{PREFIX}_N_curve.csv", index=False)
    at_n = summary[summary["N"] == N_FEATURES].drop(columns="N")
    at_n.to_csv(f"{PREFIX}_metrics.csv", index=False)
    plot_n_curve(summary, names)

    wins = win_matrix(scores, names)
    chosen, branch = choose_regressor(at_n)
    headline = at_n.set_index("regressor").loc[chosen]
    plot_oof(df, oof[chosen][:, N_FEATURES - 1, :], headline, chosen)

    chosen_folds = [
        r[:N_FEATURES] for (_, _, name), r in rankings.items() if name == chosen
    ]
    freq = Counter(f for r in chosen_folds for f in r)
    freq = pd.DataFrame(
        [
            {"descriptor": f, "frequency": c / len(chosen_folds)}
            for f, c in freq.most_common()
        ]
    )
    freq.to_csv(f"{PREFIX}_selection_frequency.csv", index=False)
    plot_selection_frequency(freq, chosen)

    # Y-randomization: shuffled target, one shuffle per outer repeat, chosen regressor only
    y_rand = [
        np.random.RandomState(RANDOM_STATE + 1000 + r).permutation(y)
        for r in range(OUTER_REPEATS)
    ]
    y_rand_perm = min(Y_RAND_PERM_REPEATS, PERM_REPEATS)
    rand_scores, _, _, _ = nested_cv(X, binary_cols, y_rand, [chosen], y_rand_perm)
    rand_summary = summarize(rand_scores)
    rand_at_n = rand_summary[rand_summary["N"] == N_FEATURES].drop(columns="N")
    y_rand_table = pd.concat(
        [
            at_n[at_n["regressor"] == chosen].assign(
                run="real", perm_repeats=PERM_REPEATS
            ),
            rand_at_n.assign(run="y-randomized", perm_repeats=y_rand_perm),
        ]
    )
    y_rand_table.to_csv(f"{PREFIX}_y_randomization.csv", index=False)

    # Q2_LOO: leave-one-out with preprocessing and ranking redone inside every fold
    q2_loo = None
    if RUN_Q2_LOO:
        loo = Parallel(n_jobs=N_JOBS)(
            delayed(loo_prediction)(X, binary_cols, y, i, chosen) for i in range(len(y))
        )
        q2_loo = float(
            1 - np.sum((y - np.array(loo)) ** 2) / np.sum((y - y.mean()) ** 2)
        )

    # Final model on all compounds: preprocessing and ranking re-run on the full set
    prep, X_full, final_counts = fit_preprocessing(X, y, binary_cols)
    final_features = permutation_select_features(
        X_full, y, REGRESSORS[chosen], PERM_REPEATS
    )[:N_FEATURES]
    X_final = X_full[final_features]
    model = REGRESSORS[chosen](RANDOM_STATE).fit(X_final.to_numpy(), y)
    in_sample = metrics(y, model.predict(X_final.to_numpy()))
    plot_shap(shap_values(model, X_final, chosen), X_final)

    fold_counts = count_summary(counts)
    manifest = {
        "parameters": {
            k: globals()[k]
            for k in [
                "N_FEATURES",
                "N_FEATURES_CURVE",
                "OUTER_SPLITS",
                "OUTER_REPEATS",
                "PERM_REPEATS",
                "N_INNER_SPLITS",
                "N_PERMUTATIONS",
                "MISSING_THRESHOLD",
                "CORR_THRESHOLD",
                "CANDIDATE_CAP",
                "RANDOM_STATE",
                "N_JOBS",
                "N_STRATA_BINS",
                "RUN_Q2_LOO",
                "LOG_TARGET",
                "EXCLUDE_CONGENERS",
            ]
        },
        "featurizers": [type(f).__name__ for f in FEATURIZERS],
        "featurizer_label": LABEL,
        "regressors": {name: repr(REGRESSORS[name](RANDOM_STATE)) for name in names},
        "n_compounds": int(len(y)),
        "n_descriptors_raw": int(X.shape[1]),
        "feature_counts_outer_folds": fold_counts,
        "feature_counts_final_model": final_counts,
        "rankings_computed": len(rankings),
        "metrics_at_N": at_n.to_dict(orient="records"),
        "win_matrix_rmse": wins,
        "rule": {"chosen": chosen, "branch": branch},
        "q2_loo": q2_loo,  # null when RUN_Q2_LOO = False
        "y_randomization": {
            "perm_repeats": y_rand_perm,
            "metrics": y_rand_table.to_dict(orient="records"),
        },
        "final_features": final_features,
        "in_sample_metrics_not_a_performance_claim": in_sample,
    }
    with open(f"{PREFIX}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Starnes HSA nested CV | {LABEL} | {len(y)} compounds | {OUTER_REPEATS}x{OUTER_SPLITS} outer folds | PERM_REPEATS={PERM_REPEATS}"
    )
    # Features dropped and remaining after each preprocessing stage
    stages = {
        "missing": f"missing > {MISSING_THRESHOLD:.0%}",
        "constant": "constant",
        "variance": "zero variance",
        "correlation": f"|r| > {CORR_THRESHOLD}",
        "cap": f"cap {CANDIDATE_CAP} (Spearman)",
    }
    fold_df = pd.DataFrame(counts)
    remaining, final_remaining = fold_df["start"], final_counts["start"]
    rows = [["featurized", "", spread(remaining), "", final_remaining]]
    for key, stage in stages.items():
        remaining = remaining - fold_df[f"dropped_{key}"]
        final_remaining -= final_counts[f"dropped_{key}"]
        rows.append(
            [
                stage,
                spread(fold_df[f"dropped_{key}"]),
                spread(remaining),
                final_counts[f"dropped_{key}"],
                final_remaining,
            ]
        )
    print(
        f"\nFeature filtering (outer folds: median [min-max] over {len(fold_df)} folds; final model: all {len(y)} compounds):"
    )
    print(
        pd.DataFrame(
            rows,
            columns=[
                "stage",
                "dropped (folds)",
                "remaining (folds)",
                "dropped (final)",
                "remaining (final)",
            ],
        ).to_string(index=False)
    )
    cols = ["regressor"] + [
        f"{m}_{s}" for m in ["RMSE", "MAE", "R2"] for s in ["median", "IQR"]
    ]
    print(f"\nNested CV at N={N_FEATURES} (median, IQR across repeats):")
    print(at_n[cols].round(3).to_string(index=False))
    print("\nRMSE wins (repeats where row beats column):")
    win_table = pd.DataFrame(wins).T.reindex(index=names, columns=names)
    print(win_table.map(lambda v: "-" if pd.isna(v) else int(v)).to_string())
    print(f"\nRule: {branch}")
    q2_text = f"{q2_loo:.3f}" if RUN_Q2_LOO else "skipped (RUN_Q2_LOO = False)"
    print(f"Q2_LOO ({chosen}, N={N_FEATURES}): {q2_text}")
    r = y_rand_table.set_index("run")
    print(
        f"Y-randomization ({chosen}, PERM_REPEATS={y_rand_perm}): RMSE {r.loc['y-randomized', 'RMSE_median']:.3f} "
        f"vs {r.loc['real', 'RMSE_median']:.3f} real; R2 {r.loc['y-randomized', 'R2_median']:.3f} vs {r.loc['real', 'R2_median']:.3f}"
    )
    print(f"Final features: {final_features}")
    print(
        "In-sample, not a performance claim: "
        + ", ".join(f"{k} {v:.3f}" for k, v in in_sample.items())
    )
    print(f"Outputs: {PREFIX}_*")


if __name__ == "__main__":
    main()
