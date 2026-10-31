"""Train the final HSA Log10Ka model and predict Log10Ka for the half-life PFAS.

The metadata below is copied from
results/HSA_nested_rdkit-mordred-topfp-maccs_N5_manifest.json, written by
HSA_nested_permutation.py. Only the dataset is read at runtime. No cross-validation is
repeated here - the performance estimate lives in that manifest - this fits one model on
every remaining compound, predicts the half-life PFAS with it and writes Ka_results.csv.
Uploading to Jaqpot is off by default: it currently fails because jaqpotpy sanitises
hyphenated feature names on ONNX export (BCUTd-1l -> BCUTd_1l) but not when looking them
up at inference, so the served model raises KeyError on every request.
Structured after PFAS_albumin_QSAR/Equilibrium_dialysis/ED_QSAR_model.py.
"""

import numpy as np
import pandas as pd
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import (
    MACCSKeysFingerprint,
    MordredDescriptors,
    RDKitDescriptors,
    TopologicalFingerprint,
)
from jaqpotpy.doa import BoundingBox, KernelBased, Leverage, Mahalanobis, MeanVar
from jaqpotpy.models import SklearnModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# --- from the N5 manifest --------------------------------------------------
DATA_PATH = "PFAS_albumin_QSAR/PFAS_HSA_QSAR/HSA_Chen_data.csv"
EXCLUDED_CONGENERS = ["PFMBA", "PFNS", "PFTrA", "FOSA"]
N_COMPOUNDS = 56  # after the exclusions above
SELECTED_FEATURES = ["MIC4", "AATS5d", "BCUTd-1l", "GATS5d", "GATS5v"]
FEATURIZERS = [
    RDKitDescriptors(),
    MordredDescriptors(),
    TopologicalFingerprint(),
    MACCSKeysFingerprint(),
]
REGRESSOR = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    max_features=0.6,
    random_state=42,
    n_jobs=1,
)
DOA = [Leverage(), MeanVar(), BoundingBox(), Mahalanobis(), KernelBased()]

MODEL_NAME = "PFAS HSA binding (Log10Ka)"
MODEL_DESCRIPTION = (
    "Random forest QSAR for human serum albumin binding of PFAS (log10 Ka, M^-1), "
    "trained on the Chen et al. 2025 equilibrium-dialysis dataset."
)
MODEL_VISIBILITY = "PRIVATE"
DEPLOY = False  # see the note above; the upload itself still works, the served model does not

# Half-life PFAS to score with the finished model; LogKa is an input to the half-life model
HALFLIFE_PATH = (
    "PFAS_HalfLife_QSAR/Restructured/data/Half-life_dataset_Human_statistics.csv"
)
KA_RESULTS_PATH = "PFAS_HalfLife_QSAR/Restructured/data/HSA_LogKa_results.csv"

# ---------------------------------------------------------------------------
# Data: the same compounds the manifest was built from
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = df[~df["Congener"].isin(EXCLUDED_CONGENERS)].reset_index(drop=True)
df["Log10Ka"] = np.log10(df["Ka"])  # Ka is already in M^-1
assert len(df) == N_COMPOUNDS, (
    f"{len(df)} compounds after exclusions, expected {N_COMPOUNDS}; "
    "the dataset has changed since the manifest was written"
)

# ---------------------------------------------------------------------------
# Featurization, cut down to the selected features
# ---------------------------------------------------------------------------
dataset = JaqpotTabularDataset(
    df=df[["SMILES", "Log10Ka"]],
    y_cols="Log10Ka",
    x_cols=[],
    smiles_cols="SMILES",
    featurizers=FEATURIZERS,
    task="REGRESSION",
    verbose=False,
)
missing = [c for c in SELECTED_FEATURES if c not in dataset.X.columns]
assert not missing, f"selected features absent from the featurized data: {missing}"
dataset.select_features(SelectColumns=SELECTED_FEATURES)

# ---------------------------------------------------------------------------
# Preprocessing, model and applicability domain
# ---------------------------------------------------------------------------
# The nested preprocessing chain is feature SELECTION apart from the scaling, and the
# selection is already baked into SELECTED_FEATURES; the scaler is the only part that
# has to travel with the deployed model.
preprocessor = ColumnTransformer(
    transformers=[("MinMaxScaler", MinMaxScaler(), SELECTED_FEATURES)],
    remainder="passthrough",
)
model = SklearnModel(
    dataset=dataset,
    model=REGRESSOR,
    preprocess_x=[preprocessor],
    doa=DOA,
)
model.fit()

# ---------------------------------------------------------------------------
# Predict Log10Ka for the half-life PFAS
# ---------------------------------------------------------------------------
# model.predict() applies the preprocessor fitted above, so the scaler stays fitted on the
# HSA training compounds and is only applied to these structures, never refitted on them.
half_life = pd.read_csv(HALFLIFE_PATH)
compounds = (
    half_life.dropna(subset=["SMILES"])[
        ["PFAS", "SMILES"]
    ]  # iso-PFOA/iso-PFOS lack SMILES
    .drop_duplicates()
    .sort_values("PFAS")
    .reset_index(drop=True)
)
assert compounds["SMILES"].is_unique, "a SMILES maps to more than one half-life name"
compounds["Log10Ka"] = 0.0  # the dataset needs the column; the values are never used

prediction_set = JaqpotTabularDataset(
    df=compounds[["SMILES", "Log10Ka"]],
    y_cols="Log10Ka",
    x_cols=[],
    smiles_cols="SMILES",
    featurizers=FEATURIZERS,
    task="REGRESSION",
    verbose=False,
)
missing = [c for c in SELECTED_FEATURES if c not in prediction_set.X.columns]
assert not missing, f"selected features absent for the half-life PFAS: {missing}"
prediction_set.select_features(SelectColumns=SELECTED_FEATURES)

# jaqpotpy writes -1000 where Mordred failed; left alone that silently poisons a prediction
compounds["n_failed_descriptors"] = (prediction_set.X == -1000.0).sum(axis=1).to_numpy()
compounds["LogKa"] = np.asarray(model.predict(prediction_set)).ravel()
compounds["in_hsa_training"] = compounds["SMILES"].isin(df["SMILES"])

compounds[
    ["SMILES", "LogKa", "PFAS", "in_hsa_training", "n_failed_descriptors"]
].to_csv(KA_RESULTS_PATH, index=False)

in_training = int(compounds["in_hsa_training"].sum())
failed = int((compounds["n_failed_descriptors"] > 0).sum())
print(
    f"Half-life Log10Ka -> {KA_RESULTS_PATH} ({len(compounds)} compounds; "
    f"{in_training} are in the HSA training set so their values are in-sample, "
    f"{len(compounds) - in_training} are genuine predictions"
    + (f"; {failed} had a descriptor Mordred could not compute)" if failed else ")")
)

# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------
if DEPLOY:
    from jaqpotpy import Jaqpot

    jaqpot = Jaqpot()
    jaqpot.login()
    model.deploy_on_jaqpot(
        jaqpot=jaqpot,
        name=MODEL_NAME,
        description=MODEL_DESCRIPTION,
        visibility=MODEL_VISIBILITY,
    )
    print(f"Deployed to Jaqpot as {MODEL_NAME!r} ({MODEL_VISIBILITY})")
