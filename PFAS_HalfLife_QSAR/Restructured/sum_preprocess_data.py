import json

import pandas as pd
import numpy as np
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import (
    MACCSKeysFingerprint,
    MordredDescriptors,
    RDKitDescriptors,
    TopologicalFingerprint,
)

# The four featurizers used by the HSA and LFABP binding models, so the half-life
# model draws candidates from the same descriptor space. Mordred and MACCS are new
# here; the dataset previously carried only RDKit descriptors and topological bits.
FEATURIZERS = [
    RDKitDescriptors(),
    MordredDescriptors(),
    TopologicalFingerprint(),
    MACCSKeysFingerprint(),
]
FEATURIZER_LABELS = {
    "RDKitDescriptors": "rdkit",
    "MordredDescriptors": "mordred",
    "TopologicalFingerprint": "topfp",
    "MACCSKeysFingerprint": "maccs",
}
# Written alongside the dataset. Downstream scripts need to know which columns are
# 0/1 fingerprint bits and which are continuous descriptors, and that is no longer
# recoverable from the names: with Mordred in the mix, matching against
# rdkit.Chem.Descriptors.descList would classify ~1800 continuous Mordred
# descriptors as fingerprint bits.
COLUMNS_PATH = "PFAS_HalfLife_QSAR/Restructured/data/sum_featurized_columns.json"
LOGKA_COLUMNS = ["LogKa_HSA", "LogKa_LFABP"]


halflife_df = pd.read_csv(
    "PFAS_HalfLife_QSAR/Restructured/data/Half-life_dataset_Human_statistics.csv"
)
hsa_Ka_data = pd.read_csv("PFAS_HalfLife_QSAR/Restructured/data/HSA_LogKa_results.csv")
lfabp_Ka_data = pd.read_csv(
    "PFAS_HalfLife_QSAR/Restructured/data/LFABP_LogKa_results.csv"
)

# Estimate the log10 half-life values (Assuming log normal distribution for half-life)
halflife_df["log_half_life"] = np.log10(halflife_df["half_life"])

# Drop rows with NaN values because no SMILES found for these rows
halflife_df.dropna(subset=["SMILES"], inplace=True)

halflife_df.drop(
    halflife_df[
        (halflife_df["Study"] == "Zhang et al. 2013") & (halflife_df["sex"] == "both")
    ].index,
    inplace=True,
)

halflife_df.drop(
    halflife_df[(halflife_df["Study"] == "Fu et al. 2016")].index,
    inplace=True,
)

halflife_df.drop(
    labels=[
        "half_life_days",
        "DOI",
        "tissue",
        "sex",
        "age (years)",
        "adult",
    ],
    axis=1,
    inplace=True,
)

# Add the two binding predictions, each named after its protein so the featurized
# dataset says which is which. Only SMILES -> LogKa is taken from each file: both
# also carry PFAS, in_*_training and n_failed_descriptors, and their PFAS column
# would collide with the half-life one and rename both to PFAS_x / PFAS_y.
halflife_df = halflife_df.merge(
    hsa_Ka_data[["SMILES", "LogKa"]].rename(columns={"LogKa": "LogKa_HSA"}),
    on="SMILES",
    how="left",
)
halflife_df = halflife_df.merge(
    lfabp_Ka_data[["SMILES", "LogKa"]].rename(columns={"LogKa": "LogKa_LFABP"}),
    on="SMILES",
    how="left",
)
for col, source in [
    ("LogKa_HSA", "HSA_deploy_model.py"),
    ("LogKa_LFABP", "LFABP_deploy_model.py"),
]:
    missing = halflife_df.loc[halflife_df[col].isna(), "PFAS"].unique()
    assert len(missing) == 0, f"no {col} for {sorted(missing)}; rerun {source}"

# --- Impute missing SE via coefficient-of-variation (CV = half_life_sd / half_life) ---
# Runs after this script's own row-cleaning (SMILES dropna, Zhang "both"-sex
# removal, Fu et al. 2016 removal) so the CV source pool only draws from
# rows actually used downstream, not from data this script already excluded.
# In-memory only - does not modify Half-life_dataset_Human_statistics.csv.
missing_se_mask = halflife_df["SE"].isna()
print(f"Rows with missing SE: {missing_se_mask.sum()}")
print(
    halflife_df.loc[
        missing_se_mask, ["Study", "PFAS", "half_life", "N_individuals"]
    ].to_string()
)

halflife_df["SE_method"] = np.where(missing_se_mask, None, "reported")

cv_source = halflife_df[
    halflife_df["half_life_sd"].notna() & halflife_df["half_life"].notna()
].copy()
cv_source["CV"] = cv_source["half_life_sd"] / cv_source["half_life"]
global_cv = cv_source["CV"].median()

se_summary_rows = []
for idx in halflife_df.index[missing_se_mask]:
    pfas = halflife_df.at[idx, "PFAS"]
    hl = halflife_df.at[idx, "half_life"]
    n_individuals = halflife_df.at[idx, "N_individuals"]

    compound_cv = cv_source.loc[
        (cv_source["PFAS"] == pfas) & (cv_source.index != idx), "CV"
    ]

    if len(compound_cv) > 0:
        cv_value = compound_cv.median()
        cv_label = "compound-specific"
        method = "imputed_CV_compound_specific"
    else:
        cv_value = global_cv
        cv_label = "global"
        method = "imputed_CV_global"

    imputed_sd = cv_value * hl
    se = imputed_sd / np.sqrt(n_individuals)

    halflife_df.at[idx, "SE"] = se
    halflife_df.at[idx, "SE_method"] = method

    se_summary_rows.append(
        {
            "PFAS": pfas,
            "half_life": hl,
            "CV_source": cv_label,
            "CV_value": cv_value,
            "imputed_SE": se,
        }
    )

print("\nImputed SE summary:")
print(pd.DataFrame(se_summary_rows).to_string(index=False))
# --- end SE imputation ---

# Prepare featurization

columns_to_keep = [
    "Study",
    "PFAS",
    "SMILES",
    "half_life_type",
    "Occupational_exposure",
    "LogKa_HSA",
    "LogKa_LFABP",
    "half_life",
    "SE",
    "SE_method",
]
halflife_combined_df = halflife_df[columns_to_keep].copy()
halflife_combined_df["log_half_life"] = np.log10(halflife_combined_df["half_life"])

# Calculate descriptors of the dataset.
#
# Each featurizer runs on its own so that the continuous/binary provenance of every
# column is known by construction rather than guessed from its name. Descriptors are
# a property of the structure alone, with nothing fitted, so computing them for every
# compound at once leaks nothing.
def featurize(smiles):
    blocks, binary_cols, seen = [], [], set()
    frame = pd.DataFrame({"SMILES": smiles, "_y": 0.0})
    for featurizer in FEATURIZERS:
        name = type(featurizer).__name__
        label = FEATURIZER_LABELS[name]
        X = JaqpotTabularDataset(
            df=frame,
            y_cols="_y",
            x_cols=[],
            smiles_cols="SMILES",
            featurizers=[featurizer],
            task="REGRESSION",
            verbose=False,
        ).X
        if X.shape[1] == 0:
            raise RuntimeError(
                f"{name} returned no descriptors; is its package installed?"
            )
        if isinstance(featurizer, MordredDescriptors):
            # jaqpotpy encodes a failed Mordred descriptor as -1000, which would
            # otherwise be read as a real and very extreme value
            X = X.replace(-1000.0, np.nan)
        if isinstance(featurizer, MACCSKeysFingerprint):
            # jaqpotpy names these f0..f166; MACCS keys have published definitions,
            # so a readable name is worth having in the downstream outputs
            X = X.rename(columns=lambda c: f"MACCS_{c.removeprefix('f')}")
        # A name produced by two featurizers gets the later one's label appended
        X = X.rename(columns=lambda c: f"{c}_{label}" if c in seen else c)
        seen.update(X.columns)
        if isinstance(featurizer, (TopologicalFingerprint, MACCSKeysFingerprint)):
            binary_cols += list(X.columns)
        blocks.append(X)
    return pd.concat(blocks, axis=1).astype(float), binary_cols


# Featurize each distinct structure once, not each row: compounds repeat across
# studies (PFOA appears 11 times), and Mordred is far too slow to run redundantly.
unique_smiles = (
    halflife_combined_df["SMILES"].drop_duplicates().reset_index(drop=True)
)
features_unique, binary_cols = featurize(unique_smiles)
features_unique.insert(0, "SMILES", unique_smiles)
n_descriptors = features_unique.shape[1] - 1
print(
    f"\nFeaturized {len(unique_smiles)} distinct structures: {n_descriptors} "
    f"descriptors ({len(binary_cols)} binary, "
    f"{n_descriptors - len(binary_cols)} continuous)"
)

features = (
    halflife_combined_df[["SMILES"]]
    .merge(features_unique, on="SMILES", how="left")
    .drop(columns="SMILES")
)
assert len(features) == len(halflife_combined_df), "featurization changed the row count"
featurized_dataset = pd.concat(
    [halflife_combined_df.reset_index(drop=True), features], axis=1
)

# Continuous features include the two binding predictions: they are descriptor-derived
# per-compound values and are averaged with the descriptors when isomers are merged.
binary_cols = [c for c in binary_cols if c in featurized_dataset.columns]
continuous_cols = [
    c
    for c in featurized_dataset.columns
    if c not in set(binary_cols) | set(columns_to_keep) | {"log_half_life"}
] + LOGKA_COLUMNS

# Merge isomers that share the same log_half_life within a study
isomer_groups = {
    "group1": ["3mPFOS", "4mPFOS", "5mPFOS"],
    "group2": ["2mPFOS", "6mPFOS"],
}

ISOMER_MERGE_KEY = [
    "Study",
    "log_half_life",
    "half_life",
    "half_life_type",
    "Occupational_exposure",
]


def merge_isomer_group(df, isomer_names):
    isomer_mask = df["PFAS"].isin(isomer_names)
    isomer_df = df[isomer_mask].copy()
    rest_df = df[~isomer_mask]

    def aggregate(group):
        if len(group) == 1:
            return group
        row = group.iloc[[0]].copy()
        row[continuous_cols] = group[continuous_cols].mean().values
        row[binary_cols] = group[binary_cols].max().values
        return row

    merged = (
        isomer_df.groupby(ISOMER_MERGE_KEY, group_keys=False)
        .apply(aggregate)
        .reset_index(drop=True)
    )
    return pd.concat([rest_df, merged], ignore_index=True)


for group_name, isomer_names in isomer_groups.items():
    featurized_dataset = merge_isomer_group(featurized_dataset, isomer_names)

featurized_dataset.to_csv(
    "PFAS_HalfLife_QSAR/Restructured/data/sum_featurized_dataset.csv", index=False
)
with open(COLUMNS_PATH, "w") as f:
    json.dump(
        {
            "featurizers": [type(x).__name__ for x in FEATURIZERS],
            "metadata": columns_to_keep + ["log_half_life"],
            "continuous": continuous_cols,
            "binary": binary_cols,
            "logka": LOGKA_COLUMNS,
        },
        f,
        indent=2,
    )
print(
    f"\nWrote {len(featurized_dataset)} rows x {featurized_dataset.shape[1]} columns; "
    f"column provenance -> {COLUMNS_PATH}"
)
