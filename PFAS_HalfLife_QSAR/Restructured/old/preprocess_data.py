import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import RDKitDescriptors, TopologicalFingerprint


halflife_df = pd.read_csv("Restructured/data/Half-life_dataset_Human_revamped.csv")
Ka_data = pd.read_csv("Restructured/data/Ka_results.csv")

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


# halflife_df.drop(
#     halflife_df[(halflife_df["PFAS"] == "6:2FTS")].index,
#     inplace=True,
# )


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
halflife_df.drop(
    halflife_df[halflife_df["Study"].isin(["Arnot et al. 2014"])].index,
    inplace=True,
)

# Add Ka values to the dataset
halflife_df = halflife_df.merge(Ka_data, on="SMILES", how="left")

# Sumarise studies that report individual measurements
halflife_individual_df = halflife_df[halflife_df["Individual_data"] == True]
halflife_not_individual_df = halflife_df[halflife_df["Individual_data"] == False]


def iqr_filter(group):
    if len(group) <= 10:
        return group
    q1 = group["log_half_life"].quantile(0.25)
    q3 = group["log_half_life"].quantile(0.75)
    iqr = q3 - q1
    return group[
        (group["log_half_life"] >= q1 - 1.5 * iqr)
        & (group["log_half_life"] <= q3 + 1.5 * iqr)
    ]


GROUP_COLS = ["Study", "PFAS", "half_life_type", "Occupational_exposure"]
n_before = halflife_individual_df.groupby(by=GROUP_COLS)["half_life"].count()

halflife_individual_df = (
    halflife_individual_df.groupby(by=GROUP_COLS, group_keys=False)
    .apply(iqr_filter)
    .reset_index(drop=True)
)

grouped = halflife_individual_df.groupby(by=GROUP_COLS)
summarised_halflife_individual_df = grouped[["half_life"]].mean()
summarised_halflife_individual_df["SMILES"] = grouped["SMILES"].first()
summarised_halflife_individual_df["LogKa"] = grouped["LogKa"].first()
summarised_halflife_individual_df["half_life_sd"] = grouped["half_life"].std()
summarised_halflife_individual_df["N_individuals"] = grouped["half_life"].count()
summarised_halflife_individual_df["N_excluded"] = (
    n_before - summarised_halflife_individual_df["N_individuals"]
)

summarised_halflife_individual_df = summarised_halflife_individual_df.reset_index()
summarised_halflife_individual_df.loc[
    summarised_halflife_individual_df["N_individuals"] <= 3, "half_life_sd"
] = float("nan")

halflife_combined_df = pd.concat(
    [halflife_not_individual_df, summarised_halflife_individual_df],
    ignore_index=True,
)


halflife_combined_df.to_csv("Restructured/data/merged_df.csv", index=False)

# Prepare featurization

columns_to_keep = [
    "Study",
    "PFAS",
    "SMILES",
    "half_life_type",
    "Occupational_exposure",
    "LogKa",
    "half_life",
]
halflife_combined_df = halflife_combined_df[columns_to_keep]
halflife_combined_df["log_half_life"] = np.log10(halflife_combined_df["half_life"])

# Create a mapping from SMILES to PFAS name
smiles_to_pfas = (
    halflife_combined_df[["SMILES", "PFAS"]]
    .drop_duplicates(subset=["SMILES"])
    .set_index("SMILES")["PFAS"]
)

# Calculate descriptors of the dataset
featurizers = [RDKitDescriptors(), TopologicalFingerprint()]

halflife_jp = JaqpotTabularDataset(
    df=halflife_combined_df,
    x_cols=["Study", "PFAS", "half_life_type", "Occupational_exposure", "LogKa"],
    y_cols=["log_half_life"],
    smiles_cols=["SMILES"],
    featurizers=featurizers,
    task="REGRESSION",
    verbose=False,
)

featurized_dataset = halflife_jp.df

# Identify RDKit descriptor vs ECFP fingerprint columns by name
rdkit_descriptor_names = {name for name, _ in Descriptors.descList}
rdkit_cols = [c for c in featurized_dataset.columns if c in rdkit_descriptor_names]
ecfp_cols = [c for c in featurized_dataset.columns if c.startswith("Bit_")]

# Merge isomers that share the same log_half_life within a study
isomer_groups = {
    "group1": ["3mPFOS", "4mPFOS", "5mPFOS"],
    "group2": ["2mPFOS", "6mPFOS"],
}

ISOMER_MERGE_KEY = ["Study", "log_half_life", "half_life_type", "Occupational_exposure"]


def merge_isomer_group(df, isomer_names):
    isomer_mask = df["PFAS"].isin(isomer_names)
    isomer_df = df[isomer_mask].copy()
    rest_df = df[~isomer_mask]

    def aggregate(group):
        if len(group) == 1:
            return group
        row = group.iloc[[0]].copy()
        row[rdkit_cols] = group[rdkit_cols].mean().values
        row[ecfp_cols] = group[ecfp_cols].max().values
        return row

    merged = (
        isomer_df.groupby(ISOMER_MERGE_KEY, group_keys=False)
        .apply(aggregate)
        .reset_index(drop=True)
    )
    return pd.concat([rest_df, merged], ignore_index=True)


for group_name, isomer_names in isomer_groups.items():
    featurized_dataset = merge_isomer_group(featurized_dataset, isomer_names)

featurized_dataset.to_csv("Restructured/data/featurized_dataset.csv", index=False)
