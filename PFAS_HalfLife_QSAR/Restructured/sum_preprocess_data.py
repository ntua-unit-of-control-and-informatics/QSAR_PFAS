import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import RDKitDescriptors, TopologicalFingerprint


halflife_df = pd.read_csv("Restructured/data/Half-life_dataset_Human_statistics.csv")
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

# Add Ka values to the dataset
halflife_df = halflife_df.merge(Ka_data, on="SMILES", how="left")

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
    "LogKa",
    "half_life",
    "SE",
    "SE_method",
]
halflife_combined_df = halflife_df[columns_to_keep]
halflife_combined_df["log_half_life"] = np.log10(halflife_combined_df["half_life"])

# Calculate descriptors of the dataset
featurizers = [RDKitDescriptors(), TopologicalFingerprint()]

halflife_jp = JaqpotTabularDataset(
    df=halflife_combined_df,
    x_cols=[
        "Study",
        "PFAS",
        "half_life_type",
        "Occupational_exposure",
        "LogKa",
        "half_life",
        "SE",
        "SE_method",
    ],
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

featurized_dataset.to_csv("Restructured/data/sum_featurized_dataset.csv", index=False)
