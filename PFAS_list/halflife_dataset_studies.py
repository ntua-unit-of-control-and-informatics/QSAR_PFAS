import pandas as pd
import numpy as np

halflife_dataset = pd.read_csv(
    "/Users/vassilis/Documents/GitHub/QSAR_PFAS/PFAS_HalfLife_QSAR/Datasets/Revamped_dataset/Half-life_dataset_Human_revamped.csv"
)
halflife_dataset = halflife_dataset.drop(
    halflife_dataset[halflife_dataset["Study"] == "Arnot et al. 2014"].index
)

studies_list = halflife_dataset["Study"].unique().tolist()

metadata_df = pd.DataFrame(columns=["Study", "Doi", "Congeners"])

for study in studies_list:
    study_df = halflife_dataset[halflife_dataset["Study"] == study]
    doi = study_df["DOI"].values[0]
    # remove https://doi.org/ prefix if present
    if doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "")
    congeners = study_df["PFAS"].unique().tolist()
    metadata_df = pd.concat(
        [
            metadata_df,
            pd.DataFrame({"Study": [study], "Doi": [doi], "Congeners": [congeners]}),
        ],
        ignore_index=True,
    )

metadata_df.to_csv("PFAS_list/Half-life_dataset_studies_metadata.csv", index=False)
