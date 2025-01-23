import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

halflife_df = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human.csv")
Ka_data = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Ka_results.csv")

# Drop rows with NaN values because no SMILES found for these rows
halflife_df.dropna(subset=["SMILES"], inplace=True)
halflife_df.drop(labels=["age (years)"], axis=1, inplace=True)
halflife_df.drop(
    halflife_df[halflife_df["Study"].isin(["Arnot et al. 2014"])].index,
    inplace=True,
)

# Add Ka values to the dataset
halflife_df = halflife_df.merge(Ka_data, on="SMILES", how="left")

# Tranform half-life to years
halflife_df["half_life_days"] = halflife_df["half_life_days"]

halflife_df = halflife_df[halflife_df["half_life_days"] > 15 * 365]
print(halflife_df[["Study", "PFAS", "sex", "adult", "half_life_days"]])
