import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

halflife_df = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human.csv")
Ka_data = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Ka_results.csv")

# Drop row with half_life == 149496.70
halflife_df = halflife_df[halflife_df["half_life_days"] != 149496.70]

# Drop rows with NaN values because no SMILES found for these rows
halflife_df.dropna(subset=["SMILES"], inplace=True)
halflife_df.drop(labels=["age (years)"], axis=1, inplace=True)
halflife_df.drop(
    halflife_df[halflife_df["Study"].isin(["Arnot et al. 2014"])].index,
    inplace=True,
)

# Add Ka values to the dataset
halflife_df = halflife_df.merge(Ka_data, on="SMILES", how="left")

# Group data by PFAS and calculate z-scores within each group
grouped = halflife_df.groupby("PFAS")
halflife_df["z_score"] = grouped["half_life_days"].transform(
    lambda x: stats.zscore(x, nan_policy="omit")
)

# Outliers detection with z-score per group
halflife_df["abs_z_score"] = abs(halflife_df["z_score"])
filtered_entries = halflife_df["abs_z_score"] > 3

# Separate outliers into a different dataframe
outliers_df = halflife_df[filtered_entries].copy()
print("Outliers:", outliers_df.shape)
# Remove outliers from the original dataframe
halflife_df = halflife_df[~filtered_entries]

# Drop the z_score and abs_z_score columns as they are no longer needed
halflife_df.drop(columns=["z_score", "abs_z_score"], inplace=True)

# Tranform half-life to years
halflife_df["half_life_days"] = halflife_df["half_life_days"] / 365
# Drop rows with half_life_days > 15*365
halflife_df = halflife_df[halflife_df["half_life_days"] <= 40]

if halflife_df.isnull().values.any():
    print("The dataframe contains NaN values.")
    print(halflife_df[halflife_df.isnull().any(axis=1)])
else:
    print("The dataframe does not contain any NaN values.")

# Transform all trues of 'adult' column to 1 and false to 0
halflife_df["adult"] = halflife_df["adult"].astype(int)
halflife_df["Occupational_exposure"] = halflife_df["Occupational_exposure"].astype(int)


# Number of studies included in dataset
N_studies = halflife_df["Study"].nunique()
print("Number of studies included in dataset:", N_studies)

# Create a dataframe that shows the number of unique PFAS and rows included in each study
study_summary_df = (
    halflife_df.groupby("Study")
    .agg(unique_PFAS=("PFAS", "nunique"), rows_included=("PFAS", "size"))
    .reset_index()
)
print(study_summary_df)
# Create a barplot that shows the number of data instances per PFAS
# pfas_counts = halflife_df["PFAS"].value_counts()
# plt.figure(figsize=(10, 6))
# pfas_counts.plot(kind="bar")
# plt.title("Number of Data Instances per PFAS")
# plt.xlabel("PFAS")
# plt.ylabel("Number of Instances")
# plt.xticks(rotation=90)
# plt.show()

# Random train-test split of the dataset
train_df, test_df = train_test_split(halflife_df, test_size=0.2, random_state=42)

# Keep specific Studies for test dataset
# test_studies = ["Abraham et al. 2024", "Li et al. 2022"]
# train_df = halflife_df[~halflife_df["Study"].isin(test_studies)].reset_index(drop=True)
# test_df = halflife_df[halflife_df["Study"].isin(test_studies)].reset_index(drop=True)

print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)
print("Test data percentage %:", test_df.shape[0] / halflife_df.shape[0] * 100)
print("Unique PFAS in train dataset:", train_df["PFAS"].nunique())
print("Unique PFAS in test dataset:", test_df["PFAS"].nunique())
# Print PFAS in test that are not included in train
pfas_in_train = set(train_df["PFAS"])
pfas_in_test = set(test_df["PFAS"])
pfas_not_in_train = pfas_in_test - pfas_in_train
print("PFAS in test that are not included in train:", pfas_not_in_train)

# Save the train and test datasets
train_df.to_csv(
    "PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human_train.csv", index=False
)
test_df.to_csv(
    "PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human_test.csv", index=False
)
