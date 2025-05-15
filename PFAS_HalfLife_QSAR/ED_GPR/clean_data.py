import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import sys


halflife_df = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human.csv")
Ka_data = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Ka_results.csv")

# Drop row with half_life == 149496.70

# Drop rows with NaN values because no SMILES found for these rows
halflife_df.dropna(subset=["SMILES"], inplace=True)
halflife_df.drop(labels=["age (years)"], axis=1, inplace=True)
halflife_df.drop(
    halflife_df[halflife_df["Study"].isin(["Arnot et al. 2014"])].index,
    inplace=True,
)
# Drop rows that are both from Zhang et al. 2013 study AND have sex as 'both'
halflife_df.drop(
    halflife_df[
        (halflife_df["Study"] == "Zhang et al. 2013") & (halflife_df["sex"] == "both")
    ].index,
    inplace=True,
)
# halflife_df.drop(
#     halflife_df[halflife_df["half_life_type"].isin(["apparent"])].index, inplace=True
# )
# halflife_df.drop(columns=["half_life_type"], inplace=True)
# Add Ka values to the dataset
halflife_df = halflife_df.merge(Ka_data, on="SMILES", how="left")

# Group data by both SMILES and half_life_type
grouped = halflife_df.groupby(["SMILES", "half_life_type"])

# Calculate the 0.25 and 0.75 quantiles for each group
quantiles = grouped["half_life_days"].quantile([0.25, 0.75]).unstack()
quantiles.columns = ["q25", "q75"]
quantiles["IQR"] = quantiles["q75"] - quantiles["q25"]

# Reset index to get SMILES and half_life_type as columns
quantiles = quantiles.reset_index()

# Merge quantiles back to the original dataframe
halflife_df = pd.merge(
    halflife_df, quantiles, on=["SMILES", "half_life_type"], how="left"
)

# Filter out rows outside the 0.25 and 0.75 quantiles
filtered_entries = (
    halflife_df["half_life_days"] < (halflife_df["q25"] - 1.5 * halflife_df["IQR"])
) | (halflife_df["half_life_days"] > (halflife_df["q75"] + 1.5 * halflife_df["IQR"]))

# Separate outliers into a different dataframe
outliers_df = halflife_df[filtered_entries].copy()
print(f"Original data: {len(halflife_df)} rows")
print(f"Outliers: {len(outliers_df)} rows")

# Remove outliers from the original dataframe
halflife_df = halflife_df[~filtered_entries]
print(f"After outlier removal: {len(halflife_df)} rows")

# Drop the quantile columns as they are no longer needed
halflife_df.drop(columns=["q25", "q75", "IQR"], inplace=True)

# Tranform half-life to years
halflife_df["half_life_days"] = halflife_df["half_life_days"] / 365


if halflife_df.isnull().values.any():
    print("The dataframe contains NaN values.")
    print(halflife_df[halflife_df.isnull().any(axis=1)])
else:
    print("The dataframe does not contain any NaN values.")

# Transform all trues of 'adult' column to 1 and false to 0
halflife_df["adult"] = halflife_df["adult"].astype(object)
halflife_df["Occupational_exposure"] = halflife_df["Occupational_exposure"].astype(
    object
)

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
# Sort the dataframe by half_life_days
halflife_df = halflife_df.sort_values(by="half_life_days")
# Remove the last row of the dataframe
halflife_df = halflife_df.iloc[:-1]
# Split the dataframe into train and test sets based on the rule
train_indices = []
test_indices = []

# Iterate over the dataframe in chunks of 10
for i in range(0, len(halflife_df), 10):
    chunk = halflife_df.iloc[i : i + 10]
    test_chunk = chunk.sample(
        frac=0.25, random_state=0
    )  # Randomly select 20% (2 out of 10) for test
    train_chunk = chunk.drop(test_chunk.index)  # The rest go to train

    # Ensure no unseen SMILES in test dataset
    # unseen_smiles = set(test_chunk["SMILES"]) - set(train_chunk["SMILES"])
    # if unseen_smiles:
    #     # Move rows with unseen SMILES from test_chunk to train_chunk
    #     for smiles in unseen_smiles:
    #         rows_to_move = test_chunk[test_chunk["SMILES"] == smiles]
    #         train_chunk = pd.concat([train_chunk, rows_to_move])
    #         test_chunk = test_chunk[test_chunk["SMILES"] != smiles]

    train_indices.extend(train_chunk.index)
    test_indices.extend(test_chunk.index)

# Create train and test dataframes
train_df = halflife_df.loc[train_indices].copy()
test_df = halflife_df.loc[test_indices].copy()
# train_df, test_df = train_test_split(halflife_df, test_size=0.2, random_state=0)

# Create a scatter plot of the train and test values of half_life with different colors
plt.figure(figsize=(10, 6))
plt.scatter(train_df["half_life_days"], train_df["PFAS"], color="blue", label="Train")
plt.scatter(test_df["half_life_days"], test_df["PFAS"], color="red", label="Test")
plt.title("Scatter Plot of Half-life Values")
plt.xlabel("Half-life (years)")
plt.ylabel("PFAS")
plt.legend()
plt.show()


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
