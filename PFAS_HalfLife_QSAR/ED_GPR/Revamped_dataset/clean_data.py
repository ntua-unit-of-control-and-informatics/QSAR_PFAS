import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import sys


halflife_df = pd.read_csv(
    "PFAS_HalfLife_QSAR/Datasets/Revamped_dataset/Half-life_dataset_Human_revamped.csv"
)
Ka_data = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Ka_results.csv")

print(halflife_df.columns)

# Drop rows with NaN values because no SMILES found for these rows
halflife_df.dropna(subset=["SMILES"], inplace=True)

halflife_df.drop(
    halflife_df[
        (halflife_df["Study"] == "Zhang et al. 2013") & (halflife_df["sex"] == "both")
    ].index,
    inplace=True,
)
halflife_df.drop(
    labels=[
        "half_life_type",
        "Occupational_exposure",
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
# Sampling for studies with summarized statistics
# Keep only rows where Population_data is False
summarised_halflife_df = halflife_df[halflife_df["Population_data"] == False]
halflife_df = halflife_df[halflife_df["Population_data"] == True]

# Perform outlier detection using IQR on halflife_df
print("\nPerforming outlier detection based on IQR method...")

# Group by SMILES for halflife_df and detect outliers
outliers_halflife = []
for smiles, group in halflife_df.groupby("SMILES"):
    Q1 = group["half_life"].quantile(0.25)
    Q3 = group["half_life"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers in this group
    outlier_indices = group[
        (group["half_life"] < lower_bound) | (group["half_life"] > upper_bound)
    ].index

    if len(outlier_indices) > 0:
        outliers_halflife.extend(outlier_indices)
        print(
            f"Found {len(outlier_indices)} outliers for {smiles} (PFAS: {group['PFAS'].iloc[0]})"
        )
        print(f"  Range: {lower_bound:.2f} to {upper_bound:.2f}")
        print(f"  Outlier values: {group.loc[outlier_indices, 'half_life'].values}")

# Group by SMILES for summarised_halflife_df and detect outliers
outliers_summarised = []
for smiles, group in summarised_halflife_df.groupby("SMILES"):
    Q1 = group["half_life"].quantile(0.25)
    Q3 = group["half_life"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers in this group
    outlier_indices = group[
        (group["half_life"] < lower_bound) | (group["half_life"] > upper_bound)
    ].index

    if len(outlier_indices) > 0:
        outliers_summarised.extend(outlier_indices)
        print(
            f"Found {len(outlier_indices)} outliers in summarised data for {smiles} (PFAS: {group['PFAS'].iloc[0]})"
        )
        print(f"  Range: {lower_bound:.2f} to {upper_bound:.2f}")
        print(f"  Outlier values: {group.loc[outlier_indices, 'half_life'].values}")

print(f"\nTotal outliers found in halflife_df: {len(outliers_halflife)}")
print(f"Total outliers found in summarised_halflife_df: {len(outliers_summarised)}")

# Remove outliers
halflife_df = halflife_df.drop(outliers_halflife)
summarised_halflife_df = summarised_halflife_df.drop(outliers_summarised)

print(
    f"After removing outliers: halflife_df shape: {halflife_df.shape}, summarised_halflife_df shape: {summarised_halflife_df.shape}"
)


sample_generated_data = pd.DataFrame()

for row in summarised_halflife_df.itertuples():
    # Generate a sample of size n from alognormal distribution
    mu = row.half_life
    sd = row.half_life_sd
    mu_log = np.log(mu**2 / np.sqrt(mu**2 + sd**2))
    sd_log = np.sqrt(np.log(1 + sd**2 / mu**2))
    mu_log = np.log(mu**2 / np.sqrt(mu**2 + sd**2))
    sd_log = np.sqrt(np.log(1 + sd**2 / mu**2))
    n = int(row.N_individuals / 2)
    sample = np.random.lognormal(mu_log, sd_log, n)

    # Create a dataframe for each sample with all columns from the original row
    for half_life_value in sample:
        # Create a new row with all columns from the original dataframe
        new_row = {
            col: getattr(row, col) for col in halflife_df.columns if col != "Index"
        }

        # Update the values that need to change
        new_row["Population_data"] = True  # Now individual data
        new_row["N_individuals"] = 1  # Individual data point
        new_row["half_life"] = half_life_value
        new_row["half_life_sd"] = 0  # No SD for individual points

        # Add the new row to the sample_generated_data
        sample_generated_data = pd.concat(
            [sample_generated_data, pd.DataFrame([new_row])], ignore_index=True
        )

# Add the generated sample data to halflife_df
halflife_df = pd.concat([halflife_df, sample_generated_data])

# Group by SMILES and calculate statistics
grouped = halflife_df.groupby("SMILES")
halflife_stats = grouped.agg(
    half_life_mean=("half_life", "mean"),
    half_life_sd=("half_life", "std"),
    count=("half_life", "count"),
)
# Transform the GroupBy result to a classic DataFrame with reset_index
halflife_stats = halflife_stats.reset_index()

# Create a mapping from SMILES to PFAS name
smiles_to_pfas = (
    halflife_df[["SMILES", "PFAS"]]
    .drop_duplicates(subset=["SMILES"])
    .set_index("SMILES")["PFAS"]
)

# Add PFAS column to the statistics DataFrame
halflife_stats["PFAS"] = halflife_stats["SMILES"].map(smiles_to_pfas)

# Add Ka values to the halflife_stats DataFrame
# First, create a mapping from SMILES to LogKa
smiles_to_ka = (
    halflife_df[["SMILES", "LogKa"]]
    .drop_duplicates(subset=["SMILES"])
    .set_index("SMILES")["LogKa"]
)
halflife_stats["SMILES"]
# Add Ka column to the statistics DataFrame
halflife_stats["LogKa"] = halflife_stats["SMILES"].map(smiles_to_ka)

# Fill NaN values in standard deviation (occurs when only one data point exists)
halflife_stats["half_life_sd"] = halflife_stats["half_life_sd"].fillna(0)
print(halflife_stats)
halflife_stats.to_csv(
    "PFAS_HalfLife_QSAR/Datasets/Revamped_dataset/Halflife_stats.csv",
    index=False,
)
# Create a new dataframe by sampling from lognormal distribution for each PFAS
sampled_halflife_df = pd.DataFrame()

for row in halflife_stats.itertuples():
    # Get parameters for lognormal distribution
    mu = row.half_life_mean
    sd = row.half_life_sd if row.half_life_sd > 0 else 0.1  # Avoid zero SD
    n = 50  # row.count  # Number of samples to generate

    # Calculate lognormal parameters
    mu_log = np.log(mu**2 / np.sqrt(mu**2 + sd**2))
    sd_log = np.sqrt(np.log(1 + sd**2 / mu**2))

    # Generate samples from lognormal distribution
    samples = np.random.lognormal(mu_log, sd_log, int(n))

    # Create dataframe with samples
    for half_life_value in samples:
        new_row = {
            "SMILES": row.SMILES,
            "PFAS": row.PFAS,
            "LogKa": row.LogKa,
            "half_life": half_life_value,
        }
        sampled_halflife_df = pd.concat(
            [sampled_halflife_df, pd.DataFrame([new_row])], ignore_index=True
        )

# Print basic statistics of the sampled data
print(
    f"Generated {len(sampled_halflife_df)} samples from {len(halflife_stats)} PFAS compounds"
)
halflife_df = sampled_halflife_df

# Random train-test split of the dataset
# Sort the dataframe by half_life_days
halflife_df = halflife_df.sort_values(by="half_life")
# Remove the last row of the dataframe
halflife_df = halflife_df.iloc[:-1]
# Split the dataframe into train and test sets based on the rule
train_indices = []
test_indices = []

# Iterate over the dataframe in chunks of 10
for i in range(0, len(halflife_df), 10):
    chunk = halflife_df.iloc[i : i + 10]
    test_chunk = chunk.sample(
        frac=0.30, random_state=0
    )  # Randomly select 20% (2 out of 10) for test
    train_chunk = chunk.drop(test_chunk.index)  # The rest go to train

    train_indices.extend(train_chunk.index)
    test_indices.extend(test_chunk.index)

# Create train and test dataframes
train_df = halflife_df.loc[train_indices].copy()
test_df = halflife_df.loc[test_indices].copy()

# Create a scatter plot of the train and test values of half_life with different colors
plt.figure(figsize=(10, 6))
plt.scatter(train_df["half_life"], train_df["PFAS"], color="blue", label="Train")
plt.scatter(test_df["half_life"], test_df["PFAS"], color="red", label="Test")
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
    "PFAS_HalfLife_QSAR/Datasets/Revamped_dataset/Half-life_dataset_Human_train.csv",
    index=False,
)
test_df.to_csv(
    "PFAS_HalfLife_QSAR/Datasets/Revamped_dataset/Half-life_dataset_Human_test.csv",
    index=False,
)
