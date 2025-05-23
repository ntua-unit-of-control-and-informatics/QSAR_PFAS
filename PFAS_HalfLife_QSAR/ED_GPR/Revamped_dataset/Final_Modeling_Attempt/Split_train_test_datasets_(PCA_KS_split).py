import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from numpy.random import RandomState
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import RDKitDescriptors, TopologicalFingerprint
from kennard_stone import train_test_split as ks_train_test_split
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# This script is to split the data based on the distibutions of features and not on
# the target variable.

halflife_df = pd.read_csv(
    "PFAS_HalfLife_QSAR/Datasets/Revamped_dataset/Half-life_dataset_Human_revamped.csv"
)
# halflife_df.drop(halflife_df[halflife_df["PFAS"] == "PFHxS"].index, inplace=True)
Ka_data = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Ka_results.csv")

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

halflife_df["log_half_life"] = np.log(halflife_df["half_life"])

# Create a histogram to better visualize the distribution
# plt.figure(figsize=(12, 6))
# sns.histplot(data=halflife_df, x="log_half_life", bins=30, kde=True)
# plt.xlabel("Log Half Life (years)")
# plt.ylabel("Frequency")
# plt.title("Histogram of Half Life Values")
# plt.tight_layout()
# plt.show()

# Group by SMILES for halflife_df and detect outliers
outliers_halflife = []
for smiles, group in halflife_df.groupby("SMILES"):
    Q1 = group["log_half_life"].quantile(0.25)
    Q3 = group["log_half_life"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers in this group
    outlier_indices = group[
        (group["log_half_life"] < lower_bound) | (group["log_half_life"] > upper_bound)
    ].index

    if len(outlier_indices) > 0:
        outliers_halflife.extend(outlier_indices)
        print(
            f"Found {len(outlier_indices)} outliers for {smiles} (PFAS: {group['PFAS'].iloc[0]})"
        )
        print(f"  Range: {lower_bound:.2f} to {upper_bound:.2f}")
        print(f"  Outlier values: {group.loc[outlier_indices, 'half_life'].values}")


# Proceed with summarised Half-life data
# Group by SMILES for summarised_halflife_df and detect outliers
## Create a histogram to better visualize the distribution
summarised_halflife_df["log_half_life"] = np.log(summarised_halflife_df["half_life"])

# plt.figure(figsize=(12, 6))
# sns.histplot(data=summarised_halflife_df, x="log_half_life", bins=30, kde=True)
# plt.xlabel("Half Life (years)")
# plt.ylabel("Frequency")
# plt.title("Histogram of Half Life Values")
# plt.tight_layout()
# plt.show()

outliers_summarised = []
for smiles, group in summarised_halflife_df.groupby("SMILES"):
    Q1 = group["log_half_life"].quantile(0.25)
    Q3 = group["log_half_life"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers in this group
    outlier_indices = group[
        (group["log_half_life"] < lower_bound) | (group["log_half_life"] > upper_bound)
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

sample_generated_data = pd.DataFrame()

for row in summarised_halflife_df.itertuples():
    # Generate a sample of size n from a lognormal distribution
    mu = row.half_life
    sd = row.half_life_sd
    mu_log = np.log(mu**2 / np.sqrt(mu**2 + sd**2))
    sd_log = np.sqrt(np.log(1 + sd**2 / mu**2))
    mu_log = np.log(mu**2 / np.sqrt(mu**2 + sd**2))
    sd_log = np.sqrt(np.log(1 + sd**2 / mu**2))
    n = int(row.N_individuals)
    np.random.seed(42)  # Set seed for reproducibility
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
# Add Ka column to the statistics DataFrame
halflife_stats["LogKa"] = halflife_stats["SMILES"].map(smiles_to_ka)

# Fill NaN values in standard deviation (occurs when only one data point exists)
halflife_stats["half_life_sd"] = halflife_stats["half_life_sd"].fillna(
    halflife_stats["half_life_mean"] * 0.5
)
print(halflife_stats)
halflife_stats.to_csv(
    "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/Halflife_stats.csv",
    index=False,
)
# Create a new dataframe by sampling from lognormal distribution for each PFAS
sampled_halflife_df = pd.DataFrame()

for row in halflife_stats.itertuples():
    # Get parameters for lognormal distribution
    mu = row.half_life_mean
    sd = (
        row.half_life_sd
        if row.half_life_sd > 0
        else 0.5 * row.half_life_mean  # halflife_stats["half_life_sd"].mean()
    )  # Avoid zero SD
    n = 50  # row.count  # Number of samples to generate

    # Calculate lognormal parameters
    mu_log = np.log(mu**2 / np.sqrt(mu**2 + sd**2))
    sd_log = np.sqrt(np.log(1 + sd**2 / mu**2))

    # Generate samples from lognormal distribution

    # Set seed for reproducibility
    np.random.seed(42)
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

# Sort the halflife_stats by half_life_mean
halflife_stats_sorted = halflife_stats.sort_values(by="half_life_mean")
all_smiles = halflife_stats_sorted["SMILES"].tolist()
all_smiles = sorted(all_smiles)
# Initialize test_smiles list
test_smiles = []

featurizers = [RDKitDescriptors(), TopologicalFingerprint()]
halflife_jp = JaqpotTabularDataset(
    df=halflife_stats_sorted,
    x_cols=["PFAS", "LogKa"],
    y_cols=["half_life_mean"],
    smiles_cols=["SMILES"],
    featurizer=featurizers,
    task="regression",
    verbose=False,
)

# Apply feature selection to non-categorical features
halflife_jp.select_features(
    FeatureSelector=VarianceThreshold(threshold=0.0), ExcludeColumns=["PFAS"]
)

# Remove highly correlated features
print("Removing highly correlated features...")
# Calculate correlation matrix on numerical features (excluding the PFAS column)
correlation_matrix = halflife_jp.X.drop(columns=["PFAS"], errors="ignore").corr().abs()

# Find pairs of features with correlation > 0.95
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append(
                (correlation_matrix.columns[i], correlation_matrix.columns[j])
            )

# Print the highly correlated pairs
if high_corr_pairs:
    print(f"Found {len(high_corr_pairs)} pairs of highly correlated features")
    for feat1, feat2 in high_corr_pairs:
        corr_val = correlation_matrix.loc[feat1, feat2]
        print(f"  {feat1} and {feat2}: {corr_val:.4f}")
else:
    print("No highly correlated features found")

# Drop the first feature from each pair
features_to_drop = set()
for feat1, feat2 in high_corr_pairs:
    features_to_drop.add(feat1)

if features_to_drop:
    print(f"Dropping {len(features_to_drop)} features:")
    for feat in features_to_drop:
        print(f"  - {feat}")
    halflife_jp.X = halflife_jp.X.drop(columns=list(features_to_drop))


# Perform PCA on the dataset (excluding categorical features)
import matplotlib.pyplot as plt

print("\nPerforming PCA analysis...")
# Extract numerical features (exclude categorical 'PFAS' column)
X_numeric = halflife_jp.X.drop(columns=["PFAS"])

# Scale the data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Perform PCA to get 5 principal components
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"])

# Add PFAS information to the PCA dataframe for visualization
pca_df["PFAS"] = halflife_jp.X["PFAS"].values

# Print explained variance for each component
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio by component:")
print(f"  PC1: {explained_variance[0]:.4f}")
print(f"  PC2: {explained_variance[1]:.4f}")
# print(f"  PC3: {explained_variance[2]:.4f}")
# print(f"  PC4: {explained_variance[3]:.4f}")
# print(f"  PC5: {explained_variance[4]:.4f}")
print(f"  Total: {sum(explained_variance):.4f}")

# Plot the first two principal components
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=halflife_jp.y.values.flatten(),
    cmap="viridis",
    alpha=0.7,
)
plt.colorbar(scatter, label="Half-life (years)")
plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.2%} variance)")
plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.2%} variance)")
plt.title("PCA: First Two Principal Components")
plt.tight_layout()

# Display top feature contributions to each principal component
feature_names = X_numeric.columns
print("\nTop feature contributions to principal components:")
for i, pc in enumerate(["PC1", "PC2"]):
    # Get the loading scores
    loadings = pd.Series(pca.components_[i], index=feature_names)
    sorted_loadings = loadings.abs().sort_values(ascending=False)

    print(f"\n{pc} - Top 5 features:")
    for feature, score in sorted_loadings[:5].items():
        print(f"  {feature}: {loadings[feature]:.4f}")

print(f"Final feature set shape: {halflife_jp.X.shape}")
# Replace X data with PCA components for the dataset
print("\nReplacing original features with PCA components...")
# Create a new dataframe with PCA components and PFAS column
pca_with_pfas = pca_df.copy()

# Replace the X data with the PCA components (keeping PFAS column)
halflife_jp.X = pca_with_pfas

print(f"Feature set after PCA reduction: {halflife_jp.X.shape}")
# Perform Kennard-Stone split on the features
print("Performing Kennard-Stone split...")

# Get the feature matrix (excluding PFAS column which is categorical)
X_features = halflife_jp.X.drop(columns=["PFAS"]).values
y = halflife_jp.y.values.flatten()

# Get all SMILES values in the same order as the features
# Use the original dataframe that was used to create the JaqpotTabularDataset
smiles_list = halflife_stats_sorted["SMILES"].values
pfas_list = halflife_stats_sorted["PFAS"].values

# Map SMILES to index for later reference
smiles_to_idx = {smiles: i for i, smiles in enumerate(smiles_list)}

# Perform Kennard-Stone split
X_train_ks, X_test_ks = ks_train_test_split(X_features, test_size=0.3)

# Get the indices from the KS split by matching the data points
train_idx = [
    i
    for i, x in enumerate(X_features)
    if any(np.array_equal(x, train_x) for train_x in X_train_ks)
]
test_idx = [
    i
    for i, x in enumerate(X_features)
    if any(np.array_equal(x, test_x) for test_x in X_test_ks)
]

# Get the SMILES strings for train and test sets
test_smiles_ks = smiles_list[test_idx]
train_smiles_ks = smiles_list[train_idx]

# Get the PFAS names for the test set
test_pfas_ks = pfas_list[test_idx]

# Print the PFAS compounds in test set
print("\nPFAS compounds in test set (Kennard-Stone):")
for pfas in test_pfas_ks:
    print(f"  - {pfas}")

print(f"Number of compounds in test set: {len(test_smiles_ks)}")
print(f"Number of compounds in train set: {len(train_smiles_ks)}")


# Create test dataset
test_df = halflife_df[halflife_df["SMILES"].isin(test_smiles_ks)]
train_df = halflife_df[halflife_df["SMILES"].isin(train_smiles_ks)]
# Reset the index of the train and test datasets
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
# Print the number of samples in train and test datasets
print(f"Number of samples in train dataset: {len(train_df)}")
print(f"Number of samples in test dataset: {len(test_df)}")
print(f"Number of unique PFAS in train dataset: {len(train_df['SMILES'].unique())}")
print(f"Number of unique PFAS in test dataset: {len(test_df['SMILES'].unique())}")

# Create PCA plot with PFAS names, different shapes for train/test, and half-life coloring
plt.figure(figsize=(12, 10))

# Create a boolean mask for test compounds
is_test = pca_with_pfas["PFAS"].isin(test_pfas_ks)

# Get half-life values for color mapping
half_life_values = halflife_jp.y.values.flatten()

# Create a colormap
cmap = plt.cm.viridis

# Get unified color scale for all data
vmin = half_life_values.min()
vmax = half_life_values.max()

# Plot train compounds with circles
scatter_train = plt.scatter(
    pca_with_pfas.loc[~is_test, "PC1"],
    pca_with_pfas.loc[~is_test, "PC2"],
    c=[
        half_life_values[i] for i in range(len(half_life_values)) if not is_test.iloc[i]
    ],
    cmap=cmap,
    marker="o",  # Circle for train
    s=100,
    alpha=0.7,
    label="Train",
    vmin=vmin,
    vmax=vmax,
)

# Plot test compounds with triangles
scatter_test = plt.scatter(
    pca_with_pfas.loc[is_test, "PC1"],
    pca_with_pfas.loc[is_test, "PC2"],
    c=[half_life_values[i] for i in range(len(half_life_values)) if is_test.iloc[i]],
    cmap=cmap,
    marker="^",  # Triangle for test
    s=100,
    alpha=0.7,
    label="Test",
    vmin=vmin,
    vmax=vmax,
)

# Add unified colorbar for half-life values (use one of the scatter objects)
cbar = plt.colorbar(scatter_train, label="Half-life (years)")

# Add PFAS names as labels
for idx, row in pca_with_pfas.iterrows():
    plt.annotate(
        row["PFAS"],
        (row["PC1"], row["PC2"]),
        fontsize=8,
        ha="center",
        va="bottom",
        xytext=(0, 5),
        textcoords="offset points",
    )

plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.2%} variance)")
plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.2%} variance)")
plt.title("PCA: Train/Test Split with Half-life Coloring")
plt.legend()
plt.tight_layout()
plt.show()

# # Find and move PFHxS from test to train set
pfhxs_mask = test_df["PFAS"] == "PFHxS"
if pfhxs_mask.any():
    # Get all PFHxS data points from test set
    pfhxs_data = test_df[pfhxs_mask].copy()

    # Add to train set
    train_df = pd.concat([train_df, pfhxs_data], ignore_index=True)

    # Remove from test set
    test_df = test_df[~pfhxs_mask].copy()

    # Get the SMILES for PFHxS
    pfhxs_smiles = pfhxs_data["SMILES"].iloc[0]

    # Update the lists of SMILES
    test_smiles_ks = [s for s in test_smiles_ks if s != pfhxs_smiles]
    train_smiles_ks = np.append(train_smiles_ks, pfhxs_smiles)


# Save the train and test datasets
train_df.to_csv(
    "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/Half-life_dataset_Human_train.csv",
    index=False,
)
test_df.to_csv(
    "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/Half-life_dataset_Human_test.csv",
    index=False,
)
