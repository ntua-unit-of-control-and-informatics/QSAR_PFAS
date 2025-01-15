import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

halflife_df = pd.read_csv("Datasets/Half-life_dataset_Human.csv")

# Drop rows with NaN values because no SMILES found for these rows
halflife_df.dropna(subset=["SMILES"], inplace=True)
halflife_df.drop(labels=["age (years)"], axis=1, inplace=True)
halflife_df.drop(
    halflife_df[halflife_df["Study"].isin(["Arnot et al. 2014"])].index,
    inplace=True,
)
halflife_df.drop(
    halflife_df[halflife_df["half_life_days"] > 15 * 365].index, inplace=True
)

if halflife_df.isnull().values.any():
    print("The dataframe contains NaN values.")
    print(halflife_df[halflife_df.isnull().any(axis=1)])
else:
    print("The dataframe does not contain any NaN values.")

# Transform all trues of 'adult' column to 1 and false to 0
halflife_df["adult"] = halflife_df["adult"].astype(int)

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
train_df, test_df = train_test_split(halflife_df, test_size=0.2, random_state=0)

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

# Create a scatter plot of half_life values and the index of each data point
# plt.figure(figsize=(10, 6))
# plt.scatter(halflife_df.index, halflife_df["half_life_days"], alpha=0.5)
# plt.title("Scatter Plot of Half-life Values and Index of Each Data Point")
# plt.xlabel("Index")
# plt.ylabel("Half-life Days")

# # Annotate points with half_life > 10000
# for i, row in halflife_df[halflife_df["half_life_days"] > 4000].iterrows():
#     plt.annotate(
#         i,
#         (i, row["half_life_days"]),
#         textcoords="offset points",
#         xytext=(0, 10),
#         ha="center",
#         fontsize=8,
#         color="red",
#     )

# plt.show()

# # Plot the distribution of the target variable 'half_life_days' in the train and test datasets on the same plot
# plt.figure(figsize=(10, 6))

# sns.kdeplot(train_df["half_life_days"], label="Train", color="blue", shade=True)
# sns.kdeplot(test_df["half_life_days"], label="Test", color="red", shade=True)

# plt.title("Distribution of Half-life Days in Train and Test Datasets")
# plt.xlabel("Half-life Days")
# plt.ylabel("Density")
# plt.legend()

# plt.show()

# Save the train and test datasets
train_df.to_csv("Datasets/Half-life_dataset_Human_train.csv", index=False)
test_df.to_csv("Datasets/Half-life_dataset_Human_test.csv", index=False)
