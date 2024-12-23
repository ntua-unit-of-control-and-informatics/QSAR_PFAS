import pandas as pd
import numpy as np
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import (
    MordredDescriptors,
    TopologicalFingerprint,
    RDKitDescriptors,
)
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("Final_Albumin_Binding_Data.csv")
dataset = dataset[dataset["Ka"] != 0]
dataset[["Ka"]] = np.log10(dataset[["Ka"]])

x_train, x_test, y_train, y_test = train_test_split(
    dataset[["SMILES", "Albumin_Type", "Method"]],
    dataset["Ka"],
    test_size=0.2,
    random_state=42,
)

df_train = pd.concat([x_train, y_train], axis=1)
df_test = pd.concat([x_test, y_test], axis=1)

# Create a JaqpotpyDataset object
x_cols = ["Albumin_Type", "Method"]
featurizers = [TopologicalFingerprint(), RDKitDescriptors()]

train_dataset = JaqpotpyDataset(
    df=df_train,
    y_cols="Ka",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

# Use VarianceThreshold to select features with a minimum variance of 0.1
FeatureSelector = VarianceThreshold(threshold=200)
train_dataset.select_features(
    FeatureSelector,
    ExcludeColumns=["Albumin_Type", "Method"],
)
train_dataset.selected_features

reduced_df = train_dataset.X
reduced_continuous_df = reduced_df.drop(columns=["Albumin_Type", "Method"])
corr_matrix = reduced_continuous_df.corr()
columns_to_drop = set()
for i in range(len(corr_matrix.columns)):
    if corr_matrix.columns[i] not in columns_to_drop:
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                col_to_drop = corr_matrix.columns[j]
                columns_to_drop.add(col_to_drop)

reduced_df = reduced_df.drop(columns=columns_to_drop)
reduced_df.columns
