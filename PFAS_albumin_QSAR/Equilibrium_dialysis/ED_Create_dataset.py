import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from jaqpotpy.datasets import JaqpotpyDataset
from sklearn.preprocessing import StandardScaler
from jaqpotpy.descriptors import (
    MordredDescriptors,
    TopologicalFingerprint,
    RDKitDescriptors,
)
from sklearn.preprocessing import StandardScaler
from kennard_stone import train_test_split as ks_train_test_split

albumin_df = pd.read_excel(
    "PFAS_albumin_QSAR/Albumin_binding_data.xlsx", sheet_name="Data"
)

final_df = albumin_df[
    ["SMILES", "Authors", "Congener", "Group", "Albumin_Type", "Method", "Ka"]
]
unique_smiles_count = final_df["SMILES"].nunique()
print(f"Unique SMILES count: {unique_smiles_count}")

final_df = final_df[
    ["Congener", "SMILES", "Group", "Albumin_Type", "Method", "Authors", "Ka"]
]
final_df = final_df[final_df["Method"] == "Equilibrium dialysis"].reset_index(drop=True)
final_df = final_df[final_df["Albumin_Type"] == "HSA"].reset_index(drop=True)

final_df.drop("Method", axis=1, inplace=True)

# Transform Ka to Log10(Ka)
final_df["Ka"] = np.log10(final_df["Ka"])
# mean_variance_df = final_df.groupby(["SMILES"])["Ka"].agg(["mean", "var"]).reset_index()

# # Merge back with the original dataframe to keep the Congener column
# mean_variance_df = mean_variance_df.merge(
#     final_df[["SMILES", "Congener"]].drop_duplicates(),
#     on=["SMILES"],
#     how="left",
# )
# mean_variance_df.rename(columns={"mean": "Ka", "var": "Ka_variance"}, inplace=True)
# final_df = mean_variance_df
# print(final_df)
# print(final_df["SMILES"].unique().shape)

# train_dataset, test_dataset = train_test_split(
#     final_df, test_size=0.20, random_state=30
# )

# train_dataset = final_df[final_df["Authors"] != "Chen et al.2025"].reset_index(
#     drop=True
# )
# test_dataset = final_df[final_df["Authors"] == "Chen et al.2025"].reset_index(drop=True)

final_df = final_df[final_df["Authors"] == "Chen et al.2025"].reset_index(drop=True)
print(final_df)

# x_cols = []
# featurizers = [RDKitDescriptors()]
# j_dataset = JaqpotpyDataset(
#     df=final_df,
#     y_cols="Ka",
#     x_cols=x_cols,
#     smiles_cols="SMILES",
#     featurizer=featurizers,
#     task="regression",
# )

# numerical_data = j_dataset.df
# # Standardize the numerical data
# scaler = StandardScaler()
# numerical_data_scaled = scaler.fit_transform(numerical_data)

# # Perform Kennard-Stone split
# train_data, test_data = ks_train_test_split(numerical_data_scaled, train_size=0.8)

# # Check which rows from numerical_data_scaled are included in test_data
# test_indexes = []
# for i, row in enumerate(numerical_data_scaled):
#     if any((row == test_row).all() for test_row in test_data):
#         test_indexes.append(i)


# train_dataset = final_df.drop(test_indexes).reset_index(drop=True)
# test_dataset = final_df.loc[test_indexes].reset_index(drop=True)

train_dataset, test_dataset = train_test_split(final_df, test_size=0.2, random_state=0)

# test_index = [2, 5, 8, 13, 18, 22, 27, 34, 42, 53, 56]
# train_dataset = final_df.drop(test_index).reset_index(drop=True)
# test_dataset = final_df.loc[test_index].reset_index(drop=True)

train_dataset.to_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Train_Albumin_Binding_Data.csv", index=False
)
test_dataset.to_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Test_Albumin_Binding_Data.csv", index=False
)

print("Train and Test datasets saved successfully.")
test_percentage = (len(test_dataset) / len(final_df)) * 100
print(
    f"Test dataset: {len(test_dataset)} rows ({test_percentage:.2f}% of the entire dataset)"
)
