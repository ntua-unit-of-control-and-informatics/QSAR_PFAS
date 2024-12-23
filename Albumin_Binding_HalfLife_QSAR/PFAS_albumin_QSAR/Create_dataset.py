import pandas as pd

albumin_df = pd.read_excel("Albumin_Binding_Data.xlsx", sheet_name="Data")

final_df = albumin_df[
    ["SMILES", "Authors", "Congener", "Albumin_Type", "Temperature", "Method", "Ka"]
]
final_df["Temperature"].fillna(-100, inplace=True)
unique_smiles_count = final_df["SMILES"].nunique()
print(f"Unique SMILES count: {unique_smiles_count}")

excluded_studies = [
    "Alesio et al.2022",
    # "Gao et al.2019",
    # "Allendorf et al.2019",
    "Qin et al.2010",
    # "MacManus-Spencer et al.2010",
    # "Jackson et al.2021",
    "Maso et al.2021",
    # "Moro et al.2022",
    "Starnes et al.2024b",
]

train_dataset = final_df[~final_df["Authors"].isin(excluded_studies)].reset_index(
    drop=True
)
test_dataset = final_df[final_df["Authors"].isin(excluded_studies)].reset_index(
    drop=True
)

train_dataset.to_csv("Train_Albumin_Binding_Data.csv", index=False)
test_dataset.to_csv("Test_Albumin_Binding_Data.csv", index=False)

print("Train and Test datasets saved successfully.")
test_percentage = (len(test_dataset) / len(final_df)) * 100
print(
    f"Test dataset: {len(test_dataset)} rows ({test_percentage:.2f}% of the entire dataset)"
)
print(f"Unique congeners in test dataset: {test_dataset['Congener'].nunique()}")
unique_congeners_test_not_in_train = set(test_dataset["Congener"]) - set(
    train_dataset["Congener"]
)
print(
    f"Number of congeners in test dataset but not in train dataset: {len(unique_congeners_test_not_in_train)}"
)
unique_congeners_in_train = set(train_dataset["Congener"])
unique_congeners_in_test = set(test_dataset["Congener"])

congeners_in_test_out_of_train = unique_congeners_in_test.intersection(
    unique_congeners_in_train
)
