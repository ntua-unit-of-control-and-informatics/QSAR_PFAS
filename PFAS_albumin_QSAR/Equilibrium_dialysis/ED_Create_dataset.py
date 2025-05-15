import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

final_df = final_df[final_df["Authors"] == "Chen et al.2025"].reset_index(drop=True)
print(final_df)

train_dataset, test_dataset = train_test_split(final_df, test_size=0.2, random_state=0)

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
