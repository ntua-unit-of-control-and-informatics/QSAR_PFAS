import pandas as pd
from dotenv import load_dotenv
from jaqpot_python_sdk.jaqpot_api_client import JaqpotApiClient

load_dotenv("Restructured/.env")

# Retrieve SMILES from Half-life dataset
dataset = pd.read_csv("Restructured/data/Half-life_dataset_Human_revamped.csv")
smiles_col = dataset["SMILES"].dropna().unique()

# Prepare dataset for Albumin Binding model
input_dataset = pd.DataFrame(
    {
        "SMILES": smiles_col,
    }
)
print(input_dataset.head())
print(input_dataset.shape)
input_dataset.to_csv("Restructured/data/input_Ka_dataset.csv", index=False)
jaqpot = JaqpotApiClient()
prediction = jaqpot.predict_with_csv_sync(
    model_id=2028, csv_path="Restructured/data/input_Ka_dataset.csv"
)
Ka_results = pd.concat(
    [input_dataset.reset_index(drop=True), pd.DataFrame(prediction)], axis=1
)

# check for PFAS outside of DOA
for row in range(len(Ka_results["jaqpotMetadata"])):
    if Ka_results["jaqpotMetadata"][row]["doa"]["majorityVoting"] == "False":
        print(f'{Ka_results["SMILES"][row]} is out of DOA.')

# Estimate mean values for each PFAS
Ka_results.drop(columns=["jaqpotMetadata"], inplace=True)
Ka_results.columns = ["SMILES", "LogKa"]
print(Ka_results.head())
Ka_results.to_csv("Restructured/data/Ka_results.csv", index=False)
