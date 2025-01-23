import pandas as pd
from dotenv import load_dotenv
from jaqpotpy.api.jaqpot_api_client import JaqpotApiClient

load_dotenv(".env")

# Retrieve SMILES from Half-life dataset
dataset = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human.csv")
smiles_col = dataset["SMILES"].dropna().unique()

# Prepare dataset for Albumin Binding model
input_dataset = pd.DataFrame(
    {
        "SMILES": smiles_col.repeat(2),
        "Temperature": -100,
        "Albumin Type": "HSA",
        "Method": ["Equilibrium dialysis", "DSF"] * len(smiles_col),
    }
)
print(input_dataset.head())
print(input_dataset.shape)
input_dataset.to_csv("PFAS_HalfLife_QSAR/Datasets/input_dataset.csv", index=False)
jaqpot = JaqpotApiClient()
prediction = jaqpot.predict_with_csv_sync(
    model_id=1988, csv_path="PFAS_HalfLife_QSAR/Datasets/input_dataset.csv"
)
Ka_results = pd.concat(
    [input_dataset.reset_index(drop=True), pd.DataFrame(prediction)], axis=1
)
Ka_results.drop("Method", axis=1, inplace=True)

# check for PFAS outside of DOA
for row in range(len(Ka_results["jaqpotMetadata"])):
    if Ka_results["jaqpotMetadata"][row]["doa"]["majorityVoting"] == "False":
        break

# Estimate mean values for each PFAS
mean_Ka = Ka_results.groupby("SMILES")["Ka"].mean().reset_index()
mean_Ka.columns = ["SMILES", "LogKa"]
print(mean_Ka.head())
mean_Ka.to_csv("PFAS_HalfLife_QSAR/Datasets/Ka_results.csv", index=False)
