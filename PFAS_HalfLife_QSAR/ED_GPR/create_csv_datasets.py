import pandas as pd
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import RDKitDescriptors, MordredDescriptors

df_train = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human_train.csv")
df_test = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human_test.csv")

x_cols = ["Study", "sex", "adult", "LogKa", "half_life_type", "Occupational_exposure"]
categorical_cols = ["sex", "adult", "half_life_type", "Occupational_exposure"]
featurizers = [RDKitDescriptors()]

train_jq = JaqpotpyDataset(
    df=df_train,
    y_cols="half_life_days",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

test_jq = JaqpotpyDataset(
    df=df_test,
    y_cols="half_life_days",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

# write datsets to csv for GPR
train_jq.df.to_csv("PFAS_HalfLife_QSAR/GPR/train_data.csv", index=False)
test_jq.df.to_csv("PFAS_HalfLife_QSAR/GPR/test_data.csv", index=False)
