import pandas as pd
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors.molecular import MordredDescriptors
from jaqpotpy.models import SklearnModel
from jaqpotpy import Jaqpot
from sklearn.ensemble import RandomForestRegressor
from jaqpotpy.doa import Leverage

train_data = pd.read_csv("Train_data_logrcf.csv")
test_data = pd.read_csv("Test_data_logrcf.csv")

smiles_col = ["SMILES"]

x_cols = [
    "Exposure_time",
    "pH",
    "PFAS_concentration (μg/kg)",
    "OC_content (%)",
    "Cultivate_mode",
    "Protein_content (%)",
]
selected_cols = [
    "Exposure_time",
    "pH",
    "ATSC6pe",
    "AATS2dv",
    "PFAS_concentration_(μg/kg)",
    "OC_content_(%)",
    "Cultivate_mode",
    "Protein_content_(%)",
]

y_cols = ["logRCF"]
featurizer = MordredDescriptors()
task = "regression"

train_dataset = JaqpotpyDataset(
    df=train_data,
    smiles_cols=smiles_col,
    x_cols=x_cols,
    y_cols=y_cols,
    featurizer=featurizer,
    task=task,
)
train_dataset.select_features(SelectColumns=selected_cols)
x_cols = [
    "Exposure_time",
    "pH",
    "PFAS_concentration (μg/kg)",
    "OC_content (%)",
    "Cultivate_mode",
    "Protein_content (%)",
]
test_dataset = JaqpotpyDataset(
    df=test_data,
    smiles_cols=smiles_col,
    x_cols=x_cols,
    y_cols=y_cols,
    featurizer=featurizer,
    task=task,
)
test_dataset.select_features(SelectColumns=selected_cols)

sklearn_model = RandomForestRegressor(n_estimators=248, random_state=42)
doa = [Leverage()]
jaqpot_model = SklearnModel(dataset=train_dataset, model=sklearn_model, doa=doa)
jaqpot_model.fit()

cv = jaqpot_model.cross_validate(train_dataset, n_splits=10)
test = jaqpot_model.evaluate(test_dataset)
jaqpot = Jaqpot()
jaqpot.login()

jaqpot_model.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="logRCF",
    description="Test model",
    visibility="PRIVATE",
)
