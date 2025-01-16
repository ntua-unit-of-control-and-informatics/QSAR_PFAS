import pandas as pd
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors.molecular import MordredDescriptors
from jaqpotpy.models import SklearnModel
from jaqpotpy import Jaqpot
from sklearn.ensemble import GradientBoostingRegressor
from jaqpotpy.doa import Leverage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

train_data = pd.read_csv("Train_data_SCF.csv")
test_data = pd.read_csv("Test_data_SCF.csv")

# train_data.rename(
#     columns={"PFAS_concentration (μg/kg)": "PFAS_concentration"}, inplace=True
# )
# test_data.rename(
#     columns={"PFAS_concentration (μg/kg)": "PFAS_concentration"}, inplace=True
# )


def decode_one_hot_columns(df, prefix):
    one_hot_cols = [col for col in df.columns if col.startswith(prefix)]
    df[prefix] = df[one_hot_cols].idxmax(axis=1).str.replace(prefix, "")
    df = df.drop(columns=one_hot_cols).drop("Unnamed: 0", axis=1)
    return df


train_data = decode_one_hot_columns(train_data, "Species")
test_data = decode_one_hot_columns(test_data, "Species")

smiles_col = ["SMILES"]
x_cols = [
    "Species",
    "Cultivate_mode",
    "pH",
    "PFAS_concentration",
    "Exposure_time",
    "Protein_content",
    "Lipid_content",
]

selected_cols = [
    "SpAbs_A",
    "SpMAD_A",
    "AATS6Z",
    "AATS7i",
    "AATS6se",
    "nO",
    "Species",
    "Cultivate_mode",
    "pH",
    "PFAS_concentration",
    "Exposure_time",
    "Protein_content",
    "Lipid_content",
]

y_cols = ["SCF"]
task = "regression"
featurizer = MordredDescriptors()
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
    "Species",
    "Cultivate_mode",
    "pH",
    "PFAS_concentration",
    "Exposure_time",
    "Protein_content",
    "Lipid_content",
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

sklearn_model = GradientBoostingRegressor(
    n_estimators=128, min_samples_leaf=2, random_state=42
)


column_transformer = [
    ColumnTransformer(
        transformers=[
            ("OneHotEncoder", OneHotEncoder(), ["Species"]),
        ],
        remainder="passthrough",
        force_int_remainder_cols=False,
    )
]
doa = [Leverage()]
jaqpot_model = SklearnModel(
    dataset=train_dataset,
    model=sklearn_model,
    doa=doa,
    preprocess_x=column_transformer,
)
jaqpot_model.fit()

cv = jaqpot_model.cross_validate(train_dataset, n_splits=10)
test = jaqpot_model.evaluate(test_dataset)

jaqpot = Jaqpot()
jaqpot.login()
jaqpot_model.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="logSCF model",
    description="Gradient Boosting model for predicting logRCF",
    visibility="PRIVATE",
)
