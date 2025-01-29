from matplotlib import pyplot as plt
import pandas as pd
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import RDKitDescriptors
from jaqpotpy.models import SklearnModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    WhiteKernel,
    Matern,
    RationalQuadratic,
    RBF,
)


df_train = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human_train.csv")
df_test = pd.read_csv("PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human_test.csv")

x_cols = ["sex", "adult", "LogKa", "half_life_type", "Occupational_exposure"]
categorical_cols = ["sex", "adult", "half_life_type", "Occupational_exposure"]
featurizers = [RDKitDescriptors()]

train_df = JaqpotpyDataset(
    df=df_train,
    y_cols="half_life_days",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

test_df = JaqpotpyDataset(
    df=df_test,
    y_cols="half_life_days",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

selected_descriptors = [
    "LogKa",
    "MaxAbsEStateIndex",
    "MaxEStateIndex",
    "MinEStateIndex",
    "qed",
    "SPS",
    "ExactMolWt",
    "MaxPartialCharge",
    "FpDensityMorgan3",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPLOW",
    "AvgIpc",
    "BalabanJ",
    "Chi0n",
    "Chi1",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi4n",
    "Chi4v",
    "Ipc",
    "Kappa2",
    "LabuteASA",
    "PEOE_VSA1",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "SMR_VSA10",
    "SMR_VSA5",
    "SMR_VSA6",
    "SlogP_VSA10",
    "SlogP_VSA3",
    "SlogP_VSA5",
    "TPSA",
    "EState_VSA1",
    "EState_VSA10",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA9",
    "VSA_EState1",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState7",
    "FractionCSP3",
    "HeavyAtomCount",
    "NOCount",
    "NumHAcceptors",
    "fr_Al_COO",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
]

categorical_columns = ["sex", "adult", "half_life_type", "Occupational_exposure"]

train_df.select_features(SelectColumns=categorical_columns + selected_descriptors)
test_df.select_features(SelectColumns=categorical_columns + selected_descriptors)

# One-hot encode the categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            "OneHotEncoder",
            OneHotEncoder(drop="first", sparse_output=False),
            categorical_columns,
        ),
        ("StandardScaler", StandardScaler(), selected_descriptors),
    ],
    remainder="passthrough",
)

kernel = (
    ConstantKernel(1.0, (1e-1, 1e1))
    * Matern(length_scale=[1.0] * 61, nu=1.5, length_scale_bounds=(1e-1, 1e2))
    + ConstantKernel(0.3, (1e-1, 1e0))  # Reduced upper bound and default value
)

jaqpot_model = SklearnModel(
    dataset=train_df,
    model=GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=20,
        random_state=42,
        alpha=0.25,  # Further reduced to balance between fit and uncertainty
    ),
    preprocess_x=preprocessor,
    preprocess_y=StandardScaler(),
)
# jaqpot_model.fit()
jaqpot_model.fit(onnx_options={GaussianProcessRegressor: {"return_std": True}})
jaqpot_model.evaluate(test_df)
print(jaqpot_model.scores)

y_pred_train, y_train_std = jaqpot_model.predict(train_df, return_std=True)
y_pred_test, y_test_std = jaqpot_model.predict(test_df, return_std=True)

y_train_onnx, y_train_onnx_sd = jaqpot_model.predict_onnx(train_df)
y_test_onnx, y_test_onnx_sd = jaqpot_model.predict_onnx(test_df)

differences = np.abs(y_test_onnx - y_pred_test)
if np.all(differences < 1e-02):
    print("ok")
else:
    print("not ok")

# only for plots
y_pred_train = y_train_onnx
y_train_std = y_train_onnx_sd
y_pred_test = y_test_onnx
y_test_std = y_test_onnx_sd

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
y_train = train_df.y
y_test = test_df.y
# Plot for training data with error bars
axes[0].errorbar(
    y_train,
    y_pred_train,
    yerr=y_train_std,
    fmt="o",
    color="blue",
    alpha=0.5,
    ecolor="lightgray",
    elinewidth=2,
    capsize=2,
)
axes[0].plot(
    [y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "k--", lw=2
)
axes[0].set_xlabel("Observed")
axes[0].set_ylabel("Predicted")
axes[0].set_title("Training Data")

# Plot for test data with error bars
axes[1].errorbar(
    y_test,
    y_pred_test,
    yerr=y_test_std,
    fmt="o",
    color="red",
    alpha=0.5,
    ecolor="lightgray",
    elinewidth=2,
    capsize=2,
)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
axes[1].set_xlabel("Observed")
axes[1].set_ylabel("Predicted")
axes[1].set_title("Test Data")

plt.tight_layout()
plt.show()

# from jaqpotpy import Jaqpot

# # Upload the pretrained model on Jaqpot
# jaqpot = Jaqpot()
# jaqpot.login()
# jaqpot_model.deploy_on_jaqpot(
#     jaqpot=jaqpot,
#     name="GPR test",
#     description="GPR test",
#     visibility="PRIVATE",
# )
