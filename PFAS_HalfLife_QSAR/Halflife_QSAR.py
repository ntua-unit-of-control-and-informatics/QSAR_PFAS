import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import (
    MordredDescriptors,
    TopologicalFingerprint,
    RDKitDescriptors,
)
from jaqpotpy.doa import (
    Leverage,
    MeanVar,
    BoundingBox,
    Mahalanobis,
    KernelBased,
    CityBlock,
)
from sklearn.feature_selection import VarianceThreshold
from jaqpotpy.models import SklearnModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df_train = pd.read_csv("Datasets/Half-life_dataset_Human_train.csv")
# Create a JaqpotpyDataset objects
# df_train["half_life_days"] = np.log(df_train["half_life_days"])
x_cols = ["sex", "adult"]
categorical_cols = ["sex", "adult"]
featurizers = [RDKitDescriptors()]

train_dataset = JaqpotpyDataset(
    df=df_train,
    y_cols="half_life_days",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

# train_dataset.select_features(SelectColumns=["sex", "adult",'MolWt'])

preprocessor_x = ColumnTransformer(
    transformers=[
        ("OneHotEncoder", OneHotEncoder(), categorical_cols),
        (
            "StandardScaler",
            StandardScaler(),
            train_dataset.X.columns.difference(categorical_cols),
        ),
    ],
    remainder="passthrough",
    force_int_remainder_cols=False,
)

model = SklearnModel(
    dataset=train_dataset,
    model=GaussianProcessRegressor(kernel=1.0 * RBF(length_scale=1.0)),
    preprocess_x=[preprocessor_x],
    # preprocess_y=[StandardScaler()],
)

model.fit()
model.cross_validate(n_splits=10, dataset=train_dataset)
# Predict the values
y_pred = model.predict(train_dataset)
y_true = train_dataset.y
# Plot y_pred vs y_true
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.7)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.show()
