import shap
import pandas as pd
import numpy as np
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import (
    TopologicalFingerprint,
    RDKitDescriptors,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Train_Albumin_Binding_Data.csv"
)

# Create a JaqpotpyDataset objects
x_cols = []
categorical_cols = []
featurizers = [RDKitDescriptors(), TopologicalFingerprint()]

train_dataset = JaqpotpyDataset(
    df=df_train,
    y_cols="Ka",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

train_dataset.select_features(
    SelectColumns=[
        "fr_quatN",
        "Bit_592",
        "Chi2v",
        "AvgIpc",
        "Bit_886",
        # "Bit_741",
        "Bit_1977",
    ]
)

X_train = train_dataset.X
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, train_dataset.y)
r2_score = model.score(X_train_scaled, train_dataset.y)
print(f"R2 score: {r2_score}")
# Create a SHAP explainer
explainer = shap.Explainer(model, X_train_scaled)

# Calculate SHAP values
shap_values = explainer(X_train_scaled)

# Plot the SHAP values for the specified features
shap.summary_plot(shap_values, X_train_scaled, feature_names=X_train.columns)
