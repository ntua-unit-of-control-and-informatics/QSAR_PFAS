from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import (
    MordredDescriptors,
    TopologicalFingerprint,
    RDKitDescriptors,
)
from jaqpotpy.models import SklearnModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # or RandomForestClassifier
from sklearn.model_selection import cross_val_score
import time


def forward_selection(dataset, estimator=None, cv=5, scoring="r2", max_features=None):
    n_features = dataset.X.shape[1]
    if max_features is None:
        max_features = n_features

    # Initialize variables
    selected_features = []
    feature_names = list(dataset.X.columns)
    remaining_features = feature_names.copy()
    scores_history = []

    for i in range(max_features):
        best_score = float("-inf")
        best_feature = None

        # Try each remaining feature
        for feature in remaining_features:
            # Create candidate feature set
            candidate_features = selected_features + [feature]

            copy_dataset = dataset.copy()
            copy_dataset.select_features(SelectColumns=candidate_features)
            # copy_dataset.X = copy_dataset.X[candidate_features]
            if estimator is None:
                estimator = RandomForestRegressor(random_state=42)
            model = SklearnModel(
                dataset=copy_dataset,
                model=RandomForestRegressor(random_state=42),
                preprocess_x=StandardScaler(),
            )
            model.fit()

            # Evaluate model with candidate feature set
            model.cross_validate(dataset=copy_dataset, n_splits=cv, random_seed=42)
            if scoring == "r2":
                avg_score = model.average_cross_val_scores["r2"]
            else:
                avg_score = None

            # Update best score and feature if improvement found
            if avg_score > best_score:
                best_score = avg_score
                best_feature = feature

        # If no improvement, stop selection
        if best_feature is None:
            break

        # Add best feature to selected set
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        scores_history.append(best_score)

        print(f"Step {i+1}: Added feature '{best_feature}' (Score: {best_score:.4f})")

    print("\nFinal selected features:", selected_features)
    print(f"Final score: {scores_history[-1]:.4f}")

    return selected_features, scores_history


df_train = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Train_Albumin_Binding_Data.csv"
)

# Create a JaqpotpyDataset objects
x_cols = []  # ["Albumin_Type"]
categorical_cols = []  # ["Albumin_Type"]
featurizers = [RDKitDescriptors()]

train_dataset = JaqpotpyDataset(
    df=df_train,
    y_cols="Ka",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

start_time = time.time()

selected_features, scores = forward_selection(
    dataset=train_dataset,
    estimator=RandomForestRegressor(random_state=42),
    cv=5,
    max_features=10,
)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken for forward selection: {elapsed_time:.2f} seconds")


# Plot the scores history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(scores) + 1), scores, marker="o")
plt.xlabel("Number of Features")
plt.ylabel("Cross-validation Score")
plt.title("Forward Selection Scores History")
plt.grid(True)
plt.show()
