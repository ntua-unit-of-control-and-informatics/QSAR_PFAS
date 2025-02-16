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
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # or RandomForestClassifier
from sklearn.model_selection import cross_val_score


def forward_selection(
    X, y, estimator=None, cv=3, scoring="neg_mean_squared_error", max_features=None
):
    """
    Perform forward feature selection using Random Forest.

    Parameters:
    -----------
    X : pandas DataFrame
        Features matrix
    y : pandas Series or numpy array
        Target variable
    estimator : sklearn estimator, optional
        If None, defaults to RandomForestRegressor()
    cv : int, optional
        Number of cross-validation folds
    scoring : str, optional
        Scoring metric to use for selection
    max_features : int, optional
        Maximum number of features to select

    Returns:
    --------
    selected_features : list
        List of names of selected features
    scores_history : list
        List of scores at each selection step
    """

    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")

    if estimator is None:
        estimator = RandomForestRegressor(random_state=42)

    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features

    # Initialize variables
    selected_features = []
    feature_names = list(X.columns)
    remaining_features = feature_names.copy()
    scores_history = []

    for i in range(max_features):
        best_score = float("-inf")
        best_feature = None

        # Try each remaining feature
        for feature in remaining_features:
            # Create candidate feature set
            candidate_features = selected_features + [feature]

            # Evaluate model with candidate feature set
            scores = cross_val_score(
                estimator, X[candidate_features], y, cv=cv, scoring=scoring
            )
            avg_score = np.mean(scores)

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
df_train = df_train[df_train["Ka"] != 0]

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

descriptors = train_dataset.X
scaler = StandardScaler()
# Exclude categorical columns from scaling
categorical_data = descriptors[categorical_cols]
numerical_data = descriptors.drop(columns=categorical_cols)

# Scale only the numerical data
scaled_numerical_data = scaler.fit_transform(numerical_data)

# Combine scaled numerical data with categorical data
scaled_descriptors_df = pd.DataFrame(
    scaled_numerical_data, columns=numerical_data.columns
)

# # Remove features with low variance
# selector = VarianceThreshold(threshold=0.01)
# selected_descriptors = selector.fit_transform(scaled_descriptors_df)
# selected_descriptors_df = pd.DataFrame(
#     selected_descriptors, columns=scaled_descriptors_df.columns[selector.get_support()]
# )
# dropped_columns = scaled_descriptors_df.columns[~selector.get_support()].tolist()
# print(f"Number of dropped columns = {len(dropped_columns)}")

# Correlation matrix

# Compute the correlation matrix
# selected_descriptors_df = selected_descriptors_df.loc[
#     :, ~selected_descriptors_df.columns.duplicated()
# ]
# corr_matrix = selected_descriptors_df.corr()


# Plot the heatmap
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
# plt.title("Correlation Matrix of Selected Descriptors")
# plt.show()

# Find columns with correlation higher than 0.9
# high_corr_pairs = (
#     corr_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)
# )
# high_corr_pairs = high_corr_pairs[high_corr_pairs < 1].reset_index()
# high_corr_pairs.columns = ["Feature1", "Feature2", "Correlation"]

# # Drop columns with higher mean correlation
# to_drop = set()
# for feature1, feature2 in high_corr_pairs[high_corr_pairs["Correlation"] > 0.9][
#     ["Feature1", "Feature2"]
# ].values:
#     if feature1 not in to_drop and feature2 not in to_drop:
#         mean_corr_feature1 = corr_matrix[feature1].mean()
#         mean_corr_feature2 = corr_matrix[feature2].mean()
#         if mean_corr_feature1 > mean_corr_feature2:
#             to_drop.add(feature1)
#         else:
#             to_drop.add(feature2)

# selected_descriptors_df = selected_descriptors_df.drop(columns=to_drop)
# print(f"Number of columns dropped due to high correlation = {len(to_drop)}")
# print(selected_descriptors_df.columns)
# print(f"Number of features after feature selection: {selected_descriptors_df.shape[1]}")

# final_train_df = pd.concat(
#     [
#         categorical_data,
#         selected_descriptors_df,
#     ],
#     axis=1,
#     ignore_index=True,
# )
# final_train_df.columns = categorical_cols + selected_descriptors_df.columns.tolist()
# final_train_df.reset_index(drop=True, inplace=True)

# # Perform one-hot encoding for categorical columns
# final_train_df = pd.get_dummies(
#     final_train_df, columns=categorical_cols, drop_first=True
# )

final_train_df = scaled_descriptors_df

selected_features, scores = forward_selection(
    final_train_df, train_dataset.y.to_numpy().ravel(), max_features=20
)


# Plot the scores history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(scores) + 1), scores, marker="o")
plt.xlabel("Number of Features")
plt.ylabel("Cross-validation Score")
plt.title("Forward Selection Scores History")
plt.grid(True)
plt.show()
