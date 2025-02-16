from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import (
    MordredDescriptors,
    TopologicalFingerprint,
    RDKitDescriptors,
)
from jaqpotpy.models import SklearnModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor


df_train = pd.read_csv("PFAS_albumin_QSAR/Train_Albumin_Binding_Data.csv")
df_train = df_train[df_train["Ka"] != 0]
df_train[["Ka"]] = np.log10(df_train[["Ka"]])

# Create a JaqpotpyDataset objects
x_cols = ["Temperature", "Albumin_Type", "Method"]
categorical_cols = ["Albumin_Type", "Method"]
featurizers = [RDKitDescriptors()]

train_dataset = JaqpotpyDataset(
    df=df_train,
    y_cols="Ka",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

descriptors = train_dataset.X.drop(columns="Temperature")
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
# Correlation matrix

# Compute the correlation matrix
corr_matrix = scaled_descriptors_df.corr()

print(corr_matrix.shape[1])


# Retrieve features that are more correlated with 'BalabanJ'
def find_highly_correlated_features(
    corr_matrix, target_feature, correlation_threshold=0.5
):
    """
    Find features that are highly correlated with the target feature.

    Parameters:
    corr_matrix (pd.DataFrame): Correlation matrix.
    target_feature (str): The feature to find correlations with.
    correlation_threshold (float): Threshold for high correlation.

    Returns:
    pd.DataFrame: DataFrame with features and their correlation with the target feature.
    """
    # Get the correlations with the target feature
    target_correlations = corr_matrix[target_feature]

    # Filter features that have a high correlation with the target feature
    highly_correlated_features = target_correlations[
        abs(target_correlations) > correlation_threshold
    ]

    # Create a DataFrame with the results
    result_df = highly_correlated_features.reset_index()
    result_df.columns = ["Feature", "Correlation"]

    return result_df


# Example usage
# "BalabanJ"
target_feature = "BalabanJ"
correlation_threshold = 0.6
highly_correlated_features_df = find_highly_correlated_features(
    corr_matrix, target_feature, correlation_threshold
)
print(
    f"Features highly correlated with {target_feature}:\n{highly_correlated_features_df}"
)
