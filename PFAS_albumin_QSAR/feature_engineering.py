from jaqpotpy.datasets import JaqpotTabularDataset
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
import shap


df_train = pd.read_csv("PFAS_albumin_QSAR/Train_Albumin_Binding_Data.csv")
df_train = df_train[df_train["Ka"] != 0]
df_train[["Ka"]] = np.log10(df_train[["Ka"]])

# Create a JaqpotpyDataset objects
x_cols = ["Temperature", "Albumin_Type", "Method"]
categorical_cols = ["Albumin_Type", "Method"]
featurizers = [RDKitDescriptors()]

train_dataset = JaqpotTabularDataset(
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

# Remove features with low variance
selector = VarianceThreshold(threshold=1)
selected_descriptors = selector.fit_transform(scaled_descriptors_df)
selected_descriptors_df = pd.DataFrame(
    selected_descriptors, columns=scaled_descriptors_df.columns[selector.get_support()]
)
dropped_columns = scaled_descriptors_df.columns[~selector.get_support()].tolist()
print(f"Number of dropped columns = {len(dropped_columns)}")

# Correlation matrix

# Compute the correlation matrix
corr_matrix = selected_descriptors_df.corr()


# Plot the heatmap
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
# plt.title("Correlation Matrix of Selected Descriptors")
# plt.show()

# Find columns with correlation higher than 0.9
high_corr_pairs = (
    corr_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)
)
high_corr_pairs = high_corr_pairs[high_corr_pairs < 1].reset_index()
high_corr_pairs.columns = ["Feature1", "Feature2", "Correlation"]

# Drop columns with lower mean correlation
to_drop = set()
for feature1, feature2 in high_corr_pairs[high_corr_pairs["Correlation"] > 0.7][
    ["Feature1", "Feature2"]
].values:
    if feature1 not in to_drop and feature2 not in to_drop:
        mean_corr_feature1 = corr_matrix[feature1].mean()
        mean_corr_feature2 = corr_matrix[feature2].mean()
        if mean_corr_feature1 < mean_corr_feature2:
            to_drop.add(feature1)
        else:
            to_drop.add(feature2)

selected_descriptors_df = selected_descriptors_df.drop(columns=to_drop)
print(f"Number of columns dropped due to high correlation = {len(to_drop)}")
print(selected_descriptors_df.columns)
print(f"Number of features after feature selection: {selected_descriptors_df.shape[1]}")

final_train_df = pd.concat(
    [
        categorical_data,
        selected_descriptors_df,
    ],
    axis=1,
    ignore_index=True,
)
final_train_df.columns = categorical_cols + selected_descriptors_df.columns.tolist()
final_train_df.reset_index(drop=True, inplace=True)

# Perform one-hot encoding for categorical columns
final_train_df = pd.get_dummies(
    final_train_df, columns=categorical_cols, drop_first=True
)

print(final_train_df.columns)
# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Initialize RFE
rfe = RFE(estimator=model, n_features_to_select=3, step=1, verbose=1)

# Fit RFE
rfe.fit(final_train_df, train_dataset.y.to_numpy().ravel())
# Get the selected features
# selected_features = final_train_df.columns[rfe.support_]

# Manually select features
selected_features = final_train_df.columns[rfe.support_]

print(f"Selected features: {selected_features}")

# Transform the dataset to contain only the selected features
cat_cols = [
    "Albumin_Type_HSA",
    "Albumin_Type_PSA",
    "Albumin_Type_RSA",
    # "Method_DSF",
    "Method_ESI-MS",
    "Method_Equilibrium dialysis",
    "Method_Isothermal Titration Calorimetry",
    "Method_Microdesalting Column Separation",
    "Method_fluorescence quenching",
    "Method_nanoESI-MS",
    "Method_ultrafiltration centrifugation",
]
final_features_df = final_train_df[selected_features.tolist() + cat_cols]
print(f"Final features: {final_features_df.columns}")
print(f"Number of features after RFE: {final_features_df.shape[1]}")

# Transform Method_DSF to 0 and 1
# Transform boolean features to integers 0 and 1
bool_cols = final_features_df.select_dtypes(include=["bool"]).columns
final_features_df[bool_cols] = (
    final_features_df[bool_cols].replace({True: 1, False: 0}).astype("int32")
)
# Fit the model on the final features
model.fit(final_features_df, train_dataset.y.to_numpy().ravel())

# sys.exit()
# Create a SHAP explainer
explainer = shap.Explainer(model, final_features_df)

# Calculate SHAP values
shap_values = explainer(final_features_df)

# Specify the features to include in the SHAP plot
features_to_include = ["BalabanJ", "PEOE_VSA2"]

# Filter the SHAP values and the final features dataframe
filtered_shap_values = shap_values[:, features_to_include]
filtered_final_features_df = final_features_df[features_to_include]

# Plot the SHAP values for the specified features
shap.summary_plot(filtered_shap_values, filtered_final_features_df)
