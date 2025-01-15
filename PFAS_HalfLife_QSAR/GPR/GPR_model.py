import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

train_df = pd.read_csv("GPR/train_data.csv")
test_df = pd.read_csv("GPR/test_data.csv")
columns_to_keep = train_df.columns.drop("Study").tolist()
categorical_columns = ["sex"]

# Encode the Study column
train_study = train_df["Study"].astype("category")
test_study = test_df["Study"].astype("category")

train_study_codes = train_study.cat.codes
test_study_codes = test_study.cat.codes

# Print the mapping of studies and indexes
study_mapping = dict(enumerate(test_study.cat.categories))
# print("Study to Index Mapping:")
# for index, study in study_mapping.items():
#     print(f"{index}: {study}")

# Drop the Study column and keep only the necessary columns
train_df = train_df[columns_to_keep]
test_df = test_df[columns_to_keep]

# One-hot encode the categorical columns
# Define the column transformer with OneHotEncoder for categorical columns
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), categorical_columns)],
    remainder="passthrough",
)

# Fit and transform the training data
train_df = pd.DataFrame(
    preprocessor.fit_transform(train_df), columns=preprocessor.get_feature_names_out()
)
test_df = pd.DataFrame(
    preprocessor.transform(test_df), columns=preprocessor.get_feature_names_out()
)

# Rename the column remainder__half_life_days to half_life_days
train_df.rename(columns={"remainder__half_life_days": "half_life_days"}, inplace=True)
test_df.rename(columns={"remainder__half_life_days": "half_life_days"}, inplace=True)

x_train, y_train = train_df.drop("half_life_days", axis=1), train_df["half_life_days"]
x_test, y_test = test_df.drop("half_life_days", axis=1), test_df["half_life_days"]

kernel = C(1.0) * RBF([0.5] * x_train.shape[1])
model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    optimizer="fmin_l_bfgs_b",
    random_state=0,
    normalize_y=True,
    alpha=1e-10,
)
model.fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred, std = model.predict(x_test, return_std=True)
y_obs = y_test

r2_train = r2_score(y_train, y_pred_train)
print(f"R2 score on train data: {r2_train:.2f}")
r2_test = r2_score(y_obs, y_pred)
print(f"R2 score on test data: {r2_test:.2f}")

# Create the plot
plt.figure(figsize=(10, 8))

# Plot the scatter points of predicted vs observed for test data
plt.scatter(y_obs, y_pred, c="blue", alpha=0.5, label="Test Predictions")

# Add error bars using the predicted standard deviation for test data
plt.errorbar(
    y_obs,
    y_pred,
    yerr=2 * std,  # 2*std for 95% confidence interval
    xerr=None,  # No error bars on x-axis
    fmt="none",  # Don't add additional markers
    ecolor="black",
    alpha=0.3,
    capsize=3,  # Add caps to the error bars
    label="95% Confidence Interval",
)

# Annotate points with the index of the Study column
# for i, study_index in enumerate(test_study_codes):
#     plt.annotate(study_index, (y_obs.iloc[i], y_pred[i]), fontsize=8, alpha=0.7)

# Plot diagonal line (perfect predictions)
min_val = 0
max_val = max(y_obs.max(), y_pred.max())
plt.plot(
    [min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Perfect Prediction"
)

# Add labels and title
plt.xlabel("Observed Values")
plt.ylabel("Predicted Values")
plt.title("GPR Predictions vs Observed Values with Uncertainty")
plt.legend()

# Make plot square with equal axes and set limits
plt.axis("equal")
plt.xlim([0, 5000])
plt.ylim([0, 5000])
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
