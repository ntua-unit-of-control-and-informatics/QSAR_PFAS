import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
)
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
import matplotlib.pyplot as plt

train_df = pd.read_csv("PFAS_HalfLife_QSAR/GPR/train_data.csv")
test_df = pd.read_csv("PFAS_HalfLife_QSAR/GPR/test_data.csv")
train_df.drop(columns=["Study"], inplace=True)
test_df.drop(columns=["Study"], inplace=True)
categorical_columns = ["sex", "adult", "half_life_type"]

# One-hot encode the categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_columns)
    ],
    remainder="passthrough",
)

# Get the non-categorical columns
numeric_columns = [col for col in train_df.columns if col not in categorical_columns]

# Fit and transform the training data
train_encoded = preprocessor.fit_transform(train_df)
test_encoded = preprocessor.transform(test_df)

# Get the feature names after transformation
encoder = preprocessor.named_transformers_["cat"]
encoded_features = encoder.get_feature_names_out(categorical_columns)
feature_names = np.concatenate([encoded_features, numeric_columns])

# Convert the encoded data back to DataFrame with proper column names
train_df = pd.DataFrame(train_encoded, columns=feature_names)
test_df = pd.DataFrame(test_encoded, columns=feature_names)

x_train, y_train = train_df.drop("half_life_days", axis=1), train_df["half_life_days"]
x_test, y_test = test_df.drop("half_life_days", axis=1), test_df["half_life_days"]

scaler = StandardScaler()
# Get the indices of the one-hot encoded features
one_hot_indices = [i for i, col in enumerate(feature_names) if col in encoded_features]

# Scale only the non-one-hot encoded features
x_train.loc[:, ~x_train.columns.isin(encoded_features)] = scaler.fit_transform(
    x_train.loc[:, ~x_train.columns.isin(encoded_features)]
)
x_test.loc[:, ~x_test.columns.isin(encoded_features)] = scaler.fit_transform(
    x_test.loc[:, ~x_test.columns.isin(encoded_features)]
)

# # Perform feature elimination with variance threshold > 1
# Exclude LogKa and one hot encoded features from the filters
excluded_from_filter = [
    "sex_female",
    "sex_male",
    "adult_1",
    "half_life_type_intrinsic",
    "LogKa",
]

# Apply VarianceThreshold to all columns except LogKa
selector = VarianceThreshold(threshold=1)
x_train_filtered = selector.fit_transform(x_train.drop(columns=excluded_from_filter))
x_train_filtered.shape
x_test_filtered = selector.transform(x_test.drop(columns=excluded_from_filter))

# Get the retained column names after variance threshold
retained_columns = x_train.drop(columns=excluded_from_filter).columns[
    selector.get_support()
]
# Convert filtered arrays back to DataFrames with retained column names
x_train_filtered_df = pd.DataFrame(x_train_filtered, columns=retained_columns)
x_test_filtered_df = pd.DataFrame(x_test_filtered, columns=retained_columns)

# Add Excluded from filtering columns back to the filtered data
x_train = pd.concat(
    [x_train.loc[:, x_train.columns.isin(excluded_from_filter)], x_train_filtered_df],
    axis=1,
)
x_test = pd.concat(
    [x_test.loc[:, x_test.columns.isin(excluded_from_filter)], x_test_filtered_df],
    axis=1,
)

print(f"Final number of dimensions: {x_train.shape[1]}")

##############Kernel##############
# kernel = ConstantKernel(1.0, (1e-1, 1e1)) * (
#     Matern(
#         length_scale=[1.0] * x_train.shape[1], nu=1.5, length_scale_bounds=(1e-2, 1e2)
#     )
#     + RationalQuadratic(length_scale=1.0, alpha=0.5, length_scale_bounds=(1e-2, 1e2))
# ) + WhiteKernel(0.5, (1e-2, 1e1))

kernel = ConstantKernel(1.0, (1e-1, 1e1)) * (
    Matern(
        length_scale=[1.3] * x_train.shape[1], nu=2.5, length_scale_bounds=(1e-2, 1e2)
    )
    + RationalQuadratic(length_scale=1.1, alpha=0.9, length_scale_bounds=(1e-2, 1e2))
) + WhiteKernel(0.1, (1e-2, 1e-1))

model = GaussianProcessRegressor(
    kernel=kernel, normalize_y=True, n_restarts_optimizer=9, alpha=1e-10
)

model.fit(x_train, y_train)

y_pred_train, y_train_std = model.predict(x_train, return_std=True)
y_pred_test, y_test_std = model.predict(x_test, return_std=True)

r2_train = r2_score(y_train, y_pred_train)
print(f"R2 score on train data: {r2_train:.2f}")
r2_test = r2_score(y_test, y_pred_test)
print(f"R2 score on test data: {r2_test:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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


def check_coverage(y_true, y_pred_test, y_std):
    # Check what percentage of true values fall within:
    # 1 SD (should be ~68%)
    in_1sd = np.mean(np.abs(y_true - y_pred_test) < y_std)
    # 2 SD (should be ~95%)
    in_2sd = np.mean(np.abs(y_true - y_pred_test) < 2 * y_std)
    # 3 SD (should be ~99.7%)
    in_3sd = np.mean(np.abs(y_true - y_pred_test) < 3 * y_std)

    return {
        "1SD_coverage": in_1sd * 100,
        "2SD_coverage": in_2sd * 100,
        "3SD_coverage": in_3sd * 100,
    }


train_coverage = check_coverage(y_train, y_pred_train, y_train_std)
test_coverage = check_coverage(y_test, y_pred_test, y_test_std)
print(train_coverage)
print(test_coverage)
