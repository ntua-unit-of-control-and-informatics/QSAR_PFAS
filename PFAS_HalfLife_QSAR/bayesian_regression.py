import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Prepare the Data
df = pd.read_csv("Datasets/bayesian_regression_df.csv")
print(df.head())
print(df.isnull().sum())
df = df.dropna()

# 2. Encode Categorical Variables
categorical_features = ["PFAS", "Sex"]
numerical_features = ["Age"]  # Add other numerical features if present

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

X = df.drop("Half_Life", axis=1)
y = df["Half_Life"]

X_transformed = preprocessor.fit_transform(X)
ohe = preprocessor.named_transformers_["cat"]
ohe_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = np.concatenate([ohe_feature_names, numerical_features])
X_transformed = pd.DataFrame(X_transformed.toarray(), columns=feature_names)
print(X_transformed.head())

# 3. Split the Data
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42, stratify=df["PFAS"]
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 4. Train the Bayesian Ridge Model
model = BayesianRidge()
model.fit(X_train, y_train)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 5. Make Predictions with Uncertainty
y_pred, y_std = model.predict(X_test, return_std=True)
results = pd.DataFrame(
    {"Actual_Half_Life": y_test, "Predicted_Mean": y_pred, "Predicted_SD": y_std}
)
print(results.head())

# 6. Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=results["Predicted_SD"], y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Standard Deviation")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs. Predicted Uncertainty")
plt.show()

nll = 0.5 * np.log(2 * np.pi * y_std**2) + (residuals**2) / (2 * y_std**2)
print(f"Average Negative Log-Likelihood (NLL): {nll.mean():.2f}")

# 7. Visualize Predictions and Uncertainties
plt.figure(figsize=(10, 8))
plt.errorbar(
    results["Predicted_Mean"],
    results["Actual_Half_Life"],
    xerr=results["Predicted_SD"],
    fmt="o",
    ecolor="lightgray",
    alpha=0.7,
    label="Data points",
)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Ideal Fit")
plt.xlabel("Predicted Mean Half-Life")
plt.ylabel("Actual Half-Life")
plt.title("Actual vs. Predicted Half-Life with Uncertainty")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(results["Predicted_SD"], bins=20, kde=True)
plt.xlabel("Predicted Standard Deviation")
plt.title("Distribution of Predicted Uncertainties")
plt.show()
