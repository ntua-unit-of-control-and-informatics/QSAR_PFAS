import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns

# Import the HGPR implementation from your files
from HGPR_Class import HeteroscedasticGPR

# Set random seed for reproducibility
np.random.seed(42)


# Function to generate synthetic heteroscedastic data
def generate_heteroscedastic_data(n_samples=100, sparse_regions=True):
    """
    Generate synthetic data with heteroscedastic noise and sparse regions.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    sparse_regions : bool
        Whether to create sparse regions in the data

    Returns:
    -----------
    X : array
        Input feature (1D)
    y : array
        Output values with heteroscedastic noise
    """

    # Define the true function
    def f(x):
        return 0.1 * x + 0.2 * x * np.sin(x)

    # Generate sparse and dense regions
    if sparse_regions:
        # Create sparse region between 2.5 and 4.5
        x_dense1 = np.random.uniform(0, 2.5, int(n_samples * 0.4))
        x_sparse = np.random.uniform(
            2.5, 4.5, int(n_samples * 0.1)
        )  # Fewer points here
        x_dense2 = np.random.uniform(4.5, 10, int(n_samples * 0.5))
        X = np.concatenate([x_dense1, x_sparse, x_dense2])
    else:
        X = np.random.uniform(0, 10, n_samples)

    # Generate y values with heteroscedastic noise
    # Noise increases with x
    true_y = f(X)
    noise_level = 0.1 + 0.05 * X**2 / 5  # Simple quadratic noise function
    noise = np.random.normal(0, noise_level)

    y = true_y + noise

    return X, y, true_y, noise_level


# Generate the dataset
X, y, true_y, noise_level = generate_heteroscedastic_data(n_samples=150)

# Reshape for sklearn
X_reshaped = X.reshape(-1, 1)

# Create a fine grid for prediction visualization
X_pred = np.linspace(0, 10, 1000).reshape(-1, 1)


# Calculate true function values on the fine grid for plotting
def true_function(x):
    return 0.1 * x + 0.2 * x * np.sin(x)


true_y_pred = true_function(X_pred.ravel())

# Calculate noise levels for the prediction grid
noise_level_pred = 0.1 + 0.05 * X_pred.ravel() ** 2 / 5

# 2. HGPR with polynomial degree = 0 (homoscedastic)
hgpr_homo = HeteroscedasticGPR(poly_degree=0)
hgpr_homo.fit(X_reshaped, y)
y_pred_homo, y_std_homo = hgpr_homo.predict(X_pred, return_std=True)
r2_homo = r2_score(y, hgpr_homo.predict(X_reshaped))

# 3. HGPR with polynomial degree = 3 (heteroscedastic)
hgpr_hetero = HeteroscedasticGPR(poly_degree=3)
hgpr_hetero.fit(X_reshaped, y)
y_pred_hetero, y_std_hetero = hgpr_hetero.predict(X_pred, return_std=True)
r2_hetero = r2_score(y, hgpr_hetero.predict(X_reshaped))

# Get decomposed uncertainties for the heteroscedastic model
y_pred_decomp, y_epis, y_alea = hgpr_hetero.predict(X_pred, return_decomposed=True)

# Visualize the results
plt.figure(figsize=(18, 16))

# 1. Data overview with heteroscedastic noise
plt.subplot(2, 2, 1)
plt.scatter(X, y, color="blue", alpha=0.6, label="Observed data")
plt.plot(X_pred.ravel(), true_y_pred, "r-", label="True function")
plt.fill_between(
    X_pred.ravel(),
    true_y_pred - 1.96 * noise_level_pred,
    true_y_pred + 1.96 * noise_level_pred,
    alpha=0.2,
    color="gray",
    label="95% CI",
)
plt.title("Synthetic Data with Heteroscedastic Noise")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 3. HGPR with poly_degree=0 prediction with uncertainty
plt.subplot(2, 2, 2)
plt.scatter(X, y, color="blue", alpha=0.5, label="Observed data")
plt.plot(
    X_pred.ravel(),
    y_pred_homo,
    "g-",
    label=f"HGPR (d=0) Prediction (R² = {r2_homo:.3f})",
)
plt.fill_between(
    X_pred.ravel(),
    y_pred_homo - 1.96 * y_std_homo,
    y_pred_homo + 1.96 * y_std_homo,
    alpha=0.2,
    color="green",
    label="95% CI",
)
plt.title("HGPR with Polynomial Degree = 0 (Homoscedastic)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 4. HGPR with poly_degree=3 prediction with uncertainty
plt.subplot(2, 2, 3)
plt.scatter(X, y, color="blue", alpha=0.5, label="Observed data")
plt.plot(
    X_pred.ravel(),
    y_pred_hetero,
    "purple",
    label=f"HGPR (d=3) Prediction (R² = {r2_hetero:.3f})",
)
plt.fill_between(
    X_pred.ravel(),
    y_pred_hetero - 1.96 * y_std_hetero,
    y_pred_hetero + 1.96 * y_std_hetero,
    alpha=0.2,
    color="purple",
    label="95% CI",
)
plt.title("HGPR with Polynomial Degree = 3 (Heteroscedastic)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 5. Decomposed uncertainty for HGPR with poly_degree=3
# plt.subplot(3, 2, 4)
# plt.plot(X_pred.ravel(), y_epis, "b-", label="Epistemic uncertainty (model)")
# plt.plot(X_pred.ravel(), y_alea, "r-", label="Aleatoric uncertainty (noise)")
# plt.title("Decomposed Uncertainty for HGPR (d=3)")
# plt.xlabel("X")
# plt.ylabel("Standard Deviation")
# plt.legend()
# plt.grid(True)

# 6. Comparison of standard deviations
plt.subplot(2, 2, 4)
# plt.plot(X_pred.ravel(), y_std_vanilla, "r-", label="Vanilla GPR")
plt.plot(X_pred.ravel(), y_std_homo, "g-", label="HGPR (d=0)")
plt.plot(X_pred.ravel(), y_std_hetero, "purple", label="HGPR (d=3)")
plt.plot(X_pred.ravel(), noise_level_pred, "k--", label="True noise level")
plt.title("Comparison of Predicted Uncertainties")
plt.xlabel("X")
plt.ylabel("Standard Deviation")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("hgpr_demonstration.png", dpi=300, bbox_inches="tight")
# plt.show()

# Create a separate figure for showing predicted vs observed with error bars
plt.figure(figsize=(18, 6))

# 2. HGPR with poly_degree=0
plt.subplot(1, 2, 1)
y_pred_homo_train = hgpr_homo.predict(X_reshaped)
_, y_std_homo_train = hgpr_homo.predict(X_reshaped, return_std=True)
plt.errorbar(
    y,
    y_pred_homo_train,
    yerr=1.96 * y_std_homo_train,
    fmt="o",
    alpha=0.5,
    capsize=3,
    label=f"HGPR d=0 (R² = {r2_homo:.3f})",
)
plt.plot([min(y), max(y)], [min(y), max(y)], "k--")
plt.title("HGPR (Homoscedastic)")
plt.xlabel("Observed y")
plt.ylabel("Predicted y")
plt.grid(True)

# 3. HGPR with poly_degree=3
plt.subplot(1, 2, 2)
y_pred_hetero_train = hgpr_hetero.predict(X_reshaped)
_, y_std_hetero_train = hgpr_hetero.predict(X_reshaped, return_std=True)
plt.errorbar(
    y,
    y_pred_hetero_train,
    yerr=1.96 * y_std_hetero_train,
    fmt="o",
    alpha=0.5,
    capsize=3,
    label=f"HGPR d=3 (R² = {r2_hetero:.3f})",
)
plt.plot([min(y), max(y)], [min(y), max(y)], "k--")
plt.title("HGPR (Heteroscedastic)")
plt.xlabel("Observed y")
plt.ylabel("Predicted y")
plt.grid(True)

plt.tight_layout()
plt.savefig("hgpr_prediction_errorbars.png", dpi=300, bbox_inches="tight")
# plt.show()

print("\nHGPR (Homoscedastic) Parameters:")
print(f"Length scales: {hgpr_homo.length_scales_}")
print(f"k_al: {hgpr_homo.k_al:.3f}")

print("\nHGPR (Heteroscedastic) Parameters:")
print(f"Length scales: {hgpr_hetero.length_scales_}")
print(f"k_al: {hgpr_hetero.k_al:.3f}")
print(f"Polynomial coefficients: {hgpr_hetero.theta_}")

# Compare model performance
print("\nModel Performance Comparison:")
# print(f"Vanilla GPR R²: {r2_vanilla:.3f}")
print(f"HGPR (Homoscedastic) R²: {r2_homo:.3f}")
print(f"HGPR (Heteroscedastic) R²: {r2_hetero:.3f}")
