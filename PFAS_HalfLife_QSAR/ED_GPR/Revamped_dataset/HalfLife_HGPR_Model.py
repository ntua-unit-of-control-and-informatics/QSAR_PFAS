import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve
from scipy.stats import norm
import warnings
from jaqpotpy.descriptors import RDKitDescriptors, TopologicalFingerprint
from jaqpotpy.datasets import JaqpotTabularDataset


def calculate_elpd(y_true, y_pred_mean, y_pred_std):
    """
    Calculate Expected Log Predictive Density (ELPD) for external validation.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred_mean : array-like
        Predicted mean values
    y_pred_std : array-like
        Predicted standard deviations

    Returns:
    --------
    elpd : float
        Total ELPD score (higher is better)
    avg_elpd : float
        Average ELPD per observation (higher is better)
    """
    # Ensure arrays
    y_true = np.array(y_true)
    y_pred_mean = np.array(y_pred_mean)
    y_pred_std = np.array(y_pred_std)

    # Handle potential numerical issues with standard deviation
    y_pred_std = np.maximum(y_pred_std, 1e-6)

    # Calculate log predictive density for each observation
    log_pred_density = norm.logpdf(y_true, loc=y_pred_mean, scale=y_pred_std)

    # Calculate ELPD (sum of log predictive densities)
    elpd = np.sum(log_pred_density)

    # Calculate average ELPD per observation
    avg_elpd = np.mean(log_pred_density)

    return elpd, avg_elpd


# Q2: Predictive Squared Correlation Coefficient
def q2_score(y_true, y_pred, y_train_mean):
    """
    Calculate the Q2 score (Predictive Squared Correlation Coefficient).

    Parameters:
    - y_true: True values of the dependent variable.
    - y_pred: Predicted values of the dependent variable.
    - y_train_mean: Mean of the training set's dependent variable.

    Returns:
    - Q2 score.
    """
    ss_total = np.sum((y_true - y_train_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    q2 = 1 - (ss_residual / ss_total)
    return q2


class HeteroscedasticGPR(BaseEstimator, RegressorMixin):
    def __init__(self, poly_degree=2, k_al=1.0, length_scale=None, priors=None):
        """
        Heteroscedastic Gaussian Process Regressor based on Ozbayram et al.

        Parameters:
        -----------
        poly_degree : int
            Degree of polynomial for noise modeling
        k_al : float
            Initial aleatoric uncertainty scaling factor
        length_scale : float or array-like
            Initial length scale(s) for ARD kernel
        priors : dict
            Prior distribution parameters for regularization
        """
        self.poly_degree = poly_degree
        self.k_al = k_al
        self.length_scale = length_scale

        # Default priors based on paper's recommendations
        default_priors = {
            "theta_mu": 0.0,
            "theta_var": 1.0,
            "k_al_mu": 0.0,
            "k_al_var": 1.0,
            "l_mu": 0.0,
            "l_var": 1.0,
        }
        self.priors = priors if priors is not None else default_priors

    def _build_polynomial_features(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        features = []

        # Start from degree 1 to match the paper's formulation
        for degree in range(1, self.poly_degree + 1):
            for i in range(n_features):
                features.append(X[:, i] ** degree)

        return np.column_stack(features) if features else np.ones((n_samples, 1))

    def _compute_noise_variance(self, X, theta):
        """Compute heteroscedastic noise variance"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_poly = self._build_polynomial_features(X)
        if self.poly_degree == 0:
            return np.full(X.shape[0], self.k_al**2 + 1e-6)

        # Clip the exponent to avoid numerical issues
        exponent = np.clip(X_poly @ theta, -10, 10)
        return (self.k_al * np.exp(exponent)) ** 2 + 1e-6

    def _compute_kernel(self, X1, X2=None):
        """Compute ARD-RBF kernel matrix"""
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2 is not None and X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        if X2 is None:
            X2 = X1

        n_dim = X1.shape[1]
        length_scales = self.length_scales_

        # Compute pairwise distances with ARD
        x1_squared = np.sum(X1**2 / length_scales.reshape(1, -1) ** 2, axis=1)
        x2_squared = np.sum(X2**2 / length_scales.reshape(1, -1) ** 2, axis=1)

        K = np.exp(
            -0.5
            * (
                x1_squared.reshape(-1, 1)
                + x2_squared.reshape(1, -1)
                - 2
                * np.dot(
                    X1 / length_scales.reshape(1, -1),
                    (X2 / length_scales.reshape(1, -1)).T,
                )
            )
        )
        return K

    def _log_marginal_likelihood(self, params):
        """Compute log marginal likelihood with proper priors"""
        try:
            # Unpack parameters
            n_features = self.X_train_.shape[1]
            log_k_al = params[0]
            log_length_scales = params[1 : n_features + 1]
            theta = params[n_features + 1 :]

            k_al = np.exp(log_k_al)
            length_scales = np.exp(log_length_scales)

            # Store current parameters
            self.k_al = k_al
            self.length_scales_ = length_scales

            # Compute kernel matrix
            K = self._compute_kernel(self.X_train_)
            K += np.eye(len(K)) * 1e-6  # Numerical stability

            # Compute noise variance
            noise_var = self._compute_noise_variance(self.X_train_, theta)
            K_noise = K + np.diag(noise_var)

            # Cholesky decomposition with robust handling
            jitter = 1e-6
            max_tries = 5
            for i in range(max_tries):
                try:
                    L = cholesky(K_noise + np.eye(len(K_noise)) * jitter, lower=True)
                    break
                except:
                    jitter *= 10
                    if i == max_tries - 1:
                        return 1e6

            # Compute log likelihood
            try:
                alpha = cho_solve((L, True), self.y_train_)
                if np.any(np.isnan(alpha)) or np.any(np.isinf(alpha)):
                    return 1e6
            except:
                return 1e6

            # Log likelihood
            log_likelihood = -0.5 * np.dot(self.y_train_, alpha)
            log_likelihood -= np.sum(np.log(np.diag(L)))
            log_likelihood -= 0.5 * self.n_samples_ * np.log(2 * np.pi)

            # Add regularization terms (prior distributions) as per paper
            # Prior for k_al (log-normal)
            R_k_al = -np.log(
                k_al * np.sqrt(2 * np.pi) * np.sqrt(self.priors["k_al_var"])
            ) - ((log_k_al - self.priors["k_al_mu"]) ** 2) / (
                2 * self.priors["k_al_var"]
            )

            # Prior for length scales (log-normal)
            R_l = 0
            for log_l in log_length_scales:
                R_l += -np.log(
                    np.exp(log_l) * np.sqrt(2 * np.pi) * np.sqrt(self.priors["l_var"])
                ) - ((log_l - self.priors["l_mu"]) ** 2) / (2 * self.priors["l_var"])

            # Prior for theta (normal)
            R_theta = -0.5 * np.sum(theta**2) / self.priors["theta_var"]

            log_likelihood += R_k_al + R_l + R_theta

            if np.isnan(log_likelihood) or np.isinf(log_likelihood):
                return 1e6

            return -float(log_likelihood)

        except Exception as e:
            return 1e6

    def fit(self, X, y):
        """Fit the model to data"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale the data for numerical stability
        self.X_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

        self.X_train_ = self.X_scaler_.fit_transform(X)
        self.y_train_ = self.y_scaler_.fit_transform(y.reshape(-1, 1)).ravel()
        self.n_samples_, self.n_features_ = self.X_train_.shape

        # Store original data ranges for rescaling
        self.x_min_ = X.min(axis=0)
        self.x_range_ = X.max(axis=0) - X.min(axis=0)
        self.y_min_ = y.min()
        self.y_range_ = y.max() - y.min()

        # Initialize length scales if not provided
        if self.length_scale is None:
            self.length_scale = np.ones(self.n_features_)
        elif np.isscalar(self.length_scale):
            self.length_scale = np.full(self.n_features_, self.length_scale)

        self.length_scales_ = np.array(self.length_scale)

        # Initial parameters: [log(k_al), log(length_scales), theta]
        n_poly_params = self._build_polynomial_features(self.X_train_).shape[1]

        initial_params = np.concatenate(
            [[np.log(self.k_al)], np.log(self.length_scales_), np.zeros(n_poly_params)]
        )

        # Parameter bounds
        bounds = (
            [(-3, 3)]  # log(k_al)
            + [(-5, 5)] * self.n_features_  # log(length_scales)
            + [(-3, 3)] * n_poly_params  # theta coefficients
        )

        # Optimize parameters
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            opt_res = minimize(
                self._log_marginal_likelihood,
                initial_params,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": 200},
            )

        # Store optimized parameters
        opt_params = opt_res.x
        self.k_al = np.exp(opt_params[0])
        self.length_scales_ = np.exp(opt_params[1 : self.n_features_ + 1])
        self.theta_ = opt_params[self.n_features_ + 1 :]

        return self

    def predict(self, X, return_std=False, return_decomposed=False):
        """
        Make predictions with uncertainty

        Parameters:
        -----------
        X : array-like
            Test points
        return_std : bool
            Whether to return standard deviation
        return_decomposed : bool
            Whether to return decomposed uncertainties (epistemic and aleatoric)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Scale the input
        X_scaled = self.X_scaler_.transform(X)

        # Compute kernel matrices
        K_train_test = self._compute_kernel(self.X_train_, X_scaled)
        K_test = self._compute_kernel(X_scaled)
        K_train = self._compute_kernel(self.X_train_)

        # Add noise variance
        noise_var_train = self._compute_noise_variance(self.X_train_, self.theta_)
        K_train_noise = K_train + np.diag(noise_var_train) + np.eye(len(K_train)) * 1e-6

        try:
            # Compute mean prediction
            L = cholesky(K_train_noise, lower=True)
            alpha = cho_solve((L, True), self.y_train_)
            y_mean = K_train_test.T @ alpha

            # Unscale predictions to original scale
            y_mean = self.y_scaler_.inverse_transform(y_mean.reshape(-1, 1)).ravel()

            if return_std or return_decomposed:
                # Compute uncertainties
                v = cho_solve((L, True), K_train_test)

                # Epistemic uncertainty (model uncertainty)
                epistemic_var = np.diag(K_test - K_train_test.T @ v)

                # Aleatoric uncertainty (noise)
                aleatoric_var = self._compute_noise_variance(X_scaled, self.theta_)

                # Scale variances back to original scale
                var_scale = self.y_scaler_.scale_**2
                epistemic_var = epistemic_var * var_scale
                aleatoric_var = aleatoric_var * var_scale

                # Total variance
                total_var = epistemic_var + aleatoric_var

                if return_decomposed:
                    return y_mean, np.sqrt(epistemic_var), np.sqrt(aleatoric_var)

                return y_mean, np.sqrt(total_var)

            return y_mean

        except Exception as e:
            print(f"Error in prediction: {e}")
            if return_decomposed:
                return (
                    np.zeros_like(X[:, 0]),
                    np.zeros_like(X[:, 0]),
                    np.zeros_like(X[:, 0]),
                )
            if return_std:
                return np.zeros_like(X[:, 0]), np.ones_like(X[:, 0])
            return np.zeros_like(X[:, 0])


if __name__ == "__main__":
    # Load datasets
    train_df = pd.read_csv(
        "PFAS_HalfLife_QSAR/Datasets/Revamped_dataset/Half-life_dataset_Human_train.csv"
    )
    test_df = pd.read_csv(
        "PFAS_HalfLife_QSAR/Datasets/Revamped_dataset/Half-life_dataset_Human_test.csv"
    )
    halflife_stats = pd.read_csv(
        "PFAS_HalfLife_QSAR/Datasets/Revamped_dataset/Halflife_stats.csv"
    )

    # Define base columns (these should be adjusted based on the forward selection results)
    x_cols = ["LogKa"]
    categorical_cols = []

    # Setup featurizers
    featurizers = [RDKitDescriptors(), TopologicalFingerprint()]

    # Create dataset
    train_dataset = JaqpotTabularDataset(
        df=train_df,
        y_cols="half_life",
        x_cols=x_cols,
        smiles_cols="SMILES",
        featurizer=featurizers,
        task="regression",
    )

    # Features selected from forward selection
    # These should be replaced with the actual features selected by the forward selection algorithm
    selected_features = ["LogKa", "BCUT2D_LOGPLOW", "fr_halogen"]

    # Prepare training data
    X = train_dataset.X[selected_features].copy()

    # Handle categorical features if any are selected
    categorical_selected = [col for col in selected_features if col in categorical_cols]
    if categorical_selected:
        X = pd.get_dummies(X, columns=categorical_selected, drop_first=True)

    X = X.values
    y = train_dataset.y.values

    # Set model parameters with appropriate priors
    model_params = {
        "poly_degree": 1,
        "k_al": 1.0,
        "priors": {
            "theta_mu": 0.0,
            "theta_var": 1.0
            / len(selected_features),  # Adjust variance based on feature count
            "k_al_mu": 0.0,
            "k_al_var": 1.0,
            "l_mu": 0.0,
            "l_var": 1.0,
        },
    }

    # Create and fit HGPR model
    model = HeteroscedasticGPR(**model_params)
    model.fit(X, y)

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = HeteroscedasticGPR(**model_params)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

        r2_val = r2_score(y_val, y_val_pred)
        rmse_val = np.sqrt(np.mean((y_val - y_val_pred.reshape(-1, 1)) ** 2))
        mae_val = np.mean(np.abs(y_val - y_val_pred.reshape(-1, 1)))
        r2_scores.append(r2_val)
        rmse_scores.append(rmse_val)
        mae_scores.append(mae_val)

    # Print average R2 score
    average_r2 = np.mean(r2_scores)
    average_rmse = np.mean(rmse_scores)
    average_mae = np.mean(mae_scores)
    print(f"Average R2 score from 5-fold cross-validation: {average_r2:.3f}")
    print(f"Average RMSE from 5-fold cross-validation: {average_rmse:.3f}")
    print(f"Average MAE from 5-fold cross-validation: {average_mae:.3f}")

    # Refit model on all training data
    model = HeteroscedasticGPR(**model_params)
    model.fit(X, y)

    # Print optimized parameters
    print("\nOptimized parameters:")
    print(f"Length scales: {model.length_scales_}")
    print(f"k_al: {model.k_al:.3f}")
    print(f"Polynomial coefficients: {model.theta_}")

    # Predict on training data
    y_train_pred, y_train_sd_pred = model.predict(X, return_std=True)
    r2_train = r2_score(y, y_train_pred)
    rmse_train = np.sqrt(np.mean((y - y_train_pred.reshape(-1, 1)) ** 2))
    mae_train = np.mean(np.abs(y - y_train_pred.reshape(-1, 1)))
    elpd_train, avg_elpd_train = calculate_elpd(
        y.flatten(), y_train_pred, y_train_sd_pred
    )
    print(f"R2 score on training data: {r2_train:.3f}")
    print(f"RMSE on training data: {rmse_train:.3f}")
    print(f"MAE on training data: {mae_train:.3f}")
    print(f"ELPD on training data: {elpd_train:.3f}")
    print(f"Average ELPD on training data: {avg_elpd_train:.3f}")
    # Prepare test dataset
    test_dataset = JaqpotTabularDataset(
        df=test_df,
        y_cols="half_life",
        x_cols=x_cols,
        smiles_cols="SMILES",
        featurizer=featurizers,
        task="regression",
    )

    X_test = test_dataset.X[selected_features].copy()

    # Handle categorical features if any are selected
    if categorical_selected:
        X_test = pd.get_dummies(X_test, columns=categorical_selected, drop_first=True)

    X_test = X_test.values
    y_test = test_dataset.y.values

    # Predict on test data with uncertainty
    y_test_mean, y_test_std = model.predict(X_test, return_std=True)

    # Calculate R2 score on test data
    r2_test = r2_score(y_test, y_test_mean)
    rmse_test = np.sqrt(np.mean((y_test - y_test_mean.reshape(-1, 1)) ** 2))
    mae_test = np.mean(np.abs(y_test - y_test_mean.reshape(-1, 1)))

    print(f"R2 score on test data: {r2_test:.3f}")
    print(
        f"Q2 score on test data: {q2_score(y_test, y_test_mean.reshape(-1,1), y.mean()):.3f}"
    )
    print(f"RMSE on test data: {rmse_test:.3f}")
    print(f"MAE on test data: {mae_test:.3f}")
    # Decomposed uncertainty prediction for test data
    y_test_mean, y_test_epis, y_test_alea = model.predict(
        X_test, return_decomposed=True
    )
    print(f"Average epistemic uncertainty: {np.mean(y_test_epis):.3f}")
    print(f"Average aleatoric uncertainty: {np.mean(y_test_alea):.3f}")

    # Save test predictions, true values, and SMILES to CSV
    test_predictions_df = pd.DataFrame(
        {
            "SMILES": test_df["SMILES"],
            "Observed_Half_Life": y_test.flatten(),
            "Predicted_Half_Life": y_test_mean,
            "Prediction_Std": y_test_std,
            "Epistemic_Uncertainty": y_test_epis,
            "Aleatoric_Uncertainty": y_test_alea,
        }
    )

    # Add the original selected features from the test set for reference
    for feature in selected_features:
        if feature in test_dataset.X.columns:
            test_predictions_df[feature] = test_dataset.X[feature].values

    # Sort the test predictions by SMILES
    test_predictions_df = test_predictions_df.sort_values("Predicted_Half_Life")

    # Save to CSV
    test_predictions_df.to_csv(
        "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/hgpr_test_predictions.csv",
        index=False,
    )
    print(
        "Test predictions saved to 'PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/hgpr_test_predictions.csv'"
    )

    # ---------------------------------------------------------------------
    #                       Plotting the results
    # ---------------------------------------------------------------------

    # Plot predicted mean and standard deviation for each pfas vs
    # the true mean and sd
    halflife_stats_dataset = JaqpotTabularDataset(
        df=halflife_stats,
        y_cols=None,
        x_cols=x_cols,
        smiles_cols="SMILES",
        featurizer=featurizers,
        task="regression",
    )
    halflife_stats_dataset = halflife_stats_dataset.X[selected_features].copy().values
    predictions_per_pfas_mean, predictions_per_pfas_mean_sd = model.predict(
        halflife_stats_dataset, return_std=True
    )

    data_to_plot = pd.DataFrame(
        {
            "SMILES": halflife_stats["SMILES"],
            "PFAS": halflife_stats["PFAS"],
            "Observed_Half_Life": halflife_stats["half_life_mean"],
            "Observed_Half_Life_SD": halflife_stats["half_life_sd"],
            "Predicted_Half_Life": predictions_per_pfas_mean,
            "Predicted_Half_Life_SD": predictions_per_pfas_mean_sd,
        }
    )
    # Create the plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

    # Common styling parameters
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10
    LEGEND_SIZE = 10

    # Calculate plot limits including error bars
    x_min = min(
        data_to_plot["Observed_Half_Life"] - data_to_plot["Observed_Half_Life_SD"]
    )
    x_max = max(
        data_to_plot["Observed_Half_Life"] + data_to_plot["Observed_Half_Life_SD"]
    )
    y_min = min(
        data_to_plot["Predicted_Half_Life"] - data_to_plot["Predicted_Half_Life_SD"]
    )
    y_max = max(
        data_to_plot["Predicted_Half_Life"] + data_to_plot["Predicted_Half_Life_SD"]
    )

    # Make sure we start at 0 and add padding
    x_min = max(0, x_min - 0.5)  # At least 0, with some padding
    y_min = max(0, y_min - 0.5)  # At least 0, with some padding
    x_max = x_max + 0.5  # Add padding
    y_max = y_max + 0.5  # Add padding

    # Use the maximum of both for square plot
    max_value = max(x_max, y_max)
    plot_range = [0, max_value]

    # First add error bars (without markers)
    ax.errorbar(
        data_to_plot["Observed_Half_Life"],
        data_to_plot["Predicted_Half_Life"],
        xerr=data_to_plot["Observed_Half_Life_SD"],
        yerr=data_to_plot["Predicted_Half_Life_SD"],
        fmt="none",  # No markers, just error bars
        ecolor="gray",
        capsize=3,
        alpha=0.4,
        elinewidth=0.8,
    )

    # Get unique PFAS compounds for coloring
    unique_pfas = data_to_plot["PFAS"].unique()

    # Create a categorical colormap for individual PFAS compounds
    # Using a qualitative colormap appropriate for categories
    from matplotlib.cm import get_cmap
    from matplotlib.colors import ListedColormap

    # Choose a colormap with distinct colors based on number of PFAS compounds
    if len(unique_pfas) <= 10:
        cmap_name = "tab10"
    elif len(unique_pfas) <= 20:
        cmap_name = "tab20"
    elif len(unique_pfas) <= 30:
        # Combine tab20 with tab20b for up to 40 colors
        cmap1 = get_cmap("tab20")
        cmap2 = get_cmap("tab20b")
        colors = [cmap1(i) for i in range(20)]
        colors.extend([cmap2(i) for i in range(len(unique_pfas) - 20)])
        custom_cmap = ListedColormap(colors[: len(unique_pfas)])
    else:
        # For even more colors, fall back to a continuous colormap
        cmap_name = "viridis"
        cmap = get_cmap(cmap_name)
        colors = [cmap(i / len(unique_pfas)) for i in range(len(unique_pfas))]
        custom_cmap = ListedColormap(colors)

    # If we're using a standard colormap without the extension
    if len(unique_pfas) <= 20:
        cmap = get_cmap(cmap_name)
        colors = [cmap(i) for i in range(len(unique_pfas))]
        custom_cmap = ListedColormap(colors[: len(unique_pfas)])

    # Create a mapping of PFAS compounds to numeric indices for coloring
    pfas_to_index = {pfas: i for i, pfas in enumerate(unique_pfas)}
    color_indices = [pfas_to_index[p] for p in data_to_plot["PFAS"]]

    # Then add scatter plot with points colored by individual PFAS compound
    scatter = ax.scatter(
        data_to_plot["Observed_Half_Life"],
        data_to_plot["Predicted_Half_Life"],
        c=color_indices,
        cmap=custom_cmap,
        s=70,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        zorder=10,  # Make sure points are on top of error bars
    )

    # Identity line
    ax.plot(plot_range, plot_range, "k--", lw=1.5)

    # Styling
    ax.set_xlabel("Observed Half-life (years)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Predicted Half-life (years)", fontsize=LABEL_SIZE)
    ax.set_title("PFAS Half-life: Observed vs Predicted", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_xlim(plot_range)
    ax.set_ylim(plot_range)

    # Add grid to match previous example
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)

    # Add legend for individual PFAS compounds
    # Create custom legend handles
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=colors[pfas_to_index[pfas]],
            edgecolor="white",
            label=pfas,
            alpha=0.8,
        )
        for pfas in unique_pfas
    ]

    # If there are many PFAS compounds, create a multi-column legend
    # if len(unique_pfas) > 15:
    #     ncols = 2  # Use 2 columns for many items
    # else:
    ncols = 1

    # Add legend in the most appropriate location (adjust as needed)
    legend = ax.legend(
        handles=legend_elements,
        loc="upper left",  # Position outside the plot
        fontsize=8,  # Smaller font for many items
        bbox_to_anchor=(1.01, 1),
        title="PFAS Compound",
        title_fontsize=10,
        framealpha=0.7,
        ncol=ncols,
    )

    # Adjust layout to make room for the legend
    plt.tight_layout(
        rect=[0, 0, 0.85 if ncols == 1 and len(unique_pfas) > 10 else 0.95, 1]
    )

    # Save the plot
    plt.savefig(
        "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/pfas_halflife_predictions_individual.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # # Create the exact plot layout shown in the example
    # plt.style.use("seaborn-v0_8-whitegrid")

    # # Create figure with two subplots
    # fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=300)

    # # Common styling parameters
    # TITLE_SIZE = 14
    # LABEL_SIZE = 12
    # TICK_SIZE = 10
    # LEGEND_SIZE = 10

    # # Calculate shared axis limits
    # y_train_1d = y.ravel() if hasattr(y, "ravel") else np.array(y).ravel()
    # y_train_pred_1d = (
    #     y_train_pred.ravel()
    #     if hasattr(y_train_pred, "ravel")
    #     else np.array(y_train_pred).ravel()
    # )
    # y_test_1d = y_test.ravel() if hasattr(y_test, "ravel") else np.array(y_test).ravel()
    # y_test_mean_1d = (
    #     y_test_mean.ravel()
    #     if hasattr(y_test_mean, "ravel")
    #     else np.array(y_test_mean).ravel()
    # )

    # all_values = np.concatenate(
    #     [y_train_1d, y_train_pred_1d, y_test_1d, y_test_mean_1d]
    # )
    # plot_min = 0  # Start at 0 like the example
    # plot_max = 20.0  # End at 20 like the example
    # plot_range = [plot_min, plot_max]

    # # Train predictions plot (left)
    # scatter_train = axes[0].scatter(
    #     y,
    #     y_train_pred,
    #     alpha=0.7,
    #     s=60,
    #     c=y_train_sd_pred,
    #     cmap="viridis",
    #     edgecolor="white",
    #     linewidth=0.5,
    #     vmin=0.0,  # Set min/max for consistent colorbar with example
    #     vmax=5.0,
    # )

    # # Error bars for train points
    # axes[0].errorbar(
    #     y,
    #     y_train_pred,
    #     yerr=y_train_sd_pred,
    #     fmt="none",
    #     ecolor="gray",
    #     alpha=0.20,
    #     capsize=3,
    #     elinewidth=0.5,
    # )

    # # Identity line for train plot
    # axes[0].plot(plot_range, plot_range, "k--", lw=1.5)

    # # Styling for train plot
    # axes[0].set_xlabel("True Half-life (years)", fontsize=LABEL_SIZE)
    # axes[0].set_ylabel("Predicted Half-life (years)", fontsize=LABEL_SIZE)
    # axes[0].set_title("Training Set Predictions", fontsize=TITLE_SIZE)
    # axes[0].tick_params(axis="both", labelsize=TICK_SIZE)
    # axes[0].set_xlim(plot_range)
    # axes[0].set_ylim(plot_range)

    # # Add R² to train plot in upper left
    # axes[0].text(
    #     0.05,
    #     0.95,
    #     f"$R^2 = {r2_train:.3f}$",
    #     transform=axes[0].transAxes,
    #     fontsize=LABEL_SIZE,
    #     verticalalignment="top",
    # )

    # # Add grid to match example
    # axes[0].grid(True, linestyle="-", linewidth=0.5, alpha=0.3)

    # # Test predictions plot (right)
    # scatter_test = axes[1].scatter(
    #     y_test,
    #     y_test_mean,
    #     alpha=0.7,
    #     s=60,
    #     c=y_test_std,
    #     cmap="viridis",
    #     edgecolor="white",
    #     linewidth=0.5,
    #     vmin=0.0,  # Set min/max for consistent colorbar with example
    #     vmax=5.0,
    # )

    # # Error bars for test points
    # axes[1].errorbar(
    #     y_test,
    #     y_test_mean,
    #     yerr=y_test_std,
    #     fmt="none",
    #     ecolor="gray",
    #     alpha=0.20,
    #     capsize=3,
    #     elinewidth=0.5,
    # )

    # # Identity line for test plot
    # axes[1].plot(plot_range, plot_range, "k--", lw=1.5)

    # # Styling for test plot
    # axes[1].set_xlabel("True Half-life (years)", fontsize=LABEL_SIZE)
    # axes[1].set_ylabel("Predicted Half-life (years)", fontsize=LABEL_SIZE)
    # axes[1].set_title("Test Set Predictions", fontsize=TITLE_SIZE)
    # axes[1].tick_params(axis="both", labelsize=TICK_SIZE)
    # axes[1].set_xlim(plot_range)
    # axes[1].set_ylim(plot_range)

    # # Add R² to test plot in upper left
    # axes[1].text(
    #     0.05,
    #     0.95,
    #     f"$R^2 = {r2_test:.3f}$",
    #     transform=axes[1].transAxes,
    #     fontsize=LABEL_SIZE,
    #     verticalalignment="top",
    # )

    # # Add grid to match example
    # axes[1].grid(True, linestyle="-", linewidth=0.5, alpha=0.3)

    # # Add single colorbar on the right side for both plots
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # cbar = fig.colorbar(scatter_test, cax=cbar_ax)
    # cbar.set_label("Prediction Uncertainty (σ)", fontsize=LABEL_SIZE)
    # cbar.ax.tick_params(labelsize=TICK_SIZE)

    # # Main title for the entire figure
    # fig.suptitle(
    #     "HGPR Model Performance: Predicting PFAS Half-life", fontsize=16, y=0.98
    # )

    # # Adjust subplot spacing
    # plt.subplots_adjust(left=0.08, right=0.9, bottom=0.11, top=0.9, wspace=0.25)

    # # Save the plot
    # plt.savefig("hgpr_model_performance.png", dpi=300, bbox_inches="tight")
    # plt.show()
