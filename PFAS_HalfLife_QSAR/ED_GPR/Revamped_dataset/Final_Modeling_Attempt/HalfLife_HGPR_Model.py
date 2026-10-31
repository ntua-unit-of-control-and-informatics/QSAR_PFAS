import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve
from scipy.stats import norm, pearsonr, spearmanr
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


def single_bin_ece(y_true, y_pred_mean, y_pred_std,
                   confidence_levels=(0.50, 0.68, 0.90, 0.95, 0.99)):
    """
    Simplified ECE for a single data point or group: average deviation
    from expected coverage across confidence levels, without binning.

    Parameters
    ----------
    y_true : float or array-like
        True target value(s).
    y_pred_mean : float or array-like
        Predicted mean value(s).
    y_pred_std : float or array-like
        Predicted standard deviation(s).
    confidence_levels : tuple of float
        Confidence levels to evaluate.

    Returns
    -------
    ece : float
        Expected calibration error for this point/group.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_mean = np.asarray(y_pred_mean).ravel()
    y_pred_std = np.maximum(np.asarray(y_pred_std).ravel(), 1e-8)

    ece = 0
    for cl in confidence_levels:
        z = norm.ppf((1 + cl) / 2)
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        actual_coverage = np.mean((y_true >= lower) & (y_true <= upper))
        ece += np.abs(cl - actual_coverage)

    return ece / len(confidence_levels)


def per_pfas_ece(y_true, y_pred_mean, y_pred_std, pfas_labels,
                 confidence_levels=(0.50, 0.68, 0.90, 0.95, 0.99)):
    """
    Calculate per-PFAS Expected Calibration Error.

    Computes single_bin_ece for each PFAS compound individually,
    then returns the mean, median, and per-PFAS values.

    Parameters
    ----------
    y_true : array-like
        True target values (one per PFAS).
    y_pred_mean : array-like
        Predicted mean values (one per PFAS).
    y_pred_std : array-like
        Predicted standard deviations (one per PFAS).
    pfas_labels : array-like
        PFAS compound identifiers (one per entry).
    confidence_levels : tuple of float
        Confidence levels to evaluate.

    Returns
    -------
    mean_ece : float
        Mean per-PFAS ECE (ideal = 0.0).
    median_ece : float
        Median per-PFAS ECE (robust to outliers).
    ece_per_pfas : dict
        Mapping {pfas_label: ece_value}.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_mean = np.asarray(y_pred_mean).ravel()
    y_pred_std = np.maximum(np.asarray(y_pred_std).ravel(), 1e-8)

    ece_per_pfas = {}
    for i, pfas in enumerate(pfas_labels):
        ece_per_pfas[pfas] = single_bin_ece(
            y_true[i], y_pred_mean[i], y_pred_std[i],
            confidence_levels=confidence_levels,
        )

    values = np.array(list(ece_per_pfas.values()))
    mean_ece = np.mean(values)
    median_ece = np.median(values)
    return mean_ece, median_ece, ece_per_pfas


def normalized_calibration_error(y_true, y_pred_mean, y_pred_std):
    """
    Calculate per-molecule Normalized Calibration Error (NCE).

    For each molecule i:
        NCE_i = |abs_error_i - sigma_i| / sigma_i

    The mean NCE across all molecules is reported.

    A well-calibrated model should have mean NCE ≈ 0.
    Individual NCE_i > 0 with |error| > sigma: locally overconfident.
    Individual NCE_i > 0 with |error| < sigma: locally underconfident.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred_mean : array-like
        Predicted mean values.
    y_pred_std : array-like
        Predicted standard deviations.

    Returns
    -------
    mean_nce : float
        Mean per-molecule NCE (ideal = 0.0).
    median_nce : float
        Median per-molecule NCE (more robust to outliers).
    nce_per_molecule : ndarray
        NCE value for each molecule.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_mean = np.asarray(y_pred_mean).ravel()
    y_pred_std = np.maximum(np.asarray(y_pred_std).ravel(), 1e-8)

    abs_errors = np.abs(y_true - y_pred_mean)
    nce_per_molecule = np.abs(abs_errors - y_pred_std) / y_pred_std
    mean_nce = np.mean(nce_per_molecule)
    median_nce = np.median(nce_per_molecule)
    return mean_nce, median_nce, nce_per_molecule


def uncertainty_pearson(observed_std, predicted_std):
    """
    Pearson correlation between observed and predicted standard deviations.

    A high positive correlation indicates that the model's uncertainty
    estimates capture the true data variability.

    Parameters
    ----------
    observed_std : array-like
        Observed standard deviations.
    predicted_std : array-like
        Predicted standard deviations.

    Returns
    -------
    r : float
        Pearson correlation coefficient.
    p_value : float
        Two-tailed p-value.
    """
    observed_std = np.asarray(observed_std).ravel()
    predicted_std = np.asarray(predicted_std).ravel()

    r, p_value = pearsonr(observed_std, predicted_std)
    return r, p_value


def uncertainty_spearman(observed_std, predicted_std):
    """
    Spearman rank correlation between observed and predicted standard
    deviations.

    A high positive correlation indicates that the ranking of predicted
    uncertainty matches the ranking of observed variability.

    Parameters
    ----------
    observed_std : array-like
        Observed standard deviations.
    predicted_std : array-like
        Predicted standard deviations.

    Returns
    -------
    rho : float
        Spearman correlation coefficient.
    p_value : float
        Two-tailed p-value.
    """
    observed_std = np.asarray(observed_std).ravel()
    predicted_std = np.asarray(predicted_std).ravel()

    rho, p_value = spearmanr(observed_std, predicted_std)
    return rho, p_value


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


def create_pfas_mapping(df, halflife_stats):
    """Create mapping from SMILES to PFAS compound names"""
    smiles_to_pfas = {}

    if "SMILES" in halflife_stats.columns and "PFAS" in halflife_stats.columns:
        for _, row in halflife_stats.iterrows():
            smiles_to_pfas[row["SMILES"]] = row["PFAS"]

    # Map SMILES to PFAS
    pfas_values = []

    for smiles in df["SMILES"]:
        pfas = smiles_to_pfas.get(smiles)
        if pfas is None:
            # If we can't find the PFAS, create a unique identifier based on SMILES
            pfas = f"Unknown_{hash(smiles) % 10000}"
        pfas_values.append(pfas)

    return pfas_values


def perform_pfas_cv(model, X, y, pfas_values, halflife_stats, n_folds=5):
    """
    Perform cross-validation based on PFAS compounds rather than random samples.
    Evaluates model performance using mean values per PFAS.

    Parameters:
    -----------
    model : HeteroscedasticGPR
        The model to evaluate
    X : array-like
        Features
    y : array-like
        Target values
    pfas_values : array-like
        PFAS compound identifiers for each sample
    halflife_stats : DataFrame
        DataFrame containing mean half-life values for each PFAS
    n_folds : int
        Number of CV folds
    """
    # Get unique PFAS compounds
    unique_pfas = np.unique(pfas_values)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_pfas)

    # Create a mapping from PFAS to mean value from halflife_stats
    pfas_to_mean = {}
    if "PFAS" in halflife_stats.columns and "half_life_mean" in halflife_stats.columns:
        for _, row in halflife_stats.iterrows():
            if not pd.isna(row.get("PFAS")) and not pd.isna(row.get("half_life_mean")):
                pfas_to_mean[row["PFAS"]] = row["half_life_mean"]

    # Define number of PFAS compounds for validation in each fold
    n_val_pfas = max(1, int(len(unique_pfas) * 0.3))  # 30% for validation

    # Store metrics for each fold
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    pfas_r2_scores = []

    for fold_idx in range(n_folds):
        # Calculate start and end indices for validation PFAS
        start_idx = (fold_idx * n_val_pfas) % len(unique_pfas)
        end_idx = min(start_idx + n_val_pfas, len(unique_pfas))

        # Handle wrap-around for last fold if needed
        if end_idx <= start_idx and fold_idx < n_folds - 1:
            val_pfas = np.concatenate(
                [
                    unique_pfas[start_idx:],
                    unique_pfas[: n_val_pfas - (len(unique_pfas) - start_idx)],
                ]
            )
        else:
            val_pfas = unique_pfas[start_idx:end_idx]

        # Create masks for train/validation split based on PFAS compounds
        val_mask = np.isin(pfas_values, val_pfas)
        train_mask = ~val_mask

        if np.sum(train_mask) == 0 or np.sum(val_mask) == 0:
            continue  # Skip empty folds

        # Split data
        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]
        pfas_train = np.array(pfas_values)[train_mask]
        pfas_val = np.array(pfas_values)[val_mask]

        print(f"\nFold {fold_idx+1}/{n_folds}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(
            f"Training PFAS compounds: {len(np.unique(pfas_train))}, Validation PFAS compounds: {len(np.unique(pfas_val))}"
        )

        # Create and fit the model
        fold_model = HeteroscedasticGPR(
            poly_degree=model.poly_degree,
            k_al=model.k_al,
            length_scale=model.length_scale,
            priors=model.priors,
        )
        fold_model.fit(X_train, y_train)

        # Predict on validation set
        y_val_pred = fold_model.predict(X_val)

        # Calculate sample-level metrics
        r2 = r2_score(y_val, y_val_pred)
        rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
        mae = np.mean(np.abs(y_val - y_val_pred))

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)

        # Calculate PFAS-level means
        val_pfas_true_means = {}
        val_pfas_pred_means = {}

        # Get unique PFAS in validation set
        unique_val_pfas = np.unique(pfas_val)

        # Calculate mean for each PFAS
        for pfas in unique_val_pfas:
            # Get samples for this PFAS
            pfas_mask = pfas_val == pfas

            # Calculate predicted mean
            if np.sum(pfas_mask) > 0:
                val_pfas_pred_means[pfas] = np.mean(y_val_pred[pfas_mask])

            # Get true mean from halflife_stats if available
            if pfas in pfas_to_mean:
                val_pfas_true_means[pfas] = pfas_to_mean[pfas]
            else:
                # Use observed mean from validation data
                if np.sum(pfas_mask) > 0:
                    val_pfas_true_means[pfas] = np.mean(y_val[pfas_mask])

        # Calculate PFAS-level R² if we have enough PFAS compounds
        if len(val_pfas_true_means) > 1:
            # Find intersection of PFAS with both true and predicted means
            common_pfas = list(
                set(val_pfas_true_means.keys()) & set(val_pfas_pred_means.keys())
            )

            if len(common_pfas) > 1:
                pfas_true = np.array(
                    [val_pfas_true_means[pfas] for pfas in common_pfas]
                )
                pfas_pred = np.array(
                    [val_pfas_pred_means[pfas] for pfas in common_pfas]
                )

                pfas_r2 = r2_score(pfas_true, pfas_pred)
                pfas_r2_scores.append(pfas_r2)
                print(
                    f"PFAS-level metrics - R² (mean values): {pfas_r2:.4f} (n={len(common_pfas)} PFAS)"
                )

        print(f"Sample-level metrics - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Calculate average metrics
    avg_r2 = np.mean(r2_scores) if r2_scores else np.nan
    avg_rmse = np.mean(rmse_scores) if rmse_scores else np.nan
    avg_mae = np.mean(mae_scores) if mae_scores else np.nan
    avg_pfas_r2 = np.mean(pfas_r2_scores) if pfas_r2_scores else np.nan

    print("\nAverage CV metrics:")
    print(f"Sample-level - R²: {avg_r2:.4f}, RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}")
    if pfas_r2_scores:
        print(f"PFAS-level - R² (mean values): {avg_pfas_r2:.4f}")

    return {
        "r2": avg_r2,
        "rmse": avg_rmse,
        "mae": avg_mae,
        "pfas_r2": avg_pfas_r2,
        "sample_scores": {"r2": r2_scores, "rmse": rmse_scores, "mae": mae_scores},
        "pfas_scores": pfas_r2_scores,
    }


if __name__ == "__main__":
    # Load datasets
    train_df = pd.read_csv(
        "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/Half-life_dataset_Human_train.csv"
    )
    test_df = pd.read_csv(
        "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/Half-life_dataset_Human_test.csv"
    )
    halflife_stats = pd.read_csv(
        "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/Halflife_stats.csv"
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

    selected_features = [
        "LogKa",
        "PEOE_VSA4",
        "Bit_1198",
        "Bit_1720",
        "Bit_486",
        "Bit_1840",
    ]

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

    # Create PFAS mapping for cross-validation
    train_pfas = create_pfas_mapping(train_df, halflife_stats)

    # Create and perform PFAS-based cross-validation
    # model = HeteroscedasticGPR(**model_params)
    # cv_results = perform_pfas_cv(model, X, y, train_pfas, halflife_stats, n_folds=5)

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

    # Get PFAS-level metrics for training data
    train_pfas_unique = np.unique(train_pfas)
    train_pfas_true_means = {}
    train_pfas_pred_means = {}
    train_pfas_pred_sd = {}

    # Create mapping from PFAS to mean and sd values
    pfas_to_mean = {}
    pfas_to_sd = {}
    if "PFAS" in halflife_stats.columns and "half_life_mean" in halflife_stats.columns:
        for _, row in halflife_stats.iterrows():
            if not pd.isna(row.get("PFAS")) and not pd.isna(row.get("half_life_mean")):
                pfas_to_mean[row["PFAS"]] = row["half_life_mean"]
            if not pd.isna(row.get("PFAS")) and not pd.isna(row.get("half_life_sd")):
                pfas_to_sd[row["PFAS"]] = row["half_life_sd"]

    # Calculate means for each PFAS in training set
    train_pfas_true_sd = {}
    for pfas in train_pfas_unique:
        pfas_mask = np.array(train_pfas) == pfas
        if np.sum(pfas_mask) > 0:
            train_pfas_pred_means[pfas] = np.mean(y_train_pred[pfas_mask])
            train_pfas_pred_sd[pfas] = np.mean(y_train_sd_pred[pfas_mask])

            # Get true mean and sd from halflife_stats if available
            train_pfas_true_means[pfas] = pfas_to_mean[pfas]
            if pfas in pfas_to_sd:
                train_pfas_true_sd[pfas] = pfas_to_sd[pfas]

    # Calculate PFAS-level R² for training
    if len(train_pfas_true_means) > 1:
        common_pfas = list(
            set(train_pfas_true_means.keys()) & set(train_pfas_pred_means.keys())
        )
        if len(common_pfas) > 1:
            train_pfas_true = np.array(
                [train_pfas_true_means[pfas] for pfas in common_pfas]
            )
            train_pfas_pred = np.array(
                [train_pfas_pred_means[pfas] for pfas in common_pfas]
            )
            train_pfas_pred_sd = np.array(
                [train_pfas_pred_sd[pfas] for pfas in common_pfas]
            )
            train_pfas_r2 = r2_score(train_pfas_true, train_pfas_pred)
            train_pfas_rmse = np.sqrt(np.mean((train_pfas_true - train_pfas_pred) ** 2))
            train_pfas_mae = np.mean(np.abs(train_pfas_true - train_pfas_pred))
            elpd_train, avg_elpd_train = calculate_elpd(
                train_pfas_true, train_pfas_pred, train_pfas_pred_sd
            )
            print(f"Training PFAS-level R² (mean values): {train_pfas_r2:.4f}")
            print(f"Training PFAS-level RMSE (mean values): {train_pfas_rmse:.4f}")
            print(f"Training PFAS-level MAE (mean values): {train_pfas_mae:.4f}")
            print(
                f"Training PFAS-level ELPD: {elpd_train:.4f} (n={len(common_pfas)} PFAS)"
            )
            print(
                f"Training PFAS-level average ELPD: {avg_elpd_train:.4f} (n={len(common_pfas)} PFAS)"
            )

            # R² on SD (train, PFAS-level)
            sd_common = [p for p in common_pfas if p in train_pfas_true_sd]
            if len(sd_common) > 1:
                train_obs_sd = np.array([train_pfas_true_sd[p] for p in sd_common])
                train_pred_sd_arr = np.array(
                    [train_pfas_pred_sd[common_pfas.index(p)] for p in sd_common]
                )
                train_sd_r2 = r2_score(train_obs_sd, train_pred_sd_arr)
                print(
                    f"Training PFAS-level R² (SD values): {train_sd_r2:.4f} (n={len(sd_common)} PFAS)"
                )

                # Uncertainty quality metrics (train, PFAS-level)
                print("\n--- Uncertainty Quality Metrics (train set, PFAS-level) ---")

                train_nce, train_nce_med, _ = normalized_calibration_error(
                    train_pfas_true, train_pfas_pred, train_pfas_pred_sd
                )
                print(f"NCE  mean   (per-PFAS Calibration Error):     {train_nce:.4f}")
                print(f"NCE  median (per-PFAS Calibration Error):     {train_nce_med:.4f}")

                train_ece_mean, train_ece_med, _ = per_pfas_ece(
                    train_pfas_true, train_pfas_pred, train_pfas_pred_sd,
                    sd_common,
                )
                print(f"ECE  mean   (per-PFAS Calibration Error):     {train_ece_mean:.4f}")
                print(f"ECE  median (per-PFAS Calibration Error):     {train_ece_med:.4f}")

                train_pfas_pearson_r, train_pfas_pearson_p = uncertainty_pearson(
                    train_obs_sd, train_pred_sd_arr
                )
                print(
                    f"Pearson  r = {train_pfas_pearson_r:.4f}  (p = {train_pfas_pearson_p:.4e})"
                )

                train_pfas_spearman_rho, train_pfas_spearman_p = uncertainty_spearman(
                    train_obs_sd, train_pred_sd_arr
                )
                print(
                    f"Spearman ρ = {train_pfas_spearman_rho:.4f}  (p = {train_pfas_spearman_p:.4e})"
                )

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

    # Create PFAS mapping for test set
    test_pfas = create_pfas_mapping(test_df, halflife_stats)

    # Predict on test data with uncertainty
    y_test_mean, y_test_std = model.predict(X_test, return_std=True)

    # Calculate PFAS-level metrics for test set
    test_pfas_unique = np.unique(test_pfas)
    test_pfas_true_means = {}
    test_pfas_pred_means = {}
    test_pfas_pred_sd = {}

    # Calculate means for each PFAS in test set
    test_pfas_true_sd = {}
    for pfas in test_pfas_unique:
        pfas_mask = np.array(test_pfas) == pfas
        if np.sum(pfas_mask) > 0:
            test_pfas_pred_means[pfas] = np.mean(y_test_mean[pfas_mask])
            test_pfas_pred_sd[pfas] = np.mean(y_test_std[pfas_mask])

            # Get true mean and sd from halflife_stats
            test_pfas_true_means[pfas] = pfas_to_mean[pfas]
            if pfas in pfas_to_sd:
                test_pfas_true_sd[pfas] = pfas_to_sd[pfas]

    # Calculate PFAS-level R² for test

    common_pfas = list(
        set(test_pfas_true_means.keys()) & set(test_pfas_pred_means.keys())
    )
    if len(common_pfas) > 1:
        test_pfas_true = np.array([test_pfas_true_means[pfas] for pfas in common_pfas])
        test_pfas_pred = np.array([test_pfas_pred_means[pfas] for pfas in common_pfas])
        test_pfas_pred_sd = np.array([test_pfas_pred_sd[pfas] for pfas in common_pfas])
        test_pfas_r2 = r2_score(test_pfas_true, test_pfas_pred)
        test_pfas_q2 = q2_score(
            test_pfas_true, test_pfas_pred, np.mean(train_pfas_true)
        )
        test_pfas_rmse = np.sqrt(np.mean((test_pfas_true - test_pfas_pred) ** 2))
        test_pfas_mae = np.mean(np.abs(test_pfas_true - test_pfas_pred))
        elpd_test, avg_elpd_test = calculate_elpd(
            test_pfas_true, test_pfas_pred, test_pfas_pred_sd
        )
        print(
            f"\nTest PFAS-level R² (mean values): {test_pfas_r2:.4f} (n={len(common_pfas)} PFAS)"
        )
        print(
            f"Test PFAS-level Q² (mean values): {test_pfas_q2:.4f} (n={len(common_pfas)} PFAS)"
        )
        print(
            f"Test PFAS-level RMSE (mean values): {test_pfas_rmse:.4f} (n={len(common_pfas)} PFAS)"
        )
        print(
            f"Test PFAS-level MAE (mean values): {test_pfas_mae:.4f} (n={len(common_pfas)} PFAS)"
        )
        print(f"Test PFAS-level ELPD: {elpd_test:.4f} (n={len(common_pfas)} PFAS)")
        print(
            f"Test PFAS-level average ELPD: {avg_elpd_test:.4f} (n={len(common_pfas)} PFAS)"
        )

    # ---------------------------------------------------------------------
    #               Uncertainty Quality Metrics (test set, PFAS-level)
    # ---------------------------------------------------------------------
    sd_common_test = [p for p in common_pfas if p in test_pfas_true_sd]
    if len(sd_common_test) > 1:
        test_obs_sd = np.array([test_pfas_true_sd[p] for p in sd_common_test])
        test_pred_sd_arr = np.array(
            [test_pfas_pred_sd[common_pfas.index(p)] for p in sd_common_test]
        )
        test_sd_r2 = r2_score(test_obs_sd, test_pred_sd_arr)
        print(
            f"Test PFAS-level R² (SD values): {test_sd_r2:.4f} (n={len(sd_common_test)} PFAS)"
        )

        print("\n--- Uncertainty Quality Metrics (test set, PFAS-level) ---")

        test_nce, test_nce_med, test_nce_per_pfas = normalized_calibration_error(
            test_pfas_true, test_pfas_pred, test_pfas_pred_sd
        )
        print(f"NCE  mean   (per-PFAS Calibration Error):     {test_nce:.4f}")
        print(f"NCE  median (per-PFAS Calibration Error):     {test_nce_med:.4f}")
        for pfas, nce_i in zip(sd_common_test, test_nce_per_pfas):
            print(f"  {pfas:>10s}: NCE = {nce_i:.4f}")

        test_ece_mean, test_ece_med, test_ece_per_pfas = per_pfas_ece(
            test_pfas_true, test_pfas_pred, test_pfas_pred_sd,
            sd_common_test,
        )
        print(f"ECE  mean   (per-PFAS Calibration Error):     {test_ece_mean:.4f}")
        print(f"ECE  median (per-PFAS Calibration Error):     {test_ece_med:.4f}")
        for pfas, ece_i in test_ece_per_pfas.items():
            print(f"  {pfas:>10s}: ECE = {ece_i:.4f}")

        pfas_pearson_r, pfas_pearson_p = uncertainty_pearson(
            test_obs_sd, test_pred_sd_arr
        )
        print(f"Pearson  r = {pfas_pearson_r:.4f}  (p = {pfas_pearson_p:.4e})")

        pfas_spearman_rho, pfas_spearman_p = uncertainty_spearman(
            test_obs_sd, test_pred_sd_arr
        )
        print(f"Spearman ρ = {pfas_spearman_rho:.4f}  (p = {pfas_spearman_p:.4e})")

    # ---------------------------------------------------------------------
    #                       Plotting the results
    # ---------------------------------------------------------------------

    # Plot predicted mean and standard deviation for each pfas vs
    # the true mean and sd
    # Filter halflife_stats to keep only PFAS compounds that are in the test set
    test_smiles = set(test_df["SMILES"])
    filtered_halflife_stats = halflife_stats[halflife_stats["SMILES"].isin(test_smiles)]

    # Check if we have any matching compounds
    if len(filtered_halflife_stats) == 0:
        print("No matching PFAS compounds found between test set and halflife_stats")
        # Create a basic plot of test predictions instead
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_test_mean, alpha=0.8)
        plt.plot(
            [0, max(y_test.max(), y_test_mean.max())],
            [0, max(y_test.max(), y_test_mean.max())],
            "k--",
        )
        plt.xlabel("Observed Half-life (years)")
        plt.ylabel("Predicted Half-life (years)")
        plt.title("PFAS Half-life: Test Set Predictions")
        plt.savefig(
            "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/pfas_halflife_predictions_test.png",
            dpi=300,
        )
        plt.show()
    else:
        # Process the filtered dataset
        halflife_stats_dataset = JaqpotTabularDataset(
            df=filtered_halflife_stats,
            y_cols=None,
            x_cols=x_cols,
            smiles_cols="SMILES",
            featurizer=featurizers,
            task="regression",
        )

        # Use all selected features
        halflife_stats_dataset = (
            halflife_stats_dataset.X[selected_features].copy().values
        )
        predictions_per_pfas_mean, predictions_per_pfas_mean_sd = model.predict(
            halflife_stats_dataset, return_std=True
        )
        print("Predicted test means", predictions_per_pfas_mean)
        # print(
        #     "observed test means", filtered_halflife_stats["half_life_mean"].flatten()
        # )

        data_to_plot = pd.DataFrame(
            {
                "SMILES": filtered_halflife_stats["SMILES"],
                "PFAS": filtered_halflife_stats["PFAS"],
                "Observed_Half_Life": filtered_halflife_stats["half_life_mean"],
                "Observed_Half_Life_SD": filtered_halflife_stats["half_life_sd"],
                "Predicted_Half_Life": predictions_per_pfas_mean,
                "Predicted_Half_Life_SD": predictions_per_pfas_mean_sd,
            }
        )

        # Create the plot
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(3.33, 4), dpi=300)

        # Common styling parameters (adjusted for single-column figure)
        # JCIM Journal Requirements:
        # - Single-column max width: 3.33 inches (240 points)
        # - Minimum font size: 4.5 points
        # - Recommended: Axis titles (8-10 pt), Tick labels (7-9 pt), Legend (7-8 pt)
        TITLE_SIZE = 12
        LABEL_SIZE = 10
        TICK_SIZE = 8
        LEGEND_SIZE = 6

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
        # # Now add text annotations for each point
        # for i, row in data_to_plot.iterrows():
        #     pfas_name = row["PFAS"]
        #     x = row["Observed_Half_Life"]
        #     y = row["Predicted_Half_Life"]

        #     # Slightly offset the text to avoid overlapping with the point
        #     ax.annotate(
        #         pfas_name,
        #         xy=(x, y),
        #         xytext=(5, 5),  # 5 points offset
        #         textcoords="offset points",
        #         fontsize=12,
        #         alpha=0.8,
        #         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
        #     )
        # Get unique PFAS compounds for coloring
        unique_pfas = data_to_plot["PFAS"].unique()

        # Create a categorical colormap for individual PFAS compounds

        # Choose a colormap with distinct colors based on number of PFAS compounds
        if len(unique_pfas) <= 10:
            cmap_name = "tab10"
        elif len(unique_pfas) <= 20:
            cmap_name = "tab20"
        else:
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
            s=60,
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
        # ax.set_title("PFAS Half-life: Observed vs Predicted", fontsize=TITLE_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.set_xlim(plot_range)
        ax.set_ylim(plot_range)

        # Format axis ticks to have consistent decimal places
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        # Add grid to match previous example
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)

        # Add legend for individual PFAS compounds
        legend_elements = [
            Patch(
                facecolor=colors[pfas_to_index[pfas]],
                edgecolor="white",
                label=pfas,
                alpha=0.8,
            )
            for pfas in unique_pfas
        ]

        # Add legend in the most appropriate location
        legend = ax.legend(
            handles=legend_elements,
            loc="upper center",
            fontsize=LEGEND_SIZE,
            bbox_to_anchor=(0.5, -0.15),
            title="PFAS Compound",
            title_fontsize=7,
            framealpha=0.7,
            ncol=3,
            handlelength=1.0,
            handleheight=0.7,
            columnspacing=1.0,
        )
        plt.setp(legend.get_title(), fontweight="normal")

        # # Add R² to the plot
        # test_r2 = r2_score(
        #     data_to_plot["Observed_Half_Life"], data_to_plot["Predicted_Half_Life"]
        # )
        # ax.text(
        #     0.05,
        #     0.95,
        #     f"$R^2 = {test_r2:.3f}$",
        #     transform=ax.transAxes,
        #     fontsize=LABEL_SIZE,
        #     verticalalignment="top",
        #     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        # )

        # Adjust layout to make room for the legend
        plt.tight_layout()

        # Save the plot
        plt.savefig(
            "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/halflife_predictions.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # Save the filtered data with predictions for reference
        data_to_plot.to_csv(
            "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/pfas_halflife_test_stats_predictions.csv",
            index=False,
        )

        # -----------------------------------------------------------------
        #       Observed SD vs Predicted SD (PFAS-level, test set)
        # -----------------------------------------------------------------
        fig_sd, ax_sd = plt.subplots(figsize=(3.33, 4), dpi=300)

        # Plot limits
        sd_x_max = data_to_plot["Observed_Half_Life_SD"].max() + 0.5
        sd_y_max = data_to_plot["Predicted_Half_Life_SD"].max() + 0.5
        sd_max_value = max(sd_x_max, sd_y_max)
        sd_plot_range = [0, sd_max_value]

        # Scatter plot coloured by PFAS compound (reuse same mapping)
        ax_sd.scatter(
            data_to_plot["Observed_Half_Life_SD"],
            data_to_plot["Predicted_Half_Life_SD"],
            c=color_indices,
            cmap=custom_cmap,
            s=60,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            zorder=10,
        )

        # Identity line
        ax_sd.plot(sd_plot_range, sd_plot_range, "k--", lw=1.5)

        # Styling (same sizes as mean plot)
        ax_sd.set_xlabel("Observed SD (years)", fontsize=LABEL_SIZE)
        ax_sd.set_ylabel("Predicted SD (years)", fontsize=LABEL_SIZE)
        ax_sd.tick_params(axis="both", labelsize=TICK_SIZE)
        ax_sd.set_xlim(sd_plot_range)
        ax_sd.set_ylim(sd_plot_range)
        ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax_sd.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax_sd.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)

        # Legend (same as mean plot)
        sd_legend_elements = [
            Patch(
                facecolor=colors[pfas_to_index[pfas]],
                edgecolor="white",
                label=pfas,
                alpha=0.8,
            )
            for pfas in unique_pfas
        ]
        sd_legend = ax_sd.legend(
            handles=sd_legend_elements,
            loc="upper center",
            fontsize=LEGEND_SIZE,
            bbox_to_anchor=(0.5, -0.15),
            title="PFAS Compound",
            title_fontsize=7,
            framealpha=0.7,
            ncol=3,
            handlelength=1.0,
            handleheight=0.7,
            columnspacing=1.0,
        )
        plt.setp(sd_legend.get_title(), fontweight="normal")

        plt.tight_layout()
        fig_sd.savefig(
            "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/halflife_sd_predictions.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    # ---------------------------------------------------------------------
    #                       Domain of Applicability
    # ---------------------------------------------------------------------

    # Get the PFAS compounds in the training set
    # train_pfas_set = set(train_pfas)

    # # Filter halflife_stats to keep only PFAS compounds that were in the training set
    # train_domain = halflife_stats[halflife_stats["PFAS"].isin(train_pfas_set)]
    # test_domain = halflife_stats[halflife_stats["PFAS"].isin(test_pfas)]

    # test_domain = JaqpotTabularDataset()

    # # Use model's optimized length scales for distance calculation
    # lengthscales = model.length_scales_

    # def ard_rbf_kernel(x, X_train, lengthscales):
    #     diffs = (X_train - x) / lengthscales
    #     dists2 = np.sum(diffs**2, axis=1)
    #     return np.exp(-0.5 * dists2)

    # # 1. Calculate max kernel similarity for each train compound
    # train_sims = []
    # for i in range(X.shape[0]):
    #     sims = ard_rbf_kernel(X[i], np.delete(X, i, axis=0), lengthscales)
    #     train_sims.append(sims.max())
    # threshold = np.percentile(train_sims, 5)

    # # 2. Calculate for test set
    # test_sims = np.array([ard_rbf_kernel(x, X, lengthscales).max() for x in X_test])
    # in_domain = test_sims >= threshold
