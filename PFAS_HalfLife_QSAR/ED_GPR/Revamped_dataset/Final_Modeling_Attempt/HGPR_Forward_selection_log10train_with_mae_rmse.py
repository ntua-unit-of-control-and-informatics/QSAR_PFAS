import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import RDKitDescriptors, TopologicalFingerprint
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve
from scipy.stats import norm
import warnings
import copy
from collections import defaultdict


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


def detect_categorical_columns(df):
    """
    Detect categorical columns in a dataframe including those that:
    1. Are actually categorical/object types
    2. Are binary (0/1) columns that might be one-hot encoded
    3. Have names starting with common prefixes for categorical columns
    """
    # Get columns that are explicitly categorical or object type
    categorical_cols = list(
        df.select_dtypes(include=["object", "category", "bool"]).columns
    )

    # Detect one-hot encoded and binary columns
    for col in df.columns:
        if col in categorical_cols:
            continue

        # Check if column is binary (0/1 or 0.0/1.0)
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            categorical_cols.append(col)
            continue

        # Check for common one-hot encoding prefixes
        if any(
            col.startswith(prefix + "_")
            for prefix in ["sex", "adult", "half_life_type", "Occupational_exposure"]
        ):
            categorical_cols.append(col)

    return categorical_cols


def create_pfas_mapping(df, halflife_stats):
    """Create mapping from SMILES to PFAS compound names"""
    smiles_to_pfas = {}

    if "SMILES" in halflife_stats.columns and "PFAS" in halflife_stats.columns:
        for _, row in halflife_stats.iterrows():
            if not pd.isna(row.get("SMILES")) and not pd.isna(row.get("PFAS")):
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


def perform_pfas_cv_for_features(
    features, dataset, halflife_stats, original_df, model_params=None, n_folds=5
):
    """
    Consistent PFAS-based cross-validation for feature evaluation.

    Parameters:
    -----------
    features : list
        List of feature names to evaluate
    dataset : JaqpotTabularDataset
        Dataset containing features and target values
    halflife_stats : DataFrame
        DataFrame containing mean values for each PFAS
    original_df : DataFrame
        Original dataframe containing SMILES column for mapping to PFAS
    model_params : dict
        Parameters for the HeteroscedasticGPR model
    n_folds : int
        Number of cross-validation folds

    Returns:
    --------
    float: PFAS-level R² score
    """
    # Create a copy of the dataset with only the selected features
    feature_df = dataset.X[features].copy()

    # Get categorical columns
    categorical_cols = detect_categorical_columns(feature_df)

    # Create copy of data for processing
    X = feature_df.values
    y_original = dataset.y.values  # Store original values for evaluation
    y = np.log10(y_original)  # Apply log10 transformation for training

    # Create PFAS mapping
    pfas_values = create_pfas_mapping(original_df, halflife_stats)

    # Create a mapping from PFAS to mean value from halflife_stats
    pfas_to_mean = {}
    if "PFAS" in halflife_stats.columns and "half_life_mean" in halflife_stats.columns:
        for _, row in halflife_stats.iterrows():
            if not pd.isna(row.get("PFAS")) and not pd.isna(row.get("half_life_mean")):
                pfas_to_mean[row["PFAS"]] = row["half_life_mean"]

    # Get unique PFAS compounds
    unique_pfas = np.unique(pfas_values)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_pfas)

    # Define number of PFAS compounds for validation in each fold
    n_val_pfas = max(1, int(len(unique_pfas) * 0.3))  # 30% for validation

    # Store metrics for each fold
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

        # Create and fit the model
        if model_params is None:
            fold_model = HeteroscedasticGPR(poly_degree=1, k_al=1.0)
        else:
            # Create a deep copy of model parameters to avoid modifying the original
            fold_model_params = copy.deepcopy(model_params)
            fold_model_params["priors"]["theta_var"] = (
                fold_model_params["priors"]["theta_var"] / X_train.shape[1]
            )
            fold_model = HeteroscedasticGPR(**fold_model_params)

        # Scale numerical features
        if categorical_cols:
            non_cat_indices = [
                i for i, f in enumerate(features) if f not in categorical_cols
            ]

            if non_cat_indices:
                scaler = StandardScaler()
                X_train_numeric = X_train[:, non_cat_indices].copy()
                X_val_numeric = X_val[:, non_cat_indices].copy()

                X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
                X_val_numeric_scaled = scaler.transform(X_val_numeric)

                X_train_scaled = X_train.copy()
                X_val_scaled = X_val.copy()
                X_train_scaled[:, non_cat_indices] = X_train_numeric_scaled
                X_val_scaled[:, non_cat_indices] = X_val_numeric_scaled
            else:
                X_train_scaled = X_train.copy()
                X_val_scaled = X_val.copy()
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

        try:
            # Train model
            fold_model.fit(X_train_scaled, y_train)

            # Predict
            y_val_pred = fold_model.predict(X_val_scaled)

            # Calculate PFAS-level means
            val_pfas_true_means = {}
            val_pfas_pred_means = {}

            # Calculate mean for each PFAS in validation set
            for pfas in np.unique(pfas_val):
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

                    pfas_r2 = r2_score(
                        10 ** np.array(pfas_true), 10 ** np.array(pfas_pred)
                    )
                    pfas_r2_scores.append(pfas_r2)

        except Exception as e:
            print(f"Error in fold {fold_idx}: {e}")
            continue

    # Return average PFAS-level R² if we have valid scores
    if pfas_r2_scores:
        avg_pfas_r2 = np.mean(pfas_r2_scores)
        return avg_pfas_r2
    else:
        return -1.0  # Return -1 if we couldn't calculate a valid score


def pfas_based_forward_selection(
    dataset,
    halflife_stats,
    original_df,
    model_params=None,
    n_folds=5,
    max_features=None,
    n_jobs=-1,
    preselected_features=None,
    tolerance=0.01,
    verbose=True,
):
    """
    Perform forward feature selection with PFAS-based cross-validation

    Parameters:
    -----------
    dataset : JaqpotTabularDataset
        Dataset containing features and target
    halflife_stats : DataFrame
        DataFrame containing mean values for each PFAS
    original_df : DataFrame
        Original dataframe containing SMILES column for mapping to PFAS
    model_params : dict or None
        Parameters for HeteroscedasticGPR model
    n_folds : int
        Number of cross-validation folds
    max_features : int or None
        Maximum number of features to select
    n_jobs : int
        Number of parallel jobs for joblib
    preselected_features : list or None
        List of feature names to always include
    tolerance : float
        Minimum improvement threshold to continue selection
    verbose : bool
        Whether to print progress information
    """
    # Initialize preselected features list if not provided
    if preselected_features is None:
        preselected_features = []

    # Handle categorical features encoding
    categorical_cols = dataset.X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns
    if len(categorical_cols) > 0:
        if verbose:
            print(f"Handling categorical columns: {categorical_cols.tolist()}")

        encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded_categorical_data = encoder.fit_transform(dataset.X[categorical_cols])
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical_data,
            columns=encoder.get_feature_names_out(categorical_cols),
        )

        # Create copy to avoid modifying original dataset
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.X = dataset_copy.X.drop(columns=categorical_cols)

        # Concatenate encoded categorical features
        dataset_copy.X = pd.concat(
            [
                encoded_categorical_df.reset_index(drop=True),
                dataset_copy.X.reset_index(drop=True),
            ],
            axis=1,
        )

        # Update dataset reference
        dataset = dataset_copy

    # Set maximum number of features
    n_features = dataset.X.shape[1]
    if max_features is None:
        max_features = n_features

    # Initialize variables
    selected_features = preselected_features.copy()
    feature_names = list(dataset.X.columns)
    remaining_features = [f for f in feature_names if f not in selected_features]
    scores_history = []

    # Calculate initial score if there are preselected features
    if selected_features:
        if verbose:
            print(f"Starting with preselected features: {selected_features}")

        initial_score = perform_pfas_cv_for_features(
            selected_features,
            dataset,
            halflife_stats,
            original_df,
            model_params,
            n_folds,
        )
        scores_history.append(initial_score)

        if verbose:
            print(
                f"Initial PFAS-level R² with preselected features: {initial_score:.4f}"
            )
    else:
        scores_history = []

    # Perform forward selection
    for i in range(len(selected_features), max_features):
        if not remaining_features:
            if verbose:
                print("No more features to select from.")
            break

        if verbose:
            print(
                f"\nStep {i+1}/{max_features}: Evaluating {len(remaining_features)} candidate features..."
            )

        # Evaluate each candidate feature in parallel
        candidate_scores = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(perform_pfas_cv_for_features)(
                selected_features + [feature],
                dataset,
                halflife_stats,
                original_df,
                model_params,
                n_folds,
            )
            for feature in remaining_features
        )

        # Find best feature
        best_score_idx = np.argmax(candidate_scores)
        best_score = candidate_scores[best_score_idx]
        best_feature = remaining_features[best_score_idx]

        # Check stopping criteria
        if scores_history and best_score < scores_history[-1] + tolerance:
            print()
            if verbose:
                print(f"\nStopping: No significant improvement (< {tolerance})")
                print(f"Previous best score: {scores_history[-1]:.4f}")
                print(f"Current best score: {best_score:.4f}")
                print(f"Improvement: {best_score - scores_history[-1]:.4f}")
            break

        if best_score < 0:  # Invalid score
            if verbose:
                print(f"\nStopping: Invalid score ({best_score:.4f})")
            break

        # Add feature to selected set
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        scores_history.append(best_score)

        if verbose:
            print(f"Added feature '{best_feature}' (PFAS-level R²: {best_score:.4f})")
            print(
                f"Current feature set ({len(selected_features)}): {selected_features}"
            )

    # Return best feature set
    if scores_history:
        best_score_idx = np.argmax(scores_history)
        best_features = selected_features[: best_score_idx + 1]

        if verbose:
            print("\nBest subset of features:", best_features)
            print(f"Best PFAS-level R²: {scores_history[best_score_idx]:.4f}")

        # Plot scores history
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(scores_history)), scores_history, marker="o")
        plt.xlabel("Number of Features")
        plt.ylabel("PFAS-level R² Score")
        plt.title("Feature Selection Performance")
        plt.grid(True)
        plt.savefig(
            "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/feature_selection_scores.png"
        )

        if verbose:
            print("Score history plot saved as 'feature_selection_scores.png'")

        return best_features, scores_history
    else:
        if verbose:
            print("\nNo features were selected.")
        return [], []


if __name__ == "__main__":
    import time

    # Load datasets
    train_df = pd.read_csv(
        "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/Half-life_dataset_Human_train.csv"
    )
    halflife_stats = pd.read_csv(
        "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/Halflife_stats.csv"
    )

    print(f"Loaded train dataset with {len(train_df)} records")
    print(f"Loaded halflife_stats with {len(halflife_stats)} records")
    print(f"Halflife stats columns: {halflife_stats.columns.tolist()}")

    # Make sure we have a copy of the original dataframe with SMILES
    original_df = train_df.copy()

    # Define columns
    x_cols = ["LogKa"]
    categorical_cols = []

    # Define featurizers
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

    # Separate categorical data for proper processing
    categorical_data = (
        train_dataset.X[categorical_cols].copy() if categorical_cols else pd.DataFrame()
    )
    if categorical_cols:
        train_dataset.X = train_dataset.X.drop(columns=categorical_cols)

    # Apply feature selection to non-categorical features
    train_dataset.select_features(FeatureSelector=VarianceThreshold(threshold=0.0))

    # Remove highly correlated features
    print("Removing highly correlated features...")
    # Calculate correlation matrix on numerical features
    correlation_matrix = train_dataset.X.corr().abs()

    # Find pairs of features with correlation > 0.99
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > 0.95:
                high_corr_pairs.append(
                    (correlation_matrix.columns[i], correlation_matrix.columns[j])
                )

    # Print the highly correlated pairs
    if high_corr_pairs:
        print(f"Found {len(high_corr_pairs)} pairs of highly correlated features")
        for feat1, feat2 in high_corr_pairs:
            corr_val = correlation_matrix.loc[feat1, feat2]
            print(f"  {feat1} and {feat2}: {corr_val:.4f}")
    else:
        print("No highly correlated features found")

    # Drop the first feature from each pair
    features_to_drop = set()
    for feat1, feat2 in high_corr_pairs:
        features_to_drop.add(feat1)

    if features_to_drop:
        print(f"Dropping {len(features_to_drop)} features:")
        for feat in features_to_drop:
            print(f"  - {feat}")
        train_dataset.X = train_dataset.X.drop(columns=list(features_to_drop))

    print(f"Final feature set shape: {train_dataset.X.shape}")

    # Add categorical data back in
    if not categorical_data.empty:
        train_dataset.X = pd.concat([categorical_data, train_dataset.X], axis=1)

    # Set model parameters
    model_params = {
        "poly_degree": 1,
        "k_al": 1.0,
        "priors": {
            "theta_mu": 0.0,
            "theta_var": 1.0,
            "k_al_mu": 0.0,
            "k_al_var": 1.0,
            "l_mu": 0.0,
            "l_var": 1.0,
        },
    }

    # Start timing
    start_time = time.time()

    # Run feature selection with PFAS-based cross-validation
    selected_features, scores = pfas_based_forward_selection(
        dataset=train_dataset,
        halflife_stats=halflife_stats,
        original_df=original_df,
        model_params=model_params,
        n_folds=5,
        max_features=7,
        n_jobs=-1,
        preselected_features=["LogKa"],
        tolerance=0.0001,
        verbose=True,
    )

    # End timing
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time)/60:.2f} minutes")

    # Save results to file
    with open("selected_features.txt", "w") as f:
        f.write("Selected features:\n")
        for feature in selected_features:
            f.write(f"- {feature}\n")
        f.write(f"\nPFAS-level R² scores:\n")
        for i, score in enumerate(scores):
            f.write(f"Features {i+1}: {score:.4f}\n")
        f.write(f"\nTime taken: {(end_time - start_time)/60:.2f} minutes\n")

    print("Results saved to 'selected_features.txt'")

    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(scores) + 1), scores, marker="o", linestyle="-")
    plt.xlabel("Number of Features")
    plt.ylabel("PFAS-level R² Score")
    plt.title("Feature Selection Performance (PFAS-based CV)")
    plt.grid(True)

    # Add feature names to the plot
    for i, txt in enumerate(selected_features):
        plt.annotate(
            txt,
            (i + 1, scores[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(
        "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/pfas_feature_selection.png",
        dpi=300,
    )
    print("Performance plot saved to 'pfas_feature_selection.png'")
