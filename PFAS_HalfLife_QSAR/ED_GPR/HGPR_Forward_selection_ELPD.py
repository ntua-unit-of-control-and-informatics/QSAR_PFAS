import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import norm
from joblib import Parallel, delayed
import warnings
import copy


def calculate_elpd_approx(model, X, y):
    """
    Calculate the approximated Expected Log Predictive Density (ELPD) for a given model.

    Parameters:
    -----------
    model : HeteroscedasticGPR
        Fitted model
    X : array-like
        Input features
    y : array-like
        Target values

    Returns:
    --------
    float
        Approximated ELPD score
    """
    n_samples = X.shape[0]
    elpd_values = np.zeros(n_samples)

    for i in range(n_samples):
        # Get the i-th sample
        x_i = X[i : i + 1]
        y_i = y[i]

        # Create datasets without i-th sample
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        X_without_i = X[mask]
        y_without_i = y[mask]

        # Compute predictions using model fitted on all data (approximation)
        # but calculate covariance with respect to X_without_i
        K_train = model._compute_kernel(X_without_i)
        K_train_test = model._compute_kernel(X_without_i, x_i)
        K_test = model._compute_kernel(x_i)

        # Add noise variance
        noise_var_train = model._compute_noise_variance(X_without_i, model.theta_)
        K_train_noise = K_train + np.diag(noise_var_train) + np.eye(len(K_train)) * 1e-6

        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(K_train_noise)

            # Solve systems
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_without_i))
            v = np.linalg.solve(L.T, np.linalg.solve(L, K_train_test))

            # Compute mean and variance
            f_i_mean = K_train_test.T @ alpha
            f_i_var = K_test - K_train_test.T @ v

            # Add aleatoric uncertainty
            aleatoric_var = model._compute_noise_variance(x_i, model.theta_)
            total_var = f_i_var + aleatoric_var

            # Compute log predictive density for point i
            log_pred_density = norm.logpdf(y_i, f_i_mean, np.sqrt(total_var))
            elpd_values[i] = log_pred_density
        except:
            elpd_values[i] = -np.inf

    # Return sum of log predictive densities
    return np.mean(elpd_values)


def evaluate_feature_set(features, dataset, model_params, cv, scoring="r2"):
    """
    Helper function to evaluate a set of features

    Parameters:
    -----------
    features : list
        List of feature names to evaluate
    dataset : JaqpotpyDataset
        Dataset containing features and target values
    model_params : dict
        Parameters for the HeteroscedasticGPR model
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric to use ("r2", "elpd", or "both")
    """
    from sklearn.preprocessing import OneHotEncoder

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Create a copy of the dataset with only the selected features
        copy_dataset = copy.deepcopy(dataset)
        feature_df = copy_dataset.X[features].copy()

        # Identify categorical columns in the selected features
        categorical_cols = detect_categorical_columns(feature_df)

        # Handle categorical features if any are selected
        if categorical_cols:
            feature_df = pd.get_dummies(
                feature_df, columns=categorical_cols, drop_first=True
            )

        # Create copy of data for processing
        X = feature_df.values.copy()
        y = dataset.y.values.copy()

        # Create the model
        if model_params is None:
            model = HeteroscedasticGPR(poly_degree=3)
        else:
            model = HeteroscedasticGPR(**model_params)

        # Perform cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        r2_scores = []
        elpd_scores = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train the model
            model.fit(X_train_scaled, y_train)

            # Calculate metrics
            y_val_pred = model.predict(X_val_scaled)
            r2_val = r2_score(y_val, y_val_pred)
            r2_scores.append(r2_val)

            if scoring in ["elpd", "both"]:
                # Calculate ELPD for validation set
                elpd_val = calculate_elpd_approx(model, X_val_scaled, y_val)
                elpd_scores.append(elpd_val)

        # Return average scores
        if scoring == "r2":
            return np.mean(r2_scores)
        elif scoring == "elpd":
            return np.mean(elpd_scores)
        else:  # both
            return np.mean(r2_scores), np.mean(elpd_scores)


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


def parallel_forward_selection(
    dataset,
    model_params=None,
    cv=5,
    scoring="both",  # Changed default to "both"
    max_features=None,
    n_jobs=-1,
    preselected_features=None,
    tolerance=0.01,
):
    """
    Perform forward feature selection with support for categorical features
    and ELPD scoring

    Parameters:
    -----------
    dataset : JaqpotpyDataset
        Dataset containing features and target
    model_params : dict or None
        Parameters for HeteroscedasticGPR model
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric to use ("r2", "elpd", or "both")
    max_features : int or None
        Maximum number of features to select
    n_jobs : int
        Number of parallel jobs for joblib
    preselected_features : list or None
        List of feature names to always include
    tolerance : float
        Minimum improvement threshold to continue selection
    """
    # Initialize preselected features list if not provided
    if preselected_features is None:
        preselected_features = []

    # Set maximum number of features
    n_features = dataset.X.shape[1]
    if max_features is None:
        max_features = n_features

    # Initialize variables
    selected_features = preselected_features.copy()
    feature_names = list(dataset.X.columns)
    remaining_features = [f for f in feature_names if f not in selected_features]
    r2_history = []
    elpd_history = []

    # Calculate initial score if there are preselected features
    if selected_features:
        if scoring == "both":
            initial_r2, initial_elpd = evaluate_feature_set(
                selected_features, dataset, model_params, cv, scoring
            )
            r2_history.append(initial_r2)
            elpd_history.append(initial_elpd)
            print(
                f"Initial scores with preselected features: R² = {initial_r2:.4f}, ELPD = {initial_elpd:.4f}"
            )
        elif scoring == "r2":
            initial_r2 = evaluate_feature_set(
                selected_features, dataset, model_params, cv, scoring
            )
            r2_history.append(initial_r2)
            print(f"Initial R² score with preselected features: {initial_r2:.4f}")
        else:  # elpd
            initial_elpd = evaluate_feature_set(
                selected_features, dataset, model_params, cv, scoring
            )
            elpd_history.append(initial_elpd)
            print(f"Initial ELPD score with preselected features: {initial_elpd:.4f}")
    else:
        if scoring in ["r2", "both"]:
            r2_history = []
        if scoring in ["elpd", "both"]:
            elpd_history = []

    # Perform forward selection
    for i in range(len(selected_features), max_features):
        if not remaining_features:
            print("No more features to select from.")
            break

        print(
            f"\nStep {i+1}/{max_features}: Evaluating {len(remaining_features)} candidate features..."
        )

        # Evaluate each candidate feature in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_feature_set)(
                selected_features + [feature], dataset, model_params, cv, scoring
            )
            for feature in remaining_features
        )

        # Process results based on scoring method
        if scoring == "both":
            # Unpack r2 and elpd scores
            candidate_r2_scores = [r[0] for r in results]
            candidate_elpd_scores = [r[1] for r in results]

            # Find best feature based on ELPD
            best_elpd_idx = np.argmax(candidate_elpd_scores)
            best_elpd_score = candidate_elpd_scores[best_elpd_idx]
            best_r2_score = candidate_r2_scores[best_elpd_idx]
            best_feature = remaining_features[best_elpd_idx]

            # Check stopping criteria for ELPD
            if elpd_history and best_elpd_score < elpd_history[-1] - tolerance:
                print(
                    f"\nStopping: ELPD score decreased by more than tolerance threshold"
                )
                print(f"Previous best ELPD: {elpd_history[-1]:.4f}")
                print(f"Current best ELPD: {best_elpd_score:.4f}")
                print(f"Decrease: {elpd_history[-1] - best_elpd_score:.4f}")
                break
        elif scoring == "r2":
            candidate_r2_scores = results
            best_r2_idx = np.argmax(candidate_r2_scores)
            best_r2_score = candidate_r2_scores[best_r2_idx]
            best_feature = remaining_features[best_r2_idx]

            # Check stopping criteria for R²
            if r2_history and best_r2_score < r2_history[-1] - tolerance:
                print(
                    f"\nStopping: R² score decreased by more than tolerance threshold"
                )
                print(f"Previous best R²: {r2_history[-1]:.4f}")
                print(f"Current best R²: {best_r2_score:.4f}")
                print(f"Decrease: {r2_history[-1] - best_r2_score:.4f}")
                break
        else:  # elpd
            candidate_elpd_scores = results
            best_elpd_idx = np.argmax(candidate_elpd_scores)
            best_elpd_score = candidate_elpd_scores[best_elpd_idx]
            best_feature = remaining_features[best_elpd_idx]

            # Check stopping criteria for ELPD
            if elpd_history and best_elpd_score < elpd_history[-1] - tolerance:
                print(
                    f"\nStopping: ELPD score decreased by more than tolerance threshold"
                )
                print(f"Previous best ELPD: {elpd_history[-1]:.4f}")
                print(f"Current best ELPD: {best_elpd_score:.4f}")
                print(f"Decrease: {elpd_history[-1] - best_elpd_score:.4f}")
                break

        # Add feature to selected set
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        # Update history based on scoring method
        if scoring == "both":
            r2_history.append(best_r2_score)
            elpd_history.append(best_elpd_score)
            print(
                f"Added feature '{best_feature}' (R²: {best_r2_score:.4f}, ELPD: {best_elpd_score:.4f})"
            )
        elif scoring == "r2":
            r2_history.append(best_r2_score)
            print(f"Added feature '{best_feature}' (R²: {best_r2_score:.4f})")
        else:  # elpd
            elpd_history.append(best_elpd_score)
            print(f"Added feature '{best_feature}' (ELPD: {best_elpd_score:.4f})")

        # Print current feature set
        print(f"Current feature set ({len(selected_features)}): {selected_features}")

    # Return best feature set based on scoring method
    if scoring == "both":
        if elpd_history:
            best_elpd_idx = np.argmax(elpd_history)
            best_features = selected_features[: best_elpd_idx + 1]

            print("\nBest subset of features based on ELPD:", best_features)
            print(f"Best ELPD score: {elpd_history[best_elpd_idx]:.4f}")
            print(f"Corresponding R² score: {r2_history[best_elpd_idx]:.4f}")

            # Create plot for both metrics
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # R² plot
            ax1.plot(range(len(r2_history)), r2_history, marker="o", color="blue")
            ax1.axvline(
                x=best_elpd_idx,
                color="red",
                linestyle="--",
                label=f"Best ELPD at {best_elpd_idx+1} features",
            )
            ax1.set_xlabel("Number of Features")
            ax1.set_ylabel("R² Score")
            ax1.set_title("R² Score vs Number of Features")
            ax1.grid(True)
            ax1.legend()

            # ELPD plot
            ax2.plot(range(len(elpd_history)), elpd_history, marker="o", color="green")
            ax2.axvline(
                x=best_elpd_idx,
                color="red",
                linestyle="--",
                label=f"Best ELPD at {best_elpd_idx+1} features",
            )
            ax2.set_xlabel("Number of Features")
            ax2.set_ylabel("ELPD Score")
            ax2.set_title("ELPD Score vs Number of Features")
            ax2.grid(True)
            ax2.legend()

            plt.tight_layout()
            plt.savefig("feature_selection_scores.png", dpi=300)

            return best_features, (r2_history, elpd_history)

    elif scoring == "r2":
        if r2_history:
            best_r2_idx = np.argmax(r2_history)
            best_features = selected_features[: best_r2_idx + 1]

            print("\nBest subset of features based on R²:", best_features)
            print(f"Best R² score: {r2_history[best_r2_idx]:.4f}")

            # Plot R² scores
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(r2_history)), r2_history, marker="o")
            plt.axvline(
                x=best_r2_idx,
                color="red",
                linestyle="--",
                label=f"Best R² at {best_r2_idx+1} features",
            )
            plt.xlabel("Number of Features")
            plt.ylabel("R² Score")
            plt.title("R² Score vs Number of Features")
            plt.grid(True)
            plt.legend()
            plt.savefig("feature_selection_r2_scores.png", dpi=300)

            return best_features, r2_history

    else:  # elpd
        if elpd_history:
            best_elpd_idx = np.argmax(elpd_history)
            best_features = selected_features[: best_elpd_idx + 1]

            print("\nBest subset of features based on ELPD:", best_features)
            print(f"Best ELPD score: {elpd_history[best_elpd_idx]:.4f}")

            # Plot ELPD scores
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(elpd_history)), elpd_history, marker="o")
            plt.axvline(
                x=best_elpd_idx,
                color="red",
                linestyle="--",
                label=f"Best ELPD at {best_elpd_idx+1} features",
            )
            plt.xlabel("Number of Features")
            plt.ylabel("ELPD Score")
            plt.title("ELPD Score vs Number of Features")
            plt.grid(True)
            plt.legend()
            plt.savefig("feature_selection_elpd_scores.png", dpi=300)

            return best_features, elpd_history

    print("\nNo features were selected.")
    if scoring == "both":
        return [], ([], [])
    else:
        return [], []


if __name__ == "__main__":
    import time
    from jaqpotpy.datasets import JaqpotpyDataset
    from jaqpotpy.descriptors import RDKitDescriptors, TopologicalFingerprint
    from sklearn.feature_selection import VarianceThreshold

    # Import the HGPR model
    from HGPR_Class import HeteroscedasticGPR

    # Load datasets
    train_df = pd.read_csv(
        "PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human_train.csv"
    )
    test_df = pd.read_csv(
        "PFAS_HalfLife_QSAR/Datasets/Half-life_dataset_Human_test.csv"
    )

    # Define columns
    x_cols = ["sex", "half_life_type", "adult", "LogKa", "Occupational_exposure"]
    categorical_cols = ["sex", "half_life_type", "adult", "Occupational_exposure"]

    # Define featurizers
    featurizers = [RDKitDescriptors(), TopologicalFingerprint()]

    # Create dataset
    train_dataset = JaqpotpyDataset(
        df=train_df,
        y_cols="half_life_days",
        x_cols=x_cols,
        smiles_cols="SMILES",
        featurizer=featurizers,
        task="regression",
    )

    # Separate categorical data for proper processing
    categorical_data = train_dataset.X[categorical_cols].copy()
    train_dataset.X = train_dataset.X.drop(columns=categorical_cols)

    # Apply feature selection to non-categorical features
    train_dataset.select_features(FeatureSelector=VarianceThreshold(threshold=0.0))

    # Add categorical data back in
    train_dataset.X = pd.concat([categorical_data, train_dataset.X], axis=1)

    # Set model parameters
    model_params = {
        "poly_degree": 2,
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

    # Run feature selection with both metrics (but selecting based on ELPD)
    selected_features, (r2_scores, elpd_scores) = parallel_forward_selection(
        dataset=train_dataset,
        model_params=model_params,
        cv=5,
        scoring="both",  # Use both metrics but select based on ELPD
        max_features=10,
        n_jobs=-1,
        preselected_features=["LogKa"],
        tolerance=0.005,
    )

    # End timing
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time)/60:.2f} minutes")

    # Save results to file
    with open("selected_features_elpd.txt", "w") as f:
        f.write("Selected features (based on ELPD):\n")
        for feature in selected_features:
            f.write(f"- {feature}\n")
        f.write(f"\nBest ELPD score: {max(elpd_scores) if elpd_scores else 'N/A'}\n")
        f.write(
            f"Corresponding R² score: {r2_scores[np.argmax(elpd_scores)] if elpd_scores else 'N/A'}\n"
        )
        f.write(f"Time taken: {(end_time - start_time)/60:.2f} minutes\n")

    print("Results saved to 'selected_features_elpd.txt'")
