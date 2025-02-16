from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import (
    MordredDescriptors,
    TopologicalFingerprint,
    RDKitDescriptors,
)
from jaqpotpy.models import SklearnModel
from sklearn.feature_selection import VarianceThreshold


def evaluate_feature_set(features, dataset, estimator, cv, scoring):
    """Helper function to evaluate a set of features"""
    copy_dataset = dataset.copy()
    copy_dataset.select_features(SelectColumns=features)

    model = SklearnModel(
        dataset=copy_dataset, model=estimator, preprocess_x=StandardScaler()
    )
    model.fit()
    model.cross_validate(dataset=copy_dataset, n_splits=cv, random_seed=42)
    return model.average_cross_val_scores["r2"] if scoring == "r2" else None


def parallel_forward_selection(
    dataset, estimator=None, cv=5, scoring="r2", max_features=None, n_jobs=-1
):
    """
    Parallel implementation of forward selection with relaxed stopping criteria

    Parameters:
    -----------
    dataset : JaqpotpyDataset
        Input dataset
    estimator : sklearn estimator, optional
        If None, defaults to RandomForestRegressor()
    cv : int, optional
        Number of cross-validation folds
    scoring : str, optional
        Scoring metric
    max_features : int, optional
        Maximum number of features to select
    n_jobs : int, optional
        Number of parallel jobs (-1 for all cores)
    """
    if estimator is None:
        estimator = RandomForestRegressor(random_state=42)

    n_features = dataset.X.shape[1]
    if max_features is None:
        max_features = n_features

    # Initialize variables
    selected_features = []
    feature_names = list(dataset.X.columns)
    remaining_features = feature_names.copy()
    scores_history = []

    for i in range(max_features):
        # Parallel evaluation of all remaining features
        candidate_scores = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_feature_set)(
                selected_features + [feature], dataset, estimator, cv, scoring
            )
            for feature in remaining_features
        )

        # Find best feature
        best_score_idx = np.argmax(candidate_scores)
        best_score = candidate_scores[best_score_idx]
        best_feature = remaining_features[best_score_idx]

        # Relaxed stopping criteria
        if (
            scores_history and best_score < scores_history[-1] - 0.01
        ):  # 0.01 is tolerance
            print(f"\nStopping: Score decreased by more than tolerance threshold")
            print(f"Previous best score: {scores_history[-1]:.4f}")
            print(f"Current best score: {best_score:.4f}")
            print(f"Decrease: {scores_history[-1] - best_score:.4f}")
            break

        # Stop if score is really poor
        if best_score < -0.5:  # You can adjust this threshold
            print(f"\nStopping: Score too low ({best_score:.4f})")
            break

        # Update feature sets
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        scores_history.append(best_score)

        print(f"Step {i+1}: Added feature '{best_feature}' (Score: {best_score:.4f})")

    print("\nFinal selected features:", selected_features)
    print(f"Final score: {scores_history[-1]:.4f}")

    return selected_features, scores_history


# Usage example:
if __name__ == "__main__":
    import time

    df_train = pd.read_csv(
        "PFAS_albumin_QSAR/Equilibrium_dialysis/Train_Albumin_Binding_Data.csv"
    )

    # Create a JaqpotpyDataset objects
    x_cols = []  # ["Albumin_Type"]
    categorical_cols = []  # ["Albumin_Type"]
    featurizers = [RDKitDescriptors(), TopologicalFingerprint()]

    train_dataset = JaqpotpyDataset(
        df=df_train,
        y_cols="Ka",
        x_cols=x_cols,
        smiles_cols="SMILES",
        featurizer=featurizers,
        task="regression",
    )
    train_dataset.select_features(FeatureSelector=VarianceThreshold(threshold=0.0))

    start_time = time.time()

    selected_features, scores = parallel_forward_selection(
        dataset=train_dataset,
        estimator=RandomForestRegressor(random_state=42),
        cv=5,
        max_features=10,
        n_jobs=-1,  # Use all available cores
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
