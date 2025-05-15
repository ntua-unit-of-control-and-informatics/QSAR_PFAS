import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import (
    TopologicalFingerprint,
    RDKitDescriptors,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from HGPR_Class import HeteroscedasticGPR

df_train = pd.read_csv(
    "PFAS_HalfLife_QSAR/Datasets/Revamped_dataset/Half-life_dataset_Human_train.csv"
)

# Create a JaqpotpyDataset objects
# Define base columns (these should be adjusted based on the forward selection results)
x_cols = ["LogKa"]
categorical_cols = []

# Setup featurizers
featurizers = [RDKitDescriptors(), TopologicalFingerprint()]

# Create dataset
train_dataset = JaqpotpyDataset(
    df=df_train,
    y_cols="half_life",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

# Features selected from forward selection
# These should be replaced with the actual features selected by the forward selection algorithm
selected_features = ["LogKa", "BCUT2D_LOGPLOW", "fr_halogen"]

# Prepare training data as a DataFrame to keep column names
X_df = train_dataset.X[selected_features].copy()

# Handle categorical features if any are selected
categorical_selected = [col for col in selected_features if col in categorical_cols]
if categorical_selected:
    X_df = pd.get_dummies(X_df, columns=categorical_selected, drop_first=True)

# Keep X as a DataFrame for SHAP visualization
X = X_df.copy()
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

# Create and fit HGPR model using numpy array
model = HeteroscedasticGPR(**model_params)
model.fit(X.values, y)


# Create a wrapper function for SHAP
def model_wrapper(X):
    if isinstance(X, pd.DataFrame):
        X = X.values
    return model.predict(X)


# Create a SHAP explainer with the wrapper function
explainer = shap.Explainer(model_wrapper, X)

# Calculate SHAP values
shap_values = explainer(X)


# Create a custom function to have more control over the plot
def custom_shap_summary_plot():
    # Set the figure size
    plt.figure(figsize=(12, 8))

    # Create the SHAP summary plot
    shap.summary_plot(
        shap_values,
        X,  # This is now a DataFrame with column names
        show=False,  # Don't show the plot yet
        alpha=0.8,  # Adjust transparency of dots
        cmap=plt.cm.viridis,  # Change colormap
        plot_size=(12, 8),  # Set plot size
    )

    # Get the current figure and its axes
    fig = plt.gcf()
    axes = fig.get_axes()

    # Make feature names bold and larger
    for ax in axes:
        # Find y-axis tick labels (feature names)
        y_labels = ax.get_yticklabels()

        # Create bold font properties
        font_prop = FontProperties()
        font_prop.set_weight("bold")
        font_prop.set_size(14)  # Increase font size

        # Set the font properties for each label
        for label in y_labels:
            label.set_fontproperties(font_prop)

    # Find all scatter plots and increase dot size
    for ax in axes:
        for collection in ax.collections:
            if isinstance(collection, plt.matplotlib.collections.PathCollection):
                # This is likely a scatter plot - significantly increase dot size
                collection.set_sizes(
                    [200]
                )  # Adjust size as needed - larger value = bigger dots

    # Add a custom title and labels
    plt.title("Feature Impact on Half-Life Prediction", fontsize=18, fontweight="bold")
    plt.xlabel("SHAP Value (impact on model output)", fontsize=16)
    plt.tight_layout()

    # Save the plot with high resolution
    plt.savefig(
        "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Halflife_SHAP.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# Run the custom plot function
custom_shap_summary_plot()
