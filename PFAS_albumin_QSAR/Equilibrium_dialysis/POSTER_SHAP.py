import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import (
    TopologicalFingerprint,
    RDKitDescriptors,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Train_Albumin_Binding_Data.csv"
)

# Create a JaqpotTabularDataset objects
x_cols = []
categorical_cols = []
featurizers = [RDKitDescriptors(), TopologicalFingerprint()]

train_dataset = JaqpotTabularDataset(
    df=df_train,
    y_cols="Ka",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizers=featurizers,
    task="REGRESSION",
)

train_dataset.select_features(
    SelectColumns=[
        "fr_quatN",
        "Bit_592",
        "Chi2v",
        "AvgIpc",
        "Bit_886",
        # "Bit_741",
        "Bit_1977",
    ]
)

X_train = train_dataset.X
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, train_dataset.y)
r2_score = model.score(X_train_scaled, train_dataset.y)
print(f"R2 score: {r2_score}")

# Create a SHAP explainer
explainer = shap.Explainer(model, X_train_scaled)

# Calculate SHAP values
shap_values = explainer(X_train_scaled)

# Set original feature names on shap_values
shap_values.feature_names = list(X_train.columns)


# Create a custom function to have more control over the plot
def custom_shap_summary_plot():
    # JCIM Journal Requirements:
    # - Single-column max width: 3.33 inches (240 points)
    # - Minimum font size: 4.5 points
    # - Recommended: Axis titles (8-10 pt), Tick labels (7-9 pt)

    # Set the figure size for single-column publication
    plt.figure(figsize=(3.33, 4))

    # Create the SHAP summary plot
    shap.summary_plot(
        shap_values,
        X_train_scaled,
        feature_names=X_train.columns,
        show=False,  # Don't show the plot yet
        alpha=0.8,  # Adjust transparency of dots
        cmap=plt.cm.viridis,  # Change colormap
        plot_size=(3.33, 4),  # Set plot size to match JCIM single-column
    )

    # Get the current figure and its axes
    fig = plt.gcf()
    axes = fig.get_axes()

    # Make feature names bold and appropriately sized for JCIM
    for ax in axes:
        # Find y-axis tick labels (feature names)
        y_labels = ax.get_yticklabels()

        # Create bold font properties
        font_prop = FontProperties()
        # font_prop.set_weight("bold")
        font_prop.set_size(8)  # Adjusted to JCIM tick label size (7-9 pt)

        # Set the font properties for each label
        for label in y_labels:
            label.set_fontproperties(font_prop)

    # Find all scatter plots and adjust dot size for smaller figure
    for ax in axes:
        for collection in ax.collections:
            if isinstance(collection, plt.matplotlib.collections.PathCollection):
                # Adjust dot size for single-column figure
                collection.set_sizes([30])  # Reduced from 200 for smaller figure

    # Add labels with JCIM-compliant font sizes
    # plt.title("Feature Impact on LogKa Prediction", fontsize=12, fontweight="bold")
    plt.xlabel(r"SHAP Value (impact on $\mathit{Log}K_\mathrm{A}$)", fontsize=10)

    # Make x-axis symmetric around zero with specific ticks
    # Only apply to the main plot axis (first axis), not the colorbar
    main_ax = axes[0]

    # Get current x-axis limits
    xlim = main_ax.get_xlim()
    # Make symmetric around zero
    max_abs = max(abs(xlim[0]), abs(xlim[1]))
    main_ax.set_xlim(-max_abs, max_abs)

    # Set specific tick locations
    main_ax.set_xticks([-0.5, 0.0, 0.5])

    # Adjust tick label size for all axes
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()

    # Save the plot with high resolution
    plt.savefig(
        "PFAS_albumin_QSAR/Equilibrium_dialysis/LogKa_SHAP.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()


def custom_shap_bar_plot():
    """Create a SHAP feature importance bar plot following JCIM guidelines."""
    # JCIM Journal Requirements:
    # - Single-column max width: 3.33 inches (240 points)
    # - Minimum font size: 4.5 points
    # - Recommended: Axis titles (8-10 pt), Tick labels (7-9 pt)

    # Create the SHAP bar plot with custom styling
    plt.figure(figsize=(3.33, 4))
    shap.summary_plot(
        shap_values,
        X_train_scaled,
        feature_names=X_train.columns,
        plot_type="bar",
        show=False,
        color="#ff0051",  # Red color for bars (SHAP default red)
    )

    # Get the current figure and axes
    fig = plt.gcf()
    ax = fig.gca()

    # Increase font sizes for readability in small figure
    # For a 3.33" figure, fonts need to be larger to be readable when printed
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    # Update axis labels with custom text and font size
    ax.set_xlabel("mean(|SHAP value|)", fontsize=16)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)

    plt.tight_layout()

    # Save the plot with high resolution
    plt.savefig(
        "PFAS_albumin_QSAR/Equilibrium_dialysis/LogKa_SHAP_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    # plt.show()


# Run the custom plot functions
custom_shap_summary_plot()
custom_shap_bar_plot()
