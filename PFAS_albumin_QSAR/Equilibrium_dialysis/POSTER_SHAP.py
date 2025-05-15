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


df_train = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Train_Albumin_Binding_Data.csv"
)

# Create a JaqpotpyDataset objects
x_cols = []
categorical_cols = []
featurizers = [RDKitDescriptors(), TopologicalFingerprint()]

train_dataset = JaqpotpyDataset(
    df=df_train,
    y_cols="Ka",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
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


# Create a custom function to have more control over the plot
def custom_shap_summary_plot():
    # Set the figure size
    plt.figure(figsize=(12, 8))

    # Create the SHAP summary plot
    shap.summary_plot(
        shap_values,
        X_train_scaled,
        feature_names=X_train.columns,
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
    plt.title("Feature Impact on LogKa Prediction", fontsize=18, fontweight="bold")
    plt.xlabel("SHAP Value (impact on model output)", fontsize=16)
    plt.tight_layout()

    # Save the plot with high resolution
    plt.savefig(
        "PFAS_albumin_QSAR/Equilibrium_dialysis/LogKa_SHAP.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# Run the custom plot function
custom_shap_summary_plot()
