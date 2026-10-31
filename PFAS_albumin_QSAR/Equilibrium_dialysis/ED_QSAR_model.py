import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import (
    MordredDescriptors,
    TopologicalFingerprint,
    RDKitDescriptors,
)
from jaqpotpy.doa import (
    Leverage,
    MeanVar,
    BoundingBox,
    Mahalanobis,
    KernelBased,
    CityBlock,
)
from sklearn.feature_selection import VarianceThreshold
from jaqpotpy.models import SklearnModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys
import shap
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.pyplot import get_cmap
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


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


df_train = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Train_Albumin_Binding_Data.csv"
)
df_test = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Test_Albumin_Binding_Data.csv"
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
        # "Bit_636",
        # "PEOE_VSA12",
        # "Bit_807",
    ]
)

test_dataset = JaqpotTabularDataset(
    df=df_test,
    y_cols="Ka",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizers=featurizers,
    task="REGRESSION",
)

test_dataset.select_features(SelectColumns=train_dataset.X.columns.tolist())
preprocessor_x = ColumnTransformer(
    transformers=[
        (
            "StandardScaler",
            StandardScaler(),
            list(set(train_dataset.X.columns) - set(categorical_cols)),
        ),
    ],
    remainder="passthrough",
    force_int_remainder_cols=False,
)

model = SklearnModel(
    dataset=train_dataset,
    model=RandomForestRegressor(random_state=42),
    preprocess_x=[preprocessor_x],
    doa=[
        Leverage(),
        MeanVar(),
        BoundingBox(),
        Mahalanobis(),
        KernelBased(),
        # CityBlock(),
    ],
)
model.fit()
model.cross_validate(dataset=train_dataset, n_splits=5, random_seed=42)
print("5-fold Cross-val Average Scores:", model.average_cross_val_scores)
model.evaluate(test_dataset)
print("Test scores", model.test_scores)
print(
    "Q2 score: ",
    q2_score(
        y_true=test_dataset.y.to_numpy().flatten(),
        y_pred=model.predict(test_dataset),
        y_train_mean=train_dataset.y.to_numpy().mean(),
    ),
)

doa_tests = model.predict_doa(test_dataset)
for doa_test_name, doa_test_results in doa_tests.items():
    doa_list = doa_test_results
    count_in_doa = sum(1 for entry in doa_list if not entry["inDoa"])
    percentage_in_doa = (count_in_doa / len(doa_list)) * 100
    print(
        f"Percentage of instances with inDoa=False for {doa_test_name}: {percentage_in_doa:.2f}%"
    )

    # Perform majority voting for each data instance based on the different doa results
    majority_voting_results = []
    for i in range(len(test_dataset.X)):
        in_doa_votes = sum(
            doa_tests[doa_test_name][i]["inDoa"] for doa_test_name in doa_tests
        )
        majority_in_doa = in_doa_votes > len(doa_tests) / 2
        majority_voting_results.append(majority_in_doa)

# Estimate the percentage of data points that are out of DOA based on majority voting
count_out_of_doa_majority = sum(1 for result in majority_voting_results if not result)
percentage_out_of_doa_majority = (
    count_out_of_doa_majority / len(majority_voting_results)
) * 100
print(
    f"Percentage of instances out of DOA based on majority voting: {percentage_out_of_doa_majority:.2f}%"
)

# Get the predictions
yy = model.predict(test_dataset)
yx = test_dataset.y
yx = yx.to_numpy()

# # Create a y_observed vs y_predicted plot and annotate the points with the congener
# plt.figure(figsize=(10, 6))

# # Plot each point with color based on DOA
# for i in range(len(yx)):
#     color = "black" if not majority_voting_results[i] else "blue"
#     plt.scatter(yx[i], yy[i], color=color, alpha=0.7)

# plt.plot(
#     [min(yx), max(yx)],
#     [min(yx), max(yx)],
#     color="red",
#     linestyle="--",
#     label="Ideal fit",
# )
# plt.xlabel("Observed Ka")
# plt.ylabel("Predicted Ka")
# plt.title("Observed vs Predicted Ka")
# plt.legend()

# # Annotate the points with the congener
# for i, txt in enumerate(df_test["Congener"]):
#     plt.annotate(txt, (yx[i], yy[i]), fontsize=8, alpha=0.7)

# plt.show()

# Ensure both arrays are 1D
if len(yx.shape) > 1:
    yx = yx.flatten()
if len(yy.shape) > 1:
    yy = yy.flatten()

# Create publication-quality visualization plot for LogKa predictions
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
LEGEND_SIZE = 8

# Calculate plot limits
all_y_values = np.concatenate([yx, yy])
min_y, max_y = np.floor(min(all_y_values)), np.ceil(max(all_y_values))
plot_range = [min_y, max_y]

# Get unique PFAS for coloring
unique_pfas = df_test["Congener"].unique()

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

# Create a mapping of PFAS to numeric indices for coloring
pfas_to_index = {pfas: i for i, pfas in enumerate(unique_pfas)}
color_indices = [pfas_to_index[p] for p in df_test["Congener"]]

# Create scatter plot with points colored by individual PFAS
scatter = ax.scatter(
    yx,
    yy,
    c=color_indices,
    cmap=custom_cmap,
    s=60,
    alpha=0.8,
    edgecolor="white",
    linewidth=0.5,
    zorder=10,  # Make sure points are on top
)

# Add text annotations for each point (matching DOA_testing.py style)
# for i, congener_name in enumerate(df_test["Congener"]):
#     x = yx[i]
#     y = yy[i]

#     # Slightly offset the text to avoid overlapping with the point
#     ax.annotate(
#         congener_name,
#         xy=(x, y),
#         xytext=(5, 5),  # 5 points offset
#         textcoords="offset points",
#         fontsize=12,
#         alpha=0.8,
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
#     )

# Identity line
ax.plot(plot_range, plot_range, "k--", lw=1.5)

# Styling
ax.set_xlabel(r"Observed $\mathit{Log}K_\mathrm{A}$", fontsize=LABEL_SIZE)
ax.set_ylabel(r"Predicted $\mathit{Log}K_\mathrm{A}$", fontsize=LABEL_SIZE)
# ax.set_title("PFAS LogKa: Observed vs Predicted", fontsize=TITLE_SIZE)
ax.tick_params(axis="both", labelsize=TICK_SIZE)
ax.set_xlim(plot_range)
ax.set_ylim(plot_range)

# Format axis ticks to have consistent decimal places at 0.5 intervals
ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

# Add grid to match DOA_testing.py
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)

# Add legend for individual PFAS compounds (matching DOA_testing.py style)
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
    fontsize=6,
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

# Add R² to the plot (matching DOA_testing.py style)
# test_r2 = r2_score(yx, yy)
# ax.text(
#     0.05,
#     0.95,
#     f"$R^2 = {0.641:.3f}$",
#     transform=ax.transAxes,
#     fontsize=LABEL_SIZE,
#     verticalalignment="top",
#     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
# )

# Adjust layout to make room for the legend
plt.tight_layout()

# Save the plot
# plt.savefig(
#     # "PFAS_albumin_QSAR/Equilibrium_dialysis/model_logka_predictions.png",
#     "PFAS_albumin_QSAR/Equilibrium_dialysis/LogKa_predictions.png",
#     dpi=300,
#     bbox_inches="tight",
# )
plt.show()

# from jaqpotpy import Jaqpot

# # Upload the pretrained model on Jaqpot
# jaqpot = Jaqpot()
# jaqpot.login()
# model.deploy_on_jaqpot(
#     jaqpot=jaqpot,
#     name="New Albumin",
#     description="None",
#     visibility="PRIVATE",
# )
