import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jaqpotpy.datasets import JaqpotpyDataset
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
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df_train = pd.read_csv("Train_Albumin_Binding_Data.csv")
df_train = df_train[df_train["Ka"] != 0]
df_train[["Ka"]] = np.log10(df_train[["Ka"]])
# df_train.drop(columns=["Authors"], inplace=True)

df_test = pd.read_csv("Test_Albumin_Binding_Data.csv")
df_test = df_test[df_test["Ka"] != 0]
df_test[["Ka"]] = np.log10(df_test[["Ka"]])
# df_test.drop(columns=["Authors"], inplace=True)

# Create a JaqpotpyDataset objects
x_cols = ["Temperature", "Albumin_Type", "Method"]
categorical_cols = ["Albumin_Type", "Method"]
featurizers = [RDKitDescriptors()]

train_dataset = JaqpotpyDataset(
    df=df_train,
    y_cols="Ka",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

# Use VarianceThreshold to select features with a minimum variance of 0.1
# FeatureSelector = VarianceThreshold(threshold=200)
# train_dataset.select_features(
#     FeatureSelector,
#     ExcludeColumns=["Albumin_Type", "Method"],
# )
# train_dataset.selected_features

train_dataset.select_features(
    SelectColumns=[
        "MolWt",
        "VSA_EState5",
        "Temperature",
        "Albumin_Type",
        "Method",
    ]
)


test_dataset = JaqpotpyDataset(
    df=df_test,
    y_cols="Ka",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)
test_dataset.select_features(SelectColumns=train_dataset.X.columns.tolist())

preprocessor_x = ColumnTransformer(
    transformers=[
        ("OneHotEncoder", OneHotEncoder(), categorical_cols),
        # ("MinMaxScaler", MinMaxScaler(), cols_to_scaling),
    ],
    remainder="passthrough",
    force_int_remainder_cols=False,
)

model = SklearnModel(
    dataset=train_dataset,
    model=RandomForestRegressor(),
    preprocess_x=[preprocessor_x],
    doa=[
        Leverage(),
        MeanVar(),
        BoundingBox(),
        Mahalanobis(),
        KernelBased(),
        CityBlock(),
    ],
)
model.fit()
model.cross_validate(dataset=train_dataset, n_splits=10)
print("10-fold Cross-val Average Scores:", model.average_cross_val_scores)
model.evaluate(test_dataset)
print("Test scores", model.test_scores)

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

yy = model.predict(test_dataset)
yx = test_dataset.y
yx = yx.to_numpy()

# Create a color map based on the "Authors" column
authors = df_test["Authors"].unique()
color_map = {author: plt.cm.tab20(i) for i, author in enumerate(authors)}
colors = df_test["Authors"].map(color_map)

plt.figure(figsize=(10, 8))
plt.scatter(yx, yy, color=colors, label="Predicted vs Actual")

# Annotate each data point with the corresponding congener from df_test
for i, txt in enumerate(df_test["Congener"]):
    plt.annotate(txt, (yx[i], yy[i]), fontsize=8, alpha=0.7)

plt.plot(
    [min(yx), max(yx)], [min(yx), max(yx)], color="red", linestyle="--", label="y=x"
)
plt.xlabel("Actual Ka (log scale)")
plt.ylabel("Predicted Ka (log scale)")
plt.title("Actual vs Predicted Ka (log scale)")

# Create a custom legend for the authors
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=color_map[author],
        markersize=10,
    )
    for author in authors
]
plt.legend(
    handles, authors, title="Authors", bbox_to_anchor=(1.05, 1), loc="upper left"
)

plt.tight_layout()
plt.show()

from jaqpotpy import Jaqpot

# # Upload the pretrained model on Jaqpot
jaqpot = Jaqpot()
jaqpot.login()
model.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Qsar PFAS Albumin Binding",
    description="Predicts log10Ka of PFAS compounds binding to albumin",
    visibility="PRIVATE",
)
