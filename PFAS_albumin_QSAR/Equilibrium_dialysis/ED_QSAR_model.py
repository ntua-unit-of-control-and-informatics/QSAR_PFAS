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
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys

df_train = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Train_Albumin_Binding_Data.csv"
)
df_test = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Test_Albumin_Binding_Data.csv"
)
df_test = df_test[df_test["Congener"] != "PFMBA"]

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

train_dataset.select_features(
    SelectColumns=[
        "fr_quatN",
        "Bit_592",
        "Chi2v",
        "AvgIpc",
        "Bit_886",
        "Bit_741",
        "Bit_1977",
        "Bit_636",
        # "PEOE_VSA12",
        # "Bit_807",
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
        CityBlock(),
    ],
)
model.fit()
model.cross_validate(dataset=train_dataset, n_splits=5, random_seed=42)
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

# Create a y_observed vs y_predicted plot and annotate the points with the congener
plt.figure(figsize=(10, 6))

# Plot each point with color based on DOA
for i in range(len(yx)):
    color = "black" if not majority_voting_results[i] else "blue"
    plt.scatter(yx[i], yy[i], color=color, alpha=0.7)

plt.plot(
    [min(yx), max(yx)],
    [min(yx), max(yx)],
    color="red",
    linestyle="--",
    label="Ideal fit",
)
plt.xlabel("Observed Ka")
plt.ylabel("Predicted Ka")
plt.title("Observed vs Predicted Ka")
plt.legend()

# Annotate the points with the congener
for i, txt in enumerate(df_test["Congener"]):
    plt.annotate(txt, (yx[i], yy[i]), fontsize=8, alpha=0.7)

plt.show()
