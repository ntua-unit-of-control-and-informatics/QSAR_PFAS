import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import (
    MordredDescriptors,
    TopologicalFingerprint,
    RDKitDescriptors,
)
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
from PIL import Image
import io


df_train = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Train_Albumin_Binding_Data.csv"
)
df_test = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Test_Albumin_Binding_Data.csv"
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
        # "Bit_636",
        # "PEOE_VSA12",
        # "Bit_807",
    ]
)
# Find Congeners that trigger specific bits
trigger_bits = [
    "Bit_592",
    "Bit_886",  # "Bit_741",
    "Bit_1977",
]
triggered_congeners = []
negative_congeners = []
for bit in trigger_bits:
    triggered_congener = train_dataset.X[train_dataset.X[bit] == 1].iloc[0]
    triggered_congener_row = train_dataset.X[train_dataset.X[bit] == 1].index[0]
    congener_info = df_train.iloc[triggered_congener_row]
    triggered_congeners.append(
        {
            "Congener": congener_info["Congener"],
            "SMILES": congener_info["SMILES"],
            "Triggered Bit": bit,
        }
    )
    # Also find negative examples
    negative_congener = train_dataset.X[train_dataset.X[bit] == 0].iloc[0]
    negative_congener_row = train_dataset.X[train_dataset.X[bit] == 0].index[0]
    congener_info = df_train.iloc[negative_congener_row]
    negative_congeners.append(
        {
            "Congener": congener_info["Congener"],
            "SMILES": congener_info["SMILES"],
            "Triggered Bit": bit,
        }
    )


triggered_congeners_df = pd.DataFrame(triggered_congeners)
print("Triggering Congeners", triggered_congeners_df)
negative_congeners_df = pd.DataFrame(negative_congeners)
print("Negative Congeners", negative_congeners_df)
# Dictionary of bit_number: SMILES that triggers it
bit_smiles_dict = {
    592: "C(C(C(C(F)(F)Cl)(F)F)(F)F)(C(C(OC(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F",  # Replace with your actual SMILES
    886: "C(C(=O)O)NS(=O)(=O)C(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F",
    # 741: "C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O",
    1977: "C(=O)(C(C(C(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
}

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 12))
axes = axes.ravel()

for idx, (bit, smiles) in enumerate(bit_smiles_dict.items()):
    mol = Chem.MolFromSmiles(smiles)

    # Generate Morgan fingerprint with bit information
    bit_info = {}
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=2048, bitInfo=bit_info
    )

    if bit in bit_info:
        # Get atoms that contribute to this bit
        highlight_atoms = [atom_idx for atom_idx, _ in bit_info[bit]]

        # Create drawing
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(500, 500)
        drawer.drawOptions().addAtomIndices = True

        # Draw molecule with highlighting
        Draw.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight_atoms)
        drawer.FinishDrawing()

        # Convert to image
        img = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img))

        # Plot in the corresponding subplot
        axes[idx].imshow(img)
        axes[idx].set_title(f"Bit {bit}")
        axes[idx].axis("off")

plt.tight_layout()
plt.show()

# Optional: Print the environments for each bit
for bit, smiles in bit_smiles_dict.items():
    mol = Chem.MolFromSmiles(smiles)
    bit_info = {}
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, bitInfo=bit_info)

    print(f"\nBit {bit} environment in {smiles}:")
    if bit in bit_info:
        for atom_idx, radius_used in bit_info[bit]:
            atom = mol.GetAtomWithIdx(atom_idx)
            print(f"Central atom: {atom.GetSymbol()} (index {atom_idx})")
            print(f"Environment radius: {radius_used}")
            neighbors = atom.GetNeighbors()
            print("Neighboring atoms:")
            for neighbor in neighbors:
                print(f"- {neighbor.GetSymbol()} (index {neighbor.GetIdx()})")
