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
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib import pyplot as plt
from PIL import Image
import io
from matplotlib.gridspec import GridSpec


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

# Dictionary of negative examples (non-triggering SMILES)
non_trigger_smiles_dict = {}
for bit, bit_name in zip([592, 886, 1977], ["Bit_592", "Bit_886", "Bit_1977"]):
    # Take the first SMILES from negative examples dataframe
    for _, row in negative_congeners_df.iterrows():
        if row["Triggered Bit"] == bit_name:
            non_trigger_smiles_dict[bit] = row["SMILES"]
            break

# If we don't have a negative example from the dataframe, use some default examples
# These are simplified structures that would likely not trigger the specific bits
if 592 not in non_trigger_smiles_dict:
    non_trigger_smiles_dict[592] = (
        "C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"  # Simple perfluorinated chain without sulfonic acid
    )
if 886 not in non_trigger_smiles_dict:
    non_trigger_smiles_dict[886] = (
        "C(C(=O)O)NC(F)(F)C(F)(F)C(F)(F)F"  # Similar but without sulfonamide
    )
if 1977 not in non_trigger_smiles_dict:
    non_trigger_smiles_dict[1977] = (
        "C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"  # Perfluorinated chain without carboxylic acid
    )

# Add descriptive labels for each bit
bit_descriptions = {
    592: "Sulfonic acid group",
    886: "Cf3-Cf2-C sub-structure",
    1977: "Carboxylic acid with perfluorinated chain",
}

# Create a figure with gridspec for more flexible layout (positive and negative examples)
fig = plt.figure(figsize=(24, 12))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1])
axes_positive = [fig.add_subplot(gs[0, i]) for i in range(3)]
axes_negative = [fig.add_subplot(gs[1, i]) for i in range(3)]

# Add a common title
fig.suptitle(
    "Morgan Fingerprint Bit Analysis: Triggering vs Non-Triggering Structures",
    fontsize=16,
)


# Function to get environment atoms up to a specified radius
def get_environment_atoms(mol, atom_idx, radius):
    """Get all atoms in the environment of the central atom up to specified radius"""
    environment = set()

    # Add the central atom itself
    environment.add(atom_idx)

    # Get neighboring atoms
    for r in range(1, radius + 1):
        # Find atoms r bonds away
        environment_prev = environment.copy()
        for curr_atom_idx in environment_prev:
            atom = mol.GetAtomWithIdx(curr_atom_idx)
            for neighbor in atom.GetNeighbors():
                environment.add(neighbor.GetIdx())

    return environment


# Function to get atoms at exact distance r from center
def get_atoms_at_distance(mol, atom_idx, r):
    """Get atoms exactly r bonds away from the central atom"""
    if r == 0:
        return {atom_idx}

    prev_atoms = get_environment_atoms(mol, atom_idx, r - 1)
    curr_atoms = get_environment_atoms(mol, atom_idx, r)
    return curr_atoms - prev_atoms


def draw_molecule_with_highlights(
    mol, bit, bit_info, axes, title_prefix="", add_description=True
):
    """Draw a molecule with properly highlighted substructures that trigger a specific fingerprint bit"""
    # Get atom indices that contribute to the bit
    highlight_atoms = []
    highlight_bonds = []
    atom_colors = {}
    bond_colors = {}

    if bit in bit_info:
        # Define color palette
        base_color = (0.0, 0.4, 1.0)  # RGB for blue

        for bit_center, radius in bit_info[bit]:
            # Add central atom with bright color
            atom_colors[bit_center] = (1.0, 0.2, 0.2)  # Bright red for center
            highlight_atoms.append(bit_center)

            # Add atoms at each distance with gradient colors
            for r in range(1, radius + 1):
                atoms_at_r = get_atoms_at_distance(mol, bit_center, r)
                # Calculate intensity based on distance - closer atoms are more intense
                intensity = 1.0 - (r / (radius + 1)) * 0.7  # Scale from 1.0 down to 0.3

                # Assign gradient color
                r_color = base_color[0] * intensity
                g_color = base_color[1] * intensity
                b_color = base_color[2]  # Keep blue channel full for better visibility

                for atom_idx in atoms_at_r:
                    highlight_atoms.append(atom_idx)
                    atom_colors[atom_idx] = (r_color, g_color, b_color)

            # Get all environment atoms for bond highlights
            all_env_atoms = get_environment_atoms(mol, bit_center, radius)
            highlight_atoms = list(
                all_env_atoms
            )  # Update highlight_atoms with all atoms

    # Create bonds between highlighted atoms
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        if begin_atom in highlight_atoms and end_atom in highlight_atoms:
            highlight_bonds.append(bond.GetIdx())

            # Assign bond color based on average of atom colors
            if begin_atom in atom_colors and end_atom in atom_colors:
                r_avg = (atom_colors[begin_atom][0] + atom_colors[end_atom][0]) / 2
                g_avg = (atom_colors[begin_atom][1] + atom_colors[end_atom][1]) / 2
                b_avg = (atom_colors[begin_atom][2] + atom_colors[end_atom][2]) / 2
                bond_colors[bond.GetIdx()] = (r_avg, g_avg, b_avg)
            else:
                bond_colors[bond.GetIdx()] = base_color

    # Draw the molecule with highlighted substructure
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)

    # Set drawing options
    drawer.drawOptions().bondLineWidth = 2.0
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().highlightRadius = 0.6

    # Draw with highlighting
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()

    # Convert to image
    img = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img))

    # Plot in the specified subplot
    axes.imshow(img)
    title = f"{title_prefix}Bit {bit} Fingerprint"
    if add_description and bit in bit_descriptions:
        title += f"\n{bit_descriptions[bit]}"
    axes.set_title(title, fontsize=14)
    axes.axis("off")

    return highlight_atoms, highlight_bonds


# Process the positive examples (molecules that trigger the bits)
for idx, (bit, smiles) in enumerate(bit_smiles_dict.items()):
    mol = Chem.MolFromSmiles(smiles)

    # Calculate Morgan fingerprint with bit info
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, bitInfo=bit_info)

    # Draw positive example
    draw_molecule_with_highlights(mol, bit, bit_info, axes_positive[idx], "Triggers ")

    # Check if this bit is actually triggered by the molecule (for verification)
    if bit in bit_info:
        axes_positive[idx].text(
            0.5,
            -0.05,
            "Bit is confirmed present",
            transform=axes_positive[idx].transAxes,
            horizontalalignment="center",
            color="green",
            fontsize=12,
        )
    else:
        axes_positive[idx].text(
            0.5,
            -0.05,
            "Warning: Bit not found in fingerprint!",
            transform=axes_positive[idx].transAxes,
            horizontalalignment="center",
            color="red",
            fontsize=12,
        )

# Process the negative examples (molecules that don't trigger the bits)
for idx, (bit, smiles) in enumerate(non_trigger_smiles_dict.items()):
    mol = Chem.MolFromSmiles(smiles)

    # Calculate Morgan fingerprint with bit info
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, bitInfo=bit_info)

    # Check if bit is actually not present (for verification)
    bit_present = bit in bit_info

    # For negative examples, we try to find similar substructures to highlight
    # We can look for the most similar atoms to what would trigger the bit
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
    drawer.drawOptions().bondLineWidth = 2.0
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Convert to image
    img = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img))

    # Plot in the specified subplot
    axes_negative[idx].imshow(img)
    title = f"Does NOT Trigger Bit {bit}"
    if bit in bit_descriptions:
        title += f"\nMissing: {bit_descriptions[bit]}"
    axes_negative[idx].set_title(title, fontsize=14)
    axes_negative[idx].axis("off")

    # Add verification message
    if bit_present:
        axes_negative[idx].text(
            0.5,
            -0.05,
            "Warning: Bit is actually present!",
            transform=axes_negative[idx].transAxes,
            horizontalalignment="center",
            color="red",
            fontsize=12,
        )
    else:
        axes_negative[idx].text(
            0.5,
            -0.05,
            "Verified: Bit is absent",
            transform=axes_negative[idx].transAxes,
            horizontalalignment="center",
            color="green",
            fontsize=12,
        )

# Add row labels
fig.text(
    0.02,
    0.75,
    "TRIGGERING\nSTRUCTURES",
    fontsize=14,
    ha="center",
    va="center",
    rotation=90,
    fontweight="bold",
)
fig.text(
    0.02,
    0.25,
    "NON-TRIGGERING\nSTRUCTURES",
    fontsize=14,
    ha="center",
    va="center",
    rotation=90,
    fontweight="bold",
)

# Add explanation of the color coding
legend_text = "Color coding: Red = central atom that triggers the fingerprint bit\nBlue gradient = surrounding atoms with intensity decreasing with distance"
fig.text(
    0.5,
    0.02,
    legend_text,
    ha="center",
    fontsize=12,
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.tight_layout(
    rect=[0.03, 0.05, 1, 0.95]
)  # Adjust layout to make room for annotations
plt.savefig("fingerprint_substructures_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Print detailed environment info for each bit
for bit, smiles in bit_smiles_dict.items():
    mol = Chem.MolFromSmiles(smiles)
    bit_info = {}
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, bitInfo=bit_info)

    print(f"\nBit {bit} environment in {Chem.MolToSmiles(mol, isomericSmiles=True)}:")
    if bit in bit_info:
        for atom_idx, radius_used in bit_info[bit]:
            atom = mol.GetAtomWithIdx(atom_idx)
            print(f"Central atom: {atom.GetSymbol()} (index {atom_idx})")
            print(f"Environment radius: {radius_used}")
            env_atoms = get_environment_atoms(mol, atom_idx, radius_used)
            env_atoms.remove(
                atom_idx
            )  # Remove the central atom since we already listed it
            print(f"All atoms in environment: {sorted(env_atoms)}")

            # Print atoms at each distance
            for r in range(1, radius_used + 1):
                atoms_at_r = get_atoms_at_distance(mol, atom_idx, r)
                print(f"Atoms at distance {r}: {sorted(atoms_at_r)}")

            neighbors = atom.GetNeighbors()
            print("Direct neighbors:")
            for neighbor in neighbors:
                print(f"- {neighbor.GetSymbol()} (index {neighbor.GetIdx()})")
            print("---")

# Save the figure with high resolution
plt.figure(fig.number)
plt.savefig("fingerprint_bit_analysis.png", dpi=300, bbox_inches="tight")
print("Figure saved as fingerprint_bit_analysis.png")
