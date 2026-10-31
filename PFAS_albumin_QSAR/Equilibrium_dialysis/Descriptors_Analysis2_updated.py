import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import (
    MordredDescriptors,
    TopologicalFingerprint,
    RDKitDescriptors,
)
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib import pyplot as plt
from PIL import Image
import io
from matplotlib.gridspec import GridSpec
from collections import deque


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

# Dictionary of bit_number: (SMILES, Congener_name) that triggers it
bit_smiles_dict = {}
for bit_name, bit in zip(["Bit_592", "Bit_886", "Bit_1977"], [592, 886, 1977]):
    for _, row in triggered_congeners_df.iterrows():
        if row["Triggered Bit"] == bit_name:
            bit_smiles_dict[bit] = (row["SMILES"], row["Congener"])
            break

# Dictionary of negative examples (non-triggering SMILES and names)
non_trigger_smiles_dict = {}
for bit_name, bit in zip(["Bit_592", "Bit_886", "Bit_1977"], [592, 886, 1977]):
    for _, row in negative_congeners_df.iterrows():
        if row["Triggered Bit"] == bit_name:
            non_trigger_smiles_dict[bit] = (row["SMILES"], row["Congener"])
            break

# If we don't have a negative example from the dataframe, use some default examples
# These are simplified structures that would likely not trigger the specific bits
if 592 not in non_trigger_smiles_dict:
    non_trigger_smiles_dict[592] = (
        "C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",  # Simple perfluorinated chain without sulfonic acid
        "Default Example",
    )
if 886 not in non_trigger_smiles_dict:
    non_trigger_smiles_dict[886] = (
        "C(C(=O)O)NC(F)(F)C(F)(F)C(F)(F)F",  # Similar but without sulfonamide
        "Default Example",
    )
if 1977 not in non_trigger_smiles_dict:
    non_trigger_smiles_dict[1977] = (
        "C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",  # Perfluorinated chain without carboxylic acid
        "Default Example",
    )

# Add descriptive labels for each bit (standardized to 2 lines for alignment)
bit_descriptions = {
    592: "\n–SO₃H as terminal group\nof the perfluorinated chain",
    886: "\nCH₃-CH₂ as terminal group\n ",  # Added empty line for alignment
    1977: "\n–COOH with an adjacent carbon\nbonded to the main carbon chain",
}

# Create a figure with gridspec for more flexible layout
# JCIM Journal Requirements: Two-column max width = 7 inches
fig = plt.figure(figsize=(7, 5), dpi=300)  # Adjusted for two-column journal format
gs = GridSpec(
    2, 3, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.3
)  # Added spacing
axes_positive = [fig.add_subplot(gs[0, i]) for i in range(3)]
axes_negative = [fig.add_subplot(gs[1, i]) for i in range(3)]

# Removed main title for cleaner publication-ready figure
# fig.suptitle(
#     "ECFPs Bit Analysis: Triggering vs Non-Triggering PFAS Structures",
#     fontsize=12,
#     fontweight="bold",
# )


def optimize_molecule_layout(mol):
    """
    Optimize the 2D layout of the molecule for better visualization
    """
    # Create a copy to avoid modifying the original
    mol_copy = Chem.Mol(mol)

    # Force coordinate generation with better parameters
    rdDepictor.Compute2DCoords(mol_copy)

    # Try to optimize the layout using different strategies
    try:
        # Strategy 1: Use RDKit's coordgen if available
        from rdkit.Chem import rdCoordGen

        rdCoordGen.AddCoords(mol_copy)
    except ImportError:
        # Strategy 2: Use standard RDKit depiction with optimization
        rdDepictor.Compute2DCoords(mol_copy)

    # Additional optimization for better spacing
    try:
        # Normalize depiction to avoid overlaps
        rdDepictor.Normalize(mol_copy)
        rdDepictor.StraightenDepiction(mol_copy)
    except:
        pass

    return mol_copy


def get_environment_atoms_corrected(mol, center_atom, radius):
    """
    Correctly get atoms within a specified radius using BFS
    """
    if radius == 0:
        return {center_atom}

    visited = set()
    atoms_by_distance = {0: {center_atom}}
    queue = deque([(center_atom, 0)])
    visited.add(center_atom)

    while queue:
        current_atom, current_distance = queue.popleft()

        if current_distance < radius:
            atom = mol.GetAtomWithIdx(current_atom)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    new_distance = current_distance + 1

                    if new_distance not in atoms_by_distance:
                        atoms_by_distance[new_distance] = set()
                    atoms_by_distance[new_distance].add(neighbor_idx)

                    queue.append((neighbor_idx, new_distance))

    # Return all atoms within radius
    all_atoms = set()
    for distance in range(radius + 1):
        if distance in atoms_by_distance:
            all_atoms.update(atoms_by_distance[distance])

    return all_atoms, atoms_by_distance


def draw_molecule_with_highlights_improved(
    mol, bit, bit_info, axes, congener_name="", title_prefix="", add_description=True
):
    """Draw a molecule with improved layout and highlighting, including congener name"""

    # Optimize the molecule layout first
    mol_optimized = optimize_molecule_layout(mol)

    highlight_atoms = []
    highlight_bonds = []
    atom_colors = {}
    bond_colors = {}

    if bit in bit_info:
        # Define color palette with milder, more transparent colors for publication
        colors = [
            (1.0, 0.4, 0.4, 0.5),  # Soft red for center (radius 0) with transparency
            (1.0, 0.7, 0.3, 0.5),  # Soft orange for radius 1 with transparency
            (0.4, 0.6, 1.0, 0.5),  # Soft blue for radius 2 with transparency
        ]

        for center_atom, radius in bit_info[bit]:
            try:
                env_atoms, atoms_by_distance = get_environment_atoms_corrected(
                    mol_optimized, center_atom, radius
                )
                highlight_atoms.extend(env_atoms)

                # Color atoms by their distance from center
                for distance, atoms_at_distance in atoms_by_distance.items():
                    if distance < len(colors):
                        color = colors[distance]
                    else:
                        # Fade to soft light blue for distances beyond our color palette
                        intensity = max(0.3, 1.0 - (distance - len(colors)) * 0.2)
                        color = (0.4 * intensity, 0.7 * intensity, 1.0, 0.4)

                    for atom_idx in atoms_at_distance:
                        atom_colors[atom_idx] = color

            except Exception as e:
                print(f"Error processing center atom {center_atom} for bit {bit}: {e}")
                continue

    # Highlight bonds between highlighted atoms
    for bond in mol_optimized.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        if begin_atom in highlight_atoms and end_atom in highlight_atoms:
            highlight_bonds.append(bond.GetIdx())

            # Average the colors of the connected atoms
            if begin_atom in atom_colors and end_atom in atom_colors:
                begin_color = atom_colors[begin_atom]
                end_color = atom_colors[end_atom]
                avg_color = tuple((begin_color[i] + end_color[i]) / 2 for i in range(3))
                bond_colors[bond.GetIdx()] = avg_color

    # Draw the molecule with improved settings
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)  # Larger canvas

    # Improved drawing options for two-column format
    drawer.drawOptions().bondLineWidth = 2.0
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().highlightRadius = 0.35
    drawer.drawOptions().atomLabelFontSize = 18
    drawer.drawOptions().bondLineWidth = 2.0
    drawer.drawOptions().multipleBondOffset = 0.15
    drawer.drawOptions().padding = 0.1

    # Set background to white for better contrast
    drawer.drawOptions().clearBackground = True
    drawer.drawOptions().backgroundColour = (1, 1, 1, 1)  # White background

    if highlight_atoms and highlight_bonds:
        # Color only bonds, keep all atom labels in black for readability
        drawer.DrawMolecule(
            mol_optimized,
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            highlightAtomColors={},  # Empty dict = no atom coloring, keeps atoms black
            highlightBondColors=bond_colors,
        )
    else:
        drawer.DrawMolecule(mol_optimized)

    drawer.FinishDrawing()

    # Convert to image and display
    img = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img))

    axes.imshow(img)
    title = f"{title_prefix}Bit {bit}"
    if add_description and bit in bit_descriptions:
        title += f"{bit_descriptions[bit]}"
    axes.set_title(title, fontsize=8, wrap=True, pad=6)

    # Add congener name well below the structure
    if congener_name:
        axes.text(
            0.5,
            -0.05,
            congener_name,
            transform=axes.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            weight="bold",
            color="darkblue",
            bbox=dict(facecolor="lightgray", alpha=0.7, pad=2, edgecolor="none"),
        )

    axes.axis("off")

    return len(highlight_atoms) > 0


def draw_molecule_simple_improved(mol, axes, title, congener_name=""):
    """Draw molecule with improved layout but no highlighting, including congener name"""

    # Optimize the molecule layout
    mol_optimized = optimize_molecule_layout(mol)

    # Draw molecule with improved settings for two-column format
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
    drawer.drawOptions().bondLineWidth = 2.0
    drawer.drawOptions().atomLabelFontSize = 18
    drawer.drawOptions().padding = 0.1
    drawer.drawOptions().clearBackground = True
    drawer.drawOptions().backgroundColour = (1, 1, 1, 1)  # White background

    drawer.DrawMolecule(mol_optimized)
    drawer.FinishDrawing()

    img = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img))

    axes.imshow(img)
    axes.set_title(title, fontsize=8, pad=6)

    # Add congener name well below the structure
    if congener_name:
        axes.text(
            0.5,
            -0.05,
            congener_name,
            transform=axes.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            weight="bold",
            color="darkred",
            bbox=dict(facecolor="lightgray", alpha=0.7, pad=2, edgecolor="none"),
        )

    axes.axis("off")


# Process the positive examples (molecules that trigger the bits)
for idx, (bit, (smiles, congener_name)) in enumerate(bit_smiles_dict.items()):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not parse SMILES for bit {bit}: {smiles}")
        continue

    # Calculate Morgan fingerprint with bit info (radius=2 for ECFP_4)
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, bitInfo=bit_info)

    # Draw and verify
    has_highlights = draw_molecule_with_highlights_improved(
        mol,
        bit,
        bit_info,
        axes_positive[idx],
        congener_name=congener_name,
        title_prefix="Triggers ",
    )

# Process the negative examples (molecules that don't trigger the bits)
for idx, (bit, (smiles, congener_name)) in enumerate(non_trigger_smiles_dict.items()):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not parse SMILES for negative bit {bit}: {smiles}")
        continue

    # Calculate Morgan fingerprint
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, bitInfo=bit_info)
    bit_present = bit in bit_info

    # Draw molecule with improved layout
    title = f"Does NOT Trigger Bit {bit}"
    draw_molecule_simple_improved(mol, axes_negative[idx], title, congener_name)

# Add labels and legend with JCIM-compliant font sizes
fig.text(
    0.02,
    0.75,
    "TRIGGERING\nSTRUCTURES",
    fontsize=8,
    ha="center",
    va="center",
    rotation=90,
    fontweight="bold",
    color="darkgreen",
)
fig.text(
    0.02,
    0.25,
    "NON-TRIGGERING\nSTRUCTURES",
    fontsize=8,
    ha="center",
    va="center",
    rotation=90,
    fontweight="bold",
    color="darkred",
)

# Removed color coding legend since we now only color bonds for clarity
# legend_text = (
#     "Color coding for triggered bits:\n"
#     "Red = Central atom (radius 0)\n"
#     "Orange = First neighbors (radius 1)\n"
#     "Blue = Second neighbors (radius 2)"
# )
# fig.text(
#     0.5,
#     0.02,
#     legend_text,
#     ha="center",
#     fontsize=9,
#     bbox=dict(facecolor="lightblue", alpha=0.8, pad=6, edgecolor="navy"),
# )

plt.tight_layout(rect=[0.03, 0.03, 1, 1])  # Adjusted for no title, more space
plt.savefig(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/LogKa_fingerprints.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()

# Print detailed environment info for each bit
for bit, (smiles, congener_name) in bit_smiles_dict.items():
    mol = Chem.MolFromSmiles(smiles)
    bit_info = {}
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, bitInfo=bit_info)

    print(
        f"\nBit {bit} environment in {Chem.MolToSmiles(mol, isomericSmiles=True)} ({congener_name}):"
    )
    if bit in bit_info:
        for atom_idx, radius_used in bit_info[bit]:
            atom = mol.GetAtomWithIdx(atom_idx)
            print(f"Central atom: {atom.GetSymbol()} (index {atom_idx})")
            print(f"Environment radius: {radius_used}")
            env_atoms, atoms_by_distance = get_environment_atoms_corrected(
                mol, atom_idx, radius_used
            )
            env_atoms.remove(
                atom_idx
            )  # Remove the central atom since we already listed it
            print(f"All atoms in environment: {sorted(env_atoms)}")

            # Print atoms at each distance
            for r in range(1, radius_used + 1):
                if r in atoms_by_distance:
                    atoms_at_r = atoms_by_distance[r]
                    print(f"Atoms at distance {r}: {sorted(atoms_at_r)}")

            neighbors = atom.GetNeighbors()
            print("Direct neighbors:")
            for neighbor in neighbors:
                print(f"- {neighbor.GetSymbol()} (index {neighbor.GetIdx()})")
            print("---")

# Print summary statistics
print(f"\nSummary:")
print(f"Processed {len(bit_smiles_dict)} fingerprint bits")
for bit_name, bit_num in zip(["Bit_592", "Bit_886", "Bit_1977"], [592, 886, 1977]):
    triggering_count = len(train_dataset.X[train_dataset.X[bit_name] == 1])
    total_count = len(train_dataset.X)
    print(
        f"{bit_name}: {triggering_count}/{total_count} molecules trigger this bit ({triggering_count/total_count*100:.1f}%)"
    )

# Save the figure with high resolution
plt.figure(fig.number)
plt.savefig(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/fingerprint_bit_analysis.png",
    dpi=300,
    bbox_inches="tight",
)
print("Figure saved as fingerprint_bit_analysis.png")
