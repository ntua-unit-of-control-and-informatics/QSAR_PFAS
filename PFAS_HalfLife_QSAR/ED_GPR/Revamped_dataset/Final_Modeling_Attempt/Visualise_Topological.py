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
    "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/Halflife_stats.csv"
)

# Create a JaqpotTabularDataset objects
x_cols = ["LogKa"]
categorical_cols = []
featurizers = [RDKitDescriptors(), TopologicalFingerprint()]

train_dataset = JaqpotTabularDataset(
    df=df_train,
    y_cols="half_life_mean",
    x_cols=x_cols,
    smiles_cols="SMILES",
    featurizer=featurizers,
    task="regression",
)

train_dataset.select_features(
    SelectColumns=[
        "LogKa",
        "PEOE_VSA4",
        "Bit_1198",
        "Bit_1720",
        "Bit_486",
        "Bit_1840",
    ]
)

# Find Congeners that trigger specific bits
trigger_bits = [
    "Bit_1198",
    "Bit_1720",
    "Bit_486",
    "Bit_1840",
]

triggered_congeners = []
negative_congeners = []

for bit in trigger_bits:
    # Get all molecules that trigger this bit
    triggering_molecules = train_dataset.X[train_dataset.X[bit] == 1]
    non_triggering_molecules = train_dataset.X[train_dataset.X[bit] == 0]

    if len(triggering_molecules) > 0:
        # Get the first triggering molecule (you might want to select differently)
        triggered_congener_row = triggering_molecules.index[0]
        congener_info = df_train.iloc[triggered_congener_row]
        triggered_congeners.append(
            {
                "Congener": congener_info["PFAS"],
                "SMILES": congener_info["SMILES"],
                "Triggered Bit": bit,
                "Count": len(triggering_molecules),
            }
        )

    if len(non_triggering_molecules) > 0:
        pfas_index = 15 if bit != "Bit_1840" else 5  # Adjusted for Bit_1840
        negative_congener_row = non_triggering_molecules.index[pfas_index]
        congener_info = df_train.iloc[negative_congener_row]
        negative_congeners.append(
            {
                "Congener": congener_info["PFAS"],
                "SMILES": congener_info["SMILES"],
                "Triggered Bit": bit,
                "Count": len(non_triggering_molecules),
            }
        )

triggered_congeners_df = pd.DataFrame(triggered_congeners)
print("Triggering Congeners:\n", triggered_congeners_df)
negative_congeners_df = pd.DataFrame(negative_congeners)
print("\nNegative Congeners:\n", negative_congeners_df)

# Extract bit numbers from bit names
bit_numbers = {"Bit_1198": 1198, "Bit_1720": 1720, "Bit_486": 486, "Bit_1840": 1840}

# Dictionary of bit_number: (SMILES, PFAS_name) that triggers it
bit_smiles_dict = {}
for bit_name, bit in bit_numbers.items():
    for _, row in triggered_congeners_df.iterrows():
        if row["Triggered Bit"] == bit_name:
            bit_smiles_dict[bit] = (row["SMILES"], row["Congener"])
            break

# Dictionary of negative examples (non-triggering SMILES and names)
non_trigger_smiles_dict = {}
for bit_name, bit in bit_numbers.items():
    for _, row in negative_congeners_df.iterrows():
        if row["Triggered Bit"] == bit_name:
            non_trigger_smiles_dict[bit] = (row["SMILES"], row["Congener"])
            break

# Add descriptive labels for each bit
bit_descriptions = {
    1198: "\n–SO₃H followed by a linear,\nthree-carbon fluorinated chain",
    1720: "\n–SO₃H followed by two carbons\nand subsequent chain branching",
    486: "\nBranching at the terminal of\nthe fluorinated carbon chain",
    1840: "\n–COOH with an adjacent carbon\nbonded to an oxygen atom",
}

# Create a figure with gridspec for more flexible layout
# JCIM Journal Requirements: Two-column max width = 7 inches
fig = plt.figure(figsize=(7, 5), dpi=300)  # Adjusted for two-column journal format
gs = GridSpec(
    2, 4, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.3
)  # Added spacing
axes_positive = [fig.add_subplot(gs[0, i]) for i in range(4)]
axes_negative = [fig.add_subplot(gs[1, i]) for i in range(4)]

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
    mol, bit, bit_info, axes, pfas_name="", title_prefix="", add_description=True
):
    """Draw a molecule with improved layout and highlighting, including PFAS name"""

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
    axes.set_title(title, fontsize=6, wrap=True, pad=6)

    # Add PFAS name well below the structure
    if pfas_name:
        axes.text(
            0.5,
            -0.05,
            pfas_name,
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


def draw_molecule_simple_improved(mol, axes, title, pfas_name=""):
    """Draw molecule with improved layout but no highlighting, including PFAS name"""

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
    axes.set_title(title, fontsize=6, pad=6)

    # Add PFAS name well below the structure
    if pfas_name:
        axes.text(
            0.5,
            -0.05,
            pfas_name,
            transform=axes.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            weight="bold",
            color="darkred",
            bbox=dict(facecolor="lightgray", alpha=0.7, pad=2, edgecolor="none"),
        )

    axes.axis("off")


# Process positive examples
for idx, (bit, (smiles, pfas_name)) in enumerate(bit_smiles_dict.items()):
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
        pfas_name=pfas_name,
        title_prefix="Triggers ",
    )

# Process negative examples
for idx, (bit, (smiles, pfas_name)) in enumerate(non_trigger_smiles_dict.items()):
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
    draw_molecule_simple_improved(mol, axes_negative[idx], title, pfas_name)

# Add labels with JCIM-compliant font sizes
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
    "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/Final_Modeling_Attempt/halflife_fingerprints.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()

# Print summary statistics
print(f"\nSummary:")
print(f"Processed {len(bit_smiles_dict)} fingerprint bits")
for bit_name, bit_num in bit_numbers.items():
    triggering_count = len(train_dataset.X[train_dataset.X[bit_name] == 1])
    total_count = len(train_dataset.X)
    print(
        f"{bit_name}: {triggering_count}/{total_count} molecules trigger this bit ({triggering_count/total_count*100:.1f}%)"
    )
