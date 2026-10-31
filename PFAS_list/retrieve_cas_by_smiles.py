import pandas as pd
from rdkit import Chem

pfas_list = pd.read_csv("PFAS_list/PFAS_list.csv")

logka_train = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Train_Albumin_Binding_Data.csv"
)
logka_test = pd.read_csv(
    "PFAS_albumin_QSAR/Equilibrium_dialysis/Test_Albumin_Binding_Data.csv"
)
logka = pd.concat([logka_train, logka_test], ignore_index=True)

halflife = pd.read_csv("PFAS_HalfLife_QSAR/Restructured/data/merged_df.csv")


def canonical_smiles(smiles):
    """Canonicalize a SMILES so equivalent strings from different sources
    (different atom ordering, etc.) match. Isotope labels are kept, so an
    isotope-tagged tracer (e.g. 13C-labeled 6:2FTS) stays distinct from its
    unlabeled parent (6:2 FTSA) - isotopologues have their own CAS numbers
    and are tracked as separate substances."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


pfas_list["canonical_SMILES"] = pfas_list["SMILES"].apply(canonical_smiles)

# Both source datasets are scanned identically for PFAS missing from
# pfas_list, so a compound found only in the LogKa data isn't silently
# dropped just because the half-life dataset doesn't have it too.
sources = [(halflife, "PFAS"), (logka, "Congener")]

new_entries = {}
for source_df, name_col in sources:
    for smiles, name in zip(source_df["SMILES"], source_df[name_col]):
        canon = canonical_smiles(smiles)
        if canon not in pfas_list["canonical_SMILES"].values and canon not in new_entries:
            print(f"PFAS with SMILES {smiles} not found in PFAS list.")
            new_entries[canon] = {
                "Substance": name,
                "SMILES": smiles,
                "canonical_SMILES": canon,
            }

pfas_list = pd.concat(
    [pfas_list, pd.DataFrame(new_entries.values())], ignore_index=True
)

logka_canonical = set(logka["SMILES"].apply(canonical_smiles))
halflife_canonical = set(halflife["SMILES"].apply(canonical_smiles))

pfas_list["in_LogKa_dataset"] = pfas_list["canonical_SMILES"].isin(logka_canonical)
pfas_list["in_Halflife_dataset"] = pfas_list["canonical_SMILES"].isin(halflife_canonical)

# drop from list rows of PFAS that are not present in either dataset
pfas_list = (
    pfas_list[pfas_list["in_LogKa_dataset"] | pfas_list["in_Halflife_dataset"]]
    .drop(columns=["canonical_SMILES"])
    .reset_index(drop=True)
)

pfas_list.to_csv("PFAS_list/PFAS_list_updated.csv", index=False)
