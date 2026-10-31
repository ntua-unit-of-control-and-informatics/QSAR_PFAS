"""Prepare the Starnes HSA data: drop non-binders, retrieve PubChem SMILES by CAS, standardize."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RAW_PATH = "PFAS_albumin_QSAR/PFAS_HSA_QSAR/Starnes_HSA/HSA_Starnes_data.csv"
OUT_PATH = "PFAS_albumin_QSAR/PFAS_HSA_QSAR/Starnes_HSA/HSA_Starnes_smiles.csv"
FOOTNOTE_MARKS = "*✧☨➰"
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}"
    "/property/SMILES,MolecularFormula,Title/JSON"
)


def split_footnotes(text):
    marks = "".join(c for c in text if c in FOOTNOTE_MARKS)
    clean = " ".join("".join(c for c in text if c not in FOOTNOTE_MARKS).split())
    return clean, marks


def pubchem_lookup(query, attempts=7):
    url = PUBCHEM_URL.format(urllib.parse.quote(query))
    for attempt in range(attempts):
        time.sleep(2**attempt)  # PubChem throttles bursts ("ServerBusy"); back off on retries
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                reply = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # PubChem has no match
                return None, 0
            continue
        if "PropertyTable" in reply:  # a busy reply can also come back with status 200
            props = reply["PropertyTable"]["Properties"]
            return props[0], len(props)
    raise RuntimeError(f"PubChem did not answer for {query!r} after {attempts} attempts")


def standardize(smiles):
    # As in the Chen data: parent compound only (no counter-ions or water) with acids
    # neutralized; permanent charges such as quaternary N are left as they are
    mol = rdMolStandardize.LargestFragmentChooser().choose(Chem.MolFromSmiles(smiles))
    return Chem.MolToSmiles(rdMolStandardize.Uncharger().uncharge(mol))


def main():
    df = pd.read_csv(RAW_PATH)
    df = df[df["Ka (mM-1)"] > 0].reset_index(drop=True)  # non-binders (Ka = 0) are dropped

    rows = []
    for _, r in df.iterrows():
        name, name_marks = split_footnotes(r["Chemical Name"])
        chem_class, class_marks = split_footnotes(r["Chemical Class"])
        cas = r["CAS ID"].strip()
        abbreviation = r["Common Abbreviation(s)"]
        hit, n_hits, lookup = *pubchem_lookup(cas), "CAS"
        if hit is None:
            hit, n_hits, lookup = *pubchem_lookup(name), "name"
        rows.append(
            {
                "Congener": abbreviation.split(",")[0].strip() if pd.notna(abbreviation) else name,
                "CAS": cas,
                "SMILES": standardize(hit["SMILES"]) if hit else None,
                "Ka": r["Ka (mM-1)"] * 1e3,  # M^-1, the unit the nested-CV script expects
                "Ka_mM": r["Ka (mM-1)"],
                "R2": r["R2"],
                "Chemical_Name": name,
                "Chemical_Class": chem_class,
                "Footnotes": name_marks,
                "Class_Footnotes": class_marks,
                "PubChem_CID": hit["CID"] if hit else None,
                "PubChem_Title": hit["Title"] if hit else None,
                "PubChem_Matches": n_hits,
                "Lookup": lookup if hit else "failed",
            }
        )
    out = pd.DataFrame(rows)
    out["Formula"] = out["SMILES"].map(lambda s: CalcMolFormula(Chem.MolFromSmiles(s)) if s else None)
    out.to_csv(OUT_PATH, index=False)

    canonical = out["SMILES"].dropna()
    print(f"{len(out)} binders written to {OUT_PATH}")
    print(f"Lookups: {out['Lookup'].value_counts().to_dict()}; multiple PubChem matches: {int((out['PubChem_Matches'] > 1).sum())}")
    print(f"Duplicated SMILES: {out.loc[canonical[canonical.duplicated(keep=False)].index, 'Congener'].tolist()}")
    print(f"Duplicated Congener names: {out.loc[out['Congener'].duplicated(keep=False), 'Congener'].tolist()}")


if __name__ == "__main__":
    main()
