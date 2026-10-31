# LFABP SALI grouping

44 congeners, 8 groups, 144 within-group pairs.

**G1 — PFCA** (11)
PFBA, PFPeA, PFHxA, PFHpA, PFOA, PFNA, PFDA, PFUnA, PFDoA, PFTrDA, PFTeA

**G2 — PFSA** (10)
PFBS, PFPeS, PFHxS, PFHpS, PFOS, PFNS, PFDS, PFDoS, PFECHS, 8Cl-PFOS

**G3 — PFESA** (3)
PFEESA, 6:2 Cl-PFESA, 8:2 Cl-PFESA

**G4 — PFECA** (8)
PF4OPeA, PF5OHxA, 3,6-OPFHpA, ADONA, GenX, HFPO-TA, HFPO-TeA, PF8O3A2

**G5 — PFPiA** (2)
6:6 PFPiA, 8:8 PFPiA

**G6 — PAPs** (3)
6:2 PAP, 6:2 diPAP, 8:2 diPAP

**G7 — FTS** (4)
4:2 FTS, 6:2 FTS, 8:2 FTS, 10:2 FTS

**G8 — FT-carboxylic acids** (3)
FHEA, FHUEA, FPePA

```python
GROUP_OF_CONGENER = {
    "PFBA": "G1", "PFPeA": "G1", "PFHxA": "G1", "PFHpA": "G1", "PFOA": "G1",
    "PFNA": "G1", "PFDA": "G1", "PFUnA": "G1", "PFDoA": "G1", "PFTrDA": "G1",
    "PFTeA": "G1",
    "PFBS": "G2", "PFPeS": "G2", "PFHxS": "G2", "PFHpS": "G2", "PFOS": "G2",
    "PFNS": "G2", "PFDS": "G2", "PFDoS": "G2", "PFECHS": "G2", "8Cl-PFOS": "G2",
    "PFEESA": "G3", "6:2 Cl-PFESA": "G3", "8:2 Cl-PFESA": "G3",
    "PF4OPeA": "G4", "PF5OHxA": "G4", "3,6-OPFHpA": "G4", "ADONA": "G4",
    "GenX": "G4", "HFPO-TA": "G4", "HFPO-TeA": "G4", "PF8O3A2": "G4",
    "6:6 PFPiA": "G5", "8:8 PFPiA": "G5",
    "6:2 PAP": "G6", "6:2 diPAP": "G6", "8:2 diPAP": "G6",
    "4:2 FTS": "G7", "6:2 FTS": "G7", "8:2 FTS": "G7", "10:2 FTS": "G7",
    "FHEA": "G8", "FHUEA": "G8", "FPePA": "G8",
}

GROUP_NAMES = {
    "G1": "PFCA", "G2": "PFSA", "G3": "PFESA", "G4": "PFECA",
    "G5": "PFPiA", "G6": "PAPs", "G7": "FTS", "G8": "FT-carboxylic acids",
}
```

Keys match the `PFAS` column in `LFABP_data.csv` exactly.

G5/G6 split: PFPiA has its perfluoroalkyl chains bonded directly to phosphorus (P-C,
fully fluorinated); PAPs attach fluorotelomer chains through an ester oxygen
(P-O-C, CH2CH2 present). Same head-group-plus-backbone rule that separates G7 from G2
and G8 from G1.
