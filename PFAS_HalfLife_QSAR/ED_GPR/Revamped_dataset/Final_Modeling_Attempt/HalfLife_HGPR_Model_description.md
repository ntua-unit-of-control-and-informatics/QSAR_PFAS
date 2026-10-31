# PFAS Human Serum Half-life: Heteroscedastic Gaussian Process Model

## Model
Predicts the human serum elimination half-life (in years) of per- and polyfluoroalkyl substances (PFAS), together with an estimate of how much reported half-lives for that compound are expected to vary.

The model is a **heteroscedastic Gaussian process regression (HGPR)** (Ozbayram et al.):
- an ARD radial-basis-function kernel describes how half-life varies with the inputs;
- an input-dependent noise term lets the expected variability of reported half-lives differ between compounds;
- hyperparameters are fitted by maximising the log marginal likelihood with weakly informative priors.

Each prediction comes with a standard deviation that combines:
- **epistemic uncertainty**: uncertainty of the model, larger for compounds unlike the training set;
- **aleatoric uncertainty**: expected spread of reported half-lives between studies and populations.

## Inputs
| Feature | Description |
|---|---|
| LogKa | log10 of the albumin association constant (M⁻¹), from the albumin-binding QSAR model |
| PEOE_VSA4 | RDKit descriptor: van der Waals surface area of atoms within one Gasteiger partial-charge range |
| Bit_486, Bit_1198, Bit_1720, Bit_1840 | presence of specific substructures (Morgan fingerprint, radius 2, 2048 bits) |

The six features were chosen by forward selection from RDKit descriptors, Morgan fingerprint bits and LogKa.

## Development

### Data curation
Starting from 9,791 reported human half-life values for 30 PFAS (linear and branched PFCAs and PFSAs, ether PFAS such as HFPO-DA, DONA, Nafion byproduct 2, PFO4DA and PFO5DoA, plus 6:2 FTS and 8-Cl-PFOS):
- studies reporting only summary statistics were expanded into individual values by drawing from a lognormal distribution matching the reported mean, SD and number of participants;
- within each compound, values more than 1.5 × IQR outside the quartiles of the log half-life were removed;
- a mean and SD were then computed per compound, and each compound was represented by **50 values drawn from a lognormal distribution** with that mean and SD. This balances the dataset, which would otherwise be dominated by PFOA (6,703 of the 9,791 reported values), and gives 1,499 modelling values in total.

### Train/test split
The split is made **between compounds, not between values**, so no test compound is seen during training:
1. all 30 compounds are described by RDKit descriptors and Morgan fingerprint bits;
2. constant features and one of each pair of features correlated above 0.95 are removed;
3. the remaining features are standardised and reduced to **6 principal components**;
4. a **Kennard–Stone selection** in that principal-component space assigns 30% of the compounds to the test set. Kennard–Stone picks compounds that are maximally spread out, so the training set covers the structural space as evenly as possible and the test set is not a random draw;
5. PFHxS was moved from the test set to the training set.

This gives **22 training compounds (1,099 values)** and **8 test compounds (400 values)**: PFHxA, PFOA, PFDA, PFUdA, PFHpS, 3mPFOS, 5mPFOS and 5mPFOA.

The split uses only the molecular features, never the half-life values.

## Performance
Per-compound mean half-life, in years.

| | Compounds | R² | Q² | RMSE | MAE |
|---|---|---|---|---|---|
| 5-fold cross-validation (grouped by compound) | 22 | 0.724 | – | – | – |
| Test set (unseen PFAS) | 8 | 0.678 | 0.784 | 0.97 | 0.81 |
| Training set (fit) | 22 | 0.962 | – | 0.45 | 0.28 |

In cross-validation, compounds are held out as whole groups, so each fold predicts compounds it has not seen. At the level of individual values, cross-validation gives R² = 0.544, RMSE = 2.62 and MAE = 1.94 years; these are higher than the per-compound errors because they also include the spread of reported half-lives within a compound.

Q² is computed with the training-set mean as reference.

**Uncertainty, test set:** the predicted SD tracks the observed spread of reported half-lives across compounds (Spearman ρ = 0.92, Pearson r = 0.86; R² = 0.42). Median expected calibration error is 0.20.

## Applicability domain
A prediction is considered inside the applicability domain when the compound is sufficiently similar to the training compounds, measured with the model's own fitted ARD kernel. The threshold is the 5th percentile of the nearest-neighbour similarities among the training compounds; a compound whose mean similarity to the training set falls below it is flagged as outside the domain.

All 8 test compounds fall inside the domain, with mean kernel similarities of 0.05–0.21 against a threshold of 0.002.

## Notes on use
- The test set contains only 8 compounds, so the test metrics carry considerable uncertainty.
- The model is trained on values resampled from each compound's distribution of reported half-lives, so predictions describe a typical population value, not a specific individual.
- Predictions are most reliable for PFAS similar to the 30 compounds used here (mainly perfluoroalkyl acids and their branched, ether and chlorinated analogues). For other structures, check the applicability domain and the predicted uncertainty.
