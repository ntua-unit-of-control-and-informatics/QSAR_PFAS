"""Within-group SALI and group activity profiles for the LFABP (Jia) dataset.

For each structural group, SALI is computed for every pair of congeners inside it, and
Log10Ka is plotted against perfluorinated carbon count. No thresholds, no flagging.
All 44 congeners are used; EXCLUDED only marks points, it never filters.
Adapted from PFAS_albumin_QSAR/PFAS_HSA_QSAR/HSA_sali_landscape.py: groups are assigned
by name rather than read from a column, and Log10Ka is derived from Kd.
"""

import os
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

DATA_PATH = "PFAS_LFABP_QSAR/LFABP_data.csv"
OUT_DIR = "PFAS_LFABP_QSAR/results"
PREFIX = os.path.join(OUT_DIR, "LFABP")

ECFP_RADIUS = 2

# The LFABP CSV carries no group column, so membership is assigned by name
GROUP_MEMBERS = {
    "G1": [
        "PFBA",
        "PFPeA",
        "PFHxA",
        "PFHpA",
        "PFOA",
        "PFNA",
        "PFDA",
        "PFUnA",
        "PFDoA",
        "PFTrDA",
        "PFTeA",
    ],
    "G2": [
        "PFBS",
        "PFPeS",
        "PFHxS",
        "PFHpS",
        "PFOS",
        "PFNS",
        "PFDS",
        "PFDoS",
        "PFECHS",
        "8Cl-PFOS",
    ],
    "G3": ["PFEESA", "6:2 Cl-PFESA", "8:2 Cl-PFESA"],
    "G4": [
        "PF4OPeA",
        "PF5OHxA",
        "3,6-OPFHpA",
        "ADONA",
        "GenX",
        "HFPO-TA",
        "HFPO-TeA",
        "PF8O3A2",
    ],
    "G5": ["6:6 PFPiA", "8:8 PFPiA"],
    "G6": ["6:2 PAP", "6:2 diPAP", "8:2 diPAP"],
    "G7": ["4:2 FTS", "6:2 FTS", "8:2 FTS", "10:2 FTS"],
    "G8": ["FHEA", "FHUEA", "FPePA"],
}
GROUP_NAMES = {
    "G1": "PFCA",
    "G2": "PFSA",
    "G3": "PFESA",
    "G4": "PFECA",
    "G5": "PFPiA",
    "G6": "PAPs",
    "G7": "FTS",
    "G8": "FT-carboxylic acids",
}
GROUP_OF_CONGENER = {n: g for g, names in GROUP_MEMBERS.items() for n in names}

# Marking only: highlighted in the plots and flagged in the CSVs, never filtered
EXCLUDED = [
    "FPePA",
    "6:2 diPAP",
]

COLORS = ["#2a78d6", "#eb6834"]
MUTED = "#9a9a95"


def perfluorinated_carbons(mol):
    return sum(
        1
        for a in mol.GetAtoms()
        if a.GetSymbol() == "C" and any(n.GetSymbol() == "F" for n in a.GetNeighbors())
    )


def load_data():
    df = pd.read_csv(DATA_PATH).rename(columns={"PFAS": "Congener"})
    df["Ka"] = 1 / (df["Kd_FABP_uM"] * 1e-06)  # Kd is reported in uM
    df["Log10Ka"] = np.log10(df["Ka"])
    df["GroupCode"] = df["Congener"].map(GROUP_OF_CONGENER)
    assert (
        df["GroupCode"].notna().all()
    ), f"ungrouped: {df.loc[df.GroupCode.isna(), 'Congener'].tolist()}"
    assert set(GROUP_OF_CONGENER) == set(
        df["Congener"]
    ), "the hard-coded groups and the CSV do not cover the same congeners"
    assert len(df) == df["Congener"].nunique() == 44
    assert set(EXCLUDED) <= set(df["Congener"]), "EXCLUDED names not found in the data"
    df["Group_name"] = df["GroupCode"].map(GROUP_NAMES)
    df["mol"] = df["SMILES"].map(Chem.MolFromSmiles)
    assert df["mol"].notna().all()
    df["nF_carbons"] = df["mol"].map(perfluorinated_carbons)
    df["excluded"] = df["Congener"].isin(EXCLUDED)
    return df.reset_index(drop=True)


def count_fingerprints(df):
    """Unhashed Morgan COUNT fingerprints.

    Binary fingerprints return Tanimoto 1.000 for adjacent homologues - adding a CF2
    creates no new atom environment, only more of an existing one - which leaves SALI
    undefined. Counts record that multiplicity; unhashed keeps collisions from equating
    different environments once fine count differences matter.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS)
    fps = [gen.GetSparseCountFingerprint(m) for m in df["mol"]]
    assert not any(
        isinstance(f, DataStructs.ExplicitBitVect) for f in fps
    ), "expected count vectors, got bit vectors"
    return fps


def tanimoto_matrix(fps):
    """On count vectors Tanimoto is sum(min(x, y)) / sum(max(x, y))."""
    T = np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])
    assert np.allclose(T, T.T) and T.min() >= 0 and T.max() <= 1
    assert np.allclose(np.diag(T), 1.0)
    off_diagonal = T[~np.eye(len(T), dtype=bool)]
    assert (
        off_diagonal.max() < 1.0
    ), "a pair has T = 1.000, leaving SALI undefined; the count fingerprint is not in use"
    return T


def within_group_pairs(df, T):
    rows = []
    for _, members in df.groupby("GroupCode", sort=True):
        for i, j in combinations(members.index, 2):
            delta = abs(df.Log10Ka[i] - df.Log10Ka[j])
            rows.append(
                {
                    "Group": df.GroupCode[i],
                    "Group_name": df.Group_name[i],
                    "Congener_1": df.Congener[i],
                    "Congener_2": df.Congener[j],
                    "nF_carbons_1": df.nF_carbons[i],
                    "nF_carbons_2": df.nF_carbons[j],
                    "Log10Ka_1": df.Log10Ka[i],
                    "Log10Ka_2": df.Log10Ka[j],
                    "T": T[i, j],
                    "abs_dLog10Ka": delta,
                    "SALI": delta / (1 - T[i, j]),
                    "either_excluded": bool(df.excluded[i] or df.excluded[j]),
                }
            )
    pairs = pd.DataFrame(rows)
    expected = int(sum(n * (n - 1) // 2 for n in df.GroupCode.value_counts()))
    assert len(pairs) == expected, f"{len(pairs)} pairs, expected {expected}"
    return pairs.sort_values(["Group", "SALI"], ascending=[True, False]).reset_index(
        drop=True
    )


def full_sali_matrix(df, T):
    """N x N SALI over every pair, not only within-group ones. Diagonal is undefined."""
    y = df["Log10Ka"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):  # the diagonal is 0/0
        S = np.abs(y[:, None] - y[None, :]) / (1 - T)
    np.fill_diagonal(S, np.nan)
    off = ~np.eye(len(S), dtype=bool)
    assert np.allclose(S[off], S.T[off]), "SALI matrix is not symmetric off-diagonal"
    assert np.isfinite(S[off]).all(), "SALI matrix has non-finite values off-diagonal"
    return S


def knn_columns(df, T, S, k=3):
    """Nearest neighbours over the WHOLE dataset, by Tanimoto.

    Group labels are head-group curation rather than a similarity measure, so a compound
    may have its closest structural neighbours in another group.
    Ties on T are broken by higher SALI, then by name, so the choice is deterministic.
    """
    names = df["Congener"].to_numpy()
    rows = []
    for i in range(len(df)):
        others = [j for j in range(len(df)) if j != i]
        nn = sorted(others, key=lambda j: (-T[i, j], -S[i, j], names[j]))[:k]
        rows.append(
            {
                "Congener": names[i],
                f"knn{k}_neighbours": ";".join(names[j] for j in nn),
                f"mean_T_knn{k}": float(np.mean([T[i, j] for j in nn])),
                f"mean_SALI_knn{k}": float(np.mean([S[i, j] for j in nn])),
                f"max_SALI_knn{k}": float(np.max([S[i, j] for j in nn])),
            }
        )
    return pd.DataFrame(rows)


def per_compound(df, pairs, T, S):
    """Existing columns stay WITHIN-GROUP; the knn3 columns are over the whole dataset."""
    rows = []
    for r in df.itertuples():
        mine = pairs[
            (pairs.Congener_1 == r.Congener) | (pairs.Congener_2 == r.Congener)
        ]
        if len(mine):  # a single-member group yields no pairs
            top = mine.loc[mine.SALI.idxmax()]
            partner = top.Congener_2 if top.Congener_1 == r.Congener else top.Congener_1
            mean_sali, max_sali = float(mine.SALI.mean()), float(mine.SALI.max())
        else:
            partner, mean_sali, max_sali = None, np.nan, np.nan
        rows.append(
            {
                "Congener": r.Congener,
                "Group": r.GroupCode,
                "Group_name": r.Group_name,
                "Log10Ka": r.Log10Ka,
                "nF_carbons": r.nF_carbons,
                "mean_SALI_in_group": mean_sali,
                "max_SALI_in_group": max_sali,
                "max_SALI_partner": partner,
                "excluded": r.excluded,
            }
        )
    out = pd.DataFrame(rows)
    out["rank_by_max_SALI_in_group"] = (
        out.groupby("Group")["max_SALI_in_group"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    out = out.merge(knn_columns(df, T, S), on="Congener", how="left")
    out["rank_by_mean_SALI_knn3"] = (  # over the whole dataset, not per group
        out["mean_SALI_knn3"].rank(ascending=False, method="min").astype(int)
    )
    return out.sort_values(["Group", "rank_by_max_SALI_in_group"]).reset_index(
        drop=True
    )


def plot_group_profiles(df):
    groups = sorted(df.GroupCode.unique())
    ncol = 3
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(13, 3.4 * nrow), dpi=300, sharey=True, squeeze=False
    )
    for ax, g in zip(axes.ravel(), groups):
        members = df[df.GroupCode == g].sort_values("nF_carbons")
        ax.plot(members.nF_carbons, members.Log10Ka, "-", color=MUTED, lw=1, zorder=1)
        for excluded, color, label in [
            (False, COLORS[0], "Congener"),
            (True, COLORS[1], "In EXCLUDED"),
        ]:
            s = members[members.excluded == excluded]
            ax.scatter(
                s.nF_carbons,
                s.Log10Ka,
                s=34,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label=label if g == groups[0] else None,
            )
        # groups that vary by headgroup (G4) stack on one carbon count, so alternate
        # the label side to keep them legible
        for k, r in enumerate(members.sort_values("Log10Ka").itertuples()):
            right = k % 2 == 0
            ax.annotate(
                r.Congener,
                (r.nF_carbons, r.Log10Ka),
                xytext=(4 if right else -4, -3),
                textcoords="offset points",
                ha="left" if right else "right",
                va="center",
                fontsize=5.5,
                color=COLORS[1] if r.excluded else "#555555",
            )
        ax.margins(x=0.16, y=0.16)
        ax.set_title(f"{g} {GROUP_NAMES[g]} (n={len(members)})", fontsize=9)
        ax.set_xlabel("Perfluorinated carbons", fontsize=8)
        ax.set_ylabel("Log10Ka", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.5, alpha=0.3)
    for ax in axes.ravel()[len(groups) :]:  # 8 groups in a 3x3 grid
        ax.axis("off")
    fig.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.text(
        0.5,
        0.04,
        "The connecting line orders congeners by carbon count as a guide to the eye; it is "
        "not a fitted trend, and these groups are not all homologous series.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(f"{PREFIX}_group_profiles.png", bbox_inches="tight")
    plt.close(fig)


def plot_sali_heatmap(df, pairs, S, compounds):
    """Lower-triangle SALI heatmap, ordered by group then perfluorinated carbon count."""
    order = df.sort_values(["GroupCode", "nF_carbons", "Congener"]).index.to_numpy()
    d = df.loc[order].reset_index(drop=True)
    S_ord = S[np.ix_(order, order)]
    n = len(d)

    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    plotted = S_ord[lower]
    vmax = float(np.nanpercentile(plotted, 99))
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("white")

    fig, (ax, axm) = plt.subplots(
        1,
        2,
        figsize=(15, 13),
        dpi=300,
        sharey=True,
        gridspec_kw={
            "width_ratios": [5, 1],
            "wspace": 0.26,  # room for the long G8 block label
        },
    )
    im = ax.imshow(
        np.ma.masked_array(S_ord, mask=~lower),
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )

    # group boundaries traced around the triangle, plus a label per block
    bounds = [i for i in range(1, n) if d.GroupCode[i] != d.GroupCode[i - 1]]
    for b in bounds:
        ax.plot([-0.5, b - 0.5], [b - 0.5, b - 0.5], color="#333333", lw=0.6, zorder=3)
        ax.plot(
            [b - 0.5, b - 0.5], [b - 0.5, n - 0.5], color="#333333", lw=0.6, zorder=3
        )
    for g, block in d.groupby("GroupCode", sort=True):
        mid = (block.index.min() + block.index.max()) / 2
        ax.text(
            mid + 0.9,
            mid,
            f"{g} {GROUP_NAMES[g]} (n={len(block)})",
            ha="left",
            va="center",
            fontsize=7,
            color="#333333",
            clip_on=False,
        )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(d.Congener, rotation=90, fontsize=5)
    ax.set_yticklabels(d.Congener, fontsize=5)
    for axis in (ax.get_xticklabels(), ax.get_yticklabels()):
        for lab, is_ex in zip(axis, d.excluded):
            lab.set_color(COLORS[1] if is_ex else "#555555")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = ax.inset_axes([0.70, 0.74, 0.022, 0.24])  # the upper triangle is empty
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(f"SALI (scale capped at p99 = {vmax:.2f})", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # marginal: mean SALI to the 3 nearest neighbours in the whole dataset
    marg = compounds.set_index("Congener").loc[d.Congener]
    median_T = float(compounds["mean_T_knn3"].median())
    weak = (marg["mean_T_knn3"] < median_T).to_numpy()
    axm.barh(
        np.arange(n),
        marg["mean_SALI_knn3"].to_numpy(),
        height=0.75,
        color=[COLORS[1] if e else COLORS[0] for e in marg["excluded"]],
        alpha=1.0,
        linewidth=0,
    )
    for bar, is_weak in zip(axm.patches, weak):  # faded = distant neighbours
        if is_weak:
            bar.set_alpha(0.45)
    axm.set_xlabel("Mean SALI, 3 nearest neighbours\n(whole dataset)", fontsize=8)
    # names repeated on the outer edge so each bar can be read without tracing back
    # to the heatmap; the y axis is shared, so the order is identical
    axm.tick_params(
        labelsize=7, left=False, labelleft=False, right=True, labelright=True
    )
    for tick, is_ex in zip(axm.yaxis.get_major_ticks(), d.excluded):
        tick.label2.set_fontsize(5)
        tick.label2.set_color(COLORS[1] if is_ex else "#555555")
    axm.tick_params(axis="y", length=2)
    axm.patch.set_visible(False)  # let a long group label overflow into the gap
    axm.grid(True, axis="x", linewidth=0.5, alpha=0.3)
    for spine in ("top", "right", "left"):
        axm.spines[spine].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[0], label="Congener"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS[1], label="In EXCLUDED"),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=COLORS[0],
            alpha=0.45,
            label=f"mean T to its 3 neighbours below the dataset median ({median_T:.2f})",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=7.5,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.055),
    )
    fig.text(
        0.5,
        0.012,
        "Rows and columns are ordered by group, then by perfluorinated carbon count within "
        "group. Cross-group cells are deflated by the low Tanimoto between different head "
        "groups, so they understate the real activity differences. The marginal bars are "
        "computed over the whole dataset, not within group, so they are not a summary of the "
        "row beside them.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
        wrap=True,
    )
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(f"{PREFIX}_sali_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    return vmax


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    T = tanimoto_matrix(count_fingerprints(df))
    S = full_sali_matrix(df, T)
    pairs = within_group_pairs(df, T)
    # the matrix and the pair table must not drift apart
    idx = {name: i for i, name in enumerate(df.Congener)}
    assert np.allclose(
        [S[idx[r.Congener_1], idx[r.Congener_2]] for r in pairs.itertuples()],
        pairs.SALI,
    ), "SALI matrix disagrees with the within-group pair table"
    compounds = per_compound(df, pairs, T, S)

    pairs.to_csv(f"{PREFIX}_sali_within_group.csv", index=False)
    compounds.to_csv(f"{PREFIX}_sali_per_compound.csv", index=False)
    plot_group_profiles(df)
    vmax = plot_sali_heatmap(df, pairs, S, compounds)

    counts = df.GroupCode.value_counts().sort_index()
    print(
        f"LFABP within-group SALI | {len(df)} congeners | {len(pairs)} within-group pairs"
    )
    print(
        "Group counts: "
        + ", ".join(f"{g} {GROUP_NAMES[g]} {n}" for g, n in counts.items())
    )
    print(
        f"Marked as EXCLUDED (never filtered): {', '.join(EXCLUDED) if EXCLUDED else 'none'}"
    )

    summary = (
        pairs.groupby(["Group", "Group_name"])["SALI"]
        .agg(n_pairs="size", min_SALI="min", median_SALI="median", max_SALI="max")
        .reset_index()
    )
    print("\nSALI per group:")
    print(summary.round(3).to_string(index=False))

    print("\nFive highest-SALI pairs overall:")
    cols = [
        "Group",
        "Congener_1",
        "Congener_2",
        "T",
        "abs_dLog10Ka",
        "SALI",
        "either_excluded",
    ]
    print(pairs.nlargest(5, "SALI")[cols].round(3).to_string(index=False))
    print(
        f"\nThirty congeners with the highest mean SALI to their 3 nearest neighbours "
        f"(whole dataset, not within group):"
    )
    knn_cols = [
        "Congener",
        "Group",
        "mean_T_knn3",
        "mean_SALI_knn3",
        "max_SALI_in_group",
        "knn3_neighbours",
        "excluded",
    ]
    print(
        compounds.nlargest(30, "mean_SALI_knn3")[knn_cols]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nHeatmap colour scale capped at the global p99 SALI = {vmax:.3f}")
    print(
        f"\nOutputs: {PREFIX}_sali_within_group.csv, {PREFIX}_sali_per_compound.csv, "
        f"{PREFIX}_group_profiles.png, {PREFIX}_sali_heatmap.png"
    )


if __name__ == "__main__":
    main()
