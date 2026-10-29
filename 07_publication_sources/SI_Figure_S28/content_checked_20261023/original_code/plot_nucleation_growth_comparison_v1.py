"""Publication figures for de novo nucleation and seeded propagation.

The de novo calculation reports the first production of a persistent
30-entity assembly.  The seeded calculation starts from a fixed crystal and
reports whether it adds 50 entities and, conditional on reaching that target,
the net propagation rate.  These are deliberately kept as distinct
observables rather than collapsed into an opaque selector.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NUC_PATH = ROOT / "common_response_surface_v2_final_condition_summary.csv"
SEED_PATH = (
    ROOT.parent
    / "simulation_review_gate_20260831"
    / "production_results_20260901"
    / "seeded_analysis_v3"
    / "seeded_v3_condition_summary.csv"
)
OUT = ROOT / "nucleation_growth_figures_v1"
OUT.mkdir(exist_ok=True)

Q_VALUES = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 3.0, 10.0])
OFF_SEEDED = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
SC_VALUES = [0.0, 0.1, 0.5, 2.0]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def centers_to_edges(values: np.ndarray, log: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    transformed = np.log10(values) if log else values
    edges = np.empty(len(values) + 1)
    edges[1:-1] = (transformed[:-1] + transformed[1:]) / 2
    edges[0] = transformed[0] - (transformed[1] - transformed[0]) / 2
    edges[-1] = transformed[-1] + (transformed[-1] - transformed[-2]) / 2
    return 10**edges if log else edges


def matrix(df: pd.DataFrame, value: str, q_values=Q_VALUES, off_values=OFF_SEEDED) -> np.ndarray:
    pivot = df.pivot(index="q", columns="off_weight", values=value)
    return pivot.reindex(index=q_values, columns=off_values).to_numpy(dtype=float)


def seed_matrix(df: pd.DataFrame, value: str) -> np.ndarray:
    pivot = df.pivot(index="q_atom", columns="off_weight", values=value)
    return pivot.reindex(index=Q_VALUES, columns=OFF_SEEDED).to_numpy(dtype=float)


def pcolor(ax, z, *, cmap, norm, title, colorbar_label=None):
    x_edges = centers_to_edges(OFF_SEEDED, log=True)
    # q is an experimentally chosen discrete ratio grid. Equal row heights keep
    # every sampled condition legible without implying logarithmic sampling.
    y_edges = np.arange(len(Q_VALUES) + 1, dtype=float) - 0.5
    cmap_obj = mpl.colormaps[cmap].copy() if isinstance(cmap, str) else cmap.copy()
    cmap_obj.set_bad("#e6e8ea")
    artist = ax.pcolormesh(x_edges, y_edges, z, cmap=cmap_obj, norm=norm, shading="flat")
    ax.set_xscale("log")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xticks(OFF_SEEDED)
    ax.set_xticklabels([f"{v:g}" for v in OFF_SEEDED])
    ax.set_yticks(np.arange(len(Q_VALUES)))
    ax.set_yticklabels([f"{v:g}" for v in Q_VALUES])
    ax.set_title(title, pad=6)
    if colorbar_label:
        cb = ax.figure.colorbar(artist, ax=ax, pad=0.025, fraction=0.048)
        cb.set_label(colorbar_label)
    return artist


def interpolated_nucleation(nuc: pd.DataFrame, system: str, sc: float) -> np.ndarray:
    """Bilinearly interpolate log Jeffreys-stabilized event rate to seed grid."""
    subset = nuc[(nuc.system == system) & np.isclose(nuc.SC, sc)].copy()
    subset["stable_rate"] = (
        subset["nucleation_event_count"].astype(float) + 0.5
    ) / subset["candidate_model_time_exposure"].astype(float)
    q_src = np.array(sorted(subset.q.unique()), dtype=float)
    off_src = np.array(sorted(subset.off_weight.unique()), dtype=float)
    source = matrix(subset, "stable_rate", q_src, off_src)
    log_source = np.log10(source)
    # First interpolate along log(off), then along the ordinary q coordinate.
    at_off = np.vstack(
        [np.interp(np.log10(OFF_SEEDED), np.log10(off_src), row) for row in log_source]
    )
    out = np.vstack([np.interp(Q_VALUES, q_src, at_off[:, j]) for j in range(len(OFF_SEEDED))]).T
    return out


def main_figure(nuc: pd.DataFrame, seed: pd.DataFrame) -> None:
    system = "uio66"
    sc = 0.5
    s = seed[(seed.system == system) & np.isclose(seed.SC, sc)]
    log_nuc = interpolated_nucleation(nuc, system, sc)
    reach = seed_matrix(s, "target_reached_fraction")
    growth = seed_matrix(s, "median_conditional_net_growth_rate")

    fig = plt.figure(figsize=(12.2, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.83])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    pcolor(
        axes[0], log_nuc, cmap="magma", norm=mpl.colors.Normalize(vmin=-8, vmax=-0.2),
        title="De novo nucleation events", colorbar_label=r"log$_{10}$ event rate (model time$^{-1}$)",
    )
    pcolor(
        axes[1], reach, cmap="viridis", norm=mpl.colors.Normalize(vmin=0, vmax=1),
        title="Seeded crystal reaches +50 entities", colorbar_label="probability",
    )
    pcolor(
        axes[2], growth, cmap="cividis", norm=mpl.colors.Normalize(vmin=0, vmax=70),
        title="Propagation rate (gray: no successful trace)", colorbar_label="entities per model time",
    )
    for ax in axes:
        ax.set_xlabel(r"link-removal/addition weight, $\rho_{\rm off}$")
    axes[0].set_ylabel("atomic Zr/BDC input ratio")

    ax = fig.add_subplot(gs[1, :])
    n_line = nuc[(nuc.system == system) & np.isclose(nuc.SC, sc) & np.isclose(nuc.q, 1.0)].copy()
    n_line = n_line.sort_values("off_weight")
    n_rate = (n_line.nucleation_event_count + 0.5) / n_line.candidate_model_time_exposure
    n_norm = n_rate / n_rate.iloc[0]
    s_line = s[np.isclose(s.q_atom, 1.0)].sort_values("off_weight")
    g_norm = s_line.median_conditional_net_growth_rate / s_line.median_conditional_net_growth_rate.iloc[0]
    ax.plot(n_line.off_weight, n_norm, "o-", lw=1.8, ms=4.5, color="#b33b32", label="de novo nucleation-event rate")
    stabilized_reach = (32 * s_line.target_reached_fraction + 0.5) / 33
    ax.plot(s_line.off_weight, stabilized_reach, "s-", lw=1.8, ms=4.5, color="#2a6f9e", label="seed reaches +50 entities")
    ax.plot(s_line.off_weight, g_norm, "^-", lw=1.8, ms=4.5, color="#317a4e", label="conditional propagation rate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.009, 40)
    ax.set_ylim(1e-8, 1.5)
    ax.set_xlabel(r"link-removal/addition weight, $\rho_{\rm off}$")
    ax.set_ylabel("normalized response")
    ax.grid(True, which="both", color="#d8dde2", lw=0.55)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.015))
    ax.text(0.012, 1.7e-7, "Zr/BDC = 1; closure coefficient = 0.5", color="#4a535b")

    labels = "abcd"
    for label, axis in zip(labels, [*axes, ax]):
        axis.text(-0.13 if axis is not ax else -0.045, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=12)
    fig.suptitle("Nucleation and propagation respond differently to reversible link removal", fontsize=13, y=1.015)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"uio66_nucleation_growth_main_v1.{ext}", dpi=400 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def full_sc_figure(nuc: pd.DataFrame, seed: pd.DataFrame, system: str) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(14.8, 10.3), constrained_layout=True, sharex=True, sharey=True)
    for col, sc in enumerate(SC_VALUES):
        s = seed[(seed.system == system) & np.isclose(seed.SC, sc)]
        log_nuc = interpolated_nucleation(nuc, system, sc)
        reach = seed_matrix(s, "target_reached_fraction")
        growth = seed_matrix(s, "median_conditional_net_growth_rate")
        pcolor(axes[0, col], log_nuc, cmap="magma", norm=mpl.colors.Normalize(-8, -0.2), title=f"closure coefficient = {sc:g}")
        pcolor(axes[1, col], reach, cmap="viridis", norm=mpl.colors.Normalize(0, 1), title="")
        pcolor(axes[2, col], growth, cmap="cividis", norm=mpl.colors.Normalize(0, 70), title="")
    row_labels = [
        r"log$_{10}$ nucleation-event rate",
        "probability seed reaches +50",
        "conditional propagation rate",
    ]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(f"{label}\natomic Zr/BDC ratio")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\rho_{\rm off}$")
    cbs = [
        (mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-8, -0.2), cmap="magma"), r"log$_{10}$ rate"),
        (mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="viridis"), "probability"),
        (mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 70), cmap="cividis"), "entities / model time"),
    ]
    for row, (sm, label) in enumerate(cbs):
        cb = fig.colorbar(sm, ax=axes[row, :], fraction=0.015, pad=0.012)
        cb.set_label(label)
    title = "UiO-66" if system == "uio66" else "BTB-MOL"
    fig.suptitle(f"{title}: complete closure-coefficient response", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{system}_nucleation_growth_full_sc_v1.{ext}", dpi=360 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def btb_layer_figure(seed: pd.DataFrame) -> None:
    data = seed[(seed.system == "btb_mol") & np.isclose(seed.SC, 0.5)]
    event = seed_matrix(data, "btb_daughter_layer_event_fraction")
    retained = seed_matrix(data, "btb_median_primary_layer_metal_fraction")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.15), constrained_layout=True, sharex=True, sharey=True)
    pcolor(axes[0], event, cmap="magma_r", norm=mpl.colors.Normalize(0, 1), title="Daughter-layer event probability", colorbar_label="probability")
    pcolor(axes[1], retained, cmap="viridis", norm=mpl.colors.Normalize(0, 1), title="Metal retained in primary layer", colorbar_label="fraction")
    for ax in axes:
        ax.set_xlabel(r"link-removal/addition weight, $\rho_{\rm off}$")
    axes[0].set_ylabel("atomic Zr/BTB input ratio")
    for label, axis in zip("ab", axes):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=12)
    fig.suptitle("Unconstrained BTB-MOL propagation reports out-of-plane growth", fontsize=12.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"btb_mol_layer_response_v1.{ext}", dpi=400 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    nuc = pd.read_csv(NUC_PATH)
    seed = pd.read_csv(SEED_PATH)
    main_figure(nuc, seed)
    full_sc_figure(nuc, seed, "uio66")
    full_sc_figure(nuc, seed, "btb_mol")
    btb_layer_figure(seed)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
