"""Build Figure 7 around one quantitative claim.

Claim: reversible link removal creates a finite window in which an unrestricted
BTB-MOL seed still grows from 30 to 80 entities while successful products become
thinner and more registered with the original seed layer. Stronger removal then
reduces the probability of completing that growth interval.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REBUILD = HERE.parent / "figure_rebuild_20260901"
SUMMARY_PATH = REBUILD / "figure7_btb_final_morphology_sc1_v1.csv"
TRACE_PATH = REBUILD / "sc1_seeded_analysis_v1" / "seeded_v3_trace_metrics.csv"
OLD_FIGURE = REBUILD / "figure7_btb_final_morphology_sc1_v1.png"
OLD_PDF = REBUILD / "figure7_btb_final_morphology_sc1_v1.pdf"
STYLE_KERNEL = Path(r"C:\Users\Cheng Wang\.codex\skills\figure-style\kernel.py")

OUTPUT_PREFIX = HERE / "figure7_kinetic_window_v2"
PANEL_DIR = HERE / "panels"
SI_DIR = HERE / "supporting_information"

BLUE = "#397FA3"
ORANGE = "#CC6B32"
GREEN = "#2A8C72"
TEXT = "#20262B"
MUTED = "#6C767E"
GRID = "#DDE3E7"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STYLE = load_module(STYLE_KERNEL, "figure7_v2_style_kernel")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_PATH).sort_values(["q", "rho"]).reset_index(drop=True)
    traces = pd.read_csv(TRACE_PATH)
    traces = traces[(traces.system == "btb_mol") & np.isclose(traces.SC, 1.0)].copy()
    if len(summary) != 42 or len(traces) != 1344:
        raise RuntimeError(f"Unexpected source shape: summary={len(summary)}, traces={len(traces)}")
    if not (traces.initial_entities.eq(30).all() and traces.target_entities.eq(80).all()):
        raise RuntimeError("The selected trajectories do not all use the declared 30-to-80 interval")

    q1 = summary[np.isclose(summary.q, 1.0)].sort_values("rho").copy()
    reference_rate = float(q1.loc[np.isclose(q1.rho, 0.1), "growth_event_rate"].iloc[0])
    q1["relative_completion_rate"] = q1.growth_event_rate / reference_rate
    intervals = [wilson_interval(int(row.successes), int(row.n)) for row in q1.itertuples()]
    q1["growth_probability_low95"] = [pair[0] for pair in intervals]
    q1["growth_probability_high95"] = [pair[1] for pair in intervals]

    successful_q1 = traces[
        np.isclose(traces.q_atom, 1.0) & traces.target_reached.astype(bool)
    ].copy()

    across = []
    for rho, group in summary.groupby("rho", sort=True):
        registered = group.median_primary_fraction_success.dropna()
        across.append(
            {
                "rho": float(rho),
                "growth_probability_median": float(group.growth_probability.median()),
                "growth_probability_q25": float(group.growth_probability.quantile(0.25)),
                "growth_probability_q75": float(group.growth_probability.quantile(0.75)),
                "primary_fraction_median": float(registered.median()),
                "primary_fraction_q25": float(registered.quantile(0.25)),
                "primary_fraction_q75": float(registered.quantile(0.75)),
                "n_compositions_growth": int(len(group)),
                "n_compositions_structure": int(len(registered)),
            }
        )
    across_summary = pd.DataFrame(across)
    return summary, q1, successful_q1, across_summary


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color=GRID, lw=0.55, zorder=0)
    ax.tick_params(which="minor", length=2.2)
    ax.margins(x=0.035)


def panel_a(ax: plt.Axes, q1: pd.DataFrame) -> None:
    x = q1.rho.to_numpy(float)
    probability = q1.growth_probability.to_numpy(float)
    low = q1.growth_probability_low95.to_numpy(float)
    high = q1.growth_probability_high95.to_numpy(float)

    probability_artist = ax.errorbar(
        x,
        probability,
        yerr=np.vstack([probability - low, high - probability]),
        color=BLUE,
        marker="o",
        ms=5.0,
        lw=1.7,
        capsize=2.5,
        label=r"$P(30\rightarrow80\ \mathrm{entities})$",
        zorder=4,
    )
    rate_artist, = ax.plot(
        x,
        q1.relative_completion_rate,
        color=ORANGE,
        marker="s",
        ms=4.6,
        lw=1.7,
        label=r"Relative 30$\rightarrow$80 completion rate",
        zorder=4,
    )

    row10 = q1[np.isclose(q1.rho, 10.0)].iloc[0]
    ax.axvspan(7.3, 13.7, color="#E5F3EC", alpha=0.75, zorder=1)
    ax.annotate(
        "32/32 reach 80\nwhile growth is slower",
        xy=(10.0, float(row10.relative_completion_rate)),
        xytext=(3.6, 0.40),
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        fontsize=7.2,
        color=TEXT,
        ha="left",
        va="center",
    )
    row30 = q1[np.isclose(q1.rho, 30.0)].iloc[0]
    ax.annotate(
        "22/32 reach 80",
        xy=(30.0, float(row30.growth_probability)),
        xytext=(16.2, 0.77),
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        fontsize=7.2,
        color=TEXT,
        ha="left",
        va="center",
    )

    ax.set_xscale("log")
    ax.set_xticks(x, [f"{value:g}" for value in x])
    ax.set_xlim(0.075, 43)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(r"$\rho_{\mathrm{off}}$")
    ax.set_ylabel("Probability or relative rate")
    ax.set_title("Growth slows before 30-to-80 success decreases", loc="left", pad=6)
    ax.legend(
        handles=[probability_artist, rate_artist],
        labels=[r"$P(30\rightarrow80\ \mathrm{entities})$", r"Relative 30$\rightarrow$80 completion rate"],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.01),
        borderaxespad=0,
        frameon=False,
    )
    style_axis(ax)


def panel_b(
    ax: plt.Axes,
    q1: pd.DataFrame,
    successful_q1: pd.DataFrame,
) -> mpl.cm.ScalarMappable:
    norm = LogNorm(vmin=float(q1.rho.min()), vmax=float(q1.rho.max()))
    cmap = mpl.colormaps["viridis"]

    ax.scatter(
        successful_q1.btb_global_rms_thickness_angstrom,
        successful_q1.btb_primary_layer_metal_fraction,
        color="#87939A",
        s=15,
        alpha=0.20,
        linewidth=0,
        zorder=1,
    )

    medians = (
        successful_q1.groupby("off_weight", as_index=False)
        .agg(
            thickness=("btb_global_rms_thickness_angstrom", "median"),
            primary_fraction=("btb_primary_layer_metal_fraction", "median"),
            successful_traces=("run_id", "count"),
        )
        .sort_values("off_weight")
        .merge(q1[["rho", "growth_probability"]], left_on="off_weight", right_on="rho", how="left")
    )

    ax.plot(medians.thickness, medians.primary_fraction, color="#626E76", lw=1.1, zorder=2)
    markers = ax.scatter(
        medians.thickness,
        medians.primary_fraction,
        c=medians.off_weight,
        cmap=cmap,
        norm=norm,
        s=118,
        marker="D",
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )

    for rho_value, xytext, align in [
        (0.1, (-9, -16), "right"),
        (10.0, (7, 6), "left"),
        (30.0, (7, 5), "left"),
    ]:
        row = medians[np.isclose(medians.off_weight, rho_value)].iloc[0]
        suffix = ""
        if rho_value in {10.0, 30.0}:
            suffix = f" ({int(row.successful_traces)}/32)"
        ax.annotate(
            rf"$\rho_{{\mathrm{{off}}}}={rho_value:g}$" + suffix,
            (float(row.thickness), float(row.primary_fraction)),
            xytext=xytext,
            textcoords="offset points",
            fontsize=7.1,
            color=TEXT,
            ha=align,
            va="center",
        )

    ax.annotate(
        "increasing " + r"$\rho_{\mathrm{off}}$",
        xy=(float(medians.iloc[-1].thickness), float(medians.iloc[-1].primary_fraction)),
        xytext=(float(medians.iloc[2].thickness) - 0.2, float(medians.iloc[2].primary_fraction) + 0.13),
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.9},
        fontsize=7.2,
        color=MUTED,
        ha="center",
    )

    ax.set_xlim(0, 12.6)
    ax.set_ylim(0.44, 1.025)
    ax.set_xlabel(r"RMS thickness ($\mathrm{\AA}$)  $\leftarrow$ thinner")
    ax.set_ylabel("Primary-layer metal fraction")
    ax.set_title("Surviving products become thinner and seed-registered", loc="left", pad=6)
    ax.text(0.03, 0.96, r"more registered $\uparrow$", transform=ax.transAxes,
            ha="left", va="top", color=MUTED, fontsize=7.2)
    style_axis(ax)
    return markers


def panel_c(ax: plt.Axes, summary: pd.DataFrame, across: pd.DataFrame) -> None:
    rhos = np.array(sorted(summary.rho.unique()), dtype=float)
    x = np.arange(len(rhos), dtype=float)
    q_values = np.array(sorted(summary.q.unique()), dtype=float)
    offsets = np.linspace(-0.075, 0.075, len(q_values))

    for i, rho in enumerate(rhos):
        group = summary[np.isclose(summary.rho, rho)].sort_values("q")
        registration = group.median_primary_fraction_success.to_numpy(float)
        probability = group.growth_probability.to_numpy(float)
        ax.scatter(
            i - 0.11 + offsets,
            probability,
            s=17,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=0.75,
            alpha=0.75,
            zorder=2,
        )
        finite = np.isfinite(registration)
        ax.scatter(
            i + 0.11 + offsets[finite],
            registration[finite],
            s=18,
            marker="D",
            facecolor="white",
            edgecolor=GREEN,
            linewidth=0.75,
            alpha=0.78,
            zorder=2,
        )

    ax.plot(x - 0.11, across.growth_probability_median, color=BLUE, marker="o", ms=4.5,
            lw=1.7, zorder=4, label=r"$P(30\rightarrow80\ \mathrm{entities})$")
    ax.plot(x + 0.11, across.primary_fraction_median, color=GREEN, marker="D", ms=4.3,
            lw=1.7, zorder=4, label="Primary-layer metal fraction")

    ax.set_xticks(x, [f"{value:g}" for value in rhos])
    ax.set_xlim(-0.42, len(rhos) - 0.58)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel(r"$\rho_{\mathrm{off}}$")
    ax.set_ylabel("Fraction")
    ax.set_title("The growth-registration trade-off persists across composition", loc="left", pad=6)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.01), borderaxespad=0, ncol=2, frameon=False)
    style_axis(ax)


def add_panel_letters(fig: plt.Figure, panels: list[tuple[str, plt.Axes]]) -> None:
    for letter, ax in panels:
        box = ax.get_position()
        fig.text(
            box.x0 - 0.032,
            box.y1 + 0.012,
            letter,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=TEXT,
        )


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=450, facecolor="white", bbox_inches="tight", pad_inches=0.045)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight", pad_inches=0.045)


def export_panels(
    summary: pd.DataFrame,
    q1: pd.DataFrame,
    successful_q1: pd.DataFrame,
    across: pd.DataFrame,
) -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    specifications = [("A", (4.2, 3.15)), ("B", (4.2, 3.15)), ("C", (7.2, 2.55))]
    for letter, size in specifications:
        fig, ax = plt.subplots(figsize=size)
        if letter == "A":
            panel_a(ax, q1)
        elif letter == "B":
            artist = panel_b(ax, q1, successful_q1)
            cbar = fig.colorbar(artist, ax=ax, pad=0.025, fraction=0.055)
            cbar.set_label(r"$\rho_{\mathrm{off}}$")
            cbar.set_ticks(q1.rho, labels=[f"{value:g}" for value in q1.rho])
        else:
            panel_c(ax, summary, across)
        STYLE.panel_letter(ax, letter, case="upper", dx=-0.08, dy=1.04, fontsize=11)
        save(fig, PANEL_DIR / f"figure7{letter}")
        plt.close(fig)


def main() -> None:
    STYLE.apply_figure_style(frame="open", font="Arial", sizes=(8.8, 7.8, 7.0), grid=False)
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    summary, q1, successful_q1, across = load_data()

    fig = plt.figure(figsize=(7.48, 5.5), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.02, 1.0],
        height_ratios=[1.12, 0.84],
        left=0.09,
        right=0.965,
        top=0.95,
        bottom=0.10,
        wspace=0.34,
        hspace=0.42,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    panel_a(ax_a, q1)
    artist_b = panel_b(ax_b, q1, successful_q1)
    cax_b = fig.add_axes([
        ax_b.get_position().x1 + 0.010,
        ax_b.get_position().y0 + 0.035,
        0.013,
        ax_b.get_position().height - 0.070,
    ])
    cbar_b = fig.colorbar(artist_b, cax=cax_b)
    cbar_b.set_label(r"$\rho_{\mathrm{off}}$", labelpad=3)
    cbar_b.set_ticks(q1.rho, labels=[f"{value:g}" for value in q1.rho])
    panel_c(ax_c, summary, across)

    add_panel_letters(fig, [("A", ax_a), ("B", ax_b), ("C", ax_c)])
    save(fig, OUTPUT_PREFIX)
    plt.close(fig)

    export_panels(summary, q1, successful_q1, across)
    summary.to_csv(HERE / "figure7_condition_source_v2.csv", index=False)
    q1.to_csv(HERE / "figure7_q1_kinetic_source_v2.csv", index=False)
    successful_q1.to_csv(HERE / "figure7_q1_successful_trajectory_source_v2.csv", index=False)
    across.to_csv(HERE / "figure7_across_composition_source_v2.csv", index=False)

    SI_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OLD_FIGURE, SI_DIR / "figureS_btb_complete_metric_grid.png")
    shutil.copy2(OLD_PDF, SI_DIR / "figureS_btb_complete_metric_grid.pdf")
    shutil.copy2(SUMMARY_PATH, SI_DIR / "figureS_btb_complete_metric_grid_source.csv")

    manifest = {
        "canonical_main_figure": str(OUTPUT_PREFIX.with_suffix(".png")),
        "trace_source": str(TRACE_PATH),
        "condition_summary_source": str(SUMMARY_PATH),
        "trace_source_sha256": sha256(TRACE_PATH),
        "condition_summary_sha256": sha256(SUMMARY_PATH),
        "conditions": int(len(summary)),
        "trajectories": int(summary.n.sum()),
        "initial_entities": 30,
        "target_entities": 80,
        "s_close": 1.0,
        "main_figure_claim": (
            "Increasing rho_off slows 30-to-80 growth before lowering its probability; "
            "successful products become thinner and more registered with the seed layer."
        ),
        "morphology_conditioning": "successful 80-entity products only",
        "no_growth_treatment": "included in growth probability; excluded from morphology statistics",
        "superseded_main_figure_retained_as_si": str(SI_DIR / "figureS_btb_complete_metric_grid.png"),
    }
    (HERE / "figure7_kinetic_window_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(OUTPUT_PREFIX.with_suffix(".png"))


if __name__ == "__main__":
    main()
