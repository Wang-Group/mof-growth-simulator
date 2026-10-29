"""Build a point-driven Figure 8 for formic-acid-modulated Hf-BTB-MOL growth.

The figure replaces the main-text concentration heatmaps with two direct tests
of the manuscript claim.  The complete grids remain Supporting Information.
No KMC result is refitted or altered.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "figure_rebuild_20260901"
SUMMARY_PATH = SOURCE / "figure8_absolute_outcomes_v1_condition_summary.csv"
TRACE_PATH = SOURCE / "figure8_absolute_outcomes_v1_trace_source.csv"
TEM_SOURCE = SOURCE / "_manuscript_v7_extract_20260901" / "word" / "media" / "image6.png"
OLD_GRID = SOURCE / "figure8_absolute_outcomes_v1.png"

OUT_STEM = HERE / "figure8_growth_window_v2"
PANEL_DIR = HERE / "panels"
SI_DIR = HERE / "supporting_information"

ACIDS = np.array([0.0, 0.33, 0.82, 1.64, 2.46, 3.28])
GROWTH = "#C96144"
MULTILAYER = "#5185C0"
NEUTRAL = "#68777F"
LIGHT = "#DDE4E8"
TEXT = "#20272B"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.2,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def crop_tem_panels() -> tuple[Image.Image, Image.Image]:
    """Crop the two original TEM fields while retaining their scale bars."""
    source = Image.open(TEM_SOURCE).convert("RGB")
    return (
        source.crop((1421, 946, 1930, 1518)),
        source.crop((2042, 946, 2595, 1518)),
    )


def build_retention_table(summary: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Endpoint ratios and trace-bootstrap intervals for eligible conditions."""
    rng = np.random.default_rng(20260904)
    records: list[dict[str, float | str | int]] = []
    n_boot = 10000
    definitions = [
        ("FIG8_FORMIC_ACID_BY_METAL", "metal_mM", "Hf"),
        ("FIG8_FORMIC_ACID_BY_LINKER", "linker_mM", "BTB"),
    ]
    for operation, varied, series in definitions:
        current = summary[summary["operation_id"] == operation]
        for value in sorted(current[varied].unique()):
            low = current[np.isclose(current[varied], value) & np.isclose(current["formic_acid_M"], 0.33)].iloc[0]
            high = current[np.isclose(current[varied], value) & np.isclose(current["formic_acid_M"], 3.28)].iloc[0]
            eligible = (
                low["multilayer_seed_fraction_mean"] >= 0.10
                and low["net_growth_rate_mean"] > 0
                and high["net_growth_rate_mean"] > 0
            )
            if not eligible:
                continue

            low_raw = raw[(raw["operation_id"] == operation) & np.isclose(raw[varied], value) & np.isclose(raw["formic_acid_M"], 0.33)]
            high_raw = raw[(raw["operation_id"] == operation) & np.isclose(raw[varied], value) & np.isclose(raw["formic_acid_M"], 3.28)]
            il = rng.integers(0, len(low_raw), size=(n_boot, len(low_raw)))
            ih = rng.integers(0, len(high_raw), size=(n_boot, len(high_raw)))
            low_g = low_raw["net_growth_rate"].to_numpy(float)
            high_g = high_raw["net_growth_rate"].to_numpy(float)
            low_m = low_raw["multilayer_seed_fraction"].to_numpy(float)
            high_m = high_raw["multilayer_seed_fraction"].to_numpy(float)
            growth_boot = high_g[ih].mean(axis=1) / low_g[il].mean(axis=1)
            multilayer_boot = high_m[ih].mean(axis=1) / low_m[il].mean(axis=1)
            growth = high["net_growth_rate_mean"] / low["net_growth_rate_mean"]
            multilayer = high["multilayer_seed_fraction_mean"] / low["multilayer_seed_fraction_mean"]
            records.append(
                {
                    "series": series,
                    "varied_quantity": varied,
                    "concentration_mM": float(value),
                    "n_0p33M": int(len(low_raw)),
                    "n_3p28M": int(len(high_raw)),
                    "growth_retained": float(growth),
                    "growth_retained_ci_low": float(np.percentile(growth_boot, 2.5)),
                    "growth_retained_ci_high": float(np.percentile(growth_boot, 97.5)),
                    "multilayer_retained": float(multilayer),
                    "multilayer_retained_ci_low": float(np.percentile(multilayer_boot, 2.5)),
                    "multilayer_retained_ci_high": float(np.percentile(multilayer_boot, 97.5)),
                    "baseline_multilayer_probability_0p33M": float(low["multilayer_seed_fraction_mean"]),
                    "eligible_rule": "P_multilayer(0.33 M)>=0.10 and positive mean growth at both endpoints",
                }
            )
    return pd.DataFrame.from_records(records)


def build_count_table(summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for acid, frame in summary.groupby("formic_acid_M", sort=True):
        records.extend(
            [
                {
                    "formic_acid_M": float(acid),
                    "outcome": "Positive mean net growth",
                    "condition_count": int((frame["net_growth_rate_mean"] > 0).sum()),
                    "n_conditions": int(len(frame)),
                    "criterion": "mean net growth rate > 0",
                },
                {
                    "formic_acid_M": float(acid),
                    "outcome": "Multilayer-producing",
                    "condition_count": int((frame["multilayer_seed_fraction_mean"] >= 0.10).sum()),
                    "n_conditions": int(len(frame)),
                    "criterion": "mean persistent multilayer probability >= 0.10",
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def draw_tem_panel(fig: plt.Figure, spec) -> list[plt.Axes]:
    low, high = crop_tem_panels()
    grid = spec.subgridspec(3, 1, height_ratios=[0.13, 1.0, 1.0], hspace=0.16)
    title_ax = fig.add_subplot(grid[0])
    title_ax.set_axis_off()
    title_ax.text(
        0.0,
        0.60,
        "Products change from intergrowth\nto thin sheets",
        ha="left",
        va="center",
        fontsize=9.4,
    )
    axes = [title_ax]
    for row, image, concentration, outcome in [
        (1, low, "0.33 M formic acid", "overlapping intergrowth"),
        (2, high, "3.28 M formic acid", "thin sheets"),
    ]:
        ax = fig.add_subplot(grid[row])
        ax.imshow(image)
        ax.set_axis_off()
        ax.text(
            0.035,
            0.96,
            concentration,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.19", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        ax.text(0.5, -0.025, outcome, transform=ax.transAxes, ha="center", va="top", fontsize=7.6)
        axes.append(ax)
    return axes


def draw_retention_panel(ax: plt.Axes, retention: pd.DataFrame) -> None:
    frame = retention.copy()
    frame["label"] = frame.apply(lambda r: f"{r['series']}  {r['concentration_mM']:g} mM", axis=1)
    y = np.arange(len(frame), dtype=float)
    ax.axhspan(-0.48, 3.48, color="#F1F6FA", zorder=0)
    ax.axhspan(3.52, 6.48, color="#FBF4EF", zorder=0)
    ax.axhline(3.5, color="white", linewidth=2.4, zorder=1)

    for i, row in frame.iterrows():
        growth = 100.0 * row["growth_retained"]
        multilayer = 100.0 * row["multilayer_retained"]
        ax.plot([multilayer, growth], [y[i], y[i]], color="#9DA9AF", linewidth=1.6, zorder=2)
        ax.errorbar(
            growth,
            y[i],
            xerr=[[growth - 100.0 * row["growth_retained_ci_low"]], [100.0 * row["growth_retained_ci_high"] - growth]],
            fmt="o",
            ms=5.7,
            mfc=GROWTH,
            mec="white",
            mew=0.7,
            ecolor=GROWTH,
            elinewidth=0.9,
            capsize=1.8,
            zorder=4,
        )
        ax.errorbar(
            multilayer,
            y[i],
            xerr=[[multilayer - 100.0 * row["multilayer_retained_ci_low"]], [100.0 * row["multilayer_retained_ci_high"] - multilayer]],
            fmt="s",
            ms=5.4,
            mfc=MULTILAYER,
            mec="white",
            mew=0.7,
            ecolor=MULTILAYER,
            elinewidth=0.9,
            capsize=1.8,
            zorder=5,
        )

    ax.set_title("Multilayers decline more than net growth\nin every matched condition", loc="left", pad=20)
    ax.text(0.0, 1.015, "3.28 M relative to 0.33 M formic acid", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.4, color=NEUTRAL)
    ax.set_xlabel("Outcome retained (%)")
    ax.set_yticks(y, frame["label"].tolist())
    ax.set_xlim(-2.5, 74)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_ylim(len(frame) - 0.45, -0.92)
    ax.grid(axis="x", color=LIGHT, linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", length=0, pad=5)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=GROWTH, markeredgecolor="white", markersize=6, label="Net growth"),
        Line2D([], [], marker="s", linestyle="", color=MULTILAYER, markeredgecolor="white", markersize=6, label="Persistent multilayers"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0.25,
        facecolor="white",
        framealpha=0.90,
    )


def draw_count_panel(ax: plt.Axes, counts: pd.DataFrame) -> None:
    styles = [
        ("Positive mean net growth", GROWTH, "o"),
        ("Multilayer-producing", MULTILAYER, "s"),
    ]
    for outcome, color, marker in styles:
        frame = counts[counts["outcome"] == outcome]
        ax.plot(
            frame["formic_acid_M"],
            frame["condition_count"],
            color=color,
            marker=marker,
            markersize=5.2,
            linewidth=2.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )
    ax.set_title("Growth persists as multilayer-producing\nconditions become rare", loc="left", pad=7)
    ax.set_xlabel("Formic acid (M)")
    ax.set_ylabel("Conditions (of 12)")
    ax.set_xticks(ACIDS, [f"{x:g}" for x in ACIDS])
    ax.set_yticks([0, 4, 8, 12])
    ax.set_xlim(-0.08, 3.98)
    ax.set_ylim(-0.35, 12.55)
    ax.grid(color=LIGHT, linewidth=0.8, zorder=0)
    growth_last = counts[(counts["outcome"] == "Positive mean net growth") & np.isclose(counts["formic_acid_M"], 3.28)].iloc[0]
    multi_last = counts[(counts["outcome"] == "Multilayer-producing") & np.isclose(counts["formic_acid_M"], 3.28)].iloc[0]
    ax.text(3.40, growth_last["condition_count"], f"growth > 0\n{int(growth_last['condition_count'])}/12", color=GROWTH, ha="left", va="center", fontsize=7.2)
    ax.text(3.40, multi_last["condition_count"], f"multilayer\n{int(multi_last['condition_count'])}/12", color=MULTILAYER, ha="left", va="center", fontsize=7.2)


def place_panel_letter(fig: plt.Figure, axes: list[plt.Axes], letter: str, dx: float = 0.035, dy: float = 0.008) -> None:
    boxes = [ax.get_position() for ax in axes]
    fig.text(min(box.x0 for box in boxes) - dx, max(box.y1 for box in boxes) + dy, letter, fontsize=14.5, fontweight="bold", ha="left", va="top")


def render_composite(retention: pd.DataFrame, counts: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.48, 5.75), facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=[0.40, 0.60], wspace=0.22)
    axes_a = draw_tem_panel(fig, outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.08, 0.78], hspace=0.42)
    ax_b = fig.add_subplot(right[0])
    ax_c = fig.add_subplot(right[1])
    draw_retention_panel(ax_b, retention)
    draw_count_panel(ax_c, counts)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.905, bottom=0.085)
    place_panel_letter(fig, axes_a, "A", dx=0.035, dy=0.010)
    place_panel_letter(fig, [ax_b], "B", dx=0.055, dy=0.075)
    place_panel_letter(fig, [ax_c], "C", dx=0.055, dy=0.055)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(OUT_STEM.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def render_standalone_panels(retention: pd.DataFrame, counts: pd.DataFrame) -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(3.05, 5.75), facecolor="white")
    axes_a = draw_tem_panel(fig, fig.add_gridspec(1, 1)[0])
    fig.subplots_adjust(left=0.08, right=0.98, top=0.965, bottom=0.04)
    place_panel_letter(fig, axes_a, "A", dx=0.055, dy=0.010)
    fig.savefig(PANEL_DIR / "figure8A.png", dpi=450, facecolor="white")
    fig.savefig(PANEL_DIR / "figure8A.svg", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.65, 3.70), facecolor="white")
    draw_retention_panel(ax, retention)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.75, bottom=0.15)
    place_panel_letter(fig, [ax], "B", dx=0.09, dy=0.018)
    fig.savefig(PANEL_DIR / "figure8B.png", dpi=450, facecolor="white")
    fig.savefig(PANEL_DIR / "figure8B.svg", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.65, 2.75), facecolor="white")
    draw_count_panel(ax, counts)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.86, bottom=0.20)
    place_panel_letter(fig, [ax], "C", dx=0.095, dy=0.018)
    fig.savefig(PANEL_DIR / "figure8C.png", dpi=450, facecolor="white")
    fig.savefig(PANEL_DIR / "figure8C.svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    configure_style()
    HERE.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    SI_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_PATH)
    raw = pd.read_csv(TRACE_PATH)
    retention = build_retention_table(summary, raw)
    counts = build_count_table(summary)
    summary.to_csv(HERE / "figure8_condition_source_v2.csv", index=False)
    retention.to_csv(HERE / "figure8_retention_source_v2.csv", index=False)
    counts.to_csv(HERE / "figure8_outcome_counts_source_v2.csv", index=False)
    shutil.copy2(OLD_GRID, SI_DIR / "figureS_btb_formic_acid_complete_metric_grid.png")
    render_standalone_panels(retention, counts)
    render_composite(retention, counts)
    manifest = {
        "canonical_figure": OUT_STEM.with_suffix(".png").as_posix(),
        "editable_figure": OUT_STEM.with_suffix(".svg").as_posix(),
        "condition_source": SUMMARY_PATH.as_posix(),
        "trajectory_source": TRACE_PATH.as_posix(),
        "tem_source": TEM_SOURCE.as_posix(),
        "n_conditions": int(len(summary)),
        "n_trajectories": int(len(raw)),
        "n_per_condition": sorted(summary["n_independent_baths"].unique().astype(int).tolist()),
        "s_close": 1,
        "panel_b_eligible_conditions": int(len(retention)),
        "panel_b_rule": "P_multilayer(0.33 M)>=0.10 and positive mean growth at both endpoints",
        "panel_c_multilayer_rule": "mean persistent multilayer probability >= 0.10",
        "main_figure_claim": "Increasing formic acid leaves a positive-growth window while preferentially suppressing persistent multilayer domains.",
        "complete_grid_status": "retained as Supporting Information audit figure",
        "supersedes": "figure8_preferential_multilayer_suppression_v1",
    }
    (HERE / "figure8_growth_window_v2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(OUT_STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
