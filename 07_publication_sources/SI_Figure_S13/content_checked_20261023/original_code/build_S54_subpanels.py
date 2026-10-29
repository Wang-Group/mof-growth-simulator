"""Generate the S54 subpanels from the audited Figure 3 plotting code.

S54 contains the two panels corresponding to Figure 3F and Figure 3G. This
small driver loads the original Figure 3 implementation without changing it,
then redraws those two panels into this folder only. The heatmap is written
using vector quadrilaterals (rather than Matplotlib's raster ``imshow``) so
that the resulting SVG remains editable and does not embed a bitmap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


S54_DIR = Path(__file__).resolve().parent
FIGURE3_SCRIPT = Path(
    r"D:\课题相关文献\课题\MOL生长模拟\论文\student_handoff_20260915_v5\MOL_handoff\03_workspace\analysis_workspace\rewrite\recalculation_campaign_20260830\figure3_rebuild_20260901\figure3_revision_20260904_v11\build_figure3_revision_v11.py"
)
STYLE_KERNEL = Path(
    r"D:\课题相关文献\课题\MOL生长模拟\论文\student_handoff_20260915_v5\MOL_handoff\03_workspace\external_helpers\figure-style\kernel.py"
)
SENSITIVITY_SOURCE = FIGURE3_SCRIPT.parent / "figure3_local_sensitivity_source_v11.csv"


def load_figure3_module():
    """Load the original script, correcting only its stale style path in memory."""
    source = FIGURE3_SCRIPT.read_text(encoding="utf-8")
    stale = r'C:\Users\Cheng Wang\.codex\skills\figure-style\kernel.py'
    source = source.replace(stale, str(STYLE_KERNEL))
    module = type(sys)("figure3_revision_v11_for_s54")
    module.__file__ = str(FIGURE3_SCRIPT)
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(source, str(FIGURE3_SCRIPT), "exec"), module.__dict__)
    return module


def vector_sensitivity_heatmap(fig3, ax, sensitivity, process, sc_value, *, norm, cmap,
                               show_y: bool, show_x: bool, row_label: str | None):
    """Figure-3F heatmap with the original values and labels, but vector cells."""
    intervals = [
        (fig3.BASE.RHO_VALUES[index], fig3.BASE.RHO_VALUES[index + 1])
        for index in range(len(fig3.BASE.RHO_VALUES) - 1)
    ]
    matrix = np.full((len(fig3.BASE.Q_VALUES), len(intervals)), np.nan)
    subset = sensitivity[
        (sensitivity.process == process) & np.isclose(sensitivity.S_close, sc_value)
    ]
    for q_index, q_value in enumerate(fig3.BASE.Q_VALUES):
        for interval_index, (rho_low, rho_high) in enumerate(intervals):
            hit = subset[
                np.isclose(subset.q_value, q_value)
                & np.isclose(subset.rho_low, rho_low)
                & np.isclose(subset.rho_high, rho_high)
            ]
            if len(hit) == 1 and int(hit.iloc[0].resolved) == 1:
                matrix[q_index, interval_index] = max(
                    -3.0, float(hit.iloc[0].beta_dlnJ_dlnrho)
                )

    # One quadrilateral per cell gives the same block layout as the original
    # imshow panel while keeping every colour editable in the SVG.
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            face = fig3.MISSING if not np.isfinite(value) else cmap(norm(value))
            ax.add_patch(
                mpl.patches.Rectangle(
                    (col_index - 0.5, row_index - 0.5),
                    1.0,
                    1.0,
                    facecolor=face,
                    edgecolor="none",
                    linewidth=0,
                    antialiased=False,
                )
            )

    ax.set_xlim(-0.5, len(intervals) - 0.5)
    ax.set_ylim(-0.5, len(fig3.BASE.Q_VALUES) - 0.5)
    ax.set_xticks(range(len(intervals)))
    ax.set_yticks(range(len(fig3.BASE.Q_VALUES)))
    if show_x:
        ax.set_xticklabels(
            [f"{low:g}-{high:g}" for low, high in intervals],
            rotation=28,
            ha="right",
        )
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_yticklabels([f"{value:g}" for value in fig3.BASE.Q_VALUES])
    else:
        ax.set_yticklabels([])
    if show_x:
        ax.set_xlabel("link-removal/addition ratio interval", labelpad=2)
    if show_y:
        ax.set_ylabel(
            (row_label + "\nZr/BDC") if row_label else "Zr/BDC",
            labelpad=4,
            fontsize=8.0,
            fontweight="bold" if row_label else "normal",
        )
    else:
        ax.set_ylabel("")
    ax.tick_params(
        labelsize=7.2,
        length=2.2,
        pad=1.5,
        labelbottom=show_x,
        labelleft=show_y,
    )
    for spine in ax.spines.values():
        spine.set_color("#AEB6BC")
        spine.set_linewidth(0.6)


def save_svg(fig, path: Path) -> None:
    fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def vector_colorbar(ax, cmap, norm) -> None:
    """Draw a smooth-looking colour bar from vector rectangles, not an image."""
    values = np.linspace(norm.vmin, norm.vmax, 256)
    for left, right in zip(values[:-1], values[1:]):
        ax.add_patch(
            mpl.patches.Rectangle(
                (left, 0.0),
                right - left,
                1.0,
                facecolor=cmap(norm((left + right) / 2)),
                edgecolor="none",
                linewidth=0,
            )
        )
    ax.set_xlim(norm.vmin, norm.vmax)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([-3, -2, -1, 0, 0.5])
    ax.tick_params(labelsize=7.2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_panel_f(fig3, sensitivity) -> None:
    fig = plt.figure(figsize=(8.0, 4.8), facecolor="white")
    grid = fig.add_gridspec(
        5,
        3,
        height_ratios=[1, 1, 0.48, 0.42, 0.22],
        wspace=0.055,
        hspace=0.17,
    )
    norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=0.5)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad(fig3.MISSING)
    axes = []
    for row, (process, row_label) in enumerate([
        ("assembly_J_1_to_30", "Nucleation"),
        ("seed_growth_J_76_to_126", "Growth"),
    ]):
        for col, sc_value in enumerate([0.1, 1.0, 2.0]):
            ax = fig.add_subplot(grid[row, col])
            vector_sensitivity_heatmap(
                fig3,
                ax,
                sensitivity,
                process,
                sc_value,
                norm=norm,
                cmap=cmap,
                show_y=col == 0,
                show_x=row == 1,
                row_label=row_label if col == 0 else None,
            )
            if row == 0:
                ax.set_title(rf"$S_{{\mathrm{{close}}}}={sc_value:g}$", fontweight="bold")
            axes.append(ax)
    shared_x = fig.add_subplot(grid[2, :])
    shared_x.axis("off")
    shared_x.text(0.5, 0.02, r"Adjacent $\rho_{\mathrm{off}}$ interval", ha="center", va="bottom", fontsize=8.0)
    key = fig.add_subplot(grid[3, :])
    key.axis("off")
    cax = key.inset_axes([0.20, 0.20, 0.60, 0.28])
    vector_colorbar(cax, cmap, norm)
    cax.set_xlabel(r"$\beta=\Delta\ln J/\Delta\ln\rho_{\mathrm{off}}$", fontsize=8.3, labelpad=3)
    cax.xaxis.set_label_position("top")
    note = fig.add_subplot(grid[4, :])
    note.axis("off")
    note.text(
        0.5,
        0.55,
        r"Gray: unresolved zero-event interval; values below $-3$ are clipped only for display.",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#5F6971",
    )
    fig3.STYLE.panel_letter(axes[0], "F", case="upper", dx=-0.13, dy=1.05, fontsize=11)
    save_svg(fig, S54_DIR / "S54_A.svg")


def draw_panel_g(fig3) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.0), facecolor="white")
    fig3.pxrd_panel(ax)
    fig3.STYLE.panel_letter(ax, "G", case="upper", dx=-0.06, dy=1.05, fontsize=11)
    save_svg(fig, S54_DIR / "S54_B.svg")


def draw_individual_heatmaps(fig3, sensitivity) -> None:
    """Write each of the six heatmaps inside Panel A as its own SVG.

    The six files are independently rendered from the same source table and
    the same cell-level plotting routine; they are not crops of S54_A.svg.
    """
    norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=0.5)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad(fig3.MISSING)
    specs = [
        (1, "assembly_J_1_to_30", "Nucleation", 0.1),
        (2, "assembly_J_1_to_30", "Nucleation", 1.0),
        (3, "assembly_J_1_to_30", "Nucleation", 2.0),
        (4, "seed_growth_J_76_to_126", "Growth", 0.1),
        (5, "seed_growth_J_76_to_126", "Growth", 1.0),
        (6, "seed_growth_J_76_to_126", "Growth", 2.0),
    ]
    for index, process, row_label, sc_value in specs:
        fig, ax = plt.subplots(figsize=(3.25, 2.85), facecolor="white")
        vector_sensitivity_heatmap(
            fig3,
            ax,
            sensitivity,
            process,
            sc_value,
            norm=norm,
            cmap=cmap,
            show_y=True,
            show_x=True,
            row_label=row_label,
        )
        ax.set_title(rf"{row_label}: $S_{{\mathrm{{close}}}}={sc_value:g}$", fontsize=9.2, fontweight="bold", pad=5)
        fig.subplots_adjust(left=0.22, right=0.98, bottom=0.25, top=0.84)
        save_svg(fig, S54_DIR / f"S54_A{index}.svg")


def main() -> None:
    fig3 = load_figure3_module()
    fig3.STYLE.apply_figure_style(frame="open", font="Arial", sizes=(8.6, 7.6, 6.8), grid=False)
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    # The original script's upstream seeded-analysis folders are not part of
    # this handoff, but its audited v11 source table is. Read that exact table
    # directly; no values are regenerated or inferred here.
    sensitivity = pd.read_csv(SENSITIVITY_SOURCE)
    for path in [*fig3.PXRD_SOURCES.values(), fig3.IDEAL_PEAKS, SENSITIVITY_SOURCE]:
        if not path.exists():
            raise FileNotFoundError(path)
    draw_panel_f(fig3, sensitivity)
    draw_individual_heatmaps(fig3, sensitivity)
    draw_panel_g(fig3)
    print(f"Generated {S54_DIR / 'S54_A.svg'}")
    print(f"Generated {S54_DIR / 'S54_B.svg'}")


if __name__ == "__main__":
    main()
