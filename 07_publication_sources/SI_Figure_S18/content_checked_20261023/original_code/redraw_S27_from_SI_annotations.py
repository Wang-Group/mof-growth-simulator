from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


LATEST = Path(r"D:\课题相关文献\课题\MOL生长模拟\论文\MOL_handoff_20260926_v2")
OUT = (
    LATEST
    / "03_workspace"
    / "si_subpanels_redrawn_20260927"
    / "S27_reconstructed_20260927"
)
SVG = OUT / "svg"
PNG = OUT / "preview_png"

OFF = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
Q = [3.0, 1.5, 1.0, 0.5]

# The old package retains the original Matplotlib plotting logic, but the
# four-closure raw condition tables are not present.  These values are a
# direct transcription of the numeric cell annotations in the current SI
# Figure S27 raster (formerly Figure S28).  They are kept explicitly here so
# the redraw is code-generated and auditable rather than a cropped raster.
PANELS = {
    "A": {
        "host": "MOF-808 host (mismatched, two shells)",
        "closure": 0.0,
        "values": [
            [0.36, 0.26, 0.00051, 0, 0, 0],
            [0.32, 0.24, 0.00026, 0, 0, 0],
            [0.29, 0.17, 0, 0, 0, 0],
            [0.20, 0.073, 0, 0, 0, 0],
        ],
    },
    "B": {
        "host": "MOF-808 host (mismatched, two shells)",
        "closure": 0.1,
        "values": [
            [0.44, 0.38, 0.091, 0, 0, 0],
            [0.43, 0.33, 0.057, 0, 0, 0],
            [0.33, 0.28, 0.030, 0, 0, 0],
            [0.23, 0.18, 0.0040, 0, 0, 0],
        ],
    },
    "C": {
        "host": "MOF-808 host (mismatched, two shells)",
        "closure": 0.5,
        "values": [
            [0.44, 0.41, 0.23, 0.023, 0.00041, 0],
            [0.40, 0.32, 0.17, 0.018, 0.00016, 0],
            [0.33, 0.27, 0.12, 0.011, 0.00018, 0],
            [0.24, 0.18, 0.059, 0.0034, 0, 0],
        ],
    },
    "D": {
        "host": "MOF-808 host (mismatched, two shells)",
        "closure": 2.0,
        "values": [
            [0.46, 0.39, 0.22, 0.029, 0.00049, 0],
            [0.43, 0.34, 0.15, 0.016, 0.00024, 0],
            [0.36, 0.28, 0.12, 0.012, 0, 0],
            [0.24, 0.18, 0.055, 0.0043, 0, 0],
        ],
    },
    "E": {
        "host": "UiO-66 host (matched)",
        "closure": 0.0,
        "values": [
            [0.46, 0.39, 0.011, 0, 0, 0],
            [0.41, 0.33, 0.0051, 0, 0, 0],
            [0.38, 0.28, 0.0017, 0, 0, 0],
            [0.28, 0.20, 0.00035, 0, 0, 0],
        ],
    },
    "F": {
        "host": "UiO-66 host (matched)",
        "closure": 0.1,
        "values": [
            [0.65, 0.57, 0.43, 0.00072, 0, 0],
            [0.54, 0.51, 0.32, 0.00045, 0, 0],
            [0.51, 0.45, 0.32, 0.00057, 0, 0],
            [0.35, 0.33, 0.19, 0.00011, 0, 0],
        ],
    },
    "G": {
        "host": "UiO-66 host (matched)",
        "closure": 0.5,
        "values": [
            [0.63, 0.57, 0.46, 0.23, 0.044, 0.00056],
            [0.55, 0.56, 0.42, 0.20, 0.039, 0.00094],
            [0.49, 0.47, 0.35, 0.16, 0.029, 0.00026],
            [0.40, 0.33, 0.23, 0.11, 0.016, 0],
        ],
    },
    "H": {
        "host": "UiO-66 host (matched)",
        "closure": 2.0,
        "values": [
            [0.60, 0.60, 0.48, 0.27, 0.048, 0.00087],
            [0.55, 0.52, 0.37, 0.20, 0.044, 0.00026],
            [0.51, 0.48, 0.37, 0.16, 0.029, 0.00054],
            [0.39, 0.33, 0.23, 0.098, 0.019, 0],
        ],
    },
}

NORM = LogNorm(vmin=1.0e-4, vmax=0.65)
CMAP = "magma"


def set_style():
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    )


def annotate(ax, image, matrix, fontsize=8.5):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            shown = max(value, NORM.vmin)
            rgba = image.cmap(image.norm(shown))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                j,
                i,
                f"{value:.2g}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color="black" if luminance > 0.52 else "white",
            )


def plot_matrix(ax, matrix):
    plotted = np.where(matrix > 0, matrix, NORM.vmin)
    x_edges = np.arange(matrix.shape[1] + 1, dtype=float) - 0.5
    y_edges = np.arange(matrix.shape[0] + 1, dtype=float) - 0.5
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        plotted,
        cmap=CMAP,
        norm=NORM,
        shading="flat",
        edgecolors="none",
    )
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_aspect("auto")
    annotate(ax, image, matrix)
    ax.set_xticks(range(len(OFF)), [f"{x:g}" for x in OFF])
    ax.set_yticks(range(len(Q)), [f"{x:g}" for x in Q])
    return image


def save(fig, name):
    SVG.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(SVG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PNG / f"{name}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def draw_vector_colorbar(cax):
    edges = np.geomspace(NORM.vmin, NORM.vmax, 257)
    centers = np.sqrt(edges[:-1] * edges[1:])[:, np.newaxis]
    cax.pcolormesh(
        [0.0, 1.0],
        edges,
        centers,
        cmap=CMAP,
        norm=NORM,
        shading="flat",
        edgecolors="none",
    )
    cax.set_yscale("log")
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(NORM.vmin, NORM.vmax)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_yticks([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])
    cax.set_ylabel("nuclei per model-time unit")


def draw_shared_colorbar():
    fig = plt.figure(figsize=(1.55, 4.6))
    cax = fig.add_axes([0.20, 0.08, 0.22, 0.86])
    draw_vector_colorbar(cax)
    save(fig, "Figure_S27_shared_colorbar")


def draw_individual_panels():
    for letter, info in PANELS.items():
        matrix = np.asarray(info["values"], dtype=float)
        fig, ax = plt.subplots(figsize=(5.0, 3.8))
        fig.subplots_adjust(left=0.22, right=0.84, bottom=0.20, top=0.79)
        image = plot_matrix(ax, matrix)
        ax.set_title(f"closure coefficient = {info['closure']:g}", pad=9)
        ax.set_xlabel("link removal / addition weight")
        ax.set_ylabel(info["host"] + "\nZr/BDC ratio")
        ax.text(
            -0.28,
            1.13,
            letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
        )
        cax = fig.add_axes([0.875, 0.20, 0.027, 0.59])
        draw_vector_colorbar(cax)
        save(fig, f"Figure_S27_{letter}")


def draw_combined():
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 6.6), sharex=True, sharey=True)
    images = []
    for index, (letter, info) in enumerate(PANELS.items()):
        row, col = divmod(index, 4)
        ax = axes[row, col]
        matrix = np.asarray(info["values"], dtype=float)
        images.append(plot_matrix(ax, matrix))
        if row == 0:
            ax.set_title(f"closure coefficient = {info['closure']:g}")
        if col == 0:
            ax.set_ylabel(info["host"] + "\nZr/BDC ratio")
        if row == 1:
            ax.set_xlabel("link removal / addition weight")
    cax = fig.add_axes([0.925, 0.16, 0.014, 0.70])
    draw_vector_colorbar(cax)
    fig.suptitle(
        "Link removal suppresses surface-nucleation events per unit model time",
        x=0.04,
        ha="left",
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.895,
        bottom=0.11,
        top=0.89,
        wspace=0.07,
        hspace=0.16,
    )
    save(fig, "Figure_S27_reconstructed_combined")


def write_source_table():
    rows = []
    for letter, info in PANELS.items():
        matrix = np.asarray(info["values"], dtype=float)
        for i, q in enumerate(Q):
            for j, off in enumerate(OFF):
                rows.append(
                    {
                        "panel": letter,
                        "host": info["host"],
                        "closure_coefficient": info["closure"],
                        "atomic_Zr_BDC_ratio": q,
                        "link_removal_addition_weight": off,
                        "nuclei_per_model_time_unit": matrix[i, j],
                        "provenance": "transcribed from numeric annotation in current SI Figure S27 raster",
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "Figure_S27_reconstructed_values.csv", index=False)
    readme = "# Figure S27 code redraw\n\n"
    readme += "The package retains the original heat-map plotting logic, but not the complete four-closure trajectory/condition table used for this historical eight-panel figure.\n\n"
    readme += "The 192 displayed cell values were therefore transcribed from the numeric annotations in the current SI Figure S27 raster (formerly Figure S28) and redrawn with the original logarithmic `magma` heat-map logic. These files are code-generated vectors, not cropped or raster-wrapped SVGs.\n"
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main():
    set_style()
    SVG.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    write_source_table()
    draw_individual_panels()
    draw_shared_colorbar()
    draw_combined()
    print(OUT)
    for path in sorted(SVG.glob("*.svg")):
        print(path.name)


if __name__ == "__main__":
    main()
