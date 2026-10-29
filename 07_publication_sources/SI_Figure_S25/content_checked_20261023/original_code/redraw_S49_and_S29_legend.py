from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext
import numpy as np
import pandas as pd


LATEST = Path(r"D:\课题相关文献\课题\MOL生长模拟\论文\MOL_handoff_20260926_v2")
WORK = LATEST / "03_workspace" / "si_subpanels_redrawn_20260927"
OUT = WORK / "corrected_S49_S29_legend_20260927"
SVG = OUT / "svg"
PNG = OUT / "preview_png"

S49_A_CSV = Path(
    r"D:\课题相关文献\课题\MOL生长模拟\论文\data1\mof-growth-simulator\UiO-66"
    r"\UiO_66_from_amorphous\initialize_structures1\output"
    r"\mixeddef800_zx2_seed800_continuous_from2124\repair_progress_vs_growth.csv"
)
S49_B_CSV = Path(
    r"D:\课题相关文献\课题\MOL生长模拟\论文\data1\mof-growth-simulator\UiO-66"
    r"\UiO_66_from_amorphous\Missing_BDC_Zr6\output"
    r"\mixeddef800_seed800_zr6only_candidate01_continuous\repair_progress_vs_growth.csv"
)

S29_SOURCE = (
    LATEST
    / "03_workspace"
    / "manuscript_revision_v34_results_provenance_20260916"
    / "source_data_steric"
)

BLUE = "#377eb8"
ORANGE = "#e67e22"
S29_COLORS = ["#4477AA", "#228833", "#AA3377"]
S29_MARKERS = ["o", "s", "^"]
S29_CAPS = ["benzoate", "4-OMe", "4-tBu"]
S29_LABELS = ["Benzoate", "4-Methoxybenzoate", "4-tert-Butylbenzoate"]


def set_style():
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "svg.fonttype": "none",
        }
    )


def save(fig, name):
    SVG.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(SVG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PNG / f"{name}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def load_repair(path):
    d = pd.read_csv(path)
    for col in ("total_entities", "bdc_fill", "zr_fill"):
        d[col] = pd.to_numeric(d[col], errors="raise")
    return d.sort_values("total_entities").reset_index(drop=True)


def draw_s49_panel(data, letter, title, subtitle, log_x, name):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.76)

    x = data["total_entities"].to_numpy()
    bdc = data["bdc_fill"].to_numpy()
    zr = data["zr_fill"].to_numpy()

    ax.plot(x, bdc, color=BLUE, lw=2.8, label="original missing BDC sites")
    ax.plot(x, zr, color=ORANGE, lw=2.8, label="original missing Zr sites")
    ax.scatter([x[-1]], [bdc[-1]], s=72, color=BLUE, zorder=4)
    ax.scatter([x[-1]], [zr[-1]], s=72, color=ORANGE, zorder=4)

    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(2.0e3, 1.10e5)
        ax.set_xticks([1.0e4, 1.0e5])
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
    else:
        ax.set_xlim(2050, 3725)
        ax.set_xticks(np.arange(2200, 3601, 200))

    ax.set_ylim(-0.02, 1.045)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel(r"assembly size, $N$")
    ax.set_ylabel("fraction of original missing sites filled")
    ax.set_title(title, fontweight="bold", pad=31, fontsize=14)
    ax.text(
        0.5,
        1.025,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#6f7780",
        fontsize=10.5,
    )
    ax.text(
        -0.16,
        1.16,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=1)
    ax.tick_params(direction="out", length=4, width=0.9)
    save(fig, name)


def draw_s49():
    a = load_repair(S49_A_CSV)
    b = load_repair(S49_B_CSV)
    assert len(a) == 130 and int(a.total_entities.iloc[[0, -1]].min()) == 2124
    assert int(a.total_entities.iloc[-1]) == 3651
    assert len(b) == 103 and int(b.total_entities.iloc[-1]) == 100001

    draw_s49_panel(
        a,
        "a",
        "Repair during continued growth",
        "fixed condition; one continuous trajectory",
        False,
        "Figure_S49_A",
    )
    draw_s49_panel(
        b,
        "b",
        "Long Zr-rich continuation",
        "fixed Zr-rich condition; one continuous trajectory",
        True,
        "Figure_S49_B",
    )


def load_s29():
    summary = pd.read_csv(S29_SOURCE / "matched_anchor_selected_9rows.csv")
    traces = pd.read_csv(S29_SOURCE / "source_trace_summary.csv")
    assert len(summary) == 9
    for row in summary.itertuples():
        for assay in ("nucleation", "growth"):
            g = traces[
                traces.identity.eq(row.cap)
                & traces.assay.eq(assay)
                & traces.rho_off.eq(1)
                & traces.q_atomic_Zr_BDC.eq(1)
                & traces.eta_S.eq(row.etaS)
            ]
            assert len(g) == 32
    return summary


def add_s29_curves(ax, summary, panel_index):
    for i, (cap, color, marker) in enumerate(zip(S29_CAPS, S29_COLORS, S29_MARKERS)):
        d = summary[summary.cap.eq(cap)].sort_values("etaS")
        x = d.etaS.to_numpy() + (i - 1) * 0.018
        if panel_index < 2:
            assay = ("nucleation", "growth")[panel_index]
            y = d[f"target_{assay}"].to_numpy() / 32
            lo = d[f"{assay}_wilson95_low"].to_numpy()
            hi = d[f"{assay}_wilson95_high"].to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=[y - lo, hi - y],
                color=color,
                marker=marker,
                ms=4.5,
                capsize=2,
                lw=1.25,
                label=S29_LABELS[i],
            )
        else:
            ax.plot(
                d.etaS,
                d.median_growth_passage_time,
                color=color,
                marker=marker,
                ms=4.5,
                lw=1.25,
                label=S29_LABELS[i],
            )


def format_s29_axis(ax, panel_index, letter):
    titles = ["Assembly to N = 30", "Growth from 76 to 126", "Successful growth time"]
    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_xlim(0.04, 0.96)
    ax.set_xlabel(r"$\eta_{\mathrm{S}}$")
    if panel_index < 2:
        ax.set_ylim(-0.05, 1.13)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylabel("Target-reaching fraction")
    else:
        ax.set_ylim(0, 11)
        ax.set_ylabel("Median growth time\n(model-time units)")
        ax.text(
            0.53,
            9.25,
            "4-tBu at 0.1:\nno successful growth",
            fontsize=7.5,
            color=S29_COLORS[2],
            ha="center",
        )
    ax.set_title(titles[panel_index], pad=8, fontsize=10)
    ax.text(
        -0.19,
        1.08,
        letter,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontweight="bold",
        fontsize=14,
    )


def draw_s29_legend(summary):
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            lw=1.5,
            markersize=6,
            label=label,
        )
        for color, marker, label in zip(S29_COLORS, S29_MARKERS, S29_LABELS)
    ]
    fig = plt.figure(figsize=(7.2, 0.65))
    fig.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=11)
    save(fig, "Figure_S29_legend")

    for k, letter in enumerate("ABC"):
        fig, ax = plt.subplots(figsize=(4.8, 3.7))
        fig.subplots_adjust(left=0.20, bottom=0.18, right=0.98, top=0.64)
        add_s29_curves(ax, summary, k)
        format_s29_axis(ax, k, letter)
        handles2, labels2 = ax.get_legend_handles_labels()
        fig.legend(
            handles2,
            labels2,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.995),
            ncol=1,
            frameon=False,
            fontsize=8.2,
            handlelength=2.8,
        )
        save(fig, f"Figure_S29_{letter}_with_legend")

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.25))
    fig.subplots_adjust(left=0.08, bottom=0.19, right=0.99, top=0.68, wspace=0.48)
    for k, (ax, letter) in enumerate(zip(axes, "ABC")):
        add_s29_curves(ax, summary, k)
        format_s29_axis(ax, k, letter)
    handles3, labels3 = axes[0].get_legend_handles_labels()
    fig.legend(
        handles3,
        labels3,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    save(fig, "Figure_S29_combined_with_legend")


def write_readme():
    text = "# Corrected SI subpanel SVGs\n\n"
    text += "- `Figure_S49_A.svg` and `Figure_S49_B.svg` were redrawn from the two original `repair_progress_vs_growth.csv` trajectory tables; no raster cropping was used.\n"
    text += "- `Figure_S29_legend.svg` is the missing shared legend.\n"
    text += "- `Figure_S29_A_with_legend.svg` through `C_with_legend.svg` contain an embedded legend for standalone use.\n"
    text += "- `Figure_S29_combined_with_legend.svg` is a three-panel reference composition.\n"
    text += "- Existing scripts and SVGs in the parent folder were not overwritten.\n"
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main():
    set_style()
    draw_s49()
    summary = load_s29()
    draw_s29_legend(summary)
    write_readme()
    print(OUT)
    for path in sorted(SVG.glob("*.svg")):
        print(path.name)


if __name__ == "__main__":
    main()
