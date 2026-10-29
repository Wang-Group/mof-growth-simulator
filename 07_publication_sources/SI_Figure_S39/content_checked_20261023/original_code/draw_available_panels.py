from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SVG_DIR = HERE / "svg"
PREVIEW_DIR = HERE / "preview_png"
SVG_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def save_panel(fig, name):
    fig.savefig(SVG_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PREVIEW_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_s29():
    source = ROOT / "03_workspace" / "manuscript_revision_v34_results_provenance_20260916" / "source_data_steric"
    s = pd.read_csv(source / "matched_anchor_selected_9rows.csv")
    traces = pd.read_csv(source / "source_trace_summary.csv")
    assert len(s) == 9
    for row in s.itertuples():
        for assay in ("nucleation", "growth"):
            g = traces[
                traces.identity.eq(row.cap) & traces.assay.eq(assay)
                & traces.rho_off.eq(1) & traces.q_atomic_Zr_BDC.eq(1)
                & traces.eta_S.eq(row.etaS)
            ]
            assert len(g) == 32
            assert g.status.eq("target_reached").sum() == getattr(row, f"target_{assay}")
            if assay == "growth" and getattr(row, "target_growth"):
                assert np.isclose(
                    g.loc[g.status.eq("target_reached"), "model_time"].median(),
                    row.median_growth_passage_time,
                )

    matplotlib.rcParams.update({
        "font.family": "Arial", "font.size": 10, "axes.labelsize": 10,
        "axes.titlesize": 11, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "axes.linewidth": 0.8,
    })
    caps = ["benzoate", "4-OMe", "4-tBu"]
    colors = ["#4477AA", "#228833", "#AA3377"]
    markers = ["o", "s", "^"]
    titles = ["Assembly to N = 30", "Growth from 76 to 126", "Successful growth time"]

    for k, letter in enumerate("ABC"):
        fig, ax = plt.subplots(figsize=(2.7, 2.8))
        fig.subplots_adjust(left=0.25, bottom=0.20, right=0.98, top=0.80)
        for i, (cap, color, marker) in enumerate(zip(caps, colors, markers)):
            a = s[s.cap.eq(cap)].sort_values("etaS")
            x = a.etaS.to_numpy() + (i - 1) * 0.018
            if k < 2:
                assay = ("nucleation", "growth")[k]
                y = a[f"target_{assay}"].to_numpy() / 32
                lo = a[f"{assay}_wilson95_low"].to_numpy()
                hi = a[f"{assay}_wilson95_high"].to_numpy()
                ax.errorbar(x, y, yerr=[y - lo, hi - y], color=color,
                            marker=marker, ms=4, capsize=2, lw=1.1)
            else:
                ax.plot(a.etaS, a.median_growth_passage_time, color=color,
                        marker=marker, ms=4, lw=1.1)
        ax.set_xticks([0.1, 0.5, 0.9])
        ax.set_xlim(0.04, 0.96)
        ax.set_xlabel(r"$\eta_{\mathrm{S}}$")
        if k < 2:
            ax.set_ylim(-0.05, 1.13)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_ylabel("Target-reaching fraction")
        else:
            ax.set_ylim(0, 11)
            ax.set_ylabel("Median growth time\n(model-time units)")
            ax.text(0.53, 9.25, "4-tBu at 0.1:\nno successful growth",
                    fontsize=7.5, color=colors[2], ha="center")
        ax.set_title(titles[k], pad=12, fontsize=9.5)
        fig.text(0.025, 0.95, letter, fontsize=13, fontweight="bold")
        save_panel(fig, f"Figure_S29_{letter}")


def draw_s51():
    source = ROOT / "03_workspace" / "author_comments_revision_20260922_v2" / "source_data"
    projection = ROOT / "03_workspace" / "si_consolidation_20260923_v1" / "n2000_projection.csv"
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300,
    })
    colors = ["#1f77b4", "#c45a0b", "#008f76", "#a146a1"]
    styles = ["-", "--", "-.", ":"]

    for k, rho in enumerate((10, 30, 50, 100)):
        fig, ax = plt.subplots(figsize=(5.1, 3.8))
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.78)
        for rep in range(4):
            f = source / f"btb_extend_rho{rho}_rep{rep:02d}_milestone_N02000.csv"
            d = pd.read_csv(f)
            mask = d.two_theta_deg.between(4.4, 6.5)
            ax.plot(d.loc[mask, "two_theta_deg"], d.loc[mask, "intensity_smoothed_normalized"],
                    color=colors[rep], ls=styles[rep], lw=1.55,
                    label=f"Trajectory {rep + 1}")
        ax.axvspan(5.55, 6.10, color="#f8edce", zorder=-1)
        if rho == 10:
            for rep, angle in ((0, 5.76), (3, 5.79)):
                d = pd.read_csv(source / f"btb_extend_rho10_rep{rep:02d}_milestone_N02000.csv")
                y = np.interp(angle, d.two_theta_deg, d.intensity_smoothed_normalized)
                ax.plot(angle, y, marker="o", ms=5, mec=colors[rep],
                        mfc="white", mew=1.3)
        ax.set_xlim(4.4, 6.5)
        ax.set_ylim(0, 1.08)
        ax.set_xticks([4.5, 5.0, 5.5, 6.0, 6.5])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(r"$2\theta$ (degrees)")
        ax.set_ylabel("Normalized intensity")
        ax.set_title(rf"$\rho_{{\mathrm{{off}}}}={rho}$", pad=8, fontsize=11)
        ax.text(-0.12, 1.05, "ABCD"[k], transform=ax.transAxes, va="bottom",
                ha="left", fontweight="bold", fontsize=14)
        h, labels = ax.get_legend_handles_labels()
        fig.legend(h, labels, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.025), fontsize=9.5)
        save_panel(fig, f"Figure_S51_{'ABCD'[k]}")

    node = pd.read_csv(projection)
    for letter, rep in (("E", 0), ("F", 3)):
        fig, ax = plt.subplots(figsize=(5.1, 3.8))
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.22, top=0.82)
        d = node[node.replicate.eq(rep)]
        ax.plot(d.two_theta_deg, d.all_nodes, color="#1f77b4", lw=1.7,
                label="All node centres")
        ax.plot(d.two_theta_deg, d.projected_nodes, color="#c45a0b",
                lw=1.7, ls="--", label="Nodes projected to best-fit plane")
        ax.axvspan(5.55, 6.10, color="#f8edce", zorder=-1)
        ax.set_xlim(4.5, 7)
        ax.set_ylim(0, 1.08)
        ax.set_xticks([4.5, 5, 5.5, 6, 6.5, 7])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(r"$2\theta$ (degrees)")
        ax.set_ylabel("Normalized node signal")
        ax.set_title(f"Node control · trajectory {rep + 1}", pad=8, fontsize=11)
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes, va="bottom",
                ha="left", fontweight="bold", fontsize=14)
        h, labels = ax.get_legend_handles_labels()
        fig.legend(h, labels, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, -0.025), fontsize=9.5)
        save_panel(fig, f"Figure_S51_{letter}")


if __name__ == "__main__":
    draw_s29()
    draw_s51()
    print(f"SVG panels: {SVG_DIR}")
