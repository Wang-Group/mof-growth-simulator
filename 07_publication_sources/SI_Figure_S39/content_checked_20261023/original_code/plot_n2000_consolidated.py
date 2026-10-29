"""One SI figure: matched N=2000 atomic PXRD and node-projection controls."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "author_comments_revision_20260922_v2" / "source_data"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300})
fig, axes = plt.subplots(3, 2, figsize=(10.2, 11.7), constrained_layout=True)
cols = ["#1f77b4", "#c45a0b", "#008f76", "#a146a1"]
styles = ["-", "--", "-.", ":"]
for panel, rho in enumerate((10, 30, 50, 100)):
    ax = axes.flat[panel]
    for rep in range(4):
        path = DATA / f"btb_extend_rho{rho}_rep{rep:02d}_milestone_N02000.csv"
        df = pd.read_csv(path)
        mask = df.two_theta_deg.between(4.4, 6.5)
        ax.plot(df.loc[mask, "two_theta_deg"],
                df.loc[mask, "intensity_smoothed_normalized"],
                color=cols[rep], ls=styles[rep], lw=1.55,
                label=f"Trajectory {rep+1}")
    ax.axvspan(5.55, 6.10, color="#f8edce", zorder=-1)
    if rho == 10:
        for rep, angle in ((0, 5.76), (3, 5.79)):
            df = pd.read_csv(DATA / f"btb_extend_rho10_rep{rep:02d}_milestone_N02000.csv")
            y = np.interp(angle, df.two_theta_deg, df.intensity_smoothed_normalized)
            ax.plot(angle, y, marker="o", ms=5, mec=cols[rep], mfc="white", mew=1.3)
    ax.set_xlim(4.4, 6.5); ax.set_ylim(0, 1.08)
    ax.set_xticks([4.5, 5.0, 5.5, 6.0, 6.5])
    ax.set_yticks([0, .5, 1])
    ax.set_xlabel(r"$2\theta$ (degrees)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(rf"$\rho_{{\mathrm{{off}}}}={rho}$", pad=8, fontsize=11)
    ax.text(-.12, 1.05, "ABCD"[panel], transform=ax.transAxes,
            va="bottom", ha="left", fontweight="bold", fontsize=14)

node = pd.read_csv(HERE / "n2000_projection.csv")
for pos, rep in enumerate((0, 3), start=4):
    ax = axes.flat[pos]
    df = node[node.replicate == rep]
    ax.plot(df.two_theta_deg, df.all_nodes, color="#1f77b4", lw=1.7,
            label="All node centres")
    ax.plot(df.two_theta_deg, df.projected_nodes, color="#c45a0b",
            lw=1.7, ls="--", label="Nodes projected to best-fit plane")
    ax.axvspan(5.55, 6.10, color="#f8edce", zorder=-1)
    ax.set_xlim(4.5, 7); ax.set_ylim(0, 1.08)
    ax.set_xticks([4.5, 5, 5.5, 6, 6.5, 7])
    ax.set_yticks([0, .5, 1])
    ax.set_xlabel(r"$2\theta$ (degrees)")
    ax.set_ylabel("Normalized node signal")
    ax.set_title(f"Node control · trajectory {rep+1}", pad=8, fontsize=11)
    ax.text(-.12, 1.05, "EF"[pos-4], transform=ax.transAxes,
            va="bottom", ha="left", fontweight="bold", fontsize=14)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(.5, 1.025), fontsize=9.5)
node_handles, node_labels = axes.flat[4].get_legend_handles_labels()
fig.legend(node_handles, node_labels, loc="lower center", ncol=2,
           frameon=False, bbox_to_anchor=(.5, -.025), fontsize=9.5)
fig.savefig(HERE / "Figure_S51_N2000_consolidated.png", bbox_inches="tight")
fig.savefig(HERE / "Figure_S51_N2000_consolidated.svg", bbox_inches="tight")
print(HERE / "Figure_S51_N2000_consolidated.png")
