"""New plotting-only helper (2026-10-30), using saved JSON; no simulation rerun.
The original replay script is retained alongside this file without modification.
The archived .replot.svg/.png are the existing source artwork, whereas outputs
from this helper are newly generated verification plots.
Requires numpy and matplotlib. Run: python replot_saved_trace.py
"""
from pathlib import Path
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    root = Path(__file__).resolve().parent.parent
    if os.name == "nt":
        prefix = chr(92) * 2
        if str(root).startswith(prefix + "?" + chr(92)):
            pass
        elif str(root).startswith(prefix):
            root = Path(prefix + "?" + chr(92) + "UNC" + chr(92) + str(root)[2:])
        else:
            root = Path(prefix + "?" + chr(92) + str(root))
    source = next((root / "source_data").glob("*.summary_trace.json"))
    q = json.loads(source.read_text(encoding="utf-8-sig"))
    grid = np.asarray([r["sim_time_seconds"] for r in q["mean_trace"]])
    for species in ["Zr12_AA", "Zr6_AA", "BDC"]:
        sampled = []
        for run in q["replay_runs"]:
            ts = np.asarray([r["sim_time_seconds"] for r in run["traces"]])
            ix = np.clip(np.searchsorted(ts, grid, side="right")-1, 0, len(ts)-1)
            sampled.append(np.asarray([r[species] for r in run["traces"]])[ix])
        expected = np.asarray([r[species+"_mean"] for r in q["mean_trace"]])
        assert np.array_equal(np.mean(sampled, axis=0), expected)
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    for sp, color, pale in [("Zr12_AA", "#c1440e", "#e0865f"), ("Zr6_AA", "#2e6ce6", "#6f9df5"), ("BDC", "#188842", "#6cbf85")]:
        for run in q["replay_runs"]:
            ax.plot([r["sim_time_seconds"] for r in run["traces"]], [r[sp] for r in run["traces"]], color=pale, alpha=.22, lw=1)
        ax.plot(grid, [r[sp+"_mean"] for r in q["mean_trace"]], color=color, lw=2.3, label=sp+" (mean)")
    zero = next(r["sim_time_seconds"] for r in q["mean_trace"] if r["Zr12_AA_mean"] == 0)
    ax.axvline(zero, color="#7b5bff", ls="--", lw=1.3, alpha=.9)
    ax.text(zero, 480, "mean Zr12 reaches 0", color="#7b5bff", ha="center", va="bottom")
    ax.set(xlabel="Simulated Time (s)", ylabel="Entity Count")
    ax.grid(color="#d6dce5", alpha=.6, lw=.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    output = root / "redrawn_verification"
    output.mkdir(exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    for ext in ["svg", "png"]:
        path = output / ("S21_saved_trace_verification."+ext)
        fig.savefig(path, dpi=180)
        print(path)
    plt.close(fig)

if __name__ == "__main__":
    main()
