from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
FIRST_PASSAGE_CSV = OUTPUT_DIR / "uio66_pre30_induction_first_passage.csv"
FIRST_PASSAGE_CURVES_CSV = OUTPUT_DIR / "uio66_induction_vs_dls_first_passage_curves.csv"
DLS_TRACE_CSV = OUTPUT_DIR / "uio66_dls_size_gualtieri_overlay_v2_trace.csv"
DLS_FIT_CSV = OUTPUT_DIR / "uio66_dls_size_gualtieri_overlay_v2_fit_curve.csv"
DLS_PARAMS_CSV = OUTPUT_DIR / "uio66_dls_size_gualtieri_overlay_v2_parameters.csv"

OUT_PNG = OUTPUT_DIR / "uio66_entity13_dls_linkage.png"
OUT_SVG = OUTPUT_DIR / "uio66_entity13_dls_linkage.svg"


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",
    }
)


def render() -> None:
    first_curves = pd.read_csv(FIRST_PASSAGE_CURVES_CSV)
    trace = pd.read_csv(DLS_TRACE_CSV)
    fit = pd.read_csv(DLS_FIT_CSV)
    params = pd.read_csv(DLS_PARAMS_CSV).iloc[0]

    thresholds = [5, 9, 13, 20, 30, 100, 300, 500, 1000]
    first_rows = []
    for case_index, case_df in first_curves.groupby("case_index"):
        case_row = {"case_index": case_index}
        for threshold in thresholds:
            reached = case_df.loc[case_df["entity"] >= threshold, "time_s"]
            case_row[f"t_entity_{threshold}_s"] = float(reached.iloc[0]) if not reached.empty else np.nan
        first_rows.append(case_row)
    first = pd.DataFrame(first_rows)
    cols = [f"t_entity_{n}_s" for n in thresholds]
    medians = first[cols].median()
    q25 = first[cols].quantile(0.25)
    q75 = first[cols].quantile(0.75)

    fig, (ax_sim, ax_dls) = plt.subplots(
        2,
        1,
        figsize=(3.65, 7.2),
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.42},
    )

    # Simulation first-passage statistics.
    rng = np.random.default_rng(7)
    y = np.asarray(thresholds, dtype=float)
    for i, col in enumerate(cols):
        jitter = rng.uniform(-0.11, 0.11, len(first))
        ax_sim.scatter(
            first[col],
            np.full(len(first), y[i]) * (1.0 + 0.030 * jitter),
            s=13,
            color="#6E6E6E",
            alpha=0.45,
            edgecolor="none",
            zorder=2,
        )

    ax_sim.plot(medians.to_numpy(), y, color="#D95F02", lw=1.8, marker="o", ms=4.2, zorder=4)
    ax_sim.fill_betweenx(
        y,
        q25.to_numpy(),
        q75.to_numpy(),
        color="#D95F02",
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )
    ax_sim.axhline(13, color="#B22222", lw=1.0, ls="--", zorder=0)
    ax_sim.set_xscale("log")
    ax_sim.set_yscale("log")
    ax_sim.set_xlim(0.01, 5.0e4)
    ax_sim.set_ylim(4, 1300)
    ax_sim.set_yticks(y)
    ax_sim.set_yticklabels([str(n) for n in thresholds])
    ax_sim.set_xlabel("First-Passage Time (s)")
    ax_sim.set_ylabel("Entity Number")
    ax_sim.set_title("Simulation: Latent Growth Threshold")
    ax_sim.tick_params(direction="out", length=3, width=0.8)
    ax_sim.set_box_aspect(1)
    for spine in ax_sim.spines.values():
        spine.set_linewidth(0.9)

    # DLS observation and crystallization-model fit.
    a_n = float(params["aN_s"])
    r2 = float(params["r2_alpha"])
    n_g = int(params["nG"])
    ax_dls.axvspan(0, a_n, color="#F4C7A1", alpha=0.45, lw=0)
    ax_dls.axvline(a_n, color="#BCA08C", lw=0.9, ls=":")
    ax_dls.axhline(float(params["baseline_nm"]), color="#BDBDBD", lw=0.8, ls=":")
    ax_dls.scatter(
        trace["time_s"],
        trace["raw_size_nm"],
        s=12,
        marker="s",
        color="#1F1A1A",
        edgecolor="none",
        label="DLS",
        zorder=3,
    )
    ax_dls.plot(
        fit["time_s"],
        fit["fit_size_nm"],
        color="#F28E2B",
        lw=1.9,
        label="Gualtieri fit",
        zorder=2,
    )
    ax_dls.text(
        a_n + 8,
        37.0,
        f"DLS onset\n$a_N$ = {a_n:.0f} s",
        color="#8C6D5A",
        ha="left",
        va="top",
        fontsize=8,
    )
    ax_dls.text(
        0.97,
        0.08,
        f"nG = {n_g}, R$^2$ = {r2:.3f}",
        transform=ax_dls.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
    )
    ax_dls.set_xlim(0, 600)
    ax_dls.set_ylim(0, 40)
    ax_dls.set_xticks([0, 200, 400, 600])
    ax_dls.set_yticks([0, 10, 20, 30, 40])
    ax_dls.set_xlabel("Time (s)")
    ax_dls.set_ylabel("DLS Size (nm)")
    ax_dls.set_title("Experiment: DLS-Visible Growth")
    ax_dls.legend(frameon=False, loc="upper left", handlelength=1.5)
    ax_dls.tick_params(direction="out", length=3, width=0.8)
    ax_dls.set_box_aspect(1)
    for spine in ax_dls.spines.values():
        spine.set_linewidth(0.9)

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.07, top=0.96, hspace=0.36)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render()
