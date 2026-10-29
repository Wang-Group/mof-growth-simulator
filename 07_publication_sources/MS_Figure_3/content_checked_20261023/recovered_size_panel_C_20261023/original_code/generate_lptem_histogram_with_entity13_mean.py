from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(r"C:\Users\SYH\Documents\Codex\2026-05-11\gitlab-ssh-gitlab-git-xmu-edu")
OUTDIR = ROOT / "outputs" / "zr_core_radius_stats"

LPTEM_CSV = OUTDIR / "lptem_particle_sizes.csv"
SIM_CSV = OUTDIR / "zr_core_outer_size_all.csv"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    lptem = pd.read_csv(LPTEM_CSV)["particle_size_nm"].astype(float)
    sim = pd.read_csv(SIM_CSV)

    entity13 = sim[(sim["entity_count"] == 13) & (sim["zr_atom_count"] > 0)][
        "max_zr_core_outer_distance_nm"
    ].astype(float)

    entity13_mean = entity13.mean()
    below_count = int((lptem < entity13_mean).sum())
    total_count = int(lptem.size)
    below_pct = below_count / total_count * 100.0
    lptem_mean = lptem.mean()

    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=200)

    bins = 12
    ax.hist(
        lptem,
        bins=bins,
        color="#8fb6d8",
        edgecolor="#333333",
        linewidth=1.2,
        alpha=0.9,
    )
    sns.kdeplot(
        lptem,
        ax=ax,
        color="#e34a42",
        linewidth=2.8,
        fill=False,
        bw_adjust=0.9,
        clip=(0, None),
    )

    ax.axvline(lptem_mean, color="#e34a42", linewidth=2.0, alpha=0.95)
    ax.axvline(
        entity13_mean,
        color="#1d4ed8",
        linewidth=2.2,
        linestyle="--",
        alpha=0.95,
    )

    ymax = ax.get_ylim()[1]
    ax.text(
        lptem_mean,
        ymax * 0.70,
        f"{lptem_mean:.3f} nm",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="black",
    )

    ax.text(
        0.98,
        0.94,
        (
            f"Entity 13 mean = {entity13_mean:.3f} nm\n"
            f"< this size: {below_count}/{total_count} ({below_pct:.1f}%)"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 6},
        color="#0f172a",
    )

    ax.text(
        entity13_mean + 0.03,
        ymax * 0.84,
        "Entity 13",
        color="#1d4ed8",
        fontsize=14,
        rotation=90,
        va="top",
        ha="left",
    )

    ax.set_xlabel("Particle diameter (nm)")
    ax.set_ylabel("Count")
    ax.set_xlim(0.9, max(4.05, lptem.max() + 0.15, entity13_mean + 0.2))
    ax.set_ylim(0, ymax)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6)

    fig.tight_layout()

    png_path = OUTDIR / "lptem_histogram_with_entity13_mean.png"
    svg_path = OUTDIR / "lptem_histogram_with_entity13_mean.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    stats_path = OUTDIR / "lptem_entity13_threshold_stats.txt"
    stats_path.write_text(
        "\n".join(
            [
                f"LPTEM N = {total_count}",
                f"LPTEM mean = {lptem_mean:.6f} nm",
                f"Entity 13 simulated mean = {entity13_mean:.6f} nm",
                f"LPTEM particles smaller than entity 13 mean = {below_count}/{total_count}",
                f"Percentage smaller than entity 13 mean = {below_pct:.6f}%",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
