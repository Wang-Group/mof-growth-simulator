from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
OUTPUT_DIR = PACKAGE_ROOT / "output"
TIME_SCALE_S = 0.175
FIRST_PASSAGE_LEVELS = (3, 5, 6, 9, 13, 20, 30, 50, 100, 300, 500, 1000)

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


def empirical_survival(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    step_x = [x[0]]
    step_y = [1.0]
    for index, value in enumerate(x):
        step_x.extend([value, value])
        step_y.extend([step_y[-1], (n - index - 1) / n])
    step_x.append(x[-1])
    step_y.append(0.0)
    return np.asarray(step_x), np.asarray(step_y)


def summarize(first_passage: pd.DataFrame) -> pd.DataFrame:
    columns = [f"t_entity_{level}_s" for level in FIRST_PASSAGE_LEVELS] + [
        "t_3_to_5_s",
        "t_5_to_6_s",
        "t_6_to_9_s",
        "t_9_to_13_s",
        "t_13_to_20_s",
        "t_20_to_30_s",
        "t_13_to_30_s",
        "t_30_to_1000_s",
    ]
    rows = []
    for column in columns:
        values = first_passage[column].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "metric": column,
                "n": values.size,
                "min_s": float(np.min(values)),
                "q25_s": float(np.quantile(values, 0.25)),
                "median_s": float(np.median(values)),
                "q75_s": float(np.quantile(values, 0.75)),
                "max_s": float(np.max(values)),
                "mean_s": float(np.mean(values)),
                "std_s": float(np.std(values, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def plot_figure(
    first_passage: pd.DataFrame, residence: pd.DataFrame, traces: pd.DataFrame
) -> None:
    colors = {11: "#0072B2", 12: "#E69F00", 14: "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.2))
    ax_trace, ax_survival, ax_interval, ax_residence = axes.ravel()

    for case_index, trace in traces.groupby("case_index", sort=True):
        case_index = int(case_index)
        t30 = float(
            first_passage.loc[first_passage["case_index"] == case_index, "t_entity_30_s"].iloc[0]
        )
        color = colors.get(case_index, "#666666")
        ax_trace.plot(
            trace["time_s"],
            trace["entity"],
            color=color,
            linewidth=0.8,
            label=f"I{case_index:02d}, t30 = {t30:.0f} s",
        )
        ax_trace.axvline(t30, color=color, linewidth=0.7, linestyle="--")
    for level in (5, 9, 13, 20, 30):
        ax_trace.axhline(level, color="black", linewidth=0.4, linestyle=":", alpha=0.4)
    ax_trace.set_title("(a) Representative Pre-30 Trajectories")
    ax_trace.set_xlabel("Simulation Time (s)")
    ax_trace.set_ylabel("Entity Number")
    ax_trace.set_ylim(0, 35)
    ax_trace.grid(True, alpha=0.2)
    ax_trace.legend(frameon=False, loc="upper left")

    survival_colors = ("#999999", "#56B4E9", "#0072B2", "#009E73", "#E69F00", "#D55E00")
    for level, color in zip((5, 6, 9, 13, 20, 30), survival_colors):
        x, y = empirical_survival(first_passage[f"t_entity_{level}_s"].to_numpy(dtype=float))
        ax_survival.step(x, y, where="post", color=color, linewidth=1.15, label=f"Entity {level}")
    ax_survival.set_xscale("log")
    ax_survival.set_title("(b) First-Passage Survival")
    ax_survival.set_xlabel("First-Passage Time (s)")
    ax_survival.set_ylabel("Probability Not Yet Reached")
    ax_survival.set_ylim(-0.03, 1.03)
    ax_survival.grid(True, alpha=0.2, which="both")
    ax_survival.legend(frameon=False, loc="lower left", ncol=2)

    interval_columns = [
        "t_3_to_5_s",
        "t_5_to_6_s",
        "t_6_to_9_s",
        "t_9_to_13_s",
        "t_13_to_20_s",
        "t_20_to_30_s",
    ]
    interval_labels = ["3-5", "5-6", "6-9", "9-13", "13-20", "20-30"]
    interval_data = [first_passage[column].to_numpy(dtype=float) for column in interval_columns]
    parts = ax_interval.violinplot(interval_data, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_edgecolor("none")
        body.set_alpha(0.38)
    parts["cmedians"].set_color("#D55E00")
    parts["cmedians"].set_linewidth(1.4)
    ax_interval.set_yscale("log")
    ax_interval.set_xticks(np.arange(1, len(interval_labels) + 1), interval_labels, rotation=20)
    ax_interval.set_title("(c) Transition Intervals")
    ax_interval.set_xlabel("Entity-Number Interval")
    ax_interval.set_ylabel("Interval Time (s)")
    ax_interval.grid(True, axis="y", alpha=0.2, which="both")

    heat = residence.pivot(index="case_index", columns="entity", values="residence_time_s").fillna(0.0)
    ordered_cases = first_passage.sort_values("t_entity_30_s")["case_index"].astype(int)
    heat = heat.loc[ordered_cases]
    image = ax_residence.imshow(
        np.log10(heat.to_numpy(dtype=float) + 1e-3),
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax_residence.set_title("(d) Residence Below Entity 30")
    ax_residence.set_xlabel("Entity Number")
    ax_residence.set_ylabel("Trajectories Sorted by t30")
    ax_residence.set_xticks(np.arange(0, 29, 4), [str(value) for value in range(1, 30, 4)])
    colorbar = fig.colorbar(image, ax=ax_residence, fraction=0.046, pad=0.03)
    colorbar.set_label("log10 Residence Time (s)")

    for axis in (ax_trace, ax_survival, ax_interval):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "uio66_long_growth_figure_s19.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "uio66_long_growth_figure_s19.svg", bbox_inches="tight")
    plt.close(fig)


def write_summary(first_passage: pd.DataFrame, statistics: pd.DataFrame) -> None:
    medians = statistics.set_index("metric")["median_s"]
    lines = [
        "# UiO-66 Long-Growth Summary",
        "",
        f"- Completed trajectories: `{len(first_passage)}` / `24`.",
        f"- Time conversion: stored KMC timing multiplied by `{TIME_SCALE_S} s`.",
        f"- Median first-passage time to entity 5: `{medians['t_entity_5_s']:.1f} s`.",
        f"- Median first-passage time to entity 9: `{medians['t_entity_9_s']:.1f} s`.",
        f"- Median first-passage time to entity 13: `{medians['t_entity_13_s']:.1f} s`.",
        f"- Median first-passage time to entity 30: `{medians['t_entity_30_s']:.1f} s`.",
        f"- Median interval from entity 13 to 20: `{medians['t_13_to_20_s']:.1f} s`.",
        f"- Median interval from entity 13 to 30: `{medians['t_13_to_30_s']:.1f} s`.",
        f"- Median interval from entity 30 to 1000: `{medians['t_30_to_1000_s']:.1f} s`.",
        "",
        "The longest median delay occurs before first passage to entity 13, especially during the entity 9-to-13 interval. Most trajectories then reach entity 30 and entity 1000 rapidly. Entity 13 is therefore an operational kinetic bottleneck and a family of candidate growth-competent motifs, not a unique or deterministic critical nucleus.",
        "",
        "The archived entity-13 mol2 files are the last saved occurrence at entity 13 in each trajectory. Their bonds are reconstructed geometrically because the mol2 snapshots contain no bond records.",
    ]
    (OUTPUT_DIR / "uio66_long_growth_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    first_passage = pd.read_csv(DATA_DIR / "first_passage_24.csv")
    residence = pd.read_csv(DATA_DIR / "residence_pre30_by_case_entity.csv")
    traces = pd.read_csv(DATA_DIR / "representative_pre30_trajectories.csv")
    if len(first_passage) != 24 or not (first_passage["max_entity"] >= 1000).all():
        raise RuntimeError("Curated data do not contain 24 completed entity-1000 trajectories")
    statistics = summarize(first_passage)
    statistics.to_csv(OUTPUT_DIR / "uio66_long_growth_summary_stats.csv", index=False)
    plot_figure(first_passage, residence, traces)
    write_summary(first_passage, statistics)
    print(OUTPUT_DIR / "uio66_long_growth_figure_s19.png")


if __name__ == "__main__":
    main()
