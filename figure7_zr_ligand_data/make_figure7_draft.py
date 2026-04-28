from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figure7_zr_ligand_data"

TREND_CSV = ROOT / "zr_ligand_one_shot_minimal_summary" / "09_experimental_trend_overlay_data.csv"
MOL_FINAL_CSV = ROOT / "zr_ligand_one_shot_minimal_summary" / "11_mol_final_experiment_vs_kmc_logy_plot_data.csv"
UIO66_SURVIVAL_CSV = (
    ROOT
    / "zr_ligand_one_shot_minimal_summary"
    / "04_uio66_one_shot_target8_experiment_survival.csv"
)
SUCCESS_CSV = ROOT / "zr_ligand_one_shot_minimal_summary" / "10_prebound_site_success_comparison.csv"
UIO66_OVERGROWTH_CSV = OUT_DIR / "UiO-66.csv"
MOL_OVERGROWTH_CSV = OUT_DIR / "BTB-MOL.csv"


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_seeded_overgrowth_series(path):
    xs = []
    ys = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            x_text = row[0].strip()
            y_text = row[1].strip()
            if not x_text or not y_text or x_text == "--" or y_text == "--":
                continue
            xs.append(float(x_text))
            ys.append(float(y_text))
    return xs, ys


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.minor.width": 0.9,
            "ytick.minor.width": 0.9,
        }
    )


def add_panel_label(ax, label):
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def make_panel_a(ax):
    add_panel_label(ax, "A")
    ax.set_title("Preassociated Motifs", loc="left", pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left = Rectangle((0.04, 0.15), 0.40, 0.68, fill=False, linewidth=1.2, linestyle="--", edgecolor="#777777")
    right = Rectangle((0.56, 0.15), 0.40, 0.68, fill=False, linewidth=1.2, linestyle="--", edgecolor="#777777")
    ax.add_patch(left)
    ax.add_patch(right)

    ax.text(0.24, 0.86, "Hf-BTB\npreassociated motif", ha="center", va="top", fontsize=9)
    ax.text(0.76, 0.86, "Zr-BDC\npreassociated motif", ha="center", va="top", fontsize=9)
    ax.text(0.24, 0.49, "Insert\nstructure", ha="center", va="center", color="#666666")
    ax.text(0.76, 0.49, "Insert\nstructure", ha="center", va="center", color="#666666")


def make_panel_b(ax, rows):
    add_panel_label(ax, "B")
    ax.set_title("Experimental Trend", loc="left", pad=8)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["system"], {"x": [], "y": []})
        grouped[row["system"]]["x"].append(float(row["zr_mM"]))
        grouped[row["system"]]["y"].append(float(row["norm_exp_time"]))

    style = {
        "Hf-BTB-MOL": {"color": "#c0362c", "marker": "s"},
        "UiO-66": {"color": "#1f1f1f", "marker": "o"},
    }
    for system, values in grouped.items():
        order = sorted(zip(values["x"], values["y"]))
        xs = [v[0] for v in order]
        ys = [v[1] for v in order]
        ax.plot(
            xs,
            ys,
            color=style[system]["color"],
            marker=style[system]["marker"],
            linewidth=2.0,
            markersize=5.2,
            label=system,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Metal Concentration (mM)")
    ax.set_ylabel("Normalized Experimental Time")
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.grid(True, which="major", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")


def make_panel_c(ax, uio66_series, mol_series):
    add_panel_label(ax, "C")
    ax.set_title("Seeded Overgrowth", loc="left", pad=8)

    mol_x, mol_y = mol_series
    uio_x, uio_y = uio66_series
    ax.plot(mol_x, mol_y, color="#c0362c", marker="s", linewidth=2.0, markersize=5.2, label="Hf-BTB-MOL")
    ax.plot(uio_x, uio_y, color="#1f1f1f", marker="o", linewidth=2.0, markersize=5.2, label="UiO-66")

    ax.set_xscale("log")
    ax.set_xlabel("Metal Concentration (mM)")
    ax.set_ylabel("Linker Loading Ratio")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.grid(True, which="major", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")


def make_panel_d(ax, rows):
    add_panel_label(ax, "D")
    ax.set_title("Hf-BTB-MOL", loc="left", pad=8)

    xs = [float(r["zr_mM"]) for r in rows]
    y_exp = [float(r["exp_min"]) for r in rows]
    y_kmc = [float(r["mean_kmc_s"]) for r in rows]

    ax2 = ax.twinx()
    l1 = ax.plot(xs, y_exp, color="#1f1f1f", marker="o", linewidth=2.0, markersize=5.0, label="Experiment")
    l2 = ax2.plot(xs, y_kmc, color="#c0362c", marker="s", linewidth=2.0, markersize=5.0, label="One-shot KMC")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xlabel("Metal Concentration (mM)")
    ax.set_ylabel("Experimental Time (min)", color="#1f1f1f")
    ax2.set_ylabel("Time to entity = 20 (s)", color="#c0362c", labelpad=-4)
    ax.tick_params(axis="y", colors="#1f1f1f")
    ax2.tick_params(axis="y", colors="#c0362c")
    ax.set_xticks(xs)
    ax.set_xticklabels(["3.38", "6.77", "13.53", "27.06", "54.13", "108.26", "216.52"])
    ax.set_ylim(10, 40)
    ax.set_yticks([10, 20, 30, 40])
    ax2.set_ylim(3, 100)
    ax2.set_yticks([5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    for axis in (ax, ax2):
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(True, which="major", color="#dddddd", linewidth=0.7)
    lines = l1 + l2
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")


def make_panel_e(ax, rows):
    add_panel_label(ax, "E")
    ax.set_title("UiO-66 Survival to entity = 8", loc="left", pad=8)

    grouped = {}
    for row in rows:
        grouped.setdefault(float(row["zr_conc"]), {"t": [], "s": []})
        grouped[float(row["zr_conc"])]["t"].append(float(row["time_seconds"]))
        grouped[float(row["zr_conc"])]["s"].append(float(row["survival_probability"]))

    colors = {
        2.0: "#1b9e77",
        4.0: "#d95f02",
        8.0: "#7570b3",
        16.0: "#e7298a",
        32.0: "#444444",
    }
    for zr, values in sorted(grouped.items()):
        order = sorted(zip(values["t"], values["s"]))
        ts = [v[0] for v in order]
        ss = [v[1] for v in order]
        ax.step(ts, ss, where="post", linewidth=2.0, color=colors[zr], label=f"{int(zr)} mM")

    ax.set_xscale("log")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0.74, 1.02)
    ax.grid(True, which="major", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=8)


def make_panel_f(ax, rows):
    add_panel_label(ax, "F")
    ax.set_title("Productive Insertion Success", loc="left", pad=8)

    system_map = {"Hf-BTB-MOL": "MOL", "UiO-66": "UiO-66"}
    categories = [f"{system_map.get(r['system'], r['system'])}\n{r['site']}" for r in rows]
    means = [float(r["mean_success_pct"]) for r in rows]
    stds = [float(r["std_success_pct"]) for r in rows]
    colors = ["#c0362c", "#ef9a9a", "#2a6f9e", "#9ecae1"]
    xpos = list(range(len(rows)))

    ax.bar(xpos, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.8, capsize=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)


def main():
    configure_style()
    trend_rows = load_csv(TREND_CSV)
    mol_rows = load_csv(MOL_FINAL_CSV)
    survival_rows = load_csv(UIO66_SURVIVAL_CSV)
    success_rows = load_csv(SUCCESS_CSV)
    uio66_overgrowth = load_seeded_overgrowth_series(UIO66_OVERGROWTH_CSV)
    mol_overgrowth = load_seeded_overgrowth_series(MOL_OVERGROWTH_CSV)

    fig = plt.figure(figsize=(13.8, 8.6), dpi=300)
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.42)

    make_panel_a(fig.add_subplot(gs[0, 0]))
    make_panel_b(fig.add_subplot(gs[0, 1]), trend_rows)
    make_panel_c(fig.add_subplot(gs[0, 2]), uio66_overgrowth, mol_overgrowth)
    make_panel_d(fig.add_subplot(gs[1, 0]), mol_rows)
    make_panel_e(fig.add_subplot(gs[1, 1]), survival_rows)
    make_panel_f(fig.add_subplot(gs[1, 2]), success_rows)

    out_png = OUT_DIR / "Figure7_draft.png"
    out_svg = OUT_DIR / "Figure7_draft.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()
