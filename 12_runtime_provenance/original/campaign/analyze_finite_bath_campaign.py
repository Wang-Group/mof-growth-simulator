from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def main() -> None:
    args = parse_args()
    records: list[dict[str, object]] = []
    for path in sorted(Path(args.input_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["manifest_row"]
        final = payload["final"]
        records.append(
            {
                **row,
                "predicted_uptake_ratio": final[
                    "new_to_retained_reference_linker_ratio"
                ],
                "new_linkers_in_solids": final["new_linkers_in_solids"],
                "reference_seed_linkers_retained": final[
                    "reference_seed_linkers_retained"
                ],
                "termination_status": payload["termination"]["status"],
                "executed_events": payload["termination"]["executed_events"],
                "model_time": payload["termination"]["model_time"],
                "node_remaining": payload["reservoir"]["node_remaining"],
                "linker_remaining": payload["reservoir"]["linker_remaining"],
            }
        )
    if not records:
        raise ValueError("no JSON results")
    trace_path = Path(f"{args.output_prefix}_trace_metrics.csv")
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    keys = (
        "mechanism_id",
        "operation_id",
        "series",
        "varied_component",
        "varied_concentration_mM",
        "observed_uptake_ratio",
        "system",
        "model_time_horizon",
    )
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(tuple(record[key] for key in keys), []).append(record)
    summary: list[dict[str, object]] = []
    for key, group in groups.items():
        values = [
            float(item["predicted_uptake_ratio"])
            for item in group
            if item["predicted_uptake_ratio"] is not None
        ]
        terminations = sorted({str(item["termination_status"]) for item in group})
        summary.append(
            {
                **dict(zip(keys, key)),
                "trace_count": len(group),
                "valid_ratio_count": len(values),
                "mean_predicted_uptake_ratio": float(np.mean(values)) if values else None,
                "median_predicted_uptake_ratio": float(np.median(values)) if values else None,
                "q025_predicted_uptake_ratio": percentile(values, 0.025) if values else None,
                "q975_predicted_uptake_ratio": percentile(values, 0.975) if values else None,
                "termination_statuses": ";".join(terminations),
            }
        )
    summary.sort(
        key=lambda row: (
            str(row["mechanism_id"]),
            str(row["operation_id"]),
            str(row["varied_component"]),
            float(row["model_time_horizon"]),
            float(row["varied_concentration_mM"]),
        )
    )
    summary_path = Path(f"{args.output_prefix}_condition_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    mechanisms = sorted({str(row["mechanism_id"]) for row in summary})
    if mechanisms == ["fixed_speciation_central"]:
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.2), constrained_layout=True)
        panels = [
            ("OP_UIO_SEED_UPTAKE", "metal", "UiO-66: Zr series"),
            ("OP_UIO_SEED_UPTAKE", "linker", "UiO-66: BDC series"),
            ("OP_BTB_SEED_UPTAKE", "metal", "BTB-MOL: Hf series"),
            ("OP_BTB_SEED_UPTAKE", "linker", "BTB-MOL: BTB series"),
        ]
        horizons = sorted({float(row["model_time_horizon"]) for row in summary})
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(horizons)))
        for ax, (operation, varied, title) in zip(axes.flat, panels):
            selected = [
                row
                for row in summary
                if row["operation_id"] == operation and row["varied_component"] == varied
            ]
            observed_rows = {}
            for row in selected:
                observed_rows[float(row["varied_concentration_mM"])] = float(
                    row["observed_uptake_ratio"]
                )
            x_obs = sorted(observed_rows)
            ax.plot(
                x_obs,
                [observed_rows[x] for x in x_obs],
                "o-",
                color="black",
                linewidth=2.0,
                markersize=5.5,
                label="experiment",
                zorder=5,
            )
            for horizon, color in zip(horizons, colors):
                curve = sorted(
                    [row for row in selected if float(row["model_time_horizon"]) == horizon],
                    key=lambda row: float(row["varied_concentration_mM"]),
                )
                x = [float(row["varied_concentration_mM"]) for row in curve]
                y = [float(row["median_predicted_uptake_ratio"]) for row in curve]
                ax.plot(x, y, "o-", color=color, linewidth=1.7, markersize=4, label=f"t={horizon:g}")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(f"{varied.capitalize()} concentration (mM)")
            ax.set_ylabel("new/reference linker ratio")
            ax.tick_params(labelsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
        fig.savefig(f"{args.output_prefix}_horizon_comparison.png", dpi=300)
        fig.savefig(f"{args.output_prefix}_horizon_comparison.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
