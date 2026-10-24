#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_curve(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        obj = pickle.load(f)

    arr = np.asarray(obj)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unsupported data shape: {arr.shape}")

    time = arr[:, 0].astype(float)
    entity = arr[:, 1].astype(float)
    return time, entity


def build_segments(entity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    change_idx = np.flatnonzero(np.diff(entity) != 0) + 1
    starts = np.r_[0, change_idx]
    ends_exclusive = np.r_[change_idx, len(entity)]
    return starts, ends_exclusive


def find_last_reach_points(
    time: np.ndarray, entity: np.ndarray, targets: List[int]
) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for target in targets:
        idx = np.where(np.isclose(entity, target, rtol=0.0, atol=1e-9))[0]
        if idx.size:
            pos_idx = idx[time[idx] > 0]
            if pos_idx.size:
                result[target] = int(pos_idx[-1])
            else:
                result[target] = int(idx[-1])
    return result


def select_plateau_segments(
    time: np.ndarray,
    starts: np.ndarray,
    ends_exclusive: np.ndarray,
    segment_entity: np.ndarray,
    top_k: int,
    min_duration: float | None,
    max_entity: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    durations = time[ends_exclusive - 1] - time[starts]
    selected = np.where(durations > 0)[0]

    if max_entity is not None:
        selected = selected[segment_entity[selected] <= max_entity]

    if min_duration is not None:
        selected = selected[durations[selected] >= min_duration]

    if top_k > 0 and selected.size > top_k:
        order = np.argsort(durations[selected])[::-1]
        selected = selected[order[:top_k]]

    selected = selected[np.argsort(starts[selected])]
    return selected, durations


def annotate_single_curve(
    pkl_file: Path,
    output_dir: Path,
    simple_only: bool,
    x_scale: float,
    targets: List[int],
    show_last_reach: bool,
    top_k_plateaus: int,
    min_plateau_duration: float | None,
    plateau_label_max: float | None,
    y_min: float | None,
    y_max: float | None,
    x_max_fraction: float | None,
    x_max_entity: float | None,
    x_min_ratio_of_xmax: float,
) -> dict:
    time, entity = load_curve(pkl_file)
    time = time * x_scale
    starts, ends_exclusive = build_segments(entity)

    seg_time_start = time[starts]
    seg_time_end = time[ends_exclusive - 1]
    seg_entity = entity[starts]

    t_min = float(np.min(time))
    t_max = float(np.max(time))
    x_lower = t_min
    x_upper = t_max
    if x_max_entity is not None:
        reach_idx = np.where(entity >= x_max_entity)[0]
        if reach_idx.size:
            x_upper = float(time[int(reach_idx[0])])
            x_lower = max(t_min, x_upper * x_min_ratio_of_xmax)
        else:
            x_upper = t_max
            x_lower = t_min
    elif x_max_fraction is not None:
        if x_max_fraction <= 0:
            raise ValueError("x_max_fraction must be > 0")
        x_upper = min(t_max, t_max * x_max_fraction)
        if x_upper <= t_min:
            x_upper = t_max
        x_lower = t_min

    # Use compressed step points from segment boundaries for faster plotting.
    x_step = np.r_[seg_time_start, seg_time_end[-1]]
    y_step = np.r_[seg_entity, seg_entity[-1]]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    if simple_only:
        ax.step(x_step, y_step, where="post", color="#1f77b4", linewidth=1.2)
    else:
        ax.step(x_step, y_step, where="post", color="#1f77b4", linewidth=1.2, label="entity count")

    if simple_only:
        last_points: Dict[int, float] = {}
        plateau_idx = np.array([], dtype=int)
        durations = np.array([], dtype=float)
    else:
        last_points = find_last_reach_points(time, entity, targets)
        point_colors = {1: "#2ca02c", 2: "#ff7f0e", 3: "#9467bd"}
        y_span = max(1.0, float(np.max(entity) - np.min(entity)))
        if show_last_reach:
            for target in targets:
                if target not in last_points:
                    continue
                idx = last_points[target]
                x = float(time[idx])
                y = float(entity[idx])
                color = point_colors.get(target, "#111111")
                ax.scatter([x], [y], color=color, s=35, zorder=5, label=f"last reach {target}")
                ax.annotate(
                    f"last {target}\n(t={x:.4g})",
                    xy=(x, y),
                    xytext=(6, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
                )

        plateau_idx, durations = select_plateau_segments(
            time=time,
            starts=starts,
            ends_exclusive=ends_exclusive,
            segment_entity=seg_entity,
            top_k=top_k_plateaus,
            min_duration=min_plateau_duration,
            max_entity=y_max,
        )

        for i in plateau_idx:
            x1 = float(seg_time_start[i])
            x2 = float(seg_time_end[i])
            y = float(seg_entity[i])

            x1p = x1
            x2p = x2
            if x2p < x1p:
                x1p, x2p = x2p, x1p

            ax.hlines(y, x1p, x2p, color="#d62728", linewidth=2.3, alpha=0.85)
            if plateau_label_max is None or y <= plateau_label_max:
                xmid = float((x1p + x2p) / 2.0)
                ax.text(
                    xmid,
                    y + 0.012 * y_span,
                    f"{int(round(y))}",
                    color="#d62728",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75),
                )

    ax.set_xlim(left=x_lower, right=x_upper)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if y_min is not None and y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    elif y_min is not None:
        ax.set_ylim(bottom=y_min)
    elif y_max is not None:
        ax.set_ylim(top=y_max)
    # Label every integer on the y-axis.
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: "" if abs(x) < 1e-9 else f"{int(round(x))}")
    )
    # Use Arial for numeric tick labels and double the default size.
    plt.setp(ax.get_xticklabels(), fontname="Arial", fontsize=20)
    plt.setp(ax.get_yticklabels(), fontname="Arial", fontsize=20)
    if not simple_only:
        ax.set_title(pkl_file.name)
        ax.grid(True, which="both", linestyle="--", alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        dedup = {}
        for h, l in zip(handles, labels):
            if l not in dedup:
                dedup[l] = h
        if dedup:
            ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8)

    fig.tight_layout()
    suffix = "entity_vs_time" if simple_only else "annotated"
    output_path = output_dir / f"{pkl_file.stem}_{suffix}.png"
    fig.savefig(output_path)
    plt.close(fig)

    return {
        "file": str(pkl_file),
        "output": str(output_path),
        "points": int(len(time)),
        "segments": int(len(starts)),
        "plateaus_marked": int(len(plateau_idx)),
        "last_points": {k: float(time[v]) for k, v in last_points.items()},
        "max_duration": float(np.max(durations)) if durations.size else 0.0,
        "x_lower": x_lower,
        "x_upper": x_upper,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot crystal growth curves from pkl files with annotations:\n"
            "1) last reach points for entity 1/2/3,\n"
            "2) highlighted plateaus with entity labels,\n"
            "3) optional truncated time axis."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing pkl files.")
    parser.add_argument(
        "--pattern",
        default="*.pkl",
        help="File pattern to match pkl files (default: *.pkl).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("annotated_plots"), help="Output directory.")
    parser.add_argument(
        "--simple-only",
        action="store_true",
        help="Only plot entity_count vs time, without any other annotation/overlay.",
    )
    parser.add_argument(
        "--x-scale",
        type=float,
        default=1.0,
        help="Multiply x-axis (time) by this factor.",
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Entity values to mark last-reach points.",
    )
    parser.add_argument(
        "--show-last-reach",
        action="store_true",
        help="Show markers and labels for last-reach points of --targets.",
    )
    parser.add_argument(
        "--top-k-plateaus",
        type=int,
        default=12,
        help="Mark the top-K longest plateaus; <=0 means all plateaus passing duration filter.",
    )
    parser.add_argument(
        "--min-plateau-duration",
        type=float,
        default=None,
        help="Only mark plateaus with duration >= this value (in time units).",
    )
    parser.add_argument(
        "--plateau-label-max",
        type=float,
        default=3.0,
        help="Only show plateau number labels for entity <= this value. Use negative to hide all.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional y-axis upper limit (e.g. 30).",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=0.0,
        help="Optional y-axis lower limit (default: 0).",
    )
    parser.add_argument(
        "--x-max-fraction",
        type=float,
        default=1.0,
        help="Show only the first fraction of total time on x-axis (e.g. 0.7).",
    )
    parser.add_argument(
        "--x-max-entity",
        type=float,
        default=20.0,
        help="If set, x_max is time when entity first reaches this value (>=).",
    )
    parser.add_argument(
        "--x-min-ratio-of-xmax",
        type=float,
        default=0.8,
        help="When --x-max-entity is used, x_min = this ratio * x_max.",
    )
    args = parser.parse_args()

    if not (0 < args.x_min_ratio_of_xmax < 1):
        raise ValueError("--x-min-ratio-of-xmax must be in (0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        print(f"[WARN] No files found with pattern: {args.pattern}")
        return

    print(f"[INFO] Found {len(files)} files")
    success = 0
    failed = 0

    for file in files:
        try:
            info = annotate_single_curve(
                pkl_file=file,
                output_dir=args.output_dir,
                simple_only=args.simple_only,
                x_scale=args.x_scale,
                targets=args.targets,
                show_last_reach=args.show_last_reach,
                top_k_plateaus=args.top_k_plateaus,
                min_plateau_duration=args.min_plateau_duration,
                plateau_label_max=args.plateau_label_max,
                y_min=args.y_min,
                y_max=args.y_max,
                x_max_fraction=args.x_max_fraction,
                x_max_entity=args.x_max_entity,
                x_min_ratio_of_xmax=args.x_min_ratio_of_xmax,
            )
            success += 1
            print(
                f"[OK] {file.name} -> {Path(info['output']).name} | "
                f"points={info['points']}, segments={info['segments']}, "
                f"plateaus_marked={info['plateaus_marked']}, "
                f"last={info['last_points']}, "
                f"xlim=({info['x_lower']:.4g}, {info['x_upper']:.4g})"
            )
        except Exception as e:
            failed += 1
            print(f"[FAIL] {file.name}: {type(e).__name__}: {e}")

    print(f"[INFO] done. success={success}, failed={failed}, output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
