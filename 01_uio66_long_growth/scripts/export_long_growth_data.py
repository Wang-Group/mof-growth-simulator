from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = (
    REPO_ROOT
    / "UiO-66"
    / "UiO66_long_growth"
    / "generated_runs_fixed"
    / "UiO66_260206_long1000"
    / "Stage_1"
)
DATA_DIR = PACKAGE_ROOT / "data"
SOURCE_DIR = PACKAGE_ROOT / "source"
CASE_RE = re.compile(r"_I(\d+)_")

# The stored KMC timing variable is dimensionless. This is the 25 degC
# exchange-time conversion used for the manuscript-facing trajectory analysis.
TIME_SCALE_S = 0.175
FIRST_PASSAGE_LEVELS = (3, 5, 6, 9, 13, 20, 30, 50, 100, 300, 500, 1000)
RESIDENCE_MAX_ENTITY = 29
REPRESENTATIVE_MAX_POINTS = 30000
ENTITY13_GROWTH_TARGETS = (30, 100, 500, 1000)


def load_case(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    with (folder / "entities_number.pkl").open("rb") as handle:
        raw = pickle.load(handle)
    if not raw:
        raise ValueError(f"Empty trajectory: {folder}")
    time_s = np.fromiter((float(row[0]) * TIME_SCALE_S for row in raw), dtype=float)
    entity = np.fromiter((int(row[1]) for row in raw), dtype=int)
    if time_s.size != entity.size or np.any(np.diff(time_s) < 0):
        raise ValueError(f"Malformed trajectory: {folder}")
    return time_s, entity


def first_crossing(time_s: np.ndarray, entity: np.ndarray, threshold: int) -> float:
    hit = np.flatnonzero(entity >= threshold)
    return float(time_s[int(hit[0])]) if hit.size else float("nan")


def residence_before_30(
    case_index: int, time_s: np.ndarray, entity: np.ndarray, t30_s: float
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    crossing = np.flatnonzero(time_s >= t30_s)
    end_index = int(crossing[0]) if crossing.size else time_s.size - 1
    durations = np.diff(time_s[: end_index + 1])
    states = entity[:end_index]

    rows: list[dict[str, float | int]] = []
    totals = np.zeros(RESIDENCE_MAX_ENTITY + 1, dtype=float)
    valid = (states >= 1) & (states <= RESIDENCE_MAX_ENTITY) & (durations > 0)
    np.add.at(totals, states[valid], durations[valid])
    for value in range(1, RESIDENCE_MAX_ENTITY + 1):
        rows.append(
            {
                "case_index": case_index,
                "entity": value,
                "residence_time_s": float(totals[value]),
            }
        )

    dominant = int(np.argmax(totals[1:]) + 1)
    dominant_row = {
        "case_index": case_index,
        "dominant_pre30_entity": dominant,
        "dominant_pre30_residence_s": float(totals[dominant]),
        "fraction_of_pre30_time": float(totals[dominant] / t30_s),
        "t_entity_30_s": t30_s,
    }
    return rows, dominant_row


def entity13_first_last_metrics(
    case_index: int, time_s: np.ndarray, entity: np.ndarray
) -> dict[str, float | int]:
    exact_indices = np.flatnonzero(entity == 13)
    if exact_indices.size == 0:
        raise ValueError(f"Case I{case_index:02d} never reaches exact entity 13")
    first_index = int(exact_indices[0])
    last_index = int(exact_indices[-1])
    episode_starts = [
        index for index in exact_indices if index == 0 or entity[index - 1] != 13
    ]
    window_entity = entity[first_index : last_index + 1]
    dt = np.diff(time_s, append=time_s[-1])
    row: dict[str, float | int] = {
        "entity": 13,
        "first_exact_s": float(time_s[first_index]),
        "last_exact_s": float(time_s[last_index]),
        "first_to_last_span_s": float(time_s[last_index] - time_s[first_index]),
        "exact_record_count": int(exact_indices.size),
        "exact_episode_count": int(len(episode_starts)),
        "exact_residence_s": float(np.sum(dt[exact_indices])),
        "window_min_entity": int(window_entity.min()),
        "window_max_entity": int(window_entity.max()),
        "window_unique_entity_count": int(np.unique(window_entity).size),
        "window_mean_entity": float(np.mean(window_entity)),
        "window_time_span_s": float(time_s[last_index] - time_s[first_index]),
    }
    for target in ENTITY13_GROWTH_TARGETS:
        later = np.flatnonzero((np.arange(entity.size) >= last_index) & (entity >= target))
        reached = float(time_s[int(later[0])]) if later.size else float("nan")
        row[f"last_to_ge_{target}_s"] = reached - float(time_s[last_index])
    row.update(
        {
            "case": case_index,
            "final_time_s": float(time_s[-1]),
            "max_entity": int(np.max(entity)),
        }
    )
    return row


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_main_body_sha256(path: Path) -> str:
    # The first 15 lines are the per-repeat parameter header. Everything after
    # that header is the common simulation implementation.
    body = "\n".join(path.read_text(encoding="utf-8").splitlines()[15:])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def export_conditions() -> None:
    workbook_path = SOURCE_DIR / "UiO66_260206_long1000_stage_1.xlsx"
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    worksheet = workbook.active
    rows = list(worksheet.iter_rows(values_only=True))
    if len(rows) != 2:
        raise ValueError(f"Expected one condition row in {workbook_path}")
    condition = dict(zip(rows[0], rows[1]))
    condition["trajectory_time_scale_s"] = TIME_SCALE_S
    condition["time_scale_basis"] = "25 degC ligand-exchange estimate"
    pd.DataFrame([condition]).to_csv(DATA_DIR / "run_conditions.csv", index=False)


def export_source_manifest() -> None:
    rows = []
    for path in sorted(SOURCE_DIR.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    pd.DataFrame(rows).to_csv(DATA_DIR / "source_manifest.csv", index=False)


def export_representative_traces(first_passage: pd.DataFrame, folders: dict[int, Path]) -> None:
    ordered = first_passage.sort_values("t_entity_30_s").reset_index(drop=True)
    selected = [
        (int(ordered.loc[0, "case_index"]), "fastest_t30"),
        (int(ordered.loc[len(ordered) // 2, "case_index"]), "median_rank_t30"),
        (int(ordered.loc[len(ordered) - 1, "case_index"]), "slowest_t30"),
    ]
    output = []
    for case_index, selection in selected:
        time_s, entity = load_case(folders[case_index])
        t30_s = float(
            first_passage.loc[first_passage["case_index"] == case_index, "t_entity_30_s"].iloc[0]
        )
        end = int(np.searchsorted(time_s, t30_s + 80.0, side="right"))
        end = max(end, 2)
        indices = np.arange(min(end, time_s.size))
        if indices.size > REPRESENTATIVE_MAX_POINTS:
            indices = np.unique(
                np.linspace(0, indices.size - 1, REPRESENTATIVE_MAX_POINTS).astype(int)
            )
        output.append(
            pd.DataFrame(
                {
                    "case_index": case_index,
                    "selection": selection,
                    "time_s": time_s[indices],
                    "entity": entity[indices],
                }
            )
        )
    pd.concat(output, ignore_index=True).to_csv(
        DATA_DIR / "representative_pre30_trajectories.csv", index=False
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    folders: dict[int, Path] = {}
    for folder in sorted(RUN_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        match = CASE_RE.search(folder.name)
        if match and (folder / "entities_number.pkl").exists():
            folders[int(match.group(1))] = folder
    if len(folders) != 24:
        raise RuntimeError(f"Expected 24 trajectories, found {len(folders)} in {RUN_ROOT}")

    first_passage_rows = []
    residence_rows = []
    dominant_rows = []
    entity13_rows = []
    manifest_rows = []
    for case_index, folder in sorted(folders.items()):
        time_s, entity = load_case(folder)
        row: dict[str, float | int] = {"case_index": case_index}
        for level in FIRST_PASSAGE_LEVELS:
            row[f"t_entity_{level}_s"] = first_crossing(time_s, entity, level)
        row.update(
            {
                "t_3_to_5_s": row["t_entity_5_s"] - row["t_entity_3_s"],
                "t_5_to_6_s": row["t_entity_6_s"] - row["t_entity_5_s"],
                "t_6_to_9_s": row["t_entity_9_s"] - row["t_entity_6_s"],
                "t_9_to_13_s": row["t_entity_13_s"] - row["t_entity_9_s"],
                "t_13_to_20_s": row["t_entity_20_s"] - row["t_entity_13_s"],
                "t_20_to_30_s": row["t_entity_30_s"] - row["t_entity_20_s"],
                "t_13_to_30_s": row["t_entity_30_s"] - row["t_entity_13_s"],
                "t_30_to_1000_s": row["t_entity_1000_s"] - row["t_entity_30_s"],
                "final_time_s": float(time_s[-1]),
                "max_entity": int(np.max(entity)),
            }
        )
        if not np.isfinite(row["t_entity_1000_s"]):
            raise RuntimeError(f"Case I{case_index:02d} did not reach entity 1000")
        first_passage_rows.append(row)

        residence, dominant = residence_before_30(
            case_index, time_s, entity, float(row["t_entity_30_s"])
        )
        residence_rows.extend(residence)
        dominant_rows.append(dominant)
        entity13_rows.append(entity13_first_last_metrics(case_index, time_s, entity))

        raw_path = folder / "entities_number.pkl"
        entity13_path = folder / "last_by_n" / "assembly_last_entity_number_13.mol2"
        assembly_source = folder / "UiO66_Assembly_Large_Correction_conc.py"
        main_source = folder / "UiO66_growth_main_conc.py"
        manifest_rows.append(
            {
                "case_index": case_index,
                "repeat_folder": folder.name,
                "raw_trajectory_path": raw_path.relative_to(REPO_ROOT).as_posix(),
                "raw_trajectory_bytes": raw_path.stat().st_size,
                "trajectory_samples": int(time_s.size),
                "max_entity": int(np.max(entity)),
                "final_time_s": float(time_s[-1]),
                "entity13_snapshot_present": entity13_path.exists(),
                "entity13_snapshot_path": entity13_path.relative_to(REPO_ROOT).as_posix(),
                "assembly_source_sha256": file_sha256(assembly_source),
                "main_body_sha256_after_parameter_header": normalized_main_body_sha256(
                    main_source
                ),
            }
        )

    first_passage = pd.DataFrame(first_passage_rows).sort_values("case_index")
    first_passage.to_csv(DATA_DIR / "first_passage_24.csv", index=False)
    pd.DataFrame(residence_rows).to_csv(
        DATA_DIR / "residence_pre30_by_case_entity.csv", index=False
    )
    pd.DataFrame(dominant_rows).sort_values("case_index").to_csv(
        DATA_DIR / "dominant_residence_pre30.csv", index=False
    )
    pd.DataFrame(entity13_rows).sort_values("case").to_csv(
        DATA_DIR / "entity13_first_last_decomposition_24.csv", index=False
    )
    pd.DataFrame(manifest_rows).sort_values("case_index").to_csv(
        DATA_DIR / "simulation_case_manifest.csv", index=False
    )
    export_representative_traces(first_passage, folders)
    export_conditions()
    export_source_manifest()
    print(f"Exported {len(first_passage)} trajectories to {DATA_DIR}")


if __name__ == "__main__":
    main()
