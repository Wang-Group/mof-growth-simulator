from __future__ import annotations

import csv
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from generate_defects import (
    assign_carboxylates,
    assign_entities,
    duplicate_composite_units_with_bdc,
    instantiate_all_entities,
    read_mol_file,
)
from mol2pkl import build_assembly_from_state
from UiO66_Assembly_Large_Correction_conc import safe_pickle_load


DEFAULT_TOLERANCE = 1.5
ENTITY_PATTERN = re.compile(r"entity_number_(\d+)")


def quiet_pickle_load(pkl_path: Path):
    sink = io.StringIO()
    with redirect_stdout(sink):
        return safe_pickle_load(str(pkl_path), rebuild_references=True)


def quiet_build_assembly_from_mol2(mol2_path: Path):
    sink = io.StringIO()
    with redirect_stdout(sink):
        elements, coordinates, connectivity_map, carboxylate_indices = read_mol_file(str(mol2_path))
        element_entity_table = assign_entities(elements, coordinates, connectivity_map, carboxylate_indices)
        (
            element_entity_table_1,
            bridge_carboxylates,
            zr6_carboxylates,
            bdc_carboxylates,
            _unnormal_carboxylates,
        ) = assign_carboxylates(
            elements,
            coordinates,
            connectivity_map,
            carboxylate_indices,
            element_entity_table,
        )
        (
            _dup_elements,
            dup_coords,
            dup_table,
            dup_carboxylate_indices,
            dup_bridge_carboxylates,
            _invalidated_entities,
        ) = duplicate_composite_units_with_bdc(
            elements,
            coordinates,
            element_entity_table_1,
            carboxylate_indices,
            bridge_carboxylates,
            Zr6_carboxylates=zr6_carboxylates,
            BDC_carboxylates=bdc_carboxylates,
        )
        (
            entities,
            free_cs,
            mc_free,
            linker_free,
            linked_pairs,
            pair_index,
            ready_pairs,
        ) = instantiate_all_entities(
            dup_table,
            dup_coords,
            dup_carboxylate_indices,
            Bridge_carboxylates=dup_bridge_carboxylates,
            Zr6_carboxylates=zr6_carboxylates,
            BDC_carboxylates=bdc_carboxylates,
        )
        assembly = build_assembly_from_state(
            entities=entities,
            free_cs=free_cs,
            MC_free=mc_free,
            Linker_free=linker_free,
            linked_pairs=linked_pairs,
            ready_pairs=ready_pairs,
            pair_index=pair_index,
            ZR6_PERCENTAGE=None,
            ENTROPY_GAIN=None,
            BUMPING_THRESHOLD=None,
        )
    return assembly


def entity_count_from_path(path: Path) -> int | None:
    match = ENTITY_PATTERN.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def first_snapshots_by_entity_count(case_dir: Path) -> list[Path]:
    seen_counts: set[int] = set()
    selected: list[Path] = []
    for path in sorted(case_dir.glob("assembly_*.pkl"), key=lambda file: (file.stat().st_mtime, file.name)):
        entity_count = entity_count_from_path(path)
        if entity_count is None or entity_count in seen_counts:
            continue
        seen_counts.add(entity_count)
        selected.append(path)
    return selected


def assembly_counts(assembly) -> tuple[int, int, int]:
    total = len(list(assembly.entities))
    bdc = sum(1 for entity in assembly.entities if getattr(entity, "entity_type", None) == "Ligand")
    zr = sum(1 for entity in assembly.entities if getattr(entity, "entity_type", None) == "Zr")
    return total, bdc, zr


def centers_for_type(assembly, entity_type: str) -> np.ndarray:
    coords = [np.asarray(entity.center, dtype=float) for entity in assembly.entities if entity.entity_type == entity_type]
    return np.vstack(coords) if coords else np.zeros((0, 3), dtype=float)


def nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0:
        return np.zeros(0, dtype=float)
    if len(target) == 0:
        return np.full(len(source), np.inf, dtype=float)
    return np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2).min(axis=1)


def missing_sites(pristine, seed, tolerance: float = DEFAULT_TOLERANCE):
    pristine_bdc = centers_for_type(pristine, "Ligand")
    pristine_zr = centers_for_type(pristine, "Zr")
    seed_bdc = centers_for_type(seed, "Ligand")
    seed_zr = centers_for_type(seed, "Zr")

    missing_bdc = pristine_bdc[nearest_distances(pristine_bdc, seed_bdc) > tolerance]
    missing_zr = pristine_zr[nearest_distances(pristine_zr, seed_zr) > tolerance]
    return missing_bdc, missing_zr


def repair_metrics(assembly, missing_bdc: np.ndarray, missing_zr: np.ndarray, tolerance: float = DEFAULT_TOLERANCE):
    assembly_bdc = centers_for_type(assembly, "Ligand")
    assembly_zr = centers_for_type(assembly, "Zr")
    total_entities, total_bdc, total_zr = assembly_counts(assembly)

    filled_bdc = int((nearest_distances(missing_bdc, assembly_bdc) <= tolerance).sum())
    filled_zr = int((nearest_distances(missing_zr, assembly_zr) <= tolerance).sum())

    return {
        "total_entities": total_entities,
        "total_bdc": total_bdc,
        "total_zr": total_zr,
        "missing_bdc_sites": int(len(missing_bdc)),
        "missing_zr_sites": int(len(missing_zr)),
        "filled_bdc_sites": filled_bdc,
        "filled_zr_sites": filled_zr,
        "bdc_fill": filled_bdc / len(missing_bdc) if len(missing_bdc) else 0.0,
        "zr_fill": filled_zr / len(missing_zr) if len(missing_zr) else 0.0,
    }


def row_for_snapshot_path(
    label: str,
    assembly_path: Path,
    missing_bdc: np.ndarray,
    missing_zr: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
):
    assembly = quiet_pickle_load(assembly_path)
    metrics = repair_metrics(assembly, missing_bdc, missing_zr, tolerance=tolerance)
    metrics["label"] = label
    metrics["file_name"] = assembly_path.name
    return metrics


def build_repair_rows(
    *,
    pristine_path: Path,
    seed_path: Path,
    checkpoint_dir: Path,
    final_path: Path,
    tracked_path: Path | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[dict[str, object]]:
    pristine = quiet_build_assembly_from_mol2(pristine_path)
    seed = quiet_pickle_load(seed_path)
    missing_bdc, missing_zr = missing_sites(pristine, seed, tolerance=tolerance)

    rows: list[dict[str, object]] = []
    seed_row = repair_metrics(seed, missing_bdc, missing_zr, tolerance=tolerance)
    seed_row["label"] = "reference_seed"
    seed_row["file_name"] = seed_path.name
    rows.append(seed_row)

    seed_entities = seed_row["total_entities"]
    final_entities = entity_count_from_path(final_path)
    tracked_resolved = tracked_path.resolve() if tracked_path is not None else None

    for path in first_snapshots_by_entity_count(checkpoint_dir):
        entity_count = entity_count_from_path(path)
        if entity_count is None:
            continue
        if entity_count <= seed_entities:
            continue
        if final_entities is not None and entity_count >= final_entities:
            continue
        label = "same_condition_continuation"
        if tracked_resolved is not None and path.resolve() == tracked_resolved:
            label = "tracked_milestone"
        rows.append(
            row_for_snapshot_path(
                label,
                path,
                missing_bdc,
                missing_zr,
                tolerance=tolerance,
            )
        )

    rows.append(
        row_for_snapshot_path(
            "final_endpoint",
            final_path,
            missing_bdc,
            missing_zr,
            tolerance=tolerance,
        )
    )
    rows.sort(key=lambda row: row["total_entities"])
    return rows


CSV_FIELDNAMES = [
    "label",
    "file_name",
    "total_entities",
    "total_bdc",
    "total_zr",
    "missing_bdc_sites",
    "missing_zr_sites",
    "filled_bdc_sites",
    "filled_zr_sites",
    "bdc_fill",
    "zr_fill",
]


def write_rows_csv(rows: list[dict[str, object]], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows_csv(csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row: dict[str, object] = dict(raw_row)
            for key in (
                "total_entities",
                "total_bdc",
                "total_zr",
                "missing_bdc_sites",
                "missing_zr_sites",
                "filled_bdc_sites",
                "filled_zr_sites",
            ):
                row[key] = int(raw_row[key])
            for key in ("bdc_fill", "zr_fill"):
                row[key] = float(raw_row[key])
            rows.append(row)
    return rows
