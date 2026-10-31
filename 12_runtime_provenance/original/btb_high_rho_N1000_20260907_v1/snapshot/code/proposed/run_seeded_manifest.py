#!/usr/bin/env python3
"""Execute a frozen-seed growth manifest with the audited molecular KMC."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import traceback
from typing import Any

from common_kmc_selector import PhysicalCoordinates
from seeded_molecular_runner import run_seeded_trace
from common_molecular_runner import sha256


def _run_row(row: dict[str, str], output_dir: str) -> dict[str, Any]:
    run_id = row["run_id"]
    destination = Path(output_dir) / f"{run_id}.json"
    if destination.is_file():
        return {"run_id": run_id, "status": "already_complete"}
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        if row.get("seed_pickle") and row.get("seed_pickle_sha256"):
            observed_seed_hash = sha256(Path(row["seed_pickle"]))
            if observed_seed_hash.upper() != row["seed_pickle_sha256"].upper():
                raise ValueError(
                    f"seed pickle hash mismatch: expected {row['seed_pickle_sha256']}, "
                    f"observed {observed_seed_hash}"
                )
        coordinates = PhysicalCoordinates(
            external_activity=float(row["external_activity"]),
            node_fraction=float(row["node_fraction"]),
            off_weight=float(row["off_weight"]),
            closure_entropy_coefficient=float(row["SC"]),
        )
        seed_generation_coordinates = None
        if row.get("seed_generation_random_seed"):
            required_generation_fields = (
                "seed_generation_external_activity",
                "seed_generation_node_fraction",
                "seed_generation_off_weight",
                "seed_generation_SC",
            )
            missing = [name for name in required_generation_fields if not row.get(name)]
            if missing:
                raise ValueError(
                    "deterministic seed regeneration requires explicit fields: "
                    + ", ".join(missing)
                )
            seed_generation_coordinates = PhysicalCoordinates(
                external_activity=float(row["seed_generation_external_activity"]),
                node_fraction=float(row["seed_generation_node_fraction"]),
                off_weight=float(row["seed_generation_off_weight"]),
                closure_entropy_coefficient=float(row["seed_generation_SC"]),
            )
        result = run_seeded_trace(
            system=row["system"],
            seed_pickle=(Path(row["seed_pickle"]) if row.get("seed_pickle") else None),
            seed_generation_random_seed=(
                int(row["seed_generation_random_seed"])
                if row.get("seed_generation_random_seed")
                else None
            ),
            seed_generation_coordinates=seed_generation_coordinates,
            initial_seed_entities=int(row["initial_seed_entities"]),
            coordinates=coordinates,
            random_seed=int(row["random_seed"]),
            target_entities=int(row["target_entities"]),
            event_cap=int(row["event_cap"]),
            wall_time_cap_s=float(row["wall_time_cap_s"]),
            site_opening_seconds=(
                float(row["site_opening_seconds"])
                if row.get("site_opening_seconds")
                else None
            ),
            seed_retention_threshold=float(row["seed_retention_threshold"]),
            m6_node_fraction=float(row.get("m6_node_fraction") or 1.0),
            structure_metric_milestones=(
                tuple(
                    int(value)
                    for value in row["structure_metric_milestones"].split(";")
                    if value.strip()
                )
                if row.get("structure_metric_milestones")
                else None
            ),
            mol2_output_path=(
                Path(output_dir) / "structures" / f"{run_id}.mol2"
                if str(row.get("save_mol2", "")).strip().lower()
                in {"1", "true", "yes", "y"}
                else None
            ),
            mol2_milestones=(
                tuple(
                    int(value)
                    for value in row["mol2_milestones"].split(";")
                    if value.strip()
                )
                if row.get("mol2_milestones")
                else None
            ),
        )
        result.update(
            {
                "run_id": run_id,
                "condition_id": row["condition_id"],
                "replicate": int(row["replicate"]),
                "continuation_replicate": int(row["continuation_replicate"]),
                "design_block": row["design_block"],
                "composition_mapping": {
                    "kind": row["composition_mapping"],
                    "atomic_metal_to_linker_ratio": float(
                        row["atomic_metal_to_linker_ratio"]
                    ),
                    "node_fraction": float(row["node_fraction"]),
                },
            }
        )
        result["bath_mapping"] = {
            key: value for key, value in row.items() if key.startswith("bath_")
        }
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return {"run_id": run_id, "status": result["termination"]["status"]}
    except Exception as exc:
        error = {
            "schema_version": "common-seeded-molecular-kmc-error-v1",
            "run_id": run_id,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        error_destination = destination.with_suffix(".error.json")
        error_destination.write_text(
            json.dumps(error, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {"run_id": run_id, "status": "error", "error": str(exc)}
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = rows[args.start : args.start + args.limit if args.limit else None]
    statuses: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_run_row, row, str(args.output_dir)) for row in selected]
        for future in as_completed(futures):
            result = future.result()
            status = result["status"]
            statuses[status] = statuses.get(status, 0) + 1
            if status == "error":
                print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"selected": len(selected), "statuses": statuses}, sort_keys=True))
    if statuses.get("error", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
