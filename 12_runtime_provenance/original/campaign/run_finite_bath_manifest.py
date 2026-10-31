from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import traceback

from chemistry_mapping import AnchoredCompetitor, CompetitiveBath
from finite_bath_population_runner import FiniteBathInputs, run_population_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-root", default=os.environ.get("MOL_GROWTH_SEED_ROOT"))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def run_row(row: dict[str, str], output_dir: str, seed_root: str) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{row['run_id']}.json"
    if destination.exists() and destination.stat().st_size > 0:
        return destination.as_posix()
    seed_directory = Path(seed_root) / row["seed_class"]
    seed_pickles = sorted(seed_directory.glob("*.pkl"))
    if not seed_pickles:
        raise FileNotFoundError(f"No seed pickles in {seed_directory}")
    formate_to_acid_ratio = as_float(row, "formate_to_acid_ratio")
    if formate_to_acid_ratio <= 0.0:
        raise ValueError("formate_to_acid_ratio must be positive")
    formic_acid_pka = as_float(row, "formic_acid_pka")
    operational_ph = formic_acid_pka + __import__("math").log10(formate_to_acid_ratio)
    bath = CompetitiveBath(
        linker=AnchoredCompetitor(
            row["linker_identity"],
            as_float(row, "linker_concentration_mM"),
            as_float(row, "linker_pka"),
            as_float(row, "linker_formate_exchange_constant"),
            as_float(row, "linker_accessibility"),
        ),
        modulator=AnchoredCompetitor(
            row["modulator_identity"],
            as_float(row, "modulator_concentration_mM"),
            as_float(row, "modulator_pka"),
            as_float(row, "modulator_formate_exchange_constant"),
            as_float(row, "modulator_accessibility"),
        ),
        solvent_weight=as_float(row, "solvent_weight"),
        operational_ph=operational_ph,
        formic_acid_pka=formic_acid_pka,
        acid_to_carboxylate_solvent_exchange_ratio=as_float(
            row, "acid_to_carboxylate_solvent_exchange_ratio"
        ),
    )
    inputs = FiniteBathInputs(
        volume_mL=as_float(row, "volume_mL"),
        seed_reference_linker_umol=as_float(row, "seed_reference_linker_umol"),
        analytical_atomic_metal_mM=as_float(row, "analytical_atomic_metal_mM"),
        free_linker_mM=as_float(row, "linker_concentration_mM"),
        reactive_m6_fraction=as_float(row, "reactive_m6_fraction"),
        node_capture_coefficient_per_mM=as_float(
            row, "node_capture_coefficient_per_mM"
        ),
        tethered_linker_weight=as_float(row, "tethered_linker_weight"),
        opened_site_rate_ratio=as_float(row, "opened_site_rate_ratio"),
        bond_opening_rate_ratio=as_float(row, "bond_opening_rate_ratio"),
        solvent_displacement_efficiency=as_float(
            row, "solvent_displacement_efficiency"
        ),
        closure_coefficient=as_float(row, "closure_coefficient"),
    )
    result = run_population_trace(
        system=row["system"],
        seed_pickles=seed_pickles,
        batch_seed_count=as_int(row, "batch_seed_count"),
        base_bath=bath,
        inputs=inputs,
        random_seed=as_int(row, "random_seed"),
        model_time_horizon=as_float(row, "model_time_horizon"),
        event_cap=as_int(row, "event_cap"),
        wall_time_cap_s=as_float(row, "wall_time_cap_s"),
        topology_mode=row["topology_mode"],
        recycle_detached_new_entities=row["recycle_detached_new_entities"].lower()
        in {"1", "true", "yes"},
    )
    result["manifest_row"] = row
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination.as_posix()


def main() -> None:
    args = parse_args()
    if not args.seed_root:
        raise ValueError("--seed-root or MOL_GROWTH_SEED_ROOT is required")
    with Path(args.manifest).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_row, row, output.as_posix(), args.seed_root): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result()
            except Exception:
                failures += 1
                error_path = output / f"{row['run_id']}.err"
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
    if failures:
        raise SystemExit(f"{failures} finite-bath rows failed")


if __name__ == "__main__":
    main()
