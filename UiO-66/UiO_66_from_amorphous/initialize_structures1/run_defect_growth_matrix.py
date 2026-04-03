import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from generate_defects import (
    assign_carboxylates,
    assign_entities,
    calculate_structure_center_from_entities,
    duplicate_composite_units_with_bdc,
    generate_defects,
    identify_shell_entities_from_objects,
    instantiate_all_entities,
    read_mol_file,
)
from mol2pkl import build_assembly_from_state
from run_seeded_growth import DEFAULTS as GROWTH_DEFAULTS
from UiO66_Assembly_Large_Correction_conc import safe_pickle_load


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "UiO-66_15x15x15_sphere_R2.mol2"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "defect_growth_matrix_R2_s20"
DEFAULT_DEFECT_TYPES = ["BDC", "Zr6", "Mixed"]
DEFAULT_RATIOS = [0.2, 0.4, 0.6, 0.8]
DEFAULT_ZR_CONC_MULTIPLIERS = [1.0]
DEFAULT_LINKER_CONC_MULTIPLIERS = [0.7, 1.0, 1.3]
DEFAULT_CAPPING_CONC_MULTIPLIERS = [0.7, 1.0, 1.3]


def parse_csv_list(raw_value, cast=str):
    items = []
    for part in raw_value.split(","):
        value = part.strip()
        if value:
            items.append(cast(value))
    return items


def infer_structure_label(input_path: Path) -> str:
    match = re.search(r"_sphere_(R[\d.]+)$", input_path.stem)
    if match:
        return match.group(1)
    return input_path.stem


def defect_tag(defect_type):
    if defect_type == "BDC":
        return "BDCdef"
    if defect_type == "Zr6":
        return "Zr6def"
    if defect_type == "Mixed":
        return "Mixeddef"
    return defect_type


def format_ratio(defect_ratio):
    return f"{defect_ratio:.2f}"


def format_float_token(value):
    token = f"{value:.2f}"
    return token.replace("-", "m").replace(".", "p")


def build_seed_stem(structure_label, defect_type, defect_ratio, defect_seed):
    return f"UiO-66_{structure_label}_{defect_tag(defect_type)}_{format_ratio(defect_ratio)}_seed{defect_seed}"


def build_case_stem(
    structure_label,
    defect_type,
    defect_ratio,
    defect_seed,
    zr_multiplier=1.0,
    linker_multiplier=1.0,
    capping_multiplier=1.0,
):
    zr_suffix = ""
    if abs(zr_multiplier - 1.0) > 1e-9:
        zr_suffix = f"_Zx{format_float_token(zr_multiplier)}"
    return (
        f"{build_seed_stem(structure_label, defect_type, defect_ratio, defect_seed)}"
        f"{zr_suffix}"
        f"_Lx{format_float_token(linker_multiplier)}_Cx{format_float_token(capping_multiplier)}"
    )


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


def assembly_counts(assembly):
    total = len(list(assembly.entities))
    bdc = sum(1 for entity in assembly.entities if getattr(entity, "entity_type", None) == "Ligand")
    zr = sum(1 for entity in assembly.entities if getattr(entity, "entity_type", None) == "Zr")
    return total, bdc, zr


def centers_for_type(assembly, entity_type):
    coords = [np.asarray(entity.center, dtype=float) for entity in assembly.entities if entity.entity_type == entity_type]
    return np.vstack(coords) if coords else np.zeros((0, 3), dtype=float)


def nearest_distances(source, target):
    if len(source) == 0:
        return np.zeros(0, dtype=float)
    if len(target) == 0:
        return np.full(len(source), np.inf, dtype=float)
    return np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2).min(axis=1)


def latest_final_pkl(growth_dir: Path):
    candidates = sorted(growth_dir.glob("assembly*.pkl"), key=lambda path: (path.stat().st_mtime, path.name))
    if not candidates:
        raise FileNotFoundError(f"No assembly pickle files found in {growth_dir}")
    return candidates[-1]


def latest_pkls_by_entity_count(growth_dir: Path):
    pattern = re.compile(r"entity_number_(\d+)")
    latest_by_count = {}
    for path in sorted(growth_dir.glob("assembly*.pkl"), key=lambda file: (file.stat().st_mtime, file.name)):
        match = pattern.search(path.name)
        if match:
            latest_by_count[int(match.group(1))] = path
    return latest_by_count


def analyze_case(pristine, seed, final, growth_dir: Path, tolerance: float):
    p_bdc = centers_for_type(pristine, "Ligand")
    p_zr = centers_for_type(pristine, "Zr")
    s_bdc = centers_for_type(seed, "Ligand")
    s_zr = centers_for_type(seed, "Zr")
    f_bdc = centers_for_type(final, "Ligand")
    f_zr = centers_for_type(final, "Zr")

    missing_bdc_sites = p_bdc[nearest_distances(p_bdc, s_bdc) > tolerance]
    missing_zr_sites = p_zr[nearest_distances(p_zr, s_zr) > tolerance]

    final_missing_bdc_dist = nearest_distances(missing_bdc_sites, f_bdc)
    final_missing_zr_dist = nearest_distances(missing_zr_sites, f_zr)

    metrics = {
        "missing_bdc_sites": int(len(missing_bdc_sites)),
        "missing_zr_sites": int(len(missing_zr_sites)),
        "filled_bdc_sites": int((final_missing_bdc_dist <= tolerance).sum()),
        "filled_zr_sites": int((final_missing_zr_dist <= tolerance).sum()),
        "pristine_bdc_sites_recovered": int((nearest_distances(p_bdc, f_bdc) <= tolerance).sum()),
        "pristine_zr_sites_recovered": int((nearest_distances(p_zr, f_zr) <= tolerance).sum()),
        "final_new_bdc_sites": int((nearest_distances(f_bdc, p_bdc) > tolerance).sum()),
        "final_new_zr_sites": int((nearest_distances(f_zr, p_zr) > tolerance).sum()),
        "first_full_fill_entity_count": "",
        "first_full_fill_file": "",
    }

    for entity_count, pkl_path in sorted(latest_pkls_by_entity_count(growth_dir).items()):
        assembly = quiet_pickle_load(pkl_path)
        bdc_centers = centers_for_type(assembly, "Ligand")
        zr_centers = centers_for_type(assembly, "Zr")
        filled_bdc = int((nearest_distances(missing_bdc_sites, bdc_centers) <= tolerance).sum())
        filled_zr = int((nearest_distances(missing_zr_sites, zr_centers) <= tolerance).sum())
        if filled_bdc == len(missing_bdc_sites) and filled_zr == len(missing_zr_sites):
            metrics["first_full_fill_entity_count"] = entity_count
            metrics["first_full_fill_file"] = pkl_path.name
            break

    return metrics


def run_growth(
    python_executable: Path,
    seed_pkl: Path,
    output_dir: Path,
    growth_seed: int,
    max_entities: int,
    output_inter: int,
    total_steps: int,
    zr_conc: float,
    linker_conc: float,
    capping_agent_conc: float,
    max_sim_time_seconds: float | None,
    dissolution_update_interval_steps: int,
    exchange_rxn_time_seconds: float,
    log_path: Path,
):
    command = [
        str(python_executable),
        str(SCRIPT_DIR / "run_seeded_growth.py"),
        "--seed-pkl",
        str(seed_pkl),
        "--output-dir",
        str(output_dir),
        "--rng-seed",
        str(growth_seed),
        "--max-entities",
        str(max_entities),
        "--output-inter",
        str(output_inter),
        "--total-steps",
        str(total_steps),
        "--zr-conc",
        str(zr_conc),
        "--linker-conc",
        str(linker_conc),
        "--capping-agent-conc",
        str(capping_agent_conc),
    ]
    if max_sim_time_seconds is not None:
        command.extend(["--max-sim-time-seconds", str(max_sim_time_seconds)])
    if dissolution_update_interval_steps is not None:
        command.extend(
            [
                "--dissolution-update-interval-steps",
                str(dissolution_update_interval_steps),
            ]
        )
    command.extend(
        [
            "--exchange-rxn-time-seconds",
            str(exchange_rxn_time_seconds),
        ]
    )
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=str(SCRIPT_DIR),
            check=True,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def generate_seed(
    input_path: Path,
    seed_dir: Path,
    structure_label: str,
    defect_type: str,
    defect_ratio: float,
    defect_seed: int,
    shell_thickness: float,
):
    case_stem = build_seed_stem(structure_label, defect_type, defect_ratio, defect_seed)
    mol2_path = seed_dir / f"{case_stem}.mol2"
    pkl_path = seed_dir / f"{case_stem}.pkl"
    metadata_path = seed_dir / f"{case_stem}.json"
    log_path = seed_dir / f"{case_stem}.generate.log"

    with log_path.open("w", encoding="utf-8") as handle:
        with redirect_stdout(handle):
            generate_defects(
                input_file=str(input_path),
                output_mol2=str(mol2_path),
                output_pkl=str(pkl_path),
                defect_ratio=defect_ratio,
                shell_thickness=shell_thickness,
                entity_type=defect_type,
                random_seed=defect_seed,
            )

    metadata_path.write_text(
        json.dumps(
            {
                "input_file": str(input_path),
                "structure_label": structure_label,
                "defect_type": defect_type,
                "defect_ratio": defect_ratio,
                "defect_seed": defect_seed,
                "shell_thickness": shell_thickness,
                "output_mol2": str(mol2_path),
                "output_pkl": str(pkl_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return case_stem, mol2_path, pkl_path


def compute_shell_counts(pristine, shell_thickness):
    center = calculate_structure_center_from_entities(pristine.entities)
    shell_entities, _max_radius = identify_shell_entities_from_objects(pristine.entities, center, shell_thickness)
    interior_entities = [entity for entity in pristine.entities if entity not in shell_entities]
    return {
        "interior_total": len(interior_entities),
        "interior_bdc": sum(1 for entity in interior_entities if getattr(entity, "entity_type", None) == "Ligand"),
        "interior_zr": sum(1 for entity in interior_entities if getattr(entity, "entity_type", None) == "Zr"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a defect matrix, run seeded growth, and summarize defect-healing outcomes."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Pristine MOL2 structure used for all experiments.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root output directory for the experiment matrix.")
    parser.add_argument("--ratios", default="0.2,0.4,0.6,0.8", help="Comma-separated defect ratios.")
    parser.add_argument("--defect-types", default=",".join(DEFAULT_DEFECT_TYPES), help="Comma-separated defect types: BDC,Zr6,Mixed.")
    parser.add_argument("--shell-thickness", type=float, default=20.0, help="Preserved outer-shell thickness in Angstroms.")
    parser.add_argument("--growth-buffer", type=int, default=20, help="Allow growth to pristine_total + buffer entities.")
    parser.add_argument("--output-inter", type=int, default=5, help="Intermediate-save interval for seeded growth.")
    parser.add_argument("--total-steps", type=int, default=200000, help="Maximum KMC steps per seeded-growth run.")
    parser.add_argument(
        "--max-sim-time-seconds",
        type=float,
        default=None,
        help="Optional physical-time cutoff in seconds. If set, growth stops when simulated time reaches this value.",
    )
    parser.add_argument(
        "--dissolution-update-interval-steps",
        type=int,
        default=None,
        help="Optional interval for recomputing dissolution/growth balance. Default keeps the historical one-time initialization.",
    )
    parser.add_argument(
        "--exchange-rxn-time-seconds",
        type=float,
        default=GROWTH_DEFAULTS["EXCHANGE_RXN_TIME_SECONDS"],
        help="Physical-time multiplier applied to the KMC timing variable.",
    )
    parser.add_argument(
        "--zr-conc-multipliers",
        default=",".join(str(value) for value in DEFAULT_ZR_CONC_MULTIPLIERS),
        help="Comma-separated multipliers applied to the default Zr concentration.",
    )
    parser.add_argument(
        "--linker-conc-multipliers",
        default=",".join(str(value) for value in DEFAULT_LINKER_CONC_MULTIPLIERS),
        help="Comma-separated multipliers applied to the default linker concentration.",
    )
    parser.add_argument(
        "--capping-agent-conc-multipliers",
        default=",".join(str(value) for value in DEFAULT_CAPPING_CONC_MULTIPLIERS),
        help="Comma-separated multipliers applied to the default capping-agent concentration.",
    )
    parser.add_argument("--defect-seed-base", type=int, default=200, help="Base RNG seed for defect generation.")
    parser.add_argument("--growth-seed-base", type=int, default=1200, help="Base RNG seed for seeded growth.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used for seeded-growth subprocesses.")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Matching tolerance in Angstroms for site recovery analysis.")
    return parser.parse_args()


def row_key(row):
    if row.get("case_stem"):
        return row["case_stem"]
    return "|".join(
        [
            str(row.get("structure_label", "")),
            str(row.get("defect_type", "")),
            str(row.get("defect_ratio", "")),
            str(row.get("defect_seed", "")),
        ]
    )


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (SCRIPT_DIR / input_path).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (SCRIPT_DIR / output_root).resolve()
    python_executable = Path(args.python_exe)

    ratios = parse_csv_list(args.ratios, float)
    defect_types = parse_csv_list(args.defect_types, str)
    zr_conc_multipliers = parse_csv_list(args.zr_conc_multipliers, float)
    linker_conc_multipliers = parse_csv_list(args.linker_conc_multipliers, float)
    capping_conc_multipliers = parse_csv_list(args.capping_agent_conc_multipliers, float)
    invalid_types = [value for value in defect_types if value not in {"BDC", "Zr6", "Mixed"}]
    if invalid_types:
        raise ValueError(f"Unsupported defect types: {invalid_types}")

    seed_dir = output_root / "seeds"
    growth_root = output_root / "growth"
    seed_dir.mkdir(parents=True, exist_ok=True)
    growth_root.mkdir(parents=True, exist_ok=True)

    structure_label = infer_structure_label(input_path)
    pristine = quiet_build_assembly_from_mol2(input_path)
    pristine_total, pristine_bdc, pristine_zr = assembly_counts(pristine)
    shell_counts = compute_shell_counts(pristine, args.shell_thickness)

    print("=" * 72)
    print("Defect Growth Matrix")
    print("=" * 72)
    print(f"input            : {input_path}")
    print(f"structure_label  : {structure_label}")
    print(f"shell_thickness  : {args.shell_thickness:.2f}")
    print(f"pristine_total   : {pristine_total}")
    print(f"pristine_bdc     : {pristine_bdc}")
    print(f"pristine_zr      : {pristine_zr}")
    print(f"interior_bdc     : {shell_counts['interior_bdc']}")
    print(f"interior_zr      : {shell_counts['interior_zr']}")
    print(f"ratios           : {ratios}")
    print(f"defect_types     : {defect_types}")
    print(f"zr_mult          : {zr_conc_multipliers}")
    print(f"linker_mult      : {linker_conc_multipliers}")
    print(f"capping_mult     : {capping_conc_multipliers}")
    print("=" * 72)

    rows = []
    case_index = 0
    for defect_type in defect_types:
        for defect_ratio in ratios:
            defect_seed = args.defect_seed_base + case_index
            case_stem_seed, seed_mol2, seed_pkl = generate_seed(
                input_path=input_path,
                seed_dir=seed_dir,
                structure_label=structure_label,
                defect_type=defect_type,
                defect_ratio=defect_ratio,
                defect_seed=defect_seed,
                shell_thickness=args.shell_thickness,
            )
            seed = quiet_pickle_load(seed_pkl)
            seed_total, seed_bdc, seed_zr = assembly_counts(seed)

            for zr_multiplier in zr_conc_multipliers:
                for linker_multiplier in linker_conc_multipliers:
                    for capping_multiplier in capping_conc_multipliers:
                        growth_seed = args.growth_seed_base + case_index
                        case_index += 1
                        zr_conc = GROWTH_DEFAULTS["Zr_conc"] * zr_multiplier
                        linker_conc = GROWTH_DEFAULTS["Linker_conc"] * linker_multiplier
                        capping_agent_conc = GROWTH_DEFAULTS["Capping_agent_conc"] * capping_multiplier
                        case_stem = build_case_stem(
                            structure_label,
                            defect_type,
                            defect_ratio,
                            defect_seed,
                            zr_multiplier=zr_multiplier,
                            linker_multiplier=linker_multiplier,
                            capping_multiplier=capping_multiplier,
                        )

                        row = {
                            "status": "ok",
                            "case_stem": case_stem,
                            "seed_case_stem": case_stem_seed,
                            "structure_label": structure_label,
                            "input_file": str(input_path),
                            "shell_thickness": args.shell_thickness,
                            "defect_type": defect_type,
                            "defect_ratio": defect_ratio,
                            "defect_seed": defect_seed,
                            "growth_seed": growth_seed,
                            "zr_conc_multiplier": zr_multiplier,
                            "zr_conc": zr_conc,
                            "linker_conc_multiplier": linker_multiplier,
                            "capping_agent_conc_multiplier": capping_multiplier,
                            "linker_conc": linker_conc,
                            "capping_agent_conc": capping_agent_conc,
                            "pristine_total": pristine_total,
                            "pristine_bdc": pristine_bdc,
                            "pristine_zr": pristine_zr,
                            "interior_bdc": shell_counts["interior_bdc"],
                            "interior_zr": shell_counts["interior_zr"],
                            "max_entities": pristine_total + args.growth_buffer,
                            "max_sim_time_seconds": args.max_sim_time_seconds,
                            "dissolution_update_interval_steps": args.dissolution_update_interval_steps,
                            "exchange_rxn_time_seconds": args.exchange_rxn_time_seconds,
                            "seed_mol2": str(seed_mol2),
                            "seed_pkl": str(seed_pkl),
                            "seed_total": seed_total,
                            "seed_bdc": seed_bdc,
                            "seed_zr": seed_zr,
                        }

                        try:
                            growth_dir = growth_root / case_stem
                            run_growth(
                                python_executable=python_executable,
                                seed_pkl=seed_pkl,
                                output_dir=growth_dir,
                                growth_seed=growth_seed,
                                max_entities=pristine_total + args.growth_buffer,
                                output_inter=args.output_inter,
                                total_steps=args.total_steps,
                                zr_conc=zr_conc,
                                linker_conc=linker_conc,
                                capping_agent_conc=capping_agent_conc,
                                max_sim_time_seconds=args.max_sim_time_seconds,
                                dissolution_update_interval_steps=args.dissolution_update_interval_steps,
                                exchange_rxn_time_seconds=args.exchange_rxn_time_seconds,
                                log_path=growth_dir.with_suffix(".growth.log"),
                            )

                            final_pkl = latest_final_pkl(growth_dir)
                            final = quiet_pickle_load(final_pkl)
                            final_total, final_bdc, final_zr = assembly_counts(final)
                            metrics = analyze_case(pristine, seed, final, growth_dir, args.tolerance)

                            row.update(
                                {
                                    "growth_dir": str(growth_dir),
                                    "final_pkl": str(final_pkl),
                                    "final_mol2": str(growth_dir / "assembly.mol2"),
                                    "final_total": final_total,
                                    "final_bdc": final_bdc,
                                    "final_zr": final_zr,
                                    **metrics,
                                }
                            )
                            print(
                                f"[{case_stem}] missing BDC/Zr = {metrics['missing_bdc_sites']}/{metrics['missing_zr_sites']}, "
                                f"filled = {metrics['filled_bdc_sites']}/{metrics['filled_zr_sites']}, "
                                f"first_full_fill = {metrics['first_full_fill_entity_count'] or 'n/a'}"
                            )
                        except Exception as exc:
                            row["status"] = "failed"
                            row["error"] = str(exc)
                            print(
                                f"[FAILED] {case_stem}: {exc}"
                            )

                        rows.append(row)

    summary_json = output_root / "summary.json"
    if summary_json.exists():
        existing_rows = json.loads(summary_json.read_text(encoding="utf-8"))
    else:
        existing_rows = []

    merged_rows = {row_key(row): row for row in existing_rows}
    for row in rows:
        merged_rows[row_key(row)] = row
    rows = list(merged_rows.values())

    summary_csv = output_root / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    metadata_json = output_root / "matrix_metadata.json"
    metadata_json.write_text(
        json.dumps(
            {
                "input_file": str(input_path),
                "structure_label": structure_label,
                "shell_thickness": args.shell_thickness,
                "pristine_total": pristine_total,
                "pristine_bdc": pristine_bdc,
                "pristine_zr": pristine_zr,
                "interior_bdc": shell_counts["interior_bdc"],
                "interior_zr": shell_counts["interior_zr"],
                "ratios": ratios,
                "defect_types": defect_types,
                "zr_conc_multipliers": zr_conc_multipliers,
                "default_zr_conc": GROWTH_DEFAULTS["Zr_conc"],
                "linker_conc_multipliers": linker_conc_multipliers,
                "capping_agent_conc_multipliers": capping_conc_multipliers,
                "default_linker_conc": GROWTH_DEFAULTS["Linker_conc"],
                "default_capping_agent_conc": GROWTH_DEFAULTS["Capping_agent_conc"],
                "growth_buffer": args.growth_buffer,
                "output_inter": args.output_inter,
                "total_steps": args.total_steps,
                "max_sim_time_seconds": args.max_sim_time_seconds,
                "dissolution_update_interval_steps": args.dissolution_update_interval_steps,
                "exchange_rxn_time_seconds": args.exchange_rxn_time_seconds,
                "defect_seed_base": args.defect_seed_base,
                "growth_seed_base": args.growth_seed_base,
                "python_executable": str(python_executable),
                "tolerance": args.tolerance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSummary files")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")
    print(f"  {metadata_json}")


if __name__ == "__main__":
    main()
