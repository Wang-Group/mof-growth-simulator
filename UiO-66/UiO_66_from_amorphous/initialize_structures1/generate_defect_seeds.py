import argparse
import json
import re
from pathlib import Path

from generate_defects import generate_defects
from UiO66_Assembly_Large_Correction_conc import safe_pickle_load


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "UiO-66_15x15x15_sphere_R1.mol2"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_SEEDS = [11, 22]
DEFAULT_ENTITY_TYPES = ["BDC", "Zr6", "Mixed"]


def parse_csv_list(raw_value, cast=str):
    items = []
    for part in raw_value.split(","):
        value = part.strip()
        if value:
            items.append(cast(value))
    return items


def infer_structure_label(input_path: Path) -> str:
    name = input_path.stem
    match = re.search(r"_sphere_(R[\d.]+)$", name)
    if match:
        return match.group(1)
    match = re.search(r"_octahedron.*$", name)
    if match:
        return "octa"
    return name


def defect_tag(entity_type):
    if entity_type == "BDC":
        return "BDCdef"
    if entity_type == "Zr6":
        return "Zr6def"
    return "Mixeddef"


def format_ratio(defect_ratio):
    return f"{defect_ratio:.2f}"


def write_roundtrip_mol2(pkl_path: Path, output_path: Path):
    assembly = safe_pickle_load(str(pkl_path), rebuild_references=True)
    if assembly is None:
        raise RuntimeError(f"Failed to load seed PKL: {pkl_path}")
    assembly.get_mol2_file(str(output_path))


def write_metadata(metadata_path: Path, payload):
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_output_stem(input_path: Path, entity_type, defect_ratio, seed, label=None):
    structure_label = label or infer_structure_label(input_path)
    return f"UiO-66_{structure_label}_{defect_tag(entity_type)}_{format_ratio(defect_ratio)}_seed{seed}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a batch of UiO-66 defect seed MOL2/PKL files."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input MOL2 structure used as the starting seed.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated seed files.",
    )
    parser.add_argument(
        "--defect-ratio",
        type=float,
        default=0.10,
        help="Fraction of interior entities to remove.",
    )
    parser.add_argument(
        "--shell-thickness",
        type=float,
        default=10.0,
        help="Outer shell thickness in Angstroms to preserve.",
    )
    parser.add_argument(
        "--entity-types",
        default=",".join(DEFAULT_ENTITY_TYPES),
        help="Comma-separated entity types: BDC,Zr6 or a subset.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated RNG seeds.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional structure label used in output names, e.g. R1.",
    )
    parser.add_argument(
        "--verify-pkl-roundtrip",
        action="store_true",
        help="Write an additional *_from_pkl.mol2 file from each generated PKL.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing seed files if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (SCRIPT_DIR / input_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input structure not found: {input_path}")
    if not 0.0 <= args.defect_ratio <= 1.0:
        raise ValueError("--defect-ratio must be between 0 and 1")

    entity_types = parse_csv_list(args.entity_types, str)
    invalid_types = [value for value in entity_types if value not in {"BDC", "Zr6", "Mixed"}]
    if invalid_types:
        raise ValueError(f"Unsupported entity types: {invalid_types}")

    seeds = parse_csv_list(args.seeds, int)
    if not seeds:
        raise ValueError("At least one RNG seed is required")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Generating UiO-66 defect seeds")
    print("=" * 72)
    print(f"input            : {input_path}")
    print(f"output_dir       : {output_dir}")
    print(f"defect_ratio     : {args.defect_ratio:.2f}")
    print(f"shell_thickness  : {args.shell_thickness:.2f}")
    print(f"entity_types     : {entity_types}")
    print(f"rng_seeds        : {seeds}")
    print(f"verify_roundtrip : {args.verify_pkl_roundtrip}")
    print("=" * 72)

    generated = []
    skipped = []

    for entity_type in entity_types:
        for seed in seeds:
            stem = build_output_stem(
                input_path=input_path,
                entity_type=entity_type,
                defect_ratio=args.defect_ratio,
                seed=seed,
                label=args.label,
            )
            mol2_path = output_dir / f"{stem}.mol2"
            pkl_path = output_dir / f"{stem}.pkl"
            roundtrip_path = output_dir / f"{stem}_from_pkl.mol2"
            metadata_path = output_dir / f"{stem}.json"

            if not args.overwrite and mol2_path.exists() and pkl_path.exists():
                skipped.append(stem)
                print(f"Skipping existing seed: {stem}")
                continue

            print(f"\nCreating seed: {stem}")
            generate_defects(
                input_file=str(input_path),
                output_mol2=str(mol2_path),
                output_pkl=str(pkl_path),
                defect_ratio=args.defect_ratio,
                shell_thickness=args.shell_thickness,
                entity_type=entity_type,
                random_seed=seed,
            )

            if args.verify_pkl_roundtrip:
                print(f"Writing round-trip MOL2: {roundtrip_path.name}")
                write_roundtrip_mol2(pkl_path, roundtrip_path)

            write_metadata(
                metadata_path,
                {
                    "input_file": str(input_path),
                    "output_mol2": str(mol2_path),
                    "output_pkl": str(pkl_path),
                    "output_roundtrip_mol2": str(roundtrip_path) if args.verify_pkl_roundtrip else None,
                    "entity_type": entity_type,
                    "defect_ratio": args.defect_ratio,
                    "shell_thickness": args.shell_thickness,
                    "random_seed": seed,
                    "label": args.label or infer_structure_label(input_path),
                },
            )

            generated.append(stem)

    print("\nSummary")
    print(f"generated : {len(generated)}")
    print(f"skipped   : {len(skipped)}")
    if generated:
        for stem in generated:
            print(f"  + {stem}")
    if skipped:
        for stem in skipped:
            print(f"  - {stem}")


if __name__ == "__main__":
    main()
