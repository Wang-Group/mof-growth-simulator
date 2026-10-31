"""Replace worker-regenerated seeds with audited, immutable pickle paths."""

from __future__ import annotations

import csv
from pathlib import Path


CAMPAIGN = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/home/wave/MOL_growth_recalculation_20260830")


def pin_manifest(
    source_name: str,
    destination_name: str,
    design_block: str,
    *,
    planar: bool,
) -> int:
    source = CAMPAIGN / source_name
    destination = CAMPAIGN / destination_name
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for field in ("topology_mode", "m6_node_fraction"):
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        index = int(row["seed_index"])
        system = row["system"]
        if planar:
            if system != "btb_mol":
                raise ValueError("Planar manifest may contain only btb_mol rows")
            subdir = "btb_planar"
            stem = "btb_mol"
            row["topology_mode"] = "planar_mol"
        elif system == "uio66":
            subdir = "uio66"
            stem = "uio66"
            row["topology_mode"] = "unrestricted"
        elif system == "btb_mol":
            subdir = "btb_network"
            stem = "btb_mol"
            row["topology_mode"] = "unrestricted"
        else:
            raise ValueError(f"Unknown system {system}")
        row["seed_pickle"] = (
            REMOTE_ROOT
            / "frozen_seed_ensemble_v2"
            / subdir
            / f"{stem}_seed_{index:02d}.pkl"
        ).as_posix()
        row["seed_generation_random_seed"] = ""
        row["m6_node_fraction"] = "1.0"
        row["design_block"] = design_block
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    counts = {
        "generic": pin_manifest(
            "manifest_seeded_propagation_v1.csv",
            "manifest_seeded_propagation_v2_fixed_seeds.csv",
            "matched_seeded_propagation_v2_fixed_seeds",
            planar=False,
        ),
        "planar": pin_manifest(
            "manifest_btb_planar_seeded_propagation_v1.csv",
            "manifest_btb_planar_seeded_propagation_v2_fixed_seeds.csv",
            "btb_planar_seeded_propagation_v2_fixed_seeds",
            planar=True,
        ),
    }
    print(counts)


if __name__ == "__main__":
    main()
