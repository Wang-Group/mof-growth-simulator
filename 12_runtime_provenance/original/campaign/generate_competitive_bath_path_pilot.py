"""Generate concentration/modulator paths in explicit competitive-bath weights.

The scan is nondimensional but chemically interpretable.  Solvent weight is
the reference (1).  ``c`` is linker competitive weight relative to solvent,
``r`` is modulator/linker competitive weight, and atomic metal/linker ratio is
fixed at the ideal framework value for each system.  No hot-DMF equilibrium
constant is selected at this stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SYSTEMS = (
    ("uio66", "unrestricted", "uio66", 1.0),
    ("btb_mol", "planar_mol", "btb_planar", 3.0),
)
MODES = ("all_bonds", "undercoordinated_metal")
FREE_LINKER_FATES = ("topology_loss", "protective_replacement")
C_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
R_VALUES = (0.0, 0.1, 1.0, 10.0, 100.0)
BOND_OPENING = (0.1, 1.0, 10.0)
SC_VALUES = (0.5, 1.0)
REPLICATES = 4
EVENT_CAP = 30_000
WALL_TIME_CAP_S = 30.0
SOLVENT_WEIGHT = 1.0
TETHERED_LINKER_WEIGHT = 1.0
SOLVENT_CAPPING_EFFICIENCY = 1.0


def coordinates(
    q: float, c: float, r: float, opening: float, free_linker_fate: str
) -> tuple[float, float, float, float, float]:
    linker_weight = c
    modulator_weight = r * c
    node_entity_activity = q * c / 6.0
    linker_activity = linker_weight / (linker_weight + modulator_weight + SOLVENT_WEIGHT)
    node_activity = (
        node_entity_activity
        * TETHERED_LINKER_WEIGHT
        / (TETHERED_LINKER_WEIGHT + linker_weight + modulator_weight + SOLVENT_WEIGHT)
    )
    opened_bond_denominator = (
        TETHERED_LINKER_WEIGHT
        + linker_weight
        + modulator_weight
        + SOLVENT_WEIGHT
    )
    loss_weight = modulator_weight + SOLVENT_CAPPING_EFFICIENCY * SOLVENT_WEIGHT
    if free_linker_fate == "topology_loss":
        loss_weight += linker_weight
    elif free_linker_fate != "protective_replacement":
        raise ValueError(f"unknown free-linker fate {free_linker_fate!r}")
    off_weight = opening * loss_weight / opened_bond_denominator
    external_activity = node_activity + linker_activity
    node_fraction = node_activity / external_activity
    return external_activity, node_fraction, off_weight, node_activity, linker_activity


def stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") & 0x7FFFFFFF


def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    de_novo: list[dict[str, object]] = []
    seeded: list[dict[str, object]] = []
    for system, topology, operation, q in SYSTEMS:
        for mode in MODES:
            for fate in FREE_LINKER_FATES:
                for ci, c in enumerate(C_VALUES):
                    for ri, r in enumerate(R_VALUES):
                        for bi, opening in enumerate(BOND_OPENING):
                            external, node_fraction, off, node_activity, linker_activity = coordinates(
                                q, c, r, opening, fate
                            )
                            for si, sc in enumerate(SC_VALUES):
                                fate_tag = "loss" if fate == "topology_loss" else "protect"
                                condition = f"{operation}_{mode}_{fate_tag}_c{ci}_r{ri}_b{bi}_s{si}"
                                common = {
                                "condition_id": condition,
                                "system": system,
                                "topology_mode": topology,
                                "bond_accessibility_mode": mode,
                                "atomic_metal_to_linker_ratio": q,
                                "node_fraction": node_fraction,
                                "external_activity": external,
                                "off_weight": off,
                                "SC": sc,
                                "m6_node_fraction": 1.0,
                                "site_opening_seconds": 0.0002405,
                                "bath_linker_to_solvent_weight": c,
                                "bath_modulator_to_linker_weight": r,
                                "bath_bond_opening_scale": opening,
                                "bath_free_linker_exchange_fate": fate,
                                "bath_solvent_weight": SOLVENT_WEIGHT,
                                "bath_tethered_linker_weight": TETHERED_LINKER_WEIGHT,
                                "bath_solvent_capping_efficiency": SOLVENT_CAPPING_EFFICIENCY,
                                "bath_node_activity": node_activity,
                                "bath_linker_activity": linker_activity,
                                "bath_mapping_equations": "WL=c; WM=r*c; WS=1; CM6=q*c/6; aL=WL/(WL+WM+WS); aM=CM6*T/(T+WL+WM+WS); Doff=T+WL+WM+WS; aoff=b*(WM+eta*WS+[WL if topology_loss else 0])/Doff",
                                }
                                for replicate in range(REPLICATES):
                                    de_novo.append(
                                    {
                                        "run_id": f"denovo_{condition}_r{replicate:02d}",
                                        "design_block": "competitive_bath_path_pilot_v2",
                                        **common,
                                        "replicate": replicate,
                                        "seed": stable_seed(f"de-novo|{condition}|{replicate}"),
                                        "composition_mapping": "explicit_competitive_bath_path",
                                        "target_entities": 30,
                                        "event_cap": EVENT_CAP,
                                        "wall_time_cap_s": WALL_TIME_CAP_S,
                                        "bumping_threshold": 2.0,
                                        "max_depth": 31,
                                    }
                                    )
                                    seed_dir = "uio66" if system == "uio66" else "btb_planar"
                                    seed_name = "uio66" if system == "uio66" else "btb_mol"
                                    seeded.append(
                                    {
                                        "run_id": f"seeded_{condition}_r{replicate:02d}",
                                        "design_block": "competitive_bath_seeded_path_pilot_v2",
                                        **common,
                                        "composition_mapping": "explicit_competitive_bath_path",
                                        "seed_index": replicate,
                                        "seed_pickle": (
                                            "/home/wave/MOL_growth_recalculation_20260830/"
                                            f"frozen_seed_ensemble_v2/{seed_dir}/{seed_name}_seed_{replicate:02d}.pkl"
                                        ),
                                        "seed_generation_random_seed": "",
                                        "initial_seed_entities": 20,
                                        "continuation_replicate": 0,
                                        "replicate": replicate,
                                        "random_seed": stable_seed(f"seeded|{condition}|{replicate}"),
                                        "target_entities": 30,
                                        "event_cap": EVENT_CAP,
                                        "wall_time_cap_s": WALL_TIME_CAP_S,
                                        "seed_retention_threshold": 0.8,
                                    }
                                    )
    write(HERE / "manifest_competitive_bath_de_novo_pilot_v2.csv", de_novo)
    write(HERE / "manifest_competitive_bath_seeded_pilot_v2.csv", seeded)
    design = {
        "schema_version": "competitive-bath-path-pilot-v2",
        "systems": [x[2] for x in SYSTEMS],
        "bond_accessibility_modes": MODES,
        "free_linker_exchange_fates": FREE_LINKER_FATES,
        "linker_to_solvent_weight": C_VALUES,
        "modulator_to_linker_weight": R_VALUES,
        "bond_opening_scale": BOND_OPENING,
        "closure_coefficient": SC_VALUES,
        "replicates": REPLICATES,
        "event_cap": EVENT_CAP,
        "wall_time_cap_s": WALL_TIME_CAP_S,
        "de_novo_traces": len(de_novo),
        "seeded_traces": len(seeded),
        "interpretation": "c changes absolute solute competition against solvent; r changes modulator dose/identity through its effective coordination weight.",
        "claim_boundary": "The scan is a nondimensional mechanism test. Actual molar recipes require uncertain exchange constants and proton activity to map concentration into c and r. Runs without a persistent event before either cap are treated as right-censored exposure, not as zero-rate observations.",
    }
    (HERE / "competitive_bath_path_pilot_design_v2.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(design, indent=2))


if __name__ == "__main__":
    main()
