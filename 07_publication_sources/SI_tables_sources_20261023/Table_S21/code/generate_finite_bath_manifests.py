from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OBSERVATIONS = ROOT / "project_seed_uptake_observations.csv"


SYSTEMS = {
    "OP_UIO_SEED_UPTAKE": {
        "system": "uio66",
        "topology_mode": "unrestricted",
        "seed_class": "uio66",
        "volume_mL": 5.0,
        "seed_reference_linker_umol": 15.96,
        "fixed_metal_mM": 4.0,
        "fixed_linker_mM": 4.0,
        "linker_identity": "BDC",
        "linker_pka": 4.74,
        "linker_formate_exchange_constant": 1.64,
        "modulator_identity": "acetic acid",
        "modulator_concentration_mM": 593.8,
        "modulator_pka": 4.76,
        "modulator_formate_exchange_constant": 1.20,
        "solvent_weight": 124.6,
        "water_concentration_mM": 0.0,
        "reaction_time_min": 60.0,
    },
    "OP_BTB_SEED_UPTAKE": {
        "system": "btb_mol",
        "topology_mode": "planar_mol",
        "seed_class": "btb_planar",
        "volume_mL": 1.645,
        "seed_reference_linker_umol": 5.8,
        "fixed_metal_mM": 27.0644103969866,
        "fixed_linker_mM": 16.0,
        "linker_identity": "BTB",
        "linker_pka": 4.74,
        "linker_formate_exchange_constant": 1.64,
        "modulator_identity": "formic acid",
        "modulator_concentration_mM": 805.5,
        "modulator_pka": 3.75,
        "modulator_formate_exchange_constant": 1.0,
        "solvent_weight": 164.0,
        "water_concentration_mM": 5061.0,
        "reaction_time_min": 45.0,
    },
}


BASE = {
    "formic_acid_pka": 3.75,
    "formate_to_acid_ratio": 0.1,
    "acid_to_carboxylate_solvent_exchange_ratio": 1.0,
    "linker_accessibility": 1.0,
    "modulator_accessibility": 1.0,
    "reactive_m6_fraction": 1.0,
    "node_capture_coefficient_per_mM": 1.0,
    "tethered_linker_weight": 10.0,
    "opened_site_rate_ratio": 1.0,
    "bond_opening_rate_ratio": 0.006,
    "solvent_displacement_efficiency": 1.0,
    "closure_coefficient": 0.5,
    "batch_seed_count": 8,
    "event_cap": 200000,
    "wall_time_cap_s": 180.0,
    "recycle_detached_new_entities": True,
}


def observation_rows() -> list[dict[str, str]]:
    with OBSERVATIONS.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["operation_id"] in SYSTEMS]


def manifest_row(
    observation: dict[str, str],
    *,
    mechanism_id: str,
    repeat: int,
    model_time_horizon: float,
    overrides: dict[str, float] | None = None,
) -> dict[str, object]:
    setup = SYSTEMS[observation["operation_id"]]
    varied = observation["varied_component"]
    concentration = float(observation["varied_concentration_mM"])
    metal = concentration if varied == "metal" else setup["fixed_metal_mM"]
    linker = concentration if varied == "linker" else setup["fixed_linker_mM"]
    values = {**setup, **BASE, **(overrides or {})}
    values["solvent_weight"] = setup["solvent_weight"] * values.pop(
        "solvent_weight_scale", 1.0
    )
    condition_token = f"{concentration:.8g}".replace(".", "p")
    run_id = (
        f"{setup['system']}_{varied}_{condition_token}_{mechanism_id}"
        f"_t{model_time_horizon:g}_r{repeat:02d}"
    ).replace(".", "p")
    return {
        "run_id": run_id,
        "mechanism_id": mechanism_id,
        "operation_id": observation["operation_id"],
        "series": observation["series"],
        "varied_component": varied,
        "varied_concentration_mM": concentration,
        "observed_uptake_ratio": float(observation["uptake_ratio"]),
        "system": setup["system"],
        "topology_mode": setup["topology_mode"],
        "seed_class": setup["seed_class"],
        "volume_mL": setup["volume_mL"],
        "seed_reference_linker_umol": setup["seed_reference_linker_umol"],
        "analytical_atomic_metal_mM": metal,
        "linker_identity": setup["linker_identity"],
        "linker_concentration_mM": linker,
        "linker_pka": setup["linker_pka"],
        "linker_formate_exchange_constant": setup[
            "linker_formate_exchange_constant"
        ],
        "linker_accessibility": values["linker_accessibility"],
        "modulator_identity": setup["modulator_identity"],
        "modulator_concentration_mM": setup["modulator_concentration_mM"],
        "modulator_pka": setup["modulator_pka"],
        "modulator_formate_exchange_constant": setup[
            "modulator_formate_exchange_constant"
        ],
        "modulator_accessibility": values["modulator_accessibility"],
        "water_concentration_mM": setup["water_concentration_mM"],
        "reaction_time_min": setup["reaction_time_min"],
        "solvent_weight": values["solvent_weight"],
        "formic_acid_pka": values["formic_acid_pka"],
        "formate_to_acid_ratio": values["formate_to_acid_ratio"],
        "acid_to_carboxylate_solvent_exchange_ratio": values[
            "acid_to_carboxylate_solvent_exchange_ratio"
        ],
        "reactive_m6_fraction": values["reactive_m6_fraction"],
        "node_capture_coefficient_per_mM": values[
            "node_capture_coefficient_per_mM"
        ],
        "tethered_linker_weight": values["tethered_linker_weight"],
        "opened_site_rate_ratio": values["opened_site_rate_ratio"],
        "bond_opening_rate_ratio": values["bond_opening_rate_ratio"],
        "solvent_displacement_efficiency": values[
            "solvent_displacement_efficiency"
        ],
        "closure_coefficient": values["closure_coefficient"],
        "batch_seed_count": values["batch_seed_count"],
        "model_time_horizon": model_time_horizon,
        "event_cap": values["event_cap"],
        "wall_time_cap_s": values["wall_time_cap_s"],
        "recycle_detached_new_entities": values[
            "recycle_detached_new_entities"
        ],
        "repeat": repeat,
        "random_seed": 9300000
        + repeat
        + int(round(concentration * 1000))
        + (0 if setup["system"] == "uio66" else 200000),
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("empty manifest")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    observations = observation_rows()
    horizon_rows = [
        manifest_row(
            observation,
            mechanism_id="fixed_speciation_central",
            repeat=repeat,
            model_time_horizon=horizon,
        )
        for observation in observations
        for horizon in (1.0, 10.0, 100.0, 1000.0)
        for repeat in range(4)
    ]
    write_manifest(ROOT / "manifest_finite_bath_horizon_pilot_v1.csv", horizon_rows)

    scenarios: list[tuple[str, dict[str, float]]] = [
        ("central", {}),
        ("reactive_m6_0p1", {"reactive_m6_fraction": 0.1}),
        ("reactive_m6_0p3", {"reactive_m6_fraction": 0.3}),
        ("formate_ratio_0p01", {"formate_to_acid_ratio": 0.01}),
        ("formate_ratio_1", {"formate_to_acid_ratio": 1.0}),
        ("acid_solvent_ratio_0p1", {"acid_to_carboxylate_solvent_exchange_ratio": 0.1}),
        ("acid_solvent_ratio_10", {"acid_to_carboxylate_solvent_exchange_ratio": 10.0}),
        ("solvent_0p3", {"solvent_weight_scale": 0.3}),
        ("solvent_3", {"solvent_weight_scale": 3.0}),
        ("tethered_3", {"tethered_linker_weight": 3.0}),
        ("tethered_30", {"tethered_linker_weight": 30.0}),
        ("bond_open_0p001", {"bond_opening_rate_ratio": 0.001}),
        ("bond_open_0p03", {"bond_opening_rate_ratio": 0.03}),
        ("node_capture_0p3", {"node_capture_coefficient_per_mM": 0.3}),
        ("node_capture_3", {"node_capture_coefficient_per_mM": 3.0}),
        ("closure_0", {"closure_coefficient": 0.0}),
        ("closure_0p1", {"closure_coefficient": 0.1}),
        ("closure_1", {"closure_coefficient": 1.0}),
        ("closure_1p5", {"closure_coefficient": 1.5}),
        ("closure_2", {"closure_coefficient": 2.0}),
    ]
    mechanism_rows = [
        manifest_row(
            observation,
            mechanism_id=mechanism_id,
            repeat=repeat,
            model_time_horizon=100.0,
            overrides=overrides,
        )
        for observation in observations
        for mechanism_id, overrides in scenarios
        for repeat in range(4)
    ]
    write_manifest(ROOT / "manifest_finite_bath_mechanism_pilot_v1.csv", mechanism_rows)


if __name__ == "__main__":
    main()
