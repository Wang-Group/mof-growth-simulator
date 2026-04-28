ZR6_PERCENTAGE = None
Zr_conc = None
entropy_correction_coefficient = None
equilibrium_constant_coefficient = None
H2O_DMF_RATIO = None
Capping_agent_conc = None
Linker_conc = None
Total_steps = None
current_folder = None
BUMPING_THRESHOLD = None
max_entities = None
output_inter = None
EXCHANGE_RXN_TIME_SECONDS = None
DISSOLUTION_UPDATE_INTERVAL_STEPS = None
DISTORTED_LINKER_ENABLED = None
DISTORTED_CHEMISTRY_MODEL = None
DISTORTED_LIGAND_ASSOCIATION_CONSTANT = None
DISTORTED_SITE_EQUILIBRIUM_CONSTANT = None
DISTORTED_SECOND_STEP_EQUIVALENTS = None
DISTORTED_NUM_SITES_ON_CLUSTER = None
DISTORTED_NUM_SITES_ON_LINKER = None

import gc
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime

import numpy as np

try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(*args, **kwargs):
        return None

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from MOL_Assembly_Large_Correction_20250811 import *
from distorted_ligand_model import (
    PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
    compute_prebound_chemistry_state,
    default_prebound_state,
    solve_linker_carboxylate_to_acid_ratio,
    zr6_cluster_add_probability,
)


if current_folder is not None:
    os.makedirs(current_folder, exist_ok=True)

"""Set parameters"""

correction_term_for_deprotonation = 10 ** (4.19 - 3.75)

H2O_pure = 55500
DMF_pure = 12900

DMF_conc = DMF_pure / (1 + H2O_DMF_RATIO)
H2O_conc = H2O_pure * H2O_DMF_RATIO / (1 + H2O_DMF_RATIO)
H2O_formate_coefficient = 0.01
DMF_formate_coefficient = 0.01
H2O_formate_coord_power = H2O_conc / Capping_agent_conc * H2O_formate_coefficient
DMF_formate_coord_power = DMF_conc / Capping_agent_conc * DMF_formate_coefficient

num_carboxylate_on_linker = 3

equilibrium_constant = 1.64
effective_equilibrium_constant = (
    equilibrium_constant_coefficient
    * equilibrium_constant
    / (1 + H2O_formate_coord_power + DMF_formate_coord_power)
)

end_DMF_decomposition_conc = 560
exp_time = 3
DMF_decomposition_rate = end_DMF_decomposition_conc / (exp_time * 3.6 * 10 ** 3)

exchange_rxn_time = EXCHANGE_RXN_TIME_SECONDS if EXCHANGE_RXN_TIME_SECONDS is not None else 0.1
LINKER_CONC_FOR_GROWTH = None


def dissolution_probability(time_passed, DMF_decomposition_rate):
    dimethylamine_conc = time_passed * DMF_decomposition_rate
    if dimethylamine_conc > end_DMF_decomposition_conc:
        dimethylamine_conc = end_DMF_decomposition_conc

    effective_linker_conc = (
        max(float(LINKER_CONC_FOR_GROWTH), 0.0)
        if LINKER_CONC_FOR_GROWTH is not None
        else max(float(Linker_conc), 0.0)
    )
    if effective_linker_conc <= 0.0:
        return 1.0, float("inf")

    linker_carboxylate_to_acid_ratio = solve_linker_carboxylate_to_acid_ratio(
        dimethylamine_conc=dimethylamine_conc,
        capping_agent_conc=Capping_agent_conc,
        linker_conc=effective_linker_conc,
        correction_term_for_deprotonation=correction_term_for_deprotonation,
        num_carboxylate_on_linker=num_carboxylate_on_linker,
    )
    formate_to_acid_ratio = linker_carboxylate_to_acid_ratio * correction_term_for_deprotonation

    linker_carboxylic_acid_conc = (
        effective_linker_conc * (1 / (1 + linker_carboxylate_to_acid_ratio)) * num_carboxylate_on_linker
    )
    if linker_carboxylic_acid_conc <= 0.0:
        return 1.0, float("inf")
    Formic_acid_conc = Capping_agent_conc * (1 / (1 + formate_to_acid_ratio))

    formate_benzoate_ratio = Formic_acid_conc / linker_carboxylic_acid_conc / effective_equilibrium_constant
    dissolution_probability_value = formate_benzoate_ratio / (formate_benzoate_ratio + 1)
    return dissolution_probability_value, formate_benzoate_ratio


ENTROPY_GAIN = 30.9
corrected_entropy_gain = ENTROPY_GAIN * entropy_correction_coefficient


def entropy_assembly(ENTROPY_GAIN, num_entity, limit=150):
    entity_extra_gain = 0.35
    if num_entity >= limit:
        return np.exp(corrected_entropy_gain)
    return np.exp(corrected_entropy_gain + entity_extra_gain * (1 - np.log(num_entity + 1) / np.log(limit)))


Entropy_correction_table = [entropy_assembly(ENTROPY_GAIN, i) for i in range(20000)]
if DISTORTED_LINKER_ENABLED:
    distorted_ligand_state = compute_prebound_chemistry_state(
        zr_conc=Zr_conc,
        linker_conc=Linker_conc,
        equilibrium_constant_coefficient=equilibrium_constant_coefficient,
        h2o_dmf_ratio=H2O_DMF_RATIO,
        capping_agent_conc=Capping_agent_conc,
        zr6_percentage=ZR6_PERCENTAGE,
        model_name=(
            DISTORTED_CHEMISTRY_MODEL
            if DISTORTED_CHEMISTRY_MODEL is not None
            else PREBOUND_MODEL_CLUSTER_ONE_TO_ONE
        ),
        association_constant_override=DISTORTED_LIGAND_ASSOCIATION_CONSTANT,
        site_equilibrium_constant_override=DISTORTED_SITE_EQUILIBRIUM_CONSTANT,
        dimethylamine_conc=0.0,
        second_step_equivalents=(
            DISTORTED_SECOND_STEP_EQUIVALENTS
            if DISTORTED_SECOND_STEP_EQUIVALENTS is not None
            else 0.0
        ),
        num_sites_on_cluster=(
            DISTORTED_NUM_SITES_ON_CLUSTER
            if DISTORTED_NUM_SITES_ON_CLUSTER is not None
            else 12
        ),
        num_sites_on_linker=(
            DISTORTED_NUM_SITES_ON_LINKER
            if DISTORTED_NUM_SITES_ON_LINKER is not None
            else num_carboxylate_on_linker
        ),
    )
else:
    distorted_ligand_state = default_prebound_state(
        zr_conc=Zr_conc,
        linker_conc=Linker_conc,
        zr6_percentage=ZR6_PERCENTAGE,
        effective_equilibrium_constant=effective_equilibrium_constant,
        second_step_equivalents=(
            DISTORTED_SECOND_STEP_EQUIVALENTS
            if DISTORTED_SECOND_STEP_EQUIVALENTS is not None
            else 0.0
        ),
        model_name=(
            DISTORTED_CHEMISTRY_MODEL
            if DISTORTED_CHEMISTRY_MODEL is not None
            else PREBOUND_MODEL_CLUSTER_ONE_TO_ONE
        ),
        num_sites_on_cluster=(
            DISTORTED_NUM_SITES_ON_CLUSTER
            if DISTORTED_NUM_SITES_ON_CLUSTER is not None
            else 12
        ),
        num_sites_on_linker=(
            DISTORTED_NUM_SITES_ON_LINKER
            if DISTORTED_NUM_SITES_ON_LINKER is not None
            else num_carboxylate_on_linker
        ),
    )

Zr6_conc = Zr_conc * ZR6_PERCENTAGE / 6
Zr12_conc = Zr_conc * (1 - ZR6_PERCENTAGE) / 12
Zr6_conc_for_growth = max(float(distorted_ligand_state["free_zr6_conc"]), 0.0)
LINKER_CONC_FOR_GROWTH = max(float(distorted_ligand_state["free_linker_conc"]), 0.0)
effective_zr6_fraction_for_growth = (
    Zr6_conc_for_growth / distorted_ligand_state["total_zr6_conc"]
    if distorted_ligand_state["total_zr6_conc"] > 0.0
    else 0.0
)
Zr6_conc_adding_probability = zr6_cluster_add_probability(
    zr6_conc=Zr6_conc_for_growth,
    linker_conc=LINKER_CONC_FOR_GROWTH,
    num_carboxylate_on_linker=num_carboxylate_on_linker,
)
LEGACY_EXTERNAL_ADDITION_ACTIVITY = float(
    np.sqrt(
        effective_zr6_fraction_for_growth
        * distorted_ligand_state["free_linker_fraction"]
    )
)
LEGACY_EXTERNAL_ADDITION_ACTIVITY = min(max(LEGACY_EXTERNAL_ADDITION_ACTIVITY, 0.0), 1.0)
FREE_ZR6_ADDITION_ACTIVITY = min(max(float(effective_zr6_fraction_for_growth), 0.0), 1.0)
FREE_LINKER_ADDITION_ACTIVITY = min(
    max(float(distorted_ligand_state["free_linker_fraction"]), 0.0),
    1.0,
)
PREBOUND_ZR_BTB_ADDITION_ACTIVITY = min(
    max(
        float(
            distorted_ligand_state.get(
                "prebound_zr6_cluster_fraction",
                distorted_ligand_state.get("prebound_zr_btb_fraction", 0.0),
            )
        ),
        0.0,
    ),
    1.0,
)
TOTAL_EXTERNAL_ADDITION_CHANNEL_ACTIVITY = (
    FREE_ZR6_ADDITION_ACTIVITY
    + FREE_LINKER_ADDITION_ACTIVITY
    + PREBOUND_ZR_BTB_ADDITION_ACTIVITY
)

chemistry_summary = {
    "ZR6_PERCENTAGE": ZR6_PERCENTAGE,
    "Zr_conc_total": Zr_conc,
    "Linker_conc_total": Linker_conc,
    "Capping_agent_conc": Capping_agent_conc,
    "equilibrium_constant_coefficient": equilibrium_constant_coefficient,
    "effective_equilibrium_constant": effective_equilibrium_constant,
    "distorted_linker_enabled": bool(DISTORTED_LINKER_ENABLED),
    "distorted_chemistry_model": distorted_ligand_state["model_name"],
    "distorted_ligand_state": distorted_ligand_state,
    "prebound_zr_bdc_fraction": distorted_ligand_state["prebound_zr_bdc_fraction"],
    "effective_zr6_conc_for_growth": Zr6_conc_for_growth,
    "effective_zr6_fraction_for_growth": effective_zr6_fraction_for_growth,
    "effective_linker_conc_for_growth": LINKER_CONC_FOR_GROWTH,
    "Zr6_conc_adding_probability": Zr6_conc_adding_probability,
    "external_addition_activity": LEGACY_EXTERNAL_ADDITION_ACTIVITY,
    "total_external_addition_channel_activity": TOTAL_EXTERNAL_ADDITION_CHANNEL_ACTIVITY,
    "legacy_external_addition_activity": LEGACY_EXTERNAL_ADDITION_ACTIVITY,
    "free_zr6_addition_activity": FREE_ZR6_ADDITION_ACTIVITY,
    "free_linker_addition_activity": FREE_LINKER_ADDITION_ACTIVITY,
    "prebound_zr_btb_addition_activity": PREBOUND_ZR_BTB_ADDITION_ACTIVITY,
    "external_channel_activity_basis": "normalized_fractions_matching_uio66",
    "prebound_external_channel_species": "prebound_zr6_cluster_fraction",
}
if current_folder is not None:
    with open(os.path.join(current_folder, "chemistry_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(chemistry_summary, handle, indent=2)

date_index = datetime.now().strftime('%Y-%m-%d_%H-%M')

assembly = Assembly(
    Zr6(),
    ZR6_PERCENTAGE,
    ENTROPY_GAIN,
    BUMPING_THRESHOLD,
    distorted_linker_fraction=distorted_ligand_state["prebound_zr_bdc_fraction"],
)

timing = 0
time_for_link = 0

event_num_link = 0
time_for_grow = 0

event_num_grow = 0
event_num_grow_success = 0
event_num_grow_fail = 0
time_for_remove = 0

event_num_remove = 0
event_num_to_do_nothing = 0
time_for_decision = 0
time_for_calculate_DMF = 0

step = 0

entities_number = []
for cycle in range(Total_steps):
    start_time = time.time()

    if max_entities is not None and len(assembly.entities) > max_entities:
        break

    if step == 0 or (
        DISSOLUTION_UPDATE_INTERVAL_STEPS is not None
        and DISSOLUTION_UPDATE_INTERVAL_STEPS > 0
        and step % DISSOLUTION_UPDATE_INTERVAL_STEPS == 0
    ):
        DISSOLUTION_PROBABILITY, FORMATE_BENZOATE_RATIO = dissolution_probability(timing, DMF_decomposition_rate)

    entities_number.append([timing, len(assembly.entities)])
    time_for_calculate_DMF += (time.time() - start_time)
    step += 1

    start_time = time.time()
    flag, selected_carboxylate, total_growth_rate, selected_pair, addition_mode = assembly.next_thing_to_do(
        FORMATE_BENZOATE_RATIO,
        Entropy_correction_table,
        Zr6_conc_adding_probability,
        external_addition_activity=TOTAL_EXTERNAL_ADDITION_CHANNEL_ACTIVITY,
        free_zr6_addition_activity=FREE_ZR6_ADDITION_ACTIVITY,
        free_linker_addition_activity=FREE_LINKER_ADDITION_ACTIVITY,
        prebound_zr_btb_addition_activity=PREBOUND_ZR_BTB_ADDITION_ACTIVITY,
    )
    end_time = time.time()
    time_for_decision += (end_time - start_time)

    if flag == 0:
        start_time = time.time()
        assembly.link_internal_carboxylate(selected_pair)
        end_time = time.time()
        time_for_link += (end_time - start_time)
        event_num_link += 1

    elif flag == 1:
        start_time = time.time()
        grow_succeeded = assembly.grow_one_step(selected_carboxylate, addition_mode=addition_mode)
        end_time = time.time()
        time_for_grow += (end_time - start_time)
        event_num_grow += 1
        if grow_succeeded:
            event_num_grow_success += 1
        else:
            event_num_grow_fail += 1

    elif flag == -1:
        start_time = time.time()
        assembly.remove_linkage(selected_pair)
        end_time = time.time()
        time_for_remove += (end_time - start_time)
        event_num_remove += 1

    else:
        event_num_to_do_nothing += 1

    timing -= np.log(random.random()) / total_growth_rate

    if cycle % 1e5 == 0:
        pass

    if cycle % 1e5 == 0:
        clear_output(wait=True)

    if cycle % 1e2 == 0:
        gc.collect()
        all_objects = gc.get_objects()
        total_memory_usage = np.sum([sys.getsizeof(obj) for obj in all_objects])
        del all_objects

    if output_inter is not None and output_inter > 0 and cycle % output_inter == 0:
        with open(current_folder + f"/entities_number_{abs(timing)}.pkl", "wb") as f:
            pickle.dump(entities_number, f)

        assembly.get_mol2_file(current_folder + f'/assembly_{abs(timing)}.mol2')

with open(current_folder + "/entities_number.pkl", "wb") as f:
    pickle.dump(entities_number, f)

assembly.get_mol2_file(current_folder + f'/assembly.mol2')

run_summary = {
    "final_entities": len(assembly.entities),
    "simulated_time_seconds": timing * exchange_rxn_time,
    "event_num_link": event_num_link,
    "event_num_grow": event_num_grow,
    "event_num_grow_success": event_num_grow_success,
    "event_num_grow_fail": event_num_grow_fail,
    "event_num_remove": event_num_remove,
    "event_num_to_do_nothing": event_num_to_do_nothing,
    "distorted_linker_enabled": bool(DISTORTED_LINKER_ENABLED),
    "distorted_chemistry_model": distorted_ligand_state["model_name"],
    "prebound_growth_attempts": assembly.prebound_growth_attempts,
    "prebound_growth_successes": assembly.prebound_growth_successes,
    "prebound_growth_failures": assembly.prebound_growth_failures,
    "prebound_metal_site_attempts": assembly.prebound_metal_site_attempts,
    "prebound_metal_site_successes": assembly.prebound_metal_site_successes,
    "prebound_metal_site_failures": assembly.prebound_metal_site_failures,
    "prebound_linker_site_attempts": assembly.prebound_linker_site_attempts,
    "prebound_linker_site_successes": assembly.prebound_linker_site_successes,
    "prebound_linker_site_failures": assembly.prebound_linker_site_failures,
    "prebound_entities_added": assembly.prebound_entities_added,
    "prebound_linkages_formed": assembly.prebound_linkages_formed,
    "prebound_free_growth_site_delta": assembly.prebound_free_growth_site_delta,
    "prebound_ready_pair_delta": assembly.prebound_ready_pair_delta,
    "prebound_zr_bdc_fraction": distorted_ligand_state["prebound_zr_bdc_fraction"],
    "distorted_linker_fraction": distorted_ligand_state["distorted_ligand_fraction"],
    "off_pathway_linker_fraction": distorted_ligand_state["off_pathway_linker_fraction"],
    "effective_zr6_conc": Zr6_conc_for_growth,
    "effective_zr6_fraction_for_growth": effective_zr6_fraction_for_growth,
    "effective_linker_conc": LINKER_CONC_FOR_GROWTH,
    "external_addition_activity": LEGACY_EXTERNAL_ADDITION_ACTIVITY,
    "total_external_addition_channel_activity": TOTAL_EXTERNAL_ADDITION_CHANNEL_ACTIVITY,
    "legacy_external_addition_activity": LEGACY_EXTERNAL_ADDITION_ACTIVITY,
    "free_zr6_addition_activity": FREE_ZR6_ADDITION_ACTIVITY,
    "free_linker_addition_activity": FREE_LINKER_ADDITION_ACTIVITY,
    "prebound_zr_btb_addition_activity": PREBOUND_ZR_BTB_ADDITION_ACTIVITY,
    "external_channel_activity_basis": "normalized_fractions_matching_uio66",
    "prebound_external_channel_species": "prebound_zr6_cluster_fraction",
    "final_free_growth_sites": len(assembly.free_carboxylates),
    "final_metal_growth_sites": len(assembly.MC_free_carboxylates),
    "final_linker_growth_sites": len(assembly.Linker_free_carboxylates),
    "final_linked_pairs": len(assembly.linked_carboxylate_pairs),
    "final_ready_to_connect_pairs": len(assembly.ready_to_connect_carboxylate_pairs),
}
with open(current_folder + "/run_summary.json", "w", encoding="utf-8") as f:
    json.dump(run_summary, f, indent=2)
