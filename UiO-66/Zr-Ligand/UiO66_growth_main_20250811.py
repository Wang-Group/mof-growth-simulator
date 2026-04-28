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
DISTORTED_LIGAND_ASSOCIATION_CONSTANT = None
DISTORTED_SECOND_STEP_EQUIVALENTS = None

import time
import gc
try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(*args, **kwargs):
        return None
from UiO66_Assembly_Large_Correction_20250811 import *
from distorted_ligand_model import (
    compute_distorted_ligand_state,
    solve_linker_carboxylate_to_acid_ratio,
    zr6_cluster_add_probability,
)
import numpy as np
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
import sys
import types
import pickle
from datetime import datetime
import json
import os

if current_folder is not None:
    os.makedirs(current_folder, exist_ok=True)

"""Set parameters"""
# Assuming the classes Assembly, Zr6, BTB, etc., and all required functions are already defined
#parameters that are less likely to change

correction_term_for_deprotonation = 10**(3.51-4.74) #L/AA

#calculating coordination power of H2O and DMF with respect to capping agent formate
H2O_pure = 55500 #mM  #this is the concentration of water in pure water
DMF_pure = 12900 #mM  #This is the concentration of DMF for pure DMF（density 0.944 g/mL, molecular weight 73.09 g/mol）

DMF_conc = DMF_pure/(1+H2O_DMF_RATIO)
H2O_conc = H2O_pure*H2O_DMF_RATIO/(1+H2O_DMF_RATIO)
H2O_formate_coefficient = 0.01
DMF_formate_coefficient = 0.01
H2O_formate_coord_power = H2O_conc/Capping_agent_conc*H2O_formate_coefficient
DMF_formate_coord_power = DMF_conc/Capping_agent_conc*DMF_formate_coefficient

#linker information
num_carboxylate_on_linker = 2                 #tricarboxylate linker

#equilibrium between the linker and capping agent
equilibrium_constant = 1.64 #equilibrium constant between formic acid and benzoic acid, the larger the benzoate coordination is more favored
#considering the water and DMF coordination
effective_equilibrium_constant = equilibrium_constant_coefficient*equilibrium_constant/(1+H2O_formate_coord_power+DMF_formate_coord_power)

#calculate DMF decomposition rate
end_DMF_decomposition_conc = 560 # mM, experimentally determined
exp_time = 3 #h
DMF_decomposition_rate = end_DMF_decomposition_conc/(exp_time* 3.6 * 10**3) # mM/s

#estimated time for the exchange reaction to happen in one step
exchange_rxn_time = EXCHANGE_RXN_TIME_SECONDS if EXCHANGE_RXN_TIME_SECONDS is not None else 0.1 #s
LINKER_CONC_FOR_GROWTH = None

def dissolution_probability(time_passed, DMF_decomposition_rate):
    dimethylamine_conc = time_passed*DMF_decomposition_rate # from DMF decomposition, needs to be determined by other models
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
    formate_to_acid_ratio = linker_carboxylate_to_acid_ratio*correction_term_for_deprotonation

    linker_carboxylic_acid_conc = (
        effective_linker_conc * (1 / (1 + linker_carboxylate_to_acid_ratio)) * num_carboxylate_on_linker
    )
    if linker_carboxylic_acid_conc <= 0.0:
        return 1.0, float("inf")
    Formic_acid_conc = Capping_agent_conc*(1/(1+formate_to_acid_ratio))

    formate_benzoate_ratio = Formic_acid_conc/linker_carboxylic_acid_conc/effective_equilibrium_constant
    dissolution_probability_value = formate_benzoate_ratio/(formate_benzoate_ratio+1) # the share of times a dissolution rather than a growth step happen

    del linker_carboxylate_to_acid_ratio, formate_to_acid_ratio, linker_carboxylic_acid_conc, Formic_acid_conc, dimethylamine_conc
    return dissolution_probability_value, formate_benzoate_ratio


########## the entropy function
#------------ modified(please check again!)--------------------
ENTROPY_GAIN = 30.9 # the translational and rotational entropy gain of an internal linkage with respect to addition of a BTB, calculated to be (29.1R - (-3.6R/2)) = 30.9R
#------------ modified(please check again!)--------------------
corrected_entropy_gain = ENTROPY_GAIN*entropy_correction_coefficient
def entropy_assembly(ENTROPY_GAIN, num_entity, limit =150):
    # return ENTROPY_GAIN/num_entity
    entity_extra_gain = 0.35 #R
    if num_entity>=limit:
        return np.exp(corrected_entropy_gain)
    else:
        #entropy change difference between external addition and internal addition
        return np.exp(corrected_entropy_gain + entity_extra_gain*(1-np.log(num_entity+1)/np.log(limit)))

Entropy_correction_table = [entropy_assembly(ENTROPY_GAIN, i) for i in range(20000)]
if DISTORTED_LINKER_ENABLED:
    distorted_ligand_state = compute_distorted_ligand_state(
        zr_conc=Zr_conc,
        linker_conc=Linker_conc,
        equilibrium_constant_coefficient=equilibrium_constant_coefficient,
        h2o_dmf_ratio=H2O_DMF_RATIO,
        capping_agent_conc=Capping_agent_conc,
        zr6_percentage=ZR6_PERCENTAGE,
        association_constant_override=DISTORTED_LIGAND_ASSOCIATION_CONSTANT,
        dimethylamine_conc=0.0,
        second_step_equivalents=(
            DISTORTED_SECOND_STEP_EQUIVALENTS
            if DISTORTED_SECOND_STEP_EQUIVALENTS is not None
            else 1.0
        ),
    )
else:
    distorted_ligand_state = {
        "association_constant": 0.0,
        "effective_exchange_equilibrium_constant": effective_equilibrium_constant,
        "acid_speciation": None,
        "second_step_equivalents": (
            DISTORTED_SECOND_STEP_EQUIVALENTS
            if DISTORTED_SECOND_STEP_EQUIVALENTS is not None
            else 1.0
        ),
        "total_zr6_conc": Zr_conc*ZR6_PERCENTAGE/6,
        "total_linker_conc": Linker_conc,
        "distorted_ligand_conc": 0.0,
        "polymerized_distorted_conc": 0.0,
        "off_pathway_zr6_conc": 0.0,
        "off_pathway_linker_conc": 0.0,
        "distorted_ligand_fraction": 0.0,
        "off_pathway_linker_fraction": 0.0,
        "off_pathway_zr6_fraction": 0.0,
        "free_zr6_conc": Zr_conc*ZR6_PERCENTAGE/6,
        "free_linker_conc": Linker_conc,
        "free_zr6_fraction": 1.0,
        "free_linker_fraction": 1.0,
    }

# the concentration of Zr in the unit of mM
Zr6_conc = Zr_conc*ZR6_PERCENTAGE/6
Zr12_conc = Zr_conc*(1-ZR6_PERCENTAGE)/12
Zr6_conc_for_growth = max(float(distorted_ligand_state["free_zr6_conc"]), 0.0)
LINKER_CONC_FOR_GROWTH = max(float(distorted_ligand_state["free_linker_conc"]), 0.0)
Zr6_conc_0 = LINKER_CONC_FOR_GROWTH*2/12
Zr12_conc_0 = Zr6_conc_0/2
D_ENTROPY_ZR6_BTB = 3.6 # the entropy difference between Zr6 and BTB in the unit of R
D_ENTROPY_ZR12_BTB = 3.6 # the entropy difference between Zr12 and BTB in the unit of R
Zr6_conc_adding_probability = zr6_cluster_add_probability(
    zr6_conc=Zr6_conc_for_growth,
    linker_conc=LINKER_CONC_FOR_GROWTH,
    num_carboxylate_on_linker=num_carboxylate_on_linker,
)
EXTERNAL_ADDITION_ACTIVITY = float(
    np.sqrt(
        distorted_ligand_state["free_zr6_fraction"]
        * distorted_ligand_state["free_linker_fraction"]
    )
)
EXTERNAL_ADDITION_ACTIVITY = min(max(EXTERNAL_ADDITION_ACTIVITY, 0.0), 1.0)
# Zr12_conc_adding_probability =np.exp(-D_ENTROPY_ZR12_BTB+np.log(Zr12_conc_0/Zr12_conc))/(1+np.exp(-D_ENTROPY_ZR12_BTB+np.log(Zr12_conc_0/Zr12_conc) ))

chemistry_summary = {
    "ZR6_PERCENTAGE": ZR6_PERCENTAGE,
    "Zr_conc_total": Zr_conc,
    "Linker_conc_total": Linker_conc,
    "Capping_agent_conc": Capping_agent_conc,
    "equilibrium_constant_coefficient": equilibrium_constant_coefficient,
    "effective_equilibrium_constant": effective_equilibrium_constant,
    "distorted_linker_enabled": bool(DISTORTED_LINKER_ENABLED),
    "distorted_ligand_state": distorted_ligand_state,
    "effective_zr6_conc_for_growth": Zr6_conc_for_growth,
    "effective_linker_conc_for_growth": LINKER_CONC_FOR_GROWTH,
    "Zr6_conc_adding_probability": Zr6_conc_adding_probability,
    "external_addition_activity": EXTERNAL_ADDITION_ACTIVITY,
}
if current_folder is not None:
    with open(os.path.join(current_folder, "chemistry_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(chemistry_summary, handle, indent=2)

# %%

# Calculation
# Generate the date index for the filename  
date_index = datetime.now().strftime('%Y-%m-%d_%H-%M')  

# Initialize Assembly with an initial Zr6 or BTB entity
assembly = Assembly(
    Zr6_AA(),
    ZR6_PERCENTAGE,
    ENTROPY_GAIN,
    BUMPING_THRESHOLD,
    distorted_linker_fraction=distorted_ligand_state["distorted_ligand_fraction"],
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
event_num_to_do_nothing=0
time_for_decision = 0
time_for_calculate_DMF = 0

step = 0

entities_number = []
# for cycle in tqdm(range(Total_steps)):
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
    
    # save the time and entity number
    entities_number.append([timing, len(assembly.entities)])
    
    time_for_calculate_DMF += (time.time()-start_time)
    step +=1

    start_time = time.time()
    flag, selected_carboxylate, total_growth_rate, selected_pair = assembly.next_thing_to_do(
        FORMATE_BENZOATE_RATIO,
        Entropy_correction_table,
        Zr6_conc_adding_probability,
        EXTERNAL_ADDITION_ACTIVITY,
    )
    end_time = time.time()
    time_for_decision += (end_time-start_time)

    if flag == 0:
        # print('link')
        start_time = time.time()
        assembly.link_internal_carboxylate(selected_pair)
        end_time = time.time()
        time_for_link += (end_time-start_time)
        event_num_link += 1
        
    elif flag == 1:
        # print('grow')
        start_time = time.time()
        grow_succeeded = assembly.grow_one_step(selected_carboxylate)
        end_time = time.time()
        time_for_grow += (end_time-start_time)
        event_num_grow += 1
        if grow_succeeded:
            event_num_grow_success += 1
        else:
            event_num_grow_fail += 1
        
    elif flag == -1:
        # print('dissolve')
        start_time = time.time()
        assembly.remove_linkage(selected_pair)
        end_time = time.time()
        time_for_remove += (end_time-start_time)
        event_num_remove += 1

    else:
        event_num_to_do_nothing+=1

    timing -= np.log(random.random())/total_growth_rate 

    if cycle%1e5 == 0: #we need to update to a variable evolution time.
        pass
        # print('step',cycle,'assembly size:',len(assembly.entities),' at ',timing*exchange_rxn_time,'s')

    if cycle%1e5 == 0:
        clear_output(wait=True)
        # print(f'number of enitities in the assembly: {len(assembly.entities)}')
        # assembly.visualize()

    if cycle%1e2 == 0:

        # Get the total memory usage
        gc.collect()
        all_objects = gc.get_objects()
        total_memory_usage = np.sum([sys.getsizeof(obj) for obj in all_objects])
        # print(f'Total memory usage: {total_memory_usage} bytes')
        del all_objects

        # # Save the assembly object to a pickle file
        # try:
        #     assembly.get_mol_file(current_folder + f'/assembly_{date_index}_{str(cycle//1e2)}.mol')
        # except:
        #     with open(current_folder + f'/assembly_{date_index}_{str(cycle//1e2)}.pkl', 'wb') as output_file:  
        #         pickle.dump(assembly, output_file)

    if output_inter is not None and output_inter > 0 and cycle % output_inter == 0:
        with open(current_folder + f"/entities_number_{abs(timing)}.pkl", "wb") as f:
            pickle.dump(entities_number, f)

        assembly.get_mol2_file(current_folder + f'/assembly_{abs(timing)}.mol2')

# Save to a pickle file
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
    "distorted_linker_fraction": distorted_ligand_state["distorted_ligand_fraction"],
    "off_pathway_linker_fraction": distorted_ligand_state["off_pathway_linker_fraction"],
    "effective_zr6_conc": Zr6_conc_for_growth,
    "effective_linker_conc": LINKER_CONC_FOR_GROWTH,
    "external_addition_activity": EXTERNAL_ADDITION_ACTIVITY,
}
with open(current_folder + "/run_summary.json", "w", encoding="utf-8") as f:
    json.dump(run_summary, f, indent=2)

# %%
# assembly.get_mol2_file(f'assembly.mol2')

# %%



