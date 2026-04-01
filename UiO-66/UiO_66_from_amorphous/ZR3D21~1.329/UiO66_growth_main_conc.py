ZR6_PERCENTAGE = 0.6
Zr_conc = 22.9154705859894
entropy_correction_coefficient = 0.789387907185137
equilibrium_constant_coefficient = 1.32975172557788
H2O_DMF_RATIO = 3e-10
Capping_agent_conc = 1473.06341756944
Linker_conc = 69.1596872253079
Total_steps = 1000000000000000
current_folder = '/mnt/syh/UiO66_growth_data/UiO66_initial_structure/20251117-Zr6-Zr12_stage1/UiO66_BDC_251117/Stage_1/Zr_22.9154705859894_FA_1473.06341756944_L_69.1596872253079_Ratio_3e-10_Step_1e15_Index_0_SC_0.789387907185137_KC_1.32975172557788_Nmax_3000_2025-11-17_20_09_48/'
BUMPING_THRESHOLD = 2
pkl_path = None
max_entities = 3000
output_inter = 200
last_saved = -1

import time
import gc
from IPython.display import clear_output
from UiO66_Assembly_Large_Correction_conc import *
import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import sys
import types
try:
    import dill
    dill.settings['recurse'] = True
    import dill as pickle
except ImportError:
    import pickle
from datetime import datetime

os.makedirs(current_folder, exist_ok=True)

# Increase recursion limit to avoid recursion depth issues when pickling large object graphs
sys.setrecursionlimit(100000)

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
exchange_rxn_time = 0.1 #s

def dissolution_probability(time_passed, DMF_decomposition_rate):
    dimethylamine_conc = time_passed*DMF_decomposition_rate # from DMF decomposition, needs to be determined by other models
    if dimethylamine_conc > end_DMF_decomposition_conc:
        dimethylamine_conc = end_DMF_decomposition_conc

    # assumption: DMF decomposition to CO + dimethylamine. Another path of decomposition to formate and protonated dimethylamine would modify this formula here
    # Define the function for the equation  
    def equation(y):  
        return y*correction_term_for_deprotonation/(1+y*correction_term_for_deprotonation)*Capping_agent_conc+y/(1+y)*Linker_conc*num_carboxylate_on_linker-dimethylamine_conc

    # Initial guess  
    initial_guess = 0  

    # Solve the equation  
    solution = fsolve(equation, initial_guess)  

    # Print the solution  
    # print(f"The solution is: {solution[0]}")  

    linker_carboxylate_to_acid_ratio = solution[0]
    formate_to_acid_ratio = solution[0]*correction_term_for_deprotonation

    linker_carboxylic_acid_conc = Linker_conc*(1/(1+linker_carboxylate_to_acid_ratio))*num_carboxylate_on_linker
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

Entropy_correction_table = [entropy_assembly(ENTROPY_GAIN, i) for i in range(25000)]
 # the concentration of Zr in the unit of mM
Zr6_conc = Zr_conc*ZR6_PERCENTAGE/6
Zr12_conc = Zr_conc*(1-ZR6_PERCENTAGE)/12
Zr6_conc_0 = Linker_conc*2/12
Zr12_conc_0 = Zr6_conc_0/2
D_ENTROPY_ZR6_BTB = 3.6 # the entropy difference between Zr6 and BTB in the unit of R
D_ENTROPY_ZR12_BTB = 3.6 # the entropy difference between Zr12 and BTB in the unit of R
#------------ modified(please check again!)--------------------
Zr6_conc_adding_probability =np.exp(-D_ENTROPY_ZR6_BTB+np.log(Zr6_conc/Zr6_conc_0))/(1+np.exp(-D_ENTROPY_ZR6_BTB+np.log(Zr6_conc/Zr6_conc_0) ))
Zr12_conc_adding_probability =np.exp(-D_ENTROPY_ZR12_BTB+np.log(Zr12_conc/Zr12_conc_0))/(1+np.exp(-D_ENTROPY_ZR12_BTB+np.log(Zr12_conc/Zr12_conc_0)))
Zr_cluster_conc_adding_probability = Zr6_conc_adding_probability+Zr12_conc_adding_probability
#------------ modified(please check again!)--------------------
# %%

# Calculation
# Generate the date index for the filename  
date_index = datetime.now().strftime('%Y-%m-%d_%H-%M')  

# Initialize Assembly with an initial Zr6 or BTB entity
# --------------------------------------modified---------------------------------------------------------
if pkl_path is not None:
    # Use standalone module's safe load function
    assembly_loaded = safe_pickle_load(pkl_path, rebuild_references=True)
    
    if assembly_loaded is None:
        print("Failed to load assembly, exiting...")
        sys.exit(1)
    
    print(assembly_loaded)
    assembly = Assembly(assembly_loaded, ZR6_PERCENTAGE, ENTROPY_GAIN, BUMPING_THRESHOLD)
   
else:
    print("Starting with initial Zr6 entity...")
    assembly = Assembly(Zr6_AA(), ZR6_PERCENTAGE, ENTROPY_GAIN, BUMPING_THRESHOLD)
# ---------------------------------------modified----------------------------------------------------------

timing = 0
time_for_link = 0

event_num_link = 0
time_for_grow = 0

event_num_grow = 0
time_for_remove = 0

event_num_remove = 0
event_num_to_do_nothing=0
time_for_decision = 0
time_for_calculate_DMF = 0

step = 0
entities_number = []

for cycle in range(Total_steps + 1):
    date_index_process = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  
    start_time = time.time()

    # Check if max_entities limit is reached (only if max_entities is not None)
    if max_entities is not None and max_entities > 0 and len(assembly.entities) > max_entities:
        break
    if step == 0:
        DISSOLUTION_PROBABILITY, FORMATE_BENZOATE_RATIO = dissolution_probability(timing, DMF_decomposition_rate)
    # save the time and entity number
    entities_number.append([timing, len(assembly.entities)])
    
    time_for_calculate_DMF += (time.time()-start_time)
    step +=1

    start_time = time.time()
    # ----------------------------------------------modified-------------------------------------------------------------------------------------------------------------
    flag, selected_carboxylate, total_growth_rate, selected_pair = assembly.next_thing_to_do(FORMATE_BENZOATE_RATIO,Entropy_correction_table,Zr_cluster_conc_adding_probability)
    end_time = time.time()
    time_for_decision += (end_time-start_time)
    # ----------------------------------------------modified-------------------------------------------------------------------------------------------------------------
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
        assembly.grow_one_step(selected_carboxylate)
        end_time = time.time()
        time_for_grow += (end_time-start_time)
        event_num_grow += 1
        
    elif flag == -1:
        # print('dissolve')
        start_time = time.time()
        assembly.remove_linkage(selected_pair)
        end_time = time.time()
        time_for_remove += (end_time-start_time)
        event_num_remove += 1


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
    
    n = len(assembly.entities)
    if n % output_inter == 0 and n != 0:  # only at multiples of output_inter, not zero
        if n != last_saved:  # only save if it's a new multiple
            assembly.get_mol2_file(current_folder + f'/assembly_{date_index_process}_entity_number_{n}.mol2')
            
            # Use standalone module's safe save function
            filepath = current_folder + f'/assembly_{date_index_process}_entity_number_{n}.pkl'
            rebuild_after = (cycle < Total_steps)  # Rebuild if continuing
            
            success = safe_pickle_save(
                assembly, 
                filepath, 
                clean_connected_entities=True,  # Clean connected entities to avoid RecursionError
                rebuild_after_save=rebuild_after,
                protocol=pickle.HIGHEST_PROTOCOL
            )
            
            if success:
                last_saved = n
            else:
                print(f"Save failed, skipping this save")
            
# Use standalone module to save final Assembly
filepath = current_folder + f'/assembly_{date_index}_{str(cycle//1e1)}.pkl'
safe_pickle_save(
    assembly, 
    filepath, 
    clean_connected_entities=True,  # Clean connected entities to avoid RecursionError
    rebuild_after_save=False,  # Final save, no rebuild needed
    protocol=pickle.HIGHEST_PROTOCOL
)
    
# Save to a pickle file
with open(current_folder + "/entities_number.pkl", "wb") as f:
    pickle.dump(entities_number, f)

assembly.get_mol2_file(current_folder + f'/assembly.mol2') 

# %%
# assembly.get_mol2_file(f'assembly.mol2')

# %%

