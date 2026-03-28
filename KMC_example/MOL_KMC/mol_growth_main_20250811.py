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

import time
import gc
from IPython.display import clear_output
from MOL_Assembly_Large_Correction_20250811 import *
import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
import types
import pickle
from datetime import datetime

"""Set parameters"""
# Assuming the classes Assembly, Zr6, BTB, etc., and all required functions are already defined
#parameters that are less likely to change

correction_term_for_deprotonation = 10**(4.19-3.75)   #BA/FA
# correction_term_for_deprotonation = 10**(3.46-3.75)   #btb/FA
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
num_carboxylate_on_linker = 3 #tricarboxylate linker

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
    initial_guess = 0.1  

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
def entropy_assembly(ENTROPY_GAIN, num_entity, limit = 150):
    # return ENTROPY_GAIN/num_entity
    entity_extra_gain = 0.35 #R
    if num_entity>=limit:
        return np.exp(corrected_entropy_gain)
    else:
        #entropy change difference between external addition and internal addition
        return np.exp(corrected_entropy_gain + entity_extra_gain*(1-np.log(num_entity+1)/np.log(limit)))

Entropy_correction_table = [entropy_assembly(ENTROPY_GAIN, i) for i in range(20000)]
 # the concentration of Zr in the unit of mM
Zr6_conc = Zr_conc*ZR6_PERCENTAGE/6
Zr12_conc = Zr_conc*(1-ZR6_PERCENTAGE)/12
Zr6_conc_0 = Linker_conc*3/12
Zr12_conc_0 = Zr6_conc_0/2
D_ENTROPY_ZR6_BTB = 3.6 # the entropy difference between Zr6 and BTB in the unit of R
D_ENTROPY_ZR12_BTB = 3.6 # the entropy difference between Zr12 and BTB in the unit of R
#------------ modified(please check again!)--------------------
Zr6_conc_adding_probability =np.exp(-D_ENTROPY_ZR6_BTB+np.log(Zr6_conc/Zr6_conc_0))/(1+np.exp(-D_ENTROPY_ZR6_BTB+np.log(Zr6_conc/Zr6_conc_0) ))
#------------ modified(please check again!)--------------------
# Zr12_conc_adding_probability =np.exp(-D_ENTROPY_ZR12_BTB+np.log(Zr12_conc_0/Zr12_conc))/(1+np.exp(-D_ENTROPY_ZR12_BTB+np.log(Zr12_conc_0/Zr12_conc) ))


# %%

# Calculation
# Generate the date index for the filename  
date_index = datetime.now().strftime('%Y-%m-%d_%H-%M')  

# Initialize Assembly with an initial Zr6 or BTB entity
assembly = Assembly(Zr6(), ZR6_PERCENTAGE, ENTROPY_GAIN, BUMPING_THRESHOLD)

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

    if len(assembly.entities) > 20:  # break if exceeding a certain size
        break

    start_time = time.time()
    if step % 1e5 == 0:
        DISSOLUTION_PROBABILITY, FORMATE_BENZOATE_RATIO = dissolution_probability(timing, DMF_decomposition_rate)
        
    entities_number.append([timing, len(assembly.entities)])
        # DISSOLUTION_PROBABILITY, FORMATE_BENZOATE_RATIO = dissolution_probability(timing, 0)
    time_for_calculate_DMF += (time.time()-start_time)
    step +=1

    start_time = time.time()
    flag, selected_carboxylate, total_growth_rate, selected_pair = assembly.next_thing_to_do(FORMATE_BENZOATE_RATIO,Entropy_correction_table,Zr6_conc_adding_probability)
    end_time = time.time()
    time_for_decision += (end_time-start_time)

    if flag==0:
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
        
    elif flag==-1:
        # print('dissolve')
        start_time = time.time()
        assembly.remove_linkage(selected_pair)
        end_time = time.time()
        time_for_remove += (end_time-start_time)
        event_num_remove += 1

    else:
        event_num_to_do_nothing+=1

    timing -= np.log(random.random())/total_growth_rate

    if cycle%1e6 == 0: #we need to update to a variable evolution time.
        pass
        # print('step',cycle,'assembly size:',len(assembly.entities),' at ',timing*exchange_rxn_time,'s')

    if cycle%1e6 == 0:
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

    if cycle%1e6 == 0:
        with open(current_folder + f"/entities_number_{abs(timing)}.pkl", "wb") as f:
            pickle.dump(entities_number, f)

        assembly.get_mol2_file(current_folder + f'/assembly_{date_index}.mol2')

            
# Save to a pickle file
with open(current_folder + "/entities_number.pkl", "wb") as f:
    pickle.dump(entities_number, f)

assembly.get_mol2_file(current_folder + f'/assembly_{date_index}.mol2') 

# %%
# assembly.get_mol2_file(f'assembly.mol2')

# %%



