import os
from Entropy import *
import pandas as pd
import numpy as np
import signal
import time
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Load the data
folder_path=["/mnt/nas/光催化小组/宋宇航/MOL_growth_data/250412-2","/mnt/nas/光催化小组/宋宇航/MOL_growth_data/250412-3",
             "/mnt/nas/光催化小组/宋宇航/MOL_growth_data/250412-A2","/mnt/nas/光催化小组/宋宇航/MOL_growth_data/250412-A4",
             "/mnt/nas/光催化小组/宋宇航/MOL_growth_data/250412-A5","/mnt/nas/光催化小组/宋宇航/MOL_growth_data/250412-A7",
             "/mnt/nas/光催化小组/宋宇航/MOL_growth_data/250412-A9","/mnt/nas/光催化小组/宋宇航/MOL_growth_data/250412-4",
             "/mnt/nas/光催化小组/宋宇航/MOL_growth_data/250414-4"]

# Set the limited time
timeout_seconds = 600
for directory in folder_path:
    results=[]
    processed_files = 0
    skipped_files = 0
   # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(directory):
        # Filter files ending with .mol2
        for filename in filenames:
            if filename.endswith(".mol2"):
                start_time = time.time()
                result = {}
                full_path = os.path.join(dirpath, filename)
                
                try:
                    with time_limit(timeout_seconds):
                        print(f"\nProcessing file: {filename}")
                        
                        # Load the data
                        try:
                            data = pd.read_csv(full_path, header=None,
                                            skiprows=8, 
                                            sep=' ',
                                            usecols=[i for i in range(1,5)],
                                            engine='python')
                            data.columns = ['atoms', 'x', 'y', 'z']
                            data = data.iloc[:-1]
                        except Exception as e:
                            print(f"Error loading {filename}: {str(e)}")
                            skipped_files += 1
                            continue

                        # Count SBUs & BTBs
                        data['atoms'] = data['atoms'].replace('Zr', 'Hf')
                        Hf_count = count_atoms(data, "Hf")
                        C_count = count_atoms(data, "C")
                        result["SBU_count"] = Hf_count / 6
                        result["BTB_count"] = (C_count - 12 * result["SBU_count"]) / 27 

                        # Remove duplicate atoms & extract coordinates
                        data_cleaned = data.drop_duplicates()
                        atoms = data_cleaned['atoms'].values
                        coords = data_cleaned[['x', 'y', 'z']].values

                        # Calculate molecular properties
                        molecular_mass = calculate_molecular_mass(atoms)
                        result["Molecular Mass (g/mol)"] = molecular_mass

                        volume = calculate_vdw_volume_optimized(data_cleaned)
                        result['Van der Waals Volume (Å^3)'] = volume

                        rg_squared = calculate_rg_squared(atoms, coords)
                        r_g = np.sqrt(rg_squared)
                        result["Radius of Gyration (r_g)"] = r_g

                        com = calculate_com(atoms, coords)
                        inertia_tensor = calculate_inertia_tensor(atoms, coords, com)
                        eigenvalues = np.linalg.eigvals(inertia_tensor)
                        I = np.prod(eigenvalues)**(1/3)
                        I = I * 1e-3 / N_A * 1e-20
                        result['Average Moment of Inertia (I)'] = I

                        solvent_property = [49.8505, 38.91547731, 1.024396285, 393]
                        solute_property = [molecular_mass, volume, r_g, 1, I]
                        S_t, S_r, S = calculate_entropy(solvent_property, solute_property)
                        result['Translational Entropy (S_t)'] = S_t
                        result['Rotational Entropy (S_r)'] = S_r
                        result['Total Entropy (S)'] = S

                        results.append(pd.DataFrame([result], index=[filename]))
                        processed_files += 1
                        
                        elapsed = time.time() - start_time
                        print(f"Successfully processed {filename} in {elapsed:.2f} seconds")
                        print("-" * 60)

                except TimeoutException:
                    print(f"\nTimeout: Skipping {filename} after {timeout_seconds} seconds")
                    skipped_files += 1
                    continue
                
                except Exception as e:
                    print(f"\nError processing {filename}: {str(e)}")
                    skipped_files += 1
                    continue
    
    # Save results if any files were processed
    if results:
        base_name = os.path.basename(os.path.normpath(directory))
        output_filename = f"/home/yyt/mol_growth/mol_growth/entropy/data/{base_name}_result.csv"
        result_df = pd.concat(results)
        result_df.to_csv(output_filename, index=True, encoding='utf-8')
        print(f"\nSaved results to {output_filename}")
    
    print(f"\nProcessing summary for {directory}:")
    print(f"  Successfully processed files: {processed_files}")
    print(f"  Skipped files: {skipped_files}")
    print("=" * 60)