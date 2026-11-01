import os.path
import numpy as np
from scipy.constants import k, h, pi, N_A,R
from scipy.spatial import cKDTree
import pandas as pd
import os
from numpy.linalg import norm
from multiprocessing import Pool, cpu_count
from itertools import combinations, islice
R_con=R

# Constants
PI = np.pi

# Define a mapping between atomic symbols and van der Waals radii.
ATOMIC_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

BONDI_VDW_RADIUS = [
    1.10, 1.40, 1.81, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 1.84, 2.10, 1.80, 1.80,
    1.75, 1.88, 2.75, 2.31, 2.30, 2.15, 2.05, 2.05, 2.05, 2.05, 2.00, 2.00, 2.00, 2.10, 1.87, 2.11,
    1.85, 1.90, 1.83, 2.02, 3.03, 2.49, 2.40, 2.30, 2.15, 2.10, 2.05, 2.05, 2.00, 2.05, 2.10, 2.20,
    1.93, 2.17, 2.06, 2.06, 1.98, 2.16, 3.43, 2.68, 2.50, 2.48, 2.47, 2.45, 2.43, 2.42, 2.40, 2.38,
    2.37, 2.35, 2.33, 2.32, 2.30, 2.28, 2.27, 2.25, 2.20, 2.10, 2.05, 2.00, 2.00, 2.05, 2.10, 2.05,
    1.96, 2.02, 2.07, 1.97, 2.02, 2.20, 3.48, 2.83, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40,
    2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40,
    2.40, 2.40, 2.40, 2.40, 2.40, 2.40
]

ELEMENT_TO_RADIUS = {symbol: radius for symbol, radius in zip(ATOMIC_SYMBOLS, BONDI_VDW_RADIUS)}

# def convert_mol2_to_xyz(input_path, output_path):
#     """
#     Convert a single .mol2 file to XYZ format
    
#     Args:
#         input_path (str): Full path to the input .mol2 file
#         output_path (str): Full path for the output .xyz file
    
#     Raises:
#         IOError: If file operations fail
#         ValueError: If file format is invalid
#     """
#     try:
#         with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
#             atoms = []
#             in_atom_section = False
            
#             for line in infile:
#                 # Detect start of atomic data section
#                 if line.startswith("@<TRIPOS>ATOM"):
#                     in_atom_section = True
#                     continue
                
#                 # Stop at next TRIPOS section
#                 if line.startswith("@<TRIPOS>") and in_atom_section:
#                     break
                
#                 # Process atom data lines
#                 if in_atom_section and line.strip():
#                     parts = line.split()
#                     if len(parts) >= 6:
#                         # Extract atom type (column 2) and coordinates (columns 3-5)
#                         atom_type = parts[1]
#                         x, y, z = parts[2:5]
#                         atoms.append(f"{atom_type} {x} {y} {z}")
            
#             # Write XYZ file format:
#             # Line 1: Number of atoms
#             # Line 2: Comment line (optional)
#             # Subsequent lines: Atom coordinates
#             outfile.write(f"{len(atoms)}\n")
#             outfile.write(f"Converted from: {input_path}\n")
#             outfile.write("\n".join(atoms))
            
#     except Exception as e:
#         print(f"Error processing {input_path}: {str(e)}")
#         raise


# def batch_convert_mol2(input_dir, output_dir=None):
#     """
#     Batch conversion example with recursive search
    
#     Args:
#         input_dir (str): Root directory to search for .mol2 files
#         output_dir (str, optional): Output directory (default: current directory)
#     """
#     # Set default output directory
#     output_dir = output_dir or os.getcwd()
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Track filename conflicts
#     file_counter = {}
    
#     # Recursive file search
#     for root, _, files in os.walk(input_dir):
#         for filename in files:
#             if filename.lower().endswith('.mol2'):
#                 full_input_path = os.path.join(root, filename)
                
#                 # Generate unique output filename
#                 base_name = os.path.splitext(filename)[0]
#                 output_name = f"{base_name}.xyz"
                
#                 # Handle filename conflicts
#                 if output_name in file_counter:
#                     file_counter[output_name] += 1
#                     output_name = f"{base_name}_{file_counter[output_name]}.xyz"
#                 else:
#                     file_counter[output_name] = 0
                
#                 full_output_path = os.path.join(output_dir, output_name)
                
#                 # Perform conversion
#                 convert_mol2_to_xyz(full_input_path, full_output_path)

# def process_xyz_file(input_file, output_file):
#     '''
#     Replace the element label "Zr" with "Hf" and remove duplicate coordinates.
#     Args:
#         input_file: The original xyz file
#         output_file: The processed xyz file
#     '''
#     # Remove the duplicate coordinates
#     unique_coords = set()
#     unique_lines = []

#     with open(input_file, 'r') as infile:
#         lines = infile.readlines()
        
#         # Skip the first two lines
#         num_atoms = int(lines[0].strip())
#         comment = lines[1].strip()
        
#         for line in lines[2:]:
#             parts = line.split()
#             if len(parts) == 4:
#                 element, x, y, z = parts
#                 # Replace "Zr" with "Hf"
#                 if element == 'Zr':
#                     element = 'Hf'
#                 coord = (element, float(x), float(y), float(z))
#                 if coord not in unique_coords:
#                     unique_coords.add(coord)
#                     unique_lines.append(f"{element} {x} {y} {z}\n")
        
#         # Update the number of atoms
#         num_atoms = len(unique_lines)

#     # Write the output file
#     with open(output_file, 'w') as outfile:
#         outfile.write(f"{num_atoms}\n")
#         outfile.write(f"{comment}\n")
#         outfile.writelines(unique_lines)

# def process_all_xyz_files(input_directory, output_directory=None):
#     '''
#     Process all .xyz files in a specified directory
    
#     Args:
#         input_directory (str): Path to directory containing input XYZ files
#         output_directory (str, optional): Path to save processed files. 
#             Defaults to input directory if not specified.
#     '''
#     # Set default output directory
#     if output_directory is None:
#         output_directory = input_directory
    
#     # Create output directory if needed
#     os.makedirs(output_directory, exist_ok=True)
    
#     # Process all XYZ files
#     for filename in os.listdir(input_directory):
#         if filename.endswith('.xyz'):
#             # Build full paths
#             input_path = os.path.join(input_directory, filename)
            
#             # Create output filename with "_processed" suffix
#             base_name = os.path.splitext(filename)[0]
#             output_filename = f"{base_name}.xyz"
#             output_path = os.path.join(output_directory, output_filename)
            
#             # Process file
#             process_xyz_file(input_path, output_path)
#             print(f"Processed: {input_path} -> {output_path}")

def count_atoms(df, target_atom):
    """
    Count the number of specific atoms in a DataFrame
    
    Parameters:
        df (pd.DataFrame): DataFrame containing atomic data
        target_atom (str): Target atomic symbol to count (e.g., 'C', 'O', 'Zr')
    
    Returns:
        int: Count of target atoms
    """
    # Ensure atomic symbols are strings and strip whitespace
    df['atoms'] = df['atoms'].astype(str).str.strip()
    
    # Count atoms matching target
    return df[df['atoms'] == target_atom].shape[0]


def bondi_vdW(element):
    """Get the van der Waals radii of atoms"""
    try:
        return ELEMENT_TO_RADIUS[element]
    except KeyError:
        raise ValueError(f"Error: van der Waals radius for element {element} not found")

def sphere_volume(r):
    """Calculate the volume of sphere"""
    return (4 / 3) * PI * r**3

def intersection_volume(r1, r2, d):
    """Calculate the intersection volume between two spheres"""
    if d >= r1 + r2:
        return 0.0
    else:
        return (PI * (r1 + r2 - d)**2 * (d**2 + 2 * d * (r1 + r2) - 3 * (r1 - r2)**2)) / (12 * d)

# def calculate_vdw_volume(df):
#     """Calculate the van der Waals volume of molecules from a DataFrame
    
#     Args:
#         df: DataFrame with columns ['atoms', 'x', 'y', 'z'] containing element symbols and coordinates
    
#     Returns:
#         float: van der Waals volume
#     """
#     # Validate input DataFrame
#     required_columns = ['atoms', 'x', 'y', 'z']
#     if not all(col in df.columns for col in required_columns):
#         raise ValueError(f"DataFrame must contain these columns: {required_columns}")
    
#     # Extract elements and coordinates
#     elements = df['atoms'].values
#     xyz = df[['x', 'y', 'z']].values
    
#     # Get the van der Waals radii
#     radii = np.array([bondi_vdW(element) for element in elements])
#     n_atom = len(elements)
    
#     volume = 0.0
#     for i in range(n_atom):
#         volume += sphere_volume(radii[i])
#         for j in range(i):
#             d = norm(xyz[i] - xyz[j])  # Distance between atoms i and j
#             volume -= intersection_volume(radii[i], radii[j], d)
    
#     return volume

def calculate_vdw_volume_optimized(df, cutoff_ratio=1.5):
    """
    Calculate vdW volume with distance cutoff for faster computation
    
    Args:
        df: DataFrame with columns ['atoms', 'x', 'y', 'z']
        cutoff_ratio: Only consider atom pairs within (r_i + r_j)*cutoff_ratio
        
    Returns:
        float: van der Waals volume
    """
    # Validate input
    required_columns = ['atoms', 'x', 'y', 'z']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain these columns: {required_columns}")
    
    # Extract data
    elements = df['atoms'].values
    xyz = df[['x', 'y', 'z']].values
    radii = np.array([bondi_vdW(element) for element in elements])
    n_atom = len(elements)
    
    # Build KD-tree for fast neighbor searching
    tree = cKDTree(xyz)
    
    volume = 0.0
    intersection_total = 0.0
    
    # Precompute sphere volumes
    sphere_volumes = (4/3) * np.pi * radii**3
    volume = np.sum(sphere_volumes)
    
    # Process each atom
    for i in range(n_atom):
        r_i = radii[i]
        pos_i = xyz[i]
        
        # Find neighbors within cutoff distance
        max_cutoff = (r_i + radii.max()) * cutoff_ratio
        neighbors = tree.query_ball_point(pos_i, r=max_cutoff)
        
        # Only consider neighbors with j < i to avoid double counting
        neighbors = [j for j in neighbors if j < i]
        
        for j in neighbors:
            r_j = radii[j]
            d = np.linalg.norm(pos_i - xyz[j])
                
            # Only calculate intersection if atoms actually overlap
            intersection_total += intersection_volume(r_i, r_j, d)
    
    volume -= intersection_total
    return volume

# Calculate the molecular mass
def calculate_molecular_mass(atoms):
    atomic_masses = {'H': 1.008, 'O': 15.999, 'C': 12.011, 'N': 14.007, 'Hf': 178.49}   
    total_mass = sum(atomic_masses.get(atom, 0) for atom in atoms)  
    return total_mass

# Calculate the center of mass
def calculate_com(atoms, coords):
    atomic_masses = {'H': 1.008, 'O': 15.999, 'C': 12.011, 'N': 14.007, 'Hf': 178.49}
    masses = np.array([atomic_masses[atom] for atom in atoms])
    com = np.sum(masses[:, None] * coords, axis=0) / np.sum(masses)
    return com

# Calculate the inertia tensor
def calculate_inertia_tensor(atoms, coords, com):
    atomic_masses = {'H': 1.008, 'O': 15.999, 'C': 12.011, 'N': 14.007, 'Hf': 178.49}
    masses = np.array([atomic_masses[atom] for atom in atoms])
    coords_centered = coords - com
    I = np.zeros((3, 3))
    for i in range(len(atoms)):
        x, y, z = coords_centered[i]
        m = masses[i]
        I[0, 0] += m * (y**2 + z**2)
        I[1, 1] += m * (x**2 + z**2)
        I[2, 2] += m * (x**2 + y**2)
        I[0, 1] -= m * x * y
        I[1, 0] -= m * x * y
        I[0, 2] -= m * x * z
        I[2, 0] -= m * x * z
        I[1, 2] -= m * y * z
        I[2, 1] -= m * y * z
    return I

# Calculate the square of the radius of gyration (r_g^2)
def calculate_rg_squared(atoms, coords):
    num_atoms = len(atoms)
    # Calculate the center of mass
    com = calculate_com(atoms, coords)
    # Calculate r_g^2
    rg_squared = 0.0
    for i in range(num_atoms):
        r_k = coords[i]
        rg_squared += np.linalg.norm(r_k - com) ** 2
    rg_squared /= num_atoms
    return rg_squared

def calculate_entropy(solvent_property,solute_property):
    """
    The function is based on this paper(https://pubs.acs.org/doi/10.1021/acs.jctc.9b00214)
    For our system, we will only consider the translation entropy and rotation entropy since we consider that
    the external addition and internal linking do not differ significantly in terms of vibration entropy. 
    As for the cavity entropy, the situation is more complicated and enthaply should be involved accounting
    for "entropy-enthalpy compensation" to matche the experimental results requiring extra work deviates from 
    our simulation objectives.So we assume that the term of external addition has little difference with the 
    one of internal linking.
    
    Args:
    Solvent_property:A list to describe the property of solvent including 
    [the molecular weight,the volume of a solvent molecule,the mass density,temperature]
    i.e.[M_S,V_S,rho,T];
    Solute_property:A list to describe the property of solute including
    [the molecular weight,the volume of a solute molecule,radius of gyration,the rotational symmetry number,the average moment of inertia]
    i.e.[m,V_M,,r_g,sigma,I];

    Return: Translational entropy, rotational entropy and their summation
 
    """
    M_S,V_S,rho,T=solvent_property
    m,V_M,r_g,sigma,I=solute_property

    """Calculate translation entropy (St:R/mol)"""
    # Calculate the volume of free space per solvent particle(unit:A^3) 
    V_free=(M_S/(N_A*rho))*1e24-V_S
    # Calculate the volume of the cavity
    V_c=(V_M**(1/3)+V_free**(1/3))**3
    # Calculate the radius of the cavity
    r_c=(3*V_c/(4*pi))**(1/3)

    # Calculate the probility per solvent particle of "hopping" to an adjacent cavity
    x=max(0,V_free**(2/3)-V_M**(2/3))/(V_free**(2/3)+V_S**(2/3))
    N_x=4*((4*PI/3)**(2/3))*((r_c**2)/(V_free**(2/3)+V_S**(2/3)))
    # Calculate the average number of accessible cavities
    N_c=1+N_x*(1/(1-x)-1)

    # Calculate the volume of the box 
    V=V_c*N_c*1e-30
    m=m*1e-3/N_A
    # Calculate the translational partition function
    q_t = (((2 * PI *m* k * T) / (h**2)) ** (3/2))*V
    # Calculate the translational entropy(the entropy penalty of bringing a gas at standard conditions to a 1 M concentration is included)
    S_t = np.log(q_t) + 2.5-6.4*4.184/R_con

    """Calculate rotational entropy(Sr:R/mol)"""
    # There are three condtions to be discussed
    if (r_c>=r_g) and (r_c-r_g>=((3 / (4 * pi)) ** (2/3)) * (V_free ** (1/3))):
        # Calculate the rotational partition function
        q_r = (pi**0.5 / sigma) * (8 * pi**2 * I * k * T / h**2) ** (3/2)
        # Calculate the translational entropy loss due to rotation
        Sr_loss=3*np.log((r_c-r_g)/r_c)
        # Calculate the rotational entropy
        S_r = np.log(q_r) +1.5+Sr_loss
    elif (r_c>=r_g) and (r_c-r_g<((3 / (4 * pi)) ** (2/3)) * (V_free ** (1/3))):
        q_r = (pi**0.5 / sigma) * (8 * pi**2 * I * k * T / h**2) ** (3/2)
        Sr_loss=3*np.log((3 / (4 * pi)) ** (2/3) * V_free ** (1/3)/r_c)
        S_r=np.log(q_r)+1.5+Sr_loss 
    else:
        """Calculate angle θ_0"""
        # Calculate the radius of the free space per solvent particle(unit:A)
        r_free = (3 * V_free / (4 * PI)) ** (1/3)
        theta_0 = 2 * np.arccos(r_g / np.sqrt(r_g**2 + r_free**2))
        q_r = (8 * PI * I * k * T / h**2) ** (3/2)*theta_0**2
        Sr_loss=3*np.log((3 / (4 * PI)) ** (2/3) * V_free ** (1/3)/r_c)
        S_r=np.log(q_r)+1.5+Sr_loss
    
    return S_t,S_r,S_t+S_r