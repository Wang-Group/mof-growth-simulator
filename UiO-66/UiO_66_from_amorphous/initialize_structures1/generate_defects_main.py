"""
Simple script to generate defects in UiO-66 structure.

This is a simplified version that can be run directly without command line arguments.
Modify the parameters in the script and run: python generate_defects_simple.py

Note: The script removes:
1. Selected defect entities from the interior (based on defect_ratio)
2. Completely isolated entities (entities with ZERO connections)

Entities with partial connections (lost some neighbors due to defects) are PRESERVED.
This maintains the shell structure while cleaning up truly isolated fragments.
"""

from generate_defects import generate_defects

# ========== CONFIGURATION PARAMETERS ==========
# Input file
base_name = "UiO-66_15x15x15_sphere_R3"
INPUT_FILE = f"data/{base_name}.mol2"

# Defect parameters
DEFECT_RATIO = 0.3  # Remove 10% of interior entities (0.0 to 1.0)
SHELL_THICKNESS = 30.0  # Preserve outer 10 Angstroms (increase to preserve more)

# Output files
OUTPUT_MOL2 = f"output/{base_name}_defective_{DEFECT_RATIO}.mol2"
OUTPUT_PKL = f"output/{base_name}_defective_{DEFECT_RATIO}.pkl"  # Set to None to skip PKL generation

# Entity type to remove: 'Zr6', 'BDC', or None (both types)
ENTITY_TYPE = None  # None means remove both Zr6 and BDC randomly

# Random seed for reproducibility (set to None for random results each time)
RANDOM_SEED = 42

# ===============================================

if __name__ == '__main__':
    print("="*60)
    print("UiO-66 Defect Generation Script")
    print("="*60)
    print(f"Configuration:")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Output MOL2: {OUTPUT_MOL2}")
    print(f"  Output PKL: {OUTPUT_PKL if OUTPUT_PKL else 'Not generated'}")
    print(f"  Defect ratio: {DEFECT_RATIO:.1%}")
    print(f"  Shell thickness: {SHELL_THICKNESS} Å")
    print(f"  Entity type: {ENTITY_TYPE if ENTITY_TYPE else 'Both Zr6 and BDC'}")
    print(f"  Random seed: {RANDOM_SEED if RANDOM_SEED else 'Random'}")
    print("="*60 + "\n")
    
    generate_defects(
        input_file=INPUT_FILE,
        output_mol2=OUTPUT_MOL2,
        output_pkl=OUTPUT_PKL,
        defect_ratio=DEFECT_RATIO,
        shell_thickness=SHELL_THICKNESS,
        entity_type=ENTITY_TYPE,
        random_seed=RANDOM_SEED
    )
