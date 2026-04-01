import os
import re
import pathlib
import datetime
import shutil
import time
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

"""
Multi-stage Template Generator
- Reference UiO66_generate_excel.py directory and parameter writing style
- Stage 1: Initialize from Zr6
- Stage N>1: Initialize from latest assembly_*.pkl from previous stage, with parameter override support
- Launch concurrent tasks for each stage until entities_number.pkl is generated

Usage:
Modify the parameters below in this script and run: python UiO67_staged_template_generator.py

Description:
- Read parameters from template Excel files for each row, replicate tasks by repeat_number
- Each stage generates its own subdirectory Stage_1, Stage_2, ..., where Stage_1 starts from Zr6, others initialize from previous stage's latest pkl
- Copy and rewrite base_main, base_assembly header parameters in each task directory
- Tasks in each stage complete before moving to next stage. Subsequent stages read the latest assembly_*.pkl from corresponding task directory in previous stage as pkl_path
"""

# ========== CONFIGURATION PARAMETERS ==========
# Modify these parameters as needed
TEMPLATE_EXCEL_BASE = "UiO66_BDC_250924"  # Base name (without .xlsx)
# PARENT_FOLDER = "/mnt/syh/UiO66_growth_data/"
PARENT_FOLDER = "/home/yyt/mol_growth/mol_growth/initialize_structures"
NUM_CORES = 5
NUM_STAGES = 3  # None = auto-detect, or manually set (e.g. 2, 3, 4, ...)
BASE_MAIN_SCRIPT = "UiO66_growth_main_conc.py"  # Main script with module support
BASE_ASSEMBLY_SCRIPT = "UiO66_Assembly_Large_Correction_conc.py"
DRY_RUN = False  # Set to True to only generate directories and scripts without running tasks
CLEAN_UP = True
# ===============================================

import subprocess


def _read_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()


def _write_lines(path: str, lines: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def _copy_file(src: str, dst: str):
    """Copy a file from src to dst"""
    shutil.copy2(src, dst)
    print(f"  ✓ Copied: {os.path.basename(src)} → {dst}")


def _overwrite_header_params(lines: List[str], kv: Dict[str, object]) -> List[str]:
    out = []
    for line in lines:
        replaced = False
        for key, val in kv.items():
            if line.startswith(f"{key}"):
                out.append(f"{key} = {val}\n")
                replaced = True
                break
        if not replaced:
            out.append(line)
    return out


def _replace_assembly_import(lines: List[str], assembly_module: str) -> List[str]:
    out = []
    pattern = re.compile(r'^from\s+\S*UiO66_Assembly\S*\s+import\s+\*')
    replaced_once = False
    for line in lines:
        if not replaced_once and pattern.match(line.strip()):
            out.append(f"from {assembly_module} import *\n")
            replaced_once = True
        else:
            out.append(line)
    return out


def _find_latest_pkl(task_dir: pathlib.Path) -> Optional[str]:
    # Only match files saved within the loop: assembly_*_entity_number*.pkl
    pkl_files = sorted(task_dir.glob('assembly_*_entity_number*.pkl'))
    if not pkl_files:
        return None
    # Get the most recently modified file
    pkl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(pkl_files[0])


def _spawn_and_wait(task_dirs: List[str], num_cores: int, entry_filename: str):
    status = {d: 'not_started' for d in task_dirs}
    procs = {}

    while True:
        # Start unstarted tasks, limit concurrency
        running = sum(1 for s in status.values() if s == 'in_process')
        for d, s in list(status.items()):
            if s == 'not_started' and running < num_cores:
                print(f"Starting: {d}")
                p = subprocess.Popen(["python", os.path.join(d, entry_filename)])
                procs[d] = p
                status[d] = 'in_process'
                running += 1

        # Check completion
        all_done = True
        for d, s in status.items():
            if s == 'completed':
                continue
            ent_pkl = pathlib.Path(d) / "entities_number.pkl"
            if ent_pkl.exists():
                status[d] = 'completed'
            else:
                all_done = False
        if all_done:
            break
        time.sleep(5)


def _get_stage_param(row, param_name: str, stage: int, default_value=None):
    """
    Get parameter value for specified stage, supporting stage-wise override
    Priority: param_name_stage > global param_name > default_value
    """
    stage_col = f"{param_name}_{stage}"
    if stage_col in row and pd.notna(row[stage_col]):
        return row[stage_col]
    elif param_name in row and pd.notna(row[param_name]):
        return row[param_name]
    else:
        return default_value


def _resolve_path(path_str: str) -> str:
    p = pathlib.Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)
    # Try relative path: current working directory
    if p.exists():
        return str(p)
    # Script directory
    script_dir = pathlib.Path(__file__).resolve().parent
    cand = script_dir / p
    if cand.exists():
        return str(cand)
    # Script parent directory (project root)
    cand2 = script_dir.parent / p
    if cand2.exists():
        return str(cand2)
    # Try working directory parent
    cand3 = pathlib.Path.cwd().parent / p
    if cand3.exists():
        return str(cand3)
    raise FileNotFoundError(f"Cannot resolve path: {path_str}")

def _get_last_saved_from_pkl(pkl_path: str) -> int:
    """
    Extract the last saved entity number from pkl file path
    File format: assembly_date_entity_number_.pkl
    If extraction fails, return -1
    """
    try:
        # Extract entity number from filename: assembly_date_entity_number_N.pkl
        import re
        filename = pathlib.Path(pkl_path).name
        match = re.search(r'entity_number(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            return -1
    except Exception as e:
        print(f"Warning: Failed to extract entity number from {pkl_path}: {e}")
        return -1

def _cleanup_stage_folder(root_dir: pathlib.Path, stage: int):
    """
    Clean up all folders in the specified stage directory
    """
    stage_dir = root_dir / f"Stage_{stage}"
    if stage_dir.exists():
        print(f"Cleaning up Stage_{stage} directory: {stage_dir}")
        shutil.rmtree(stage_dir)
        print(f"Successfully cleaned up Stage_{stage}")
    else:
        print(f"Stage_{stage} directory does not exist: {stage_dir}")


def _detect_num_stages(template_base: str, template_dir: pathlib.Path) -> int:
    """
    Auto-detect number of stages by checking for Excel files
    
    Returns:
    - Number of detected stages
    """
    stage = 1
    while True:
        stage_excel_name = f"{template_base}_stage_{stage}.xlsx"
        stage_excel_path = template_dir / stage_excel_name
        
        # Try to resolve path
        try:
            resolved_path = _resolve_path(str(stage_excel_path))
            if not pathlib.Path(resolved_path).exists():
                break
        except:
            break
        
        stage += 1
    
    num_stages = stage - 1
    
    if num_stages == 0:
        print(f"   Warning: No stage Excel files found ({template_base}_stage_*.xlsx)")
        print(f"   Please ensure at least {template_base}_stage_1.xlsx exists")
    
    return num_stages


def _get_experiment_key(row, repeat: int) -> str:
    """
    Build a stable key for an experiment folder name without timestamp.
    Must mirror folder_name components except the timestamp suffix.
    """
    # Keep fields in sync with folder_name composition in main()
    step_mag = int(np.log10(int(row['Total_steps']))) if int(row['Total_steps']) > 0 else 0
    
    # Handle max_entities: convert None/NaN/"None" to "None" string for folder name
    max_entities_val = row.get('max_entities')
    if pd.isna(max_entities_val) or (isinstance(max_entities_val, str) and max_entities_val.strip().lower() == 'none'):
        max_entities_str = "None"
    else:
        max_entities_str = str(max_entities_val)
    
    return (
        f"Zr_{row['Zr_conc']}_FA_{row['Capping_agent_conc']}_L_{row['Linker_conc']}"
        f"_Ratio_{row['H2O_DMF_RATIO']}_Step_1e{step_mag}_Index_{repeat}"
        f"_SC_{row['entropy_correction_coefficient']}_KC_{row['equilibrium_constant_coefficient']}"
        f"_Nmax_{max_entities_str}"
    )

def _build_prev_stage_map_from_existing(root_dir: pathlib.Path, prev_stage: int) -> Dict[tuple, str]:
    """
    Build prev_stage_map by scanning existing Stage_1 folders
    """
    prev_stage_map = {}
    stage_1_dir = root_dir / f"Stage_{prev_stage}"
    
    if not stage_1_dir.exists():
        print(f"Warning: Stage_{prev_stage} directory does not exist: {stage_1_dir}")
        return prev_stage_map
    
    # Read the Excel file for the previous stage to get the mapping
    template_base = TEMPLATE_EXCEL_BASE
    template_dir = pathlib.Path(".")
    stage_excel_name = f"{template_base}_stage_{prev_stage}.xlsx"
    stage_excel_path = template_dir / stage_excel_name
    
    try:
        stage_excel_resolved = _resolve_path(str(stage_excel_path))
        tasks = pd.read_excel(stage_excel_resolved)
        
        # Build mapping based on existing folders
        for idx, row in tasks.iterrows():
            for repeat in range(int(row["repeat_number"])):
                # Build a key without timestamp to match folders
                experiment_key = _get_experiment_key(row, repeat)
                # Match any folder that contains the key; timestamp suffix varies
                matching_folders = list(stage_1_dir.glob(f"*{experiment_key}_*"))
                
                if matching_folders:
                    # Use the most recent folder if multiple matches
                    matching_folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    prev_stage_map[(idx, repeat)] = str(matching_folders[0])
                    print(f"Found Stage_{prev_stage} folder for (idx={idx}, repeat={repeat}): {matching_folders[0]}")
                else:
                    print(f"Warning: No Stage_{prev_stage} folder found for (idx={idx}, repeat={repeat}) with key {experiment_key}")
                    
    except FileNotFoundError:
        print(f"Warning: Excel file for stage {prev_stage} not found: {stage_excel_path}")
    
    return prev_stage_map


def main(stage: int):
    """
    Run a single specified stage
    
    Parameters:
    - stage: Stage number to run (1, 2, 3, ...)
    """
    # Parse template base name (remove .xlsx suffix)
    template_base = TEMPLATE_EXCEL_BASE
    template_dir = pathlib.Path(".")  # Current directory
    
    parent_folder = PARENT_FOLDER.rstrip('/') + '/'
    number_of_cores = NUM_CORES
    
    print(f"\n{'='*60}")
    print(f"Preparing to run Stage {stage}")
    print(f"{'='*60}\n")
    
    base_main = _resolve_path(BASE_MAIN_SCRIPT)
    base_assembly = _resolve_path(BASE_ASSEMBLY_SCRIPT)

    # Stage root directory
    root_dir = pathlib.Path(parent_folder) / template_base
    root_dir.mkdir(parents=True, exist_ok=True)

    # Clean up the specified stage
    if CLEAN_UP:
        print(f"Cleaning up old data for Stage {stage}...")
        _cleanup_stage_folder(root_dir, stage)
    
    # Build prev_stage_map if this is not stage 1
    prev_stage_map: Dict[tuple, str] = {}
    if stage > 1:
        print(f"Loading data mapping from Stage {stage-1}...")
        prev_stage_map = _build_prev_stage_map_from_existing(root_dir, stage - 1)

    # Read Excel file for the specified stage
    stage_excel_name = f"{template_base}_stage_{stage}.xlsx"
    stage_excel_path = template_dir / stage_excel_name
     
    try:
        stage_excel_resolved = _resolve_path(str(stage_excel_path))
        tasks = pd.read_excel(stage_excel_resolved)
        print(f"Reading Stage {stage} configuration: {stage_excel_resolved}\n")
    except FileNotFoundError:
        print(f"Error: Cannot find configuration file for Stage {stage}: {stage_excel_path}")
        return

    stage_dir = root_dir / f"Stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    current_stage_task_dirs: List[str] = []

    # Generate all task directories for current stage
    for idx, row in tasks.iterrows():
        # Each row may be repeated multiple times
        for repeat in range(int(row["repeat_number"])):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
            # Handle max_entities for folder name (same logic as _get_experiment_key)
            max_entities_val = row.get('max_entities')
            if pd.isna(max_entities_val) or (isinstance(max_entities_val, str) and max_entities_val.strip().lower() == 'none'):
                max_entities_str = "None"
            else:
                max_entities_str = str(max_entities_val)
            folder_name = f"Zr_{row['Zr_conc']}_FA_{row['Capping_agent_conc']}_L_{row['Linker_conc']}_Ratio_{row['H2O_DMF_RATIO']}_Step_1e{int(np.log10(int(row['Total_steps'])))}_Index_{repeat}_SC_{row['entropy_correction_coefficient']}_KC_{row['equilibrium_constant_coefficient']}_Nmax_{max_entities_str}_{timestamp}"
            task_dir = stage_dir / folder_name
            task_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Setting up task: {folder_name}")
            print(f"{'='*60}")

            # Generate main and assembly files
            main_lines = _read_lines(base_main)
            assembly_lines = _read_lines(base_assembly)

            # Variable parameter table - read directly from current stage Excel
            # Helper function to safely convert max_entities
            def _parse_max_entities(val):
                """Parse max_entities value from Excel, supporting None/NaN/'None' string"""
                if pd.isna(val):
                    return None  # Default to None (no limit)
                if isinstance(val, str):
                    val_lower = val.strip().lower()
                    if val_lower == 'none' or val_lower == '':
                        return None
                    try:
                        return int(float(val))  # Try to convert string number to int
                    except (ValueError, TypeError):
                        return None
                try:
                    return int(float(val)) if val is not None else None
                except (ValueError, TypeError):
                    return None
            
            max_entities_val = _parse_max_entities(row.get("max_entities"))
            max_entities_str = "None" if max_entities_val is None else str(max_entities_val)
            
            new_values_main = {
                "ZR6_PERCENTAGE": row.get("ZR6_PERCENTAGE", 1.0),
                "Zr_conc": row.get("Zr_conc", 20),
                "entropy_correction_coefficient": row.get("entropy_correction_coefficient", 0.35),
                "equilibrium_constant_coefficient": row.get("equilibrium_constant_coefficient", 0.68),
                "H2O_DMF_RATIO": row.get("H2O_DMF_RATIO", 0.0),
                "Capping_agent_conc": row.get("Capping_agent_conc", 295),
                "Linker_conc": row.get("Linker_conc", 110),
                "Total_steps": int(row.get("Total_steps", 1000000)),
                "current_folder": "'" + str(task_dir) + "/'",
                "BUMPING_THRESHOLD": row.get("BUMPING_THRESHOLD", 2.0),
                "max_entities": max_entities_val,  # Can be None or int
                "output_inter": int(row.get("output_inter", 200)),
            }

            pkl_value = row.get("pkl_path")
            excel_pkl = str(pkl_value).strip() if pd.notna(pkl_value) and pkl_value else ""
            if excel_pkl and excel_pkl.lower() != "none":
                new_values_main["pkl_path"] = f"'{excel_pkl}'"
                new_values_main["last_saved"] = _get_last_saved_from_pkl(excel_pkl)
            # Stage 1: pkl_path=None; subsequent stages fill with latest pkl from corresponding index in previous stage
            elif stage == 1:
                new_values_main["pkl_path"] = None
                new_values_main["last_saved"] = -1
            else:
                prev_task_dir_str = prev_stage_map.get((idx, repeat))
                if prev_task_dir_str:
                    prev_task_dir = pathlib.Path(prev_task_dir_str)
                    latest_pkl = _find_latest_pkl(prev_task_dir)
                    if latest_pkl:
                        new_values_main["pkl_path"] = f"'{latest_pkl}'"
                        new_values_main["last_saved"] = _get_last_saved_from_pkl(latest_pkl)
                    else:
                        new_values_main["pkl_path"] = None
                        new_values_main["last_saved"] = -1
                else:
                    new_values_main["pkl_path"] = None
                    new_values_main["last_saved"] = -1

            new_values_assembly = {
                "MAX_DEPTH": int(row.get("MAX_DEPTH", 3))
            }

            # Override and write
            main_out = _overwrite_header_params(main_lines, new_values_main)
            assembly_out = _overwrite_header_params(assembly_lines, new_values_assembly)

            # Replace assembly module import with current copied file module name
            assembly_module = pathlib.Path(base_assembly).stem
            main_out = _replace_assembly_import(main_out, assembly_module)

            # Write files
            _write_lines(str(task_dir / pathlib.Path(base_main).name), main_out)
            _write_lines(str(task_dir / pathlib.Path(base_assembly).name), assembly_out)

            current_stage_task_dirs.append(str(task_dir))

    # Run all tasks for current stage
    if not DRY_RUN:
        print(f"\n{'='*60}")
        print(f"Starting to run {len(current_stage_task_dirs)} tasks for Stage {stage}")
        print(f"{'='*60}\n")
        _spawn_and_wait(current_stage_task_dirs, number_of_cores, pathlib.Path(base_main).name)
        print(f"\n{'='*60}")
        print(f"✓ Stage {stage} completed!")
        print(f"{'='*60}\n")
    else:
        print(f"\n✓ DRY_RUN mode: Generated {len(current_stage_task_dirs)} task directories (not executed)")


if __name__ == '__main__':
    import sys
    
    # Detect number of stages (for parameter validation)
    template_base = TEMPLATE_EXCEL_BASE
    template_dir = pathlib.Path(".")
    
    if NUM_STAGES is None:
        # Auto-detect number of stages
        detected_stages = _detect_num_stages(template_base, template_dir)
        max_stages = detected_stages
    else:
        max_stages = NUM_STAGES
    
    if max_stages == 0:
        print("    Error: No stage files found")
        print(f"   Please ensure file exists: {template_base}_stage_1.xlsx")
        sys.exit(1)
    
    # Parse command line arguments: default to Stage 1 if not provided
    if len(sys.argv) < 2:
        stage = 1
        print(f"\n💡 No stage number specified, defaulting to Stage 1")
        print(f"   Tip: Use 'python3 {sys.argv[0]} <number>' to run other stages")
        print(f"   Detected stages: {max_stages}\n")
    else:
        try:
            stage = int(sys.argv[1])
            if stage < 1 or stage > max_stages:
                print(f"   Error: Stage number must be between 1 and {max_stages}")
                print(f"   Detected stages: {max_stages}")
                sys.exit(1)
        except ValueError:
            print(f"  Error: Stage number must be an integer")
            sys.exit(1)
    
    # Run specified stage
    main(stage)

