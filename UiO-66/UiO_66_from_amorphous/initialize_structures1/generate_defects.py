"""
Script to generate defects in UiO-66 structure while maintaining outer shell integrity.

This script:
1. Loads a UiO-66 MOL2 file
2. Identifies entities (Zr6 clusters and BDC linkers) and their positions
3. Randomly removes entities from the interior while preserving the outer shell
4. Saves the defective structure as a new MOL2 file and PKL file

Usage:
    python generate_defects.py --input data/UiO-66_15x15x15_sphere_R6.mol2 \
                                --output output/UiO-66_defective.mol2 \
                                --defect_ratio 0.1 \
                                --shell_thickness 10.0
"""

import numpy as np
import argparse
from collections import defaultdict, Counter
from mol2pkl import *
import pickle
import os


def calculate_structure_center_from_entities(entities):
    """
    Calculate the geometric center of the structure from entity objects.
    
    Parameters:
        entities: collection of entity objects with 'center' attribute
    
    Returns:
        center coordinates (3D array)
    """
    if len(entities) == 0:
        return np.array([0.0, 0.0, 0.0])
    
    centers = np.array([entity.center for entity in entities])
    return np.mean(centers, axis=0)


def identify_shell_entities_from_objects(entities, structure_center, shell_thickness):
    """
    Identify entities in the outer shell based on distance from center.
    Works directly with entity objects.
    
    Parameters:
        entities: collection of entity objects with 'center' attribute
        structure_center: center of the entire structure
        shell_thickness: thickness of the outer shell to preserve (in Angstroms)
    
    Returns:
        set: entity objects in the outer shell
        float: maximum radius of the structure
    """
    # Calculate distances from structure center
    distances = {}
    for entity in entities:
        dist = np.linalg.norm(entity.center - structure_center)
        distances[entity] = dist
    
    if len(distances) == 0:
        return set(), 0.0
    
    # Find maximum radius
    max_radius = max(distances.values())
    
    # Entities in the outer shell
    shell_entities = {entity for entity, dist in distances.items() 
                     if dist >= (max_radius - shell_thickness)}
    
    return shell_entities, max_radius


def identify_shell_entities(entity_centers, structure_center, shell_thickness):
    """
    Identify entities in the outer shell based on distance from center.
    
    Parameters:
        entity_centers: dict of entity labels to their center coordinates
        structure_center: center of the entire structure
        shell_thickness: thickness of the outer shell to preserve (in Angstroms)
    
    Returns:
        set: entity labels in the outer shell
        float: maximum radius of the structure
    """
    # Calculate distances from structure center
    distances = {}
    for entity, center in entity_centers.items():
        dist = np.linalg.norm(center - structure_center)
        distances[entity] = dist
    
    # Find maximum radius
    max_radius = max(distances.values())
    
    # Entities in the outer shell
    shell_entities = {entity for entity, dist in distances.items() 
                     if dist >= (max_radius - shell_thickness)}
    
    return shell_entities, max_radius


def select_defect_entities_from_objects(entities, shell_entities, 
                                        defect_ratio, entity_type=None):
    """
    Randomly select entity objects to remove as defects from the interior.
    Works directly with entity objects.
    
    Parameters:
        entities: collection of entity objects
        shell_entities: set of entity objects in the outer shell (to preserve)
        defect_ratio: fraction of interior entities to remove (0.0 to 1.0)
        entity_type: 'Zr6', 'BDC', 'Mixed', or None
    
    Returns:
        set: entity objects to remove
    """
    interior_entities = []
    interior_bdc = []
    interior_zr = []
    for entity in entities:
        if entity in shell_entities:
            continue

        ent_type = getattr(entity, 'entity_type', None)
        if ent_type == 'Ligand':
            interior_bdc.append(entity)
        elif ent_type == 'Zr':
            interior_zr.append(entity)

        if entity_type is None:
            interior_entities.append(entity)
        elif entity_type == 'Zr6' and ent_type == 'Zr':
            interior_entities.append(entity)
        elif entity_type == 'BDC' and ent_type == 'Ligand':
            interior_entities.append(entity)

    if entity_type == 'Mixed':
        selected = set()
        num_bdc_defects = int(len(interior_bdc) * defect_ratio)
        num_zr_defects = int(len(interior_zr) * defect_ratio)
        if num_bdc_defects > 0:
            selected.update(np.random.choice(interior_bdc, size=num_bdc_defects, replace=False))
        if num_zr_defects > 0:
            selected.update(np.random.choice(interior_zr, size=num_zr_defects, replace=False))
        return selected

    num_defects = int(len(interior_entities) * defect_ratio)
    if num_defects > 0:
        return set(np.random.choice(interior_entities, size=num_defects, replace=False))
    return set()


def remove_defect_entities_from_assembly(entities, defect_entities, 
                                          free_cs, MC_free, Linker_free,
                                          linked_pairs, pair_index, ready_pairs):
    """
    Remove defect entities from the assembly data structures.
    
    Parameters:
        entities: RandomizedSet of entity objects
        defect_entities: set of entity objects to remove
        free_cs, MC_free, Linker_free: carboxylate sets
        linked_pairs: RandomizedSet of linked carboxylate pairs
        pair_index: dict mapping carboxylates to pairs
        ready_pairs: RandomizedSet of ready-to-connect pairs
    
    Returns:
        updated versions of all input data structures
    """
    # Collect all carboxylates from defect entities
    defect_carboxylates = set()
    for entity in defect_entities:
        if hasattr(entity, 'carboxylates'):
            defect_carboxylates.update(entity.carboxylates)
    
    # Helper function to create updated set (preserving type)
    def create_updated_set(original_set, filter_func):
        if isinstance(original_set, RandomizedSet):
            new_set = RandomizedSet()
            new_set.update([item for item in original_set if filter_func(item)])
            return new_set
        else:
            return set([item for item in original_set if filter_func(item)])
    
    # Collect counterpart carboxylates on surviving entities. These become
    # newly exposed growth sites when their paired defect-side linker/cluster
    # is removed.
    released_carboxylates = set()
    for pair_collection in (linked_pairs, ready_pairs):
        for pair in pair_collection:
            left, right = pair
            left_is_defect = left in defect_carboxylates
            right_is_defect = right in defect_carboxylates
            if left_is_defect == right_is_defect:
                continue

            surviving = right if left_is_defect else left
            surviving_entity = getattr(surviving, "belonging_entity", None)
            if surviving_entity is None or surviving_entity in defect_entities:
                continue
            released_carboxylates.add(surviving)

    # Remove defect entities
    updated_entities = create_updated_set(entities, lambda e: e not in defect_entities)
    
    # Remove carboxylates from defect entities
    updated_free_cs = create_updated_set(free_cs, lambda c: c not in defect_carboxylates)
    updated_MC_free = create_updated_set(MC_free, lambda c: c not in defect_carboxylates)
    updated_Linker_free = create_updated_set(Linker_free, lambda c: c not in defect_carboxylates)
    
    # Remove pairs containing defect carboxylates
    updated_linked_pairs = create_updated_set(
        linked_pairs,
        lambda pair: pair[0] not in defect_carboxylates and pair[1] not in defect_carboxylates
    )
    
    updated_ready_pairs = create_updated_set(
        ready_pairs,
        lambda pair: pair[0] not in defect_carboxylates and pair[1] not in defect_carboxylates
    )
    
    # Update pair_index
    updated_pair_index = {
        carb: pair for carb, pair in pair_index.items()
        if carb not in defect_carboxylates and 
           pair[0] not in defect_carboxylates and 
           pair[1] not in defect_carboxylates
    }

    # Re-expose newly released carboxylates as free growth sites.
    for carb in released_carboxylates:
        updated_free_cs.add(carb)
        entity = getattr(carb, "belonging_entity", None)
        if entity is None:
            continue
        if getattr(entity, "entity_type", None) == "Zr":
            updated_MC_free.add(carb)
        elif getattr(entity, "entity_type", None) == "Ligand":
            updated_Linker_free.add(carb)
    
    return (updated_entities, updated_free_cs, updated_MC_free, updated_Linker_free,
            updated_linked_pairs, updated_pair_index, updated_ready_pairs)


def remove_entities_from_structure(elements, coordinates, connectivity_map, 
                                   carboxylate_indices, element_entity_table,
                                   entities_to_remove):
    """
    Remove specified entities from the structure.
    
    Returns:
        new_elements, new_coordinates, new_connectivity_map, 
        new_carboxylate_indices, new_element_entity_table, atom_mapping
    """
    labels = element_entity_table[:, 1]
    n_atoms = len(elements)
    
    # Create mask for atoms to keep
    keep_mask = np.ones(n_atoms, dtype=bool)
    
    for entity in entities_to_remove:
        if entity.startswith("Zr6_"):
            # Remove all atoms belonging to this Zr6 (including bridges)
            for i, label in enumerate(labels):
                if label == entity or label.startswith(f"{entity}_"):
                    keep_mask[i] = False
        elif entity.startswith("BDC_"):
            # Remove all atoms belonging to this BDC (including bridges)
            for i, label in enumerate(labels):
                if label == entity or label.endswith(f"_{entity}") or f"_{entity}_" in label:
                    keep_mask[i] = False
    
    # Create mapping from old indices to new indices
    old_to_new = {}
    new_idx = 0
    for old_idx in range(n_atoms):
        if keep_mask[old_idx]:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    
    # Filter elements and coordinates
    new_elements = elements[keep_mask]
    new_coordinates = coordinates[keep_mask]
    new_element_entity_table = element_entity_table[keep_mask]
    
    # Update connectivity map
    new_connectivity_map = []
    for a, b, bond_order in connectivity_map:
        if keep_mask[a] and keep_mask[b]:
            new_connectivity_map.append((old_to_new[a], old_to_new[b], bond_order))
    new_connectivity_map = np.array(new_connectivity_map)
    
    # Update carboxylate indices
    new_carboxylate_indices = []
    for group in carboxylate_indices:
        c_idx, o1_idx, o2_idx = group
        if keep_mask[c_idx] and keep_mask[o1_idx] and keep_mask[o2_idx]:
            new_group = (old_to_new[c_idx], old_to_new[o1_idx], old_to_new[o2_idx])
            new_carboxylate_indices.append(new_group)
    
    return (new_elements, new_coordinates, new_connectivity_map, 
            new_carboxylate_indices, new_element_entity_table, old_to_new)


def write_mol2_file(filepath, elements, coordinates, connectivity_map, element_entity_table=None):
    """
    Write structure to MOL2 file format.
    
    Parameters:
        filepath: output file path
        elements: array of element symbols
        coordinates: array of atomic coordinates
        connectivity_map: array of bonds (atom1, atom2, bond_order)
        element_entity_table: optional, array with entity labels for each atom
    """
    n_atoms = len(elements)
    n_bonds = len(connectivity_map)
    
    with open(filepath, 'w') as f:
        # Header
        f.write("@<TRIPOS>MOLECULE\n")
        f.write("UiO-66_defective\n")
        f.write(f"{n_atoms} {n_bonds} 0 0 0\n")
        f.write("SMALL\n")
        f.write("NO_CHARGES\n\n")
        
        # Atoms
        f.write("@<TRIPOS>ATOM\n")
        for i, (elem, coord) in enumerate(zip(elements, coordinates)):
            x, y, z = coord
            # Get entity label if available
            if element_entity_table is not None:
                entity_label = element_entity_table[i, 1]  # Column 1 is the label
                subst_name = entity_label if entity_label != "UNASSIGNED" else "U66"
            else:
                subst_name = "U66"
            
            f.write(f"{i+1:7d} {elem}{i+1:<4d} {x:10.4f} {y:10.4f} {z:10.4f} "
                   f"{elem:>2s} 1 {subst_name} 0.0000\n")
        
        # Bonds
        if n_bonds > 0:
            f.write("@<TRIPOS>BOND\n")
            for i, (a, b, bond_order) in enumerate(connectivity_map):
                f.write(f"{i+1:7d} {a+1:7d} {b+1:7d} {bond_order:2d}\n")


def generate_defects(input_file, output_mol2, output_pkl, defect_ratio, 
                    shell_thickness, entity_type=None, random_seed=None):
    """
    Main function to generate defects in UiO-66 structure.
    
    NEW WORKFLOW:
    1. Load structure and assign entities
    2. Duplicate composite units (clean up entity labels)
    3. Select and remove defect entities from clean structure
    4. Instantiate and remove isolated entities
    
    Parameters:
        input_file: path to input MOL2 file
        output_mol2: path to output MOL2 file
        output_pkl: path to output PKL file
        defect_ratio: fraction of interior entities to remove (0.0 to 1.0)
        shell_thickness: thickness of outer shell to preserve (Angstroms)
        entity_type: 'Zr6', 'BDC', or None (both types)
        random_seed: random seed for reproducibility
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print(f"Loading structure from {input_file}...")
    elements, coordinates, connectivity_map, carboxylate_indices = read_mol_file(input_file)
    print(f"  Loaded {len(elements)} atoms, {len(connectivity_map)} bonds")
    
    print("\n=== STEP 1: Assigning entities ===")
    element_entity_table = assign_entities(elements, coordinates, connectivity_map, 
                                          carboxylate_indices)
    
    print("Assigning carboxylates...")
    (element_entity_table_1, Bridge_carboxylates, Zr6_carboxylates, 
     BDC_carboxylates, Unnormal_carboxylates) = assign_carboxylates(
        elements, coordinates, connectivity_map, carboxylate_indices, 
        element_entity_table)
    
    # Debug: count original structure
    all_labels = set(element_entity_table_1[:, 1])
    composite_labels = [l for l in all_labels if "_BDC_" in l and l.startswith("Zr6_")]
    print(f"  Total unique labels: {len(all_labels)}")
    print(f"  Composite labels (Zr6_X_BDC_Y): {len(composite_labels)}")
    
    print("\n=== STEP 2: Duplicating composite units ===")
    (dup_elements, dup_coords, dup_table, dup_carboxylate_indices, 
     dup_Bridge_carboxylates, invalidated_entities) = duplicate_composite_units_with_bdc(
        elements, coordinates, element_entity_table_1, 
        carboxylate_indices, Bridge_carboxylates,
        Zr6_carboxylates=Zr6_carboxylates, 
        BDC_carboxylates=BDC_carboxylates)
    
    # Count entities after duplication
    dup_labels = set(dup_table[:, 1])
    dup_valid_labels = [l for l in dup_labels if l != "UNASSIGNED" and (l.startswith("Zr6_") or l.startswith("BDC_"))]
    print(f"  Valid entity labels after duplication: {len(dup_valid_labels)}")
    print(f"  Invalidated entities: {len(invalidated_entities)}")
    if invalidated_entities:
        print(f"  WARNING: Found incomplete entities: {invalidated_entities}")
    
    print("\n=== STEP 3: Instantiating entities ===")
    # Instantiate BEFORE selecting defects - this is the key improvement!
    (entities, free_cs, MC_free, Linker_free, linked_pairs, 
     pair_index, ready_to_connect_carboxylate_pairs) = instantiate_all_entities(
        dup_table, dup_coords, dup_carboxylate_indices,
        Bridge_carboxylates=dup_Bridge_carboxylates,
        Zr6_carboxylates=Zr6_carboxylates,
        BDC_carboxylates=BDC_carboxylates)
    
    print(f"  Instantiated entities: {len(entities)}")
    num_zr6 = sum(1 for e in entities if getattr(e, 'entity_type', None) == 'Zr')
    num_bdc = sum(1 for e in entities if getattr(e, 'entity_type', None) == 'Ligand')
    print(f"  Zr6 clusters: {num_zr6}, BDC linkers: {num_bdc}")
    print(f"  Linked pairs: {len(linked_pairs)}")
    print(f"  Free carboxylates: {len(free_cs)}")
    
    print(f"\n=== STEP 4: Selecting defect entities ===")
    # Calculate structure center from entity objects
    structure_center = calculate_structure_center_from_entities(entities)
    
    # Identify shell entities
    print(f"Identifying shell entities (thickness={shell_thickness} Å)...")
    shell_entities, max_radius = identify_shell_entities_from_objects(
        entities, structure_center, shell_thickness)
    print(f"  Structure radius: {max_radius:.2f} Å")
    print(f"  Shell entities: {len(shell_entities)}")
    print(f"  Interior entities: {len(entities) - len(shell_entities)}")
    
    # Select defect entities
    print(f"\nSelecting defect entities (ratio={defect_ratio})...")
    defect_entities = select_defect_entities_from_objects(
        entities, shell_entities, defect_ratio, entity_type)
    print(f"  Selected {len(defect_entities)} entities to remove")
    
    if len(defect_entities) == 0:
        print("Warning: No entities selected for removal!")
        # Still need to save even if no defects
        defect_elements = dup_elements
        defect_coordinates = dup_coords
        defect_element_entity_table = dup_table
        updated_entities = entities
        updated_free_cs = free_cs
        updated_MC_free = MC_free
        updated_Linker_free = Linker_free
        updated_linked_pairs = linked_pairs
        updated_pair_index = pair_index
        updated_ready_to_connect_carboxylate_pairs = ready_to_connect_carboxylate_pairs
    else:
        print("\n=== STEP 5: Removing defect entities from assembly ===")
        # Remove from assembly data structures (much simpler!)
        (updated_entities, updated_free_cs, updated_MC_free, updated_Linker_free,
         updated_linked_pairs, updated_pair_index, 
         updated_ready_to_connect_carboxylate_pairs) = remove_defect_entities_from_assembly(
            entities, defect_entities, free_cs, MC_free, Linker_free,
            linked_pairs, pair_index, ready_to_connect_carboxylate_pairs)
        
        print(f"  Remaining entities: {len(updated_entities)} (removed {len(defect_entities)})")
        print(f"  Remaining linked pairs: {len(updated_linked_pairs)}")
        print(f"  Remaining free carboxylates: {len(updated_free_cs)}")
        
        # Remove from MOL2 structure (by entity labels)
        print("\n  Removing defect entities from MOL2 structure...")
        defect_labels = set()
        for entity in defect_entities:
            # Get entity label from entity object
            if hasattr(entity, 'entity_type') and hasattr(entity, 'entity_subtype'):
                if entity.entity_type == 'Zr':
                    defect_labels.add(f"Zr6_{entity.entity_subtype}")
                elif entity.entity_type == 'Ligand':
                    defect_labels.add(f"BDC_{entity.entity_subtype}")
        
        # Filter atoms by label
        labels = dup_table[:, 1]
        keep_mask = np.array([label not in defect_labels for label in labels], dtype=bool)
        
        defect_elements = dup_elements[keep_mask]
        defect_coordinates = dup_coords[keep_mask]
        defect_element_entity_table = dup_table[keep_mask]
        
        print(f"  Remaining atoms in MOL2: {len(defect_elements)} (removed {len(dup_elements) - len(defect_elements)})")
    
    # Now handle unconnected entities and save PKL
    if output_pkl:
        print(f"\n=== STEP 6: Removing unconnected entities and building PKL ===")
        
        # Remove unconnected entities (entities that lost all connections)
        print("Removing unconnected entities...")
        (defect_element_entity_table, final_entities, final_free_cs, final_MC_free, 
         final_Linker_free, final_linked_pairs, final_pair_index, 
         final_ready_pairs, removed_labels) = remove_unconnected_entities(
            defect_element_entity_table, defect_coordinates, updated_linked_pairs, 
            updated_entities, updated_free_cs, updated_MC_free, updated_Linker_free,
            updated_pair_index, updated_ready_to_connect_carboxylate_pairs, 
            verbose=True)
        
        # If unconnected entities were removed, also remove them from the MOL2 structure
        if len(removed_labels) > 0:
            print(f"\n  Removing {len(removed_labels)} unconnected entities from MOL2 structure...")
            print(f"  Unconnected entity labels: {sorted(removed_labels)}")
            
            # Simple removal by label matching
            iso_labels = defect_element_entity_table[:, 1]
            iso_keep_mask = np.array([label not in removed_labels for label in iso_labels], dtype=bool)
            
            defect_elements = defect_elements[iso_keep_mask]
            defect_coordinates = defect_coordinates[iso_keep_mask]
            defect_element_entity_table = defect_element_entity_table[iso_keep_mask]
            print(f"  MOL2 atoms after removing unconnected: {len(defect_elements)}")
        
        # Build assembly
        print("\n  Building assembly...")
        print(f"  Final entities in assembly: {len(final_entities)}")
        print(f"  Final linked pairs: {len(final_linked_pairs)}")
        print(f"  Final free carboxylates: {len(final_free_cs)}")
        
        asm = build_assembly_from_state(
            entities=final_entities,
            free_cs=final_free_cs,
            MC_free=final_MC_free,
            Linker_free=final_Linker_free,
            linked_pairs=final_linked_pairs,
            ready_pairs=final_ready_pairs,
            pair_index=final_pair_index,
            ZR6_PERCENTAGE=None,
            ENTROPY_GAIN=None,
            BUMPING_THRESHOLD=None
        )
        
        print(f"Writing PKL file to {output_pkl}...")
        os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
        save_ok = safe_pickle_save(
            assembly=asm,
            filepath=output_pkl,
            clean_connected_entities=True,         
            rebuild_after_save=True,               
            protocol=pickle.HIGHEST_PROTOCOL 
        )
        
        if save_ok:
            print(f"  Successfully saved assembly with {len(list(asm.entities))} entities")
        else:
            print("  Warning: Failed to save PKL file")
    else:
        # If not generating PKL, still need to save MOL2
        pass
    
    # Save MOL2 file
    print(f"\nWriting MOL2 file to {output_mol2}...")
    os.makedirs(os.path.dirname(output_mol2), exist_ok=True)
    
    if output_pkl:
        # Generate MOL2 from the final assembly (guaranteed to be consistent with PKL)
        print("  Generating MOL2 from assembly entities...")
        
        # Collect all atoms from entities in the assembly
        mol2_elements = []
        mol2_coordinates = []
        mol2_entity_labels = []
        
        for entity in final_entities:
            entity_label = f"{entity.entity_type}_{entity.entity_subtype}"
            # Convert entity_type to proper label format
            if entity.entity_type == 'Zr':
                entity_label = f"Zr6_{entity.entity_subtype}"
            elif entity.entity_type == 'Ligand':
                entity_label = f"BDC_{entity.entity_subtype}"
            
            # Get atoms from entity
            if hasattr(entity, 'elements') and hasattr(entity, 'coordinates'):
                for elem, coord in zip(entity.elements, entity.coordinates):
                    mol2_elements.append(elem)
                    mol2_coordinates.append(coord)
                    mol2_entity_labels.append(entity_label)
        
        # Convert to numpy arrays
        final_elements = np.array(mol2_elements)
        final_coordinates = np.array(mol2_coordinates, dtype=np.float32)
        
        # Create entity table
        final_entity_table = np.array([
            [str(i), label] for i, label in enumerate(mol2_entity_labels)
        ])
        
        print(f"  Collected {len(final_elements)} atoms from {len(final_entities)} entities")
    else:
        # If no PKL, filter out UNASSIGNED atoms
        print("  Filtering out UNASSIGNED atoms...")
        labels = defect_element_entity_table[:, 1]
        assigned_mask = np.array([label != "UNASSIGNED" for label in labels], dtype=bool)
        
        final_elements = defect_elements[assigned_mask]
        final_coordinates = defect_coordinates[assigned_mask]
        final_entity_table = defect_element_entity_table[assigned_mask]
        
        print(f"  Atoms before filtering: {len(defect_elements)}")
        print(f"  UNASSIGNED atoms: {len(defect_elements) - len(final_elements)}")
        print(f"  Atoms after filtering: {len(final_elements)}")
    
    empty_connectivity = np.array([])  # No bonds in output MOL2
    write_mol2_file(output_mol2, final_elements, final_coordinates, empty_connectivity, 
                   element_entity_table=final_entity_table)
    print(f"  Successfully saved MOL2 with {len(final_elements)} atoms (no bonds)")
    if output_pkl:
        print(f"  Note: MOL2 generated from assembly entities (guaranteed consistent with PKL)")
    else:
        print(f"  Note: Bonds are not included because connectivity was not updated after duplication")
    
    print("\n" + "="*60)
    print("Defect generation completed!")
    print("="*60)
    print(f"Input:  {input_file}")
    print(f"Output: {output_mol2}")
    if output_pkl:
        print(f"        {output_pkl}")
    print(f"Defect ratio: {defect_ratio:.2%}")
    print(f"Shell thickness: {shell_thickness} Å")
    print(f"Defect entities removed: {len(defect_entities)}")
    if output_pkl and len(removed_labels) > 0:
        print(f"Unconnected entities removed: {len(removed_labels)}")
        print(f"Total entities removed: {len(defect_entities) + len(removed_labels)}")
    print(f"Atoms in final MOL2 (assigned only): {len(final_elements)}")
    print(f"Total atoms removed (including UNASSIGNED): {len(dup_elements) - len(final_elements)}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate defects in UiO-66 structure while preserving outer shell')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input MOL2 file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output MOL2 file path')
    parser.add_argument('--pkl', '-p', type=str, default=None,
                       help='Output PKL file path (optional)')
    parser.add_argument('--defect_ratio', '-d', type=float, default=0.1,
                       help='Fraction of interior entities to remove (0.0-1.0, default: 0.1)')
    parser.add_argument('--shell_thickness', '-s', type=float, default=10.0,
                       help='Thickness of outer shell to preserve in Angstroms (default: 10.0)')
    parser.add_argument('--entity_type', '-t', type=str, default=None,
                       choices=['Zr6', 'BDC', None],
                       help='Type of entities to remove: Zr6, BDC, or both (default: both)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate defect ratio
    if not 0.0 <= args.defect_ratio <= 1.0:
        parser.error("defect_ratio must be between 0.0 and 1.0")
    
    generate_defects(
        input_file=args.input,
        output_mol2=args.output,
        output_pkl=args.pkl,
        defect_ratio=args.defect_ratio,
        shell_thickness=args.shell_thickness,
        entity_type=args.entity_type,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
