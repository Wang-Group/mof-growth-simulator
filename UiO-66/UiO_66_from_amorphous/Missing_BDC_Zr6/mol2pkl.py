import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque,defaultdict,Counter
import networkx as nx
from UiO66_Assembly_Large_Correction_conc import *

def read_mol_file(file_path):
    """
    Parse a .mol2 file and extract:
        1. Atom coordinates and element types
        2. Bond connectivity map (zero-based indices)
        3. Carboxylate groups (C bonded to two O atoms)

    Returns:
        elements (list[str]): list of atomic symbols
        coordinates (np.ndarray): shape (N, 3), atomic coordinates
        carboxylate_indices (list[np.ndarray]): list of arrays [C_idx, O_idx1, O_idx2]
    """
    elements = []
    coordinates = []
    connectivity_map = []

    in_atom_block = False
    in_bond_block = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # --- Block start/end detection ---
            if line.startswith('@<TRIPOS>ATOM'):
                in_atom_block = True
                in_bond_block = False
                continue
            elif line.startswith('@<TRIPOS>BOND'):
                in_atom_block = False
                in_bond_block = True
                continue
            elif line.startswith('@<TRIPOS>'):
                # Some other block, stop both atom/bond reading
                in_atom_block = False
                in_bond_block = False
                continue

            # --- Parse ATOM block ---
            if in_atom_block and line:
                # Example line:
                # 1 O1 19.4630 82.3280 82.3280 O 1 U66 0.0000
                parts = line.split()
                if len(parts) < 6:
                    continue
                x, y, z = map(np.float32, parts[2:5])
                atom_type = parts[5]  # "O" or "Zr", etc.
                elements.append(atom_type)
                coordinates.append([x, y, z])

            # --- Parse BOND block ---
            elif in_bond_block and line:
                # Example line:
                # 1 1 141 1
                parts = line.split()
                if len(parts) < 4:
                    continue
                atom1 = int(parts[1]) - 1  # zero-based
                atom2 = int(parts[2]) - 1
                bond_order = int(parts[3])
                connectivity_map.append((atom1, atom2, bond_order))

    elements = np.array(elements)
    coordinates = np.array(coordinates, dtype=np.float32)
    connectivity_map = np.array(connectivity_map)

    # --- Identify carboxylate groups ---
    oxygen_atoms = np.where(elements == 'O')[0]
    carbon_atoms = np.where(elements == 'C')[0]

    # Build C–O connectivity lookup
    connections = {}
    for a, b, _ in connectivity_map:
        if a in carbon_atoms and b in oxygen_atoms:
            connections.setdefault(a, []).append(b)
        elif b in carbon_atoms and a in oxygen_atoms:
            connections.setdefault(b, []).append(a)

    carboxylate_indices = []
    for c_idx, o_list in connections.items():
        if len(o_list) == 2:
            carboxylate_indices.append(tuple([c_idx] + o_list))
            # carboxylate_indices.append(np.array([c_idx] + o_list))

    return elements, coordinates, connectivity_map,carboxylate_indices

def assign_entities(elements, coordinates, connectivity_map, carboxylate_indices):
    n_atoms = len(elements)

    # --- Step 1. Collect all atoms in carboxylates ---
    carboxy_oxygens = set()
    carboxy_carbons = set()
    carboxy_atoms = set()  # All atoms in carboxylates (C + O)
    is_carboxylate_atom = np.zeros(n_atoms, dtype=bool)
    for group in carboxylate_indices:
        carboxy_carbons.add(group[0])
        carboxy_oxygens.update(group[1:])
        carboxy_atoms.update(group)  # Include all atoms in carboxylate group
        for idx in group:
            is_carboxylate_atom[idx] = True

    # --- Step 2. Build adjacency list with "carboxylate bridges removed" ---
    adj = [[] for _ in range(n_atoms)]
    for a, b, _ in connectivity_map:
        # Skip Zr–O(carboxylate) bonds
        if (elements[a] == 'Zr' and b in carboxy_oxygens) or (elements[b] == 'Zr' and a in carboxy_oxygens):
            continue
        adj[a].append(b)
        adj[b].append(a)

    # --- Step 3. Connected component search ---
    seen = list(is_carboxylate_atom)  # Carboxylate atoms are marked as already visited
    comps = []
    for i in range(n_atoms):
        if not seen[i]:
            q = deque([i])
            seen[i] = True
            comp = []
            while q:
                u = q.popleft()
                comp.append(u)
                for v in adj[u]:
                    if not seen[v] and not is_carboxylate_atom[v]:
                        seen[v] = True
                        q.append(v)
            comps.append(comp)

    # --- Step 4. Entity identification ---
    entity_labels = np.array(["UNASSIGNED"] * n_atoms, dtype=object)
    entity_counter = {"Zr6": 0, "BDC": 0, "CARBOXYLATE": 0}

    for comp in comps:
        comp_elems = elements[comp]
        zr_count = np.sum(comp_elems == 'Zr')
        o_count = np.sum(comp_elems == 'O')
        c_count = np.sum(comp_elems == 'C')

        # 1. Zr6O8 cluster (only count non-carboxylate atoms)
        if zr_count >= 6 and o_count >= 8:
            # Validate strict Zr6O8 composition: exactly 6 Zr, 8 O, and no other elements
            if not (zr_count == 6 and o_count == 8 and np.all(np.isin(comp_elems, ['Zr', 'O']))):
                continue
            entity_counter["Zr6"] += 1
            label = f"Zr6_{entity_counter['Zr6']}"
            entity_labels[comp] = label
            continue

        # 2. BDC skeleton (only search in non-carboxylate carbon atoms)
        if c_count >= 6:
            # Find six-membered carbon ring (exclude carboxylate carbons)
            carbon_nodes = [i for i in comp if elements[i] == 'C']
            if len(carbon_nodes) < 6:
                continue
            carbon_adj = {i: [j for j in adj[i] if elements[j] == 'C' and j not in carboxy_atoms] for i in carbon_nodes}

            def is_valid_6_ring(cycle):
                """
                Verify if it is a strict 6-membered ring: each carbon atom in the ring has exactly 2 neighbors within the ring.
                This excludes cases containing side chains like carboxylates.
                """
                cycle_set = set(cycle)
                for node in cycle:
                    # Count the number of neighbors of this node within the ring
                    ring_neighbors = [nbr for nbr in carbon_adj[node] if nbr in cycle_set]
                    # For a 6-membered ring, each carbon should have exactly 2 neighbors within the ring
                    if len(ring_neighbors) != 2:
                        return False
                return True

            def dfs(path, start, depth):
                if depth == 6 and start in carbon_adj[path[-1]]:
                    return path
                if depth >= 6:
                    return None
                for nbr in carbon_adj[path[-1]]:
                    if nbr in path:
                        continue
                    res = dfs(path + [nbr], start, depth + 1)
                    if res is not None:
                        return res
                return None

            found_cycle = None
            for c in carbon_nodes:
                cycle = dfs([c], c, 1)
                if cycle and is_valid_6_ring(cycle):
                    found_cycle = cycle
                    break

            if found_cycle:
                # Validate strict benzene ring: exactly 6 carbons and no other elements
                if not (len(comp) == 6 and np.all(comp_elems == 'C')):
                    continue
                entity_counter["BDC"] += 1
                label = f"BDC_{entity_counter['BDC']}"
                entity_labels[comp] = label
                continue

    # --- Step 5. Label carboxylates ---
    for group in carboxylate_indices:
        entity_counter["CARBOXYLATE"] += 1
        label = f"CARBOXYLATE_{entity_counter['CARBOXYLATE']}"
        entity_labels[list(group)] = label

    # --- Step 6. Output atom-entity correspondence ---
    element_entity_table = np.column_stack((elements, entity_labels))
    return element_entity_table

def visualize_entity(entity_name, element_entity_table, coordinates, connectivity_map,
                     figsize=(7, 6), show_bonds=True):
    """
    Visualize a specific molecular entity (e.g., Zr6_1 or BDC_2) in 3D.

    Parameters
    ----------
    entity_name : str
        Target entity label, e.g., 'Zr6_1', 'BDC_2'.
    element_entity_table : np.ndarray
        Array of shape (N, 2), where each row is [element_symbol, entity_label].
    coordinates : np.ndarray
        Atomic coordinates with shape (N, 3).
    connectivity_map : list of tuples
        Each entry is (atom1_index, atom2_index, bond_order).
    figsize : tuple
        Figure size for visualization.
    show_bonds : bool
        Whether to render bonds between atoms.
    """
    elements = element_entity_table[:, 0]
    entities = element_entity_table[:, 1]

    # --- 1. Select the target entity ---
    target_mask = (entities == entity_name)
    if not np.any(target_mask):
        print(f"[!] Entity '{entity_name}' not found.")
        return

    target_indices = np.where(target_mask)[0]
    target_coords = coordinates[target_mask]
    target_elements = elements[target_mask]

    # --- 2. Build graph and add bonds (within the same entity only) ---
    G = nx.Graph()
    for i in target_indices:
        G.add_node(i)

    if show_bonds:
        for a, b, _ in connectivity_map:
            if a in target_indices and b in target_indices:
                G.add_edge(a, b)

    # --- 3. Create 3D figure ---
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    plt.title(f"Entity: {entity_name}")

    # Define color scheme by element type
    color_map = {'Zr': 'orange', 'O': 'red', 'C': 'gray', 'H': 'white'}
    colors = [color_map.get(e, 'green') for e in target_elements]

    # Draw atoms
    for i, (x, y, z) in zip(target_indices, target_coords):
        ax.scatter(x, y, z, color=color_map.get(elements[i], 'green'), s=80, depthshade=True)
        ax.text(x, y, z, f"{elements[i]}{i}", fontsize=7)

    # Draw bonds
    if show_bonds:
        for a, b in G.edges():
            ax.plot(
                [coordinates[a][0], coordinates[b][0]],
                [coordinates[a][1], coordinates[b][1]],
                [coordinates[a][2], coordinates[b][2]],
                color='k', linewidth=1, alpha=0.6
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    plt.show()

def assign_carboxylates(elements, coordinates, connectivity_map, carboxylate_indices, element_entity_table):
    """
    Identify bridging carboxylate groups (C + 2 O) that connect BDC ligands and Zr6 clusters,
    and label their three atoms as Zr6_{i}_BDC_{j}.

    Parameters
    ----------
    elements : array-like
        Atom types (e.g., "Zr", "O", "C").
    coordinates : array-like
        Atom coordinates (N,3)
    connectivity_map : dict or list
        Atom neighbors: connectivity_map[i] = [j1, j2, ...]
    carboxylate_indices : list of tuples
        Each tuple contains indices of one carboxylate (C_idx, O1_idx, O2_idx)
    element_entity_table : array-like
        Current entity assignment of atoms (e.g., "Zr6_1", "BDC_2", or "UNASSIGNED")

    Returns
    -------
    new_table : array-like
        Updated entity table with bridge carboxylates labeled.
    """

    new_table = element_entity_table.copy()
    visited = set()

    n_atoms = len(elements)
    adj = [[] for _ in range(n_atoms)]
    for a, b, _ in connectivity_map:
        a, b = int(a), int(b)
        adj[a].append(b)
        adj[b].append(a)
    
    Bridge_carboxylates=[]
    Zr6_carboxylates=[]
    BDC_carboxylates=[]
    Unnormal_carboxylates=[]
    # Iterate over all carboxylate groups
    for group in carboxylate_indices:
        # group is a tuple: (C_idx, O1_idx, O2_idx)
        if group in visited:
            continue
        visited.add(group)

        c_idx, o1_idx, o2_idx = group
        o_indices = [o1_idx, o2_idx]

        # --- Step 1: find BDC entity connected to the C atom ---
        bdc_entities = set()
        bdc_neighbors = [n for n in adj[c_idx] if element_entity_table[n][1].startswith("BDC_")]
        for n in bdc_neighbors:
            bdc_entities.add(element_entity_table[n][1])

        # --- Step 2: find Zr6 entities connected to the O atoms ---
        zr_sets = [set(element_entity_table[n][1] for n in adj[o_idx] if element_entity_table[n][1].startswith("Zr6_")) for o_idx in o_indices]
        zr_entities = set.intersection(*zr_sets) if zr_sets else set()

        # --- Step 3: create bridge label ---
        if len(zr_entities) == 1 and len(bdc_entities)==1:
            bridge_label = f"{list(zr_entities)[0]}_{list(bdc_entities)[0]}"
            Bridge_carboxylates.append(group)
        elif len(zr_entities) == 1 and len(bdc_entities)==0:
            bridge_label = list(zr_entities)[0]
            Zr6_carboxylates.append(group)
        elif len(zr_entities) == 0 and len(bdc_entities)==1:
            bridge_label = list(bdc_entities)[0]
            BDC_carboxylates.append(group)
        else:
            print("There is something wrong with the carboxylate!!!")
            Unnormal_carboxylates.append(group)
            continue

        # --- Step 4: assign label to all three atoms ---
        for a in group:
            new_table[a,1] = bridge_label

    # --- Step 5: assign terminal alkyl carbons for Zr6-bound carboxylates ---
    for group in Zr6_carboxylates:
        c_idx, o1_idx, o2_idx = group
        zr_label = new_table[c_idx, 1]  # should be like "Zr6_1"

        for neighbor in adj[c_idx]:
            if elements[neighbor] == "C" and new_table[neighbor, 1] == "UNASSIGNED":
                new_table[neighbor, 1] = zr_label
                print(f"Assigned terminal carbon {neighbor} to {zr_label}")

    # --- Step 6: extend bridge labels to the directly bonded BDC carbon ---
    for group in Bridge_carboxylates:
        c_idx, o1_idx, o2_idx = group
        bridge_label = new_table[c_idx, 1]  # e.g. "Zr6_1_BDC_2"

        if "_BDC_" not in bridge_label:
            continue

        for neighbor in adj[c_idx]:
            if elements[neighbor] != "C":
                continue
            if new_table[neighbor, 1].startswith("BDC_"):
                old_label = new_table[neighbor, 1]
                new_table[neighbor, 1] = bridge_label
                print(f"Extended bridge label to BDC carbon {neighbor}: {old_label} → {bridge_label}")

    return new_table,Bridge_carboxylates,Zr6_carboxylates,BDC_carboxylates,Unnormal_carboxylates

def visualize_entities_by_type(entity_keyword, element_entity_table, coordinates, connectivity_map,
                               figsize=(7,6), show_bonds=True):
    """
    Visualize a molecular entity and its directly connected bridging atoms,
    based on pre-assigned composite labels.
    
    Parameters
    ----------
    entity_keyword : str
        Target entity, e.g., "Zr6_1" or "BDC_2".
    element_entity_table : np.ndarray
        Array of shape (N,2): [element_symbol, entity_label].
    coordinates : np.ndarray
        Atomic coordinates (N,3).
    connectivity_map : list of tuples
        Each entry: (atom1_index, atom2_index, bond_order)
    figsize : tuple
        Figure size for 3D plot.
    show_bonds : bool
        Whether to draw bonds.
    """

    elements = element_entity_table[:,0]
    labels = element_entity_table[:,1]

    # --- 1. Select main entity atoms and associated bridges ---
    if entity_keyword.startswith("Zr6"):
        target_mask = np.array([
            (label == entity_keyword) or label.startswith(f"{entity_keyword}_")
            for label in labels
        ])
    elif entity_keyword.startswith("BDC"):
        target_mask = np.array([
            (label == entity_keyword) or label.endswith(f"_{entity_keyword}")
            for label in labels
        ])
    target_indices = np.where(target_mask)[0]
    
    if len(target_indices) == 0:
        print(f"[!] Entity '{entity_keyword}' not found.")
        return

    target_coords = coordinates[target_indices]
    target_elements = elements[target_indices]
    target_labels = labels[target_indices]

    # --- 2. Build graph for bonds (only between selected atoms) ---
    G = nx.Graph()
    for i in target_indices:
        G.add_node(i)
    if show_bonds:
        for a,b,_ in connectivity_map:
            if a in target_indices and b in target_indices:
                G.add_edge(a,b)

    # --- 3. 3D visualization ---
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    plt.title(f"Entity: {entity_keyword} + connected bridges")

    color_map = {'Zr': 'orange', 'O':'red', 'C':'gray','H':'white'}

    for i, coords,e,label in zip(target_indices, target_coords, target_elements, target_labels):
        x,y,z=coords
        ax.scatter(x, y, z, color=color_map.get(e,'green'), s=80, depthshade=True)
        ax.text(x, y, z, f"{e}{i}", fontsize=7)

    if show_bonds:
        for a,b in G.edges():
            ax.plot(
                [coordinates[a][0], coordinates[b][0]],
                [coordinates[a][1], coordinates[b][1]],
                [coordinates[a][2], coordinates[b][2]],
                color='k', linewidth=1, alpha=0.6
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1,1,1])
    plt.show()

def duplicate_composite_units_with_bdc(elements, coords, element_entity_table, carboxylate_indices,Bridge_carboxylates,
                                       Zr6_carboxylates=None, BDC_carboxylates=None):
    """
    Duplicate composite-labeled units (COOC including BDC carbon) for Zr6-side.
    Original atoms are relabeled to BDC-side. Returns updated elements, coords, table, new_carboxylate_indices.
    Policy:
    - Start from a full copy of original carboxylate_indices to preserve all existing (Zr/BDC/free) groups
    - Append only the duplicated bridge carboxylate groups for the Zr side
    - Detect and handle incomplete entities (BDC != 12 atoms, Zr6 != 62 atoms)
    
    Parameters
    ----------
    elements : np.ndarray
        Atomic elements
    coords : np.ndarray
        Atomic coordinates
    element_entity_table : np.ndarray
        Entity assignment table
    carboxylate_indices : list
        List of carboxylate groups
    Bridge_carboxylates : list
        List of bridge carboxylate groups
    Zr6_carboxylates : list, optional
        List of Zr6 free carboxylate groups (for cleanup of incomplete entities)
    BDC_carboxylates : list, optional
        List of BDC free carboxylate groups (for cleanup of incomplete entities)
    
    Returns
    -------
    new_elements : np.ndarray
        Updated elements array
    new_coords : np.ndarray
        Updated coordinates array
    new_table : np.ndarray
        Updated entity table
    new_carboxylate_indices : list
        Updated carboxylate indices
    new_Bridge_carboxylates : list
        Updated bridge carboxylates (with incomplete entity bridges removed)
    invalidated_entities : set
        Set of entity labels that were invalidated (marked as UNASSIGNED)
    """
    new_elements = elements.copy()
    new_coords = coords.copy()
    new_table = element_entity_table.copy()
    # Preserve all original carboxylate groups by default
    new_carboxylate_indices = carboxylate_indices.copy()
    new_Bridge_carboxylates = Bridge_carboxylates.copy()
    
    # Track bridge carboxylate groups and their associated entity labels for later cleanup
    bridge_group_to_entities = {}  # {bridge_group: (zr_label, bdc_label)}

    for group in Bridge_carboxylates:
        c_idx, o1_idx, o2_idx = group

        # --- Step 1: Get carboxylate composite label ---
        label = new_table[c_idx, 1]

        parts = label.split("_")
        zr_label = "_".join(parts[:2])
        bdc_label = "_".join(parts[2:4])
        
        # Track which entities this bridge connects
        bridge_group_to_entities[group] = (zr_label, bdc_label)

        # --- Step 2: Update atom labels to BDC side ---
        group_indices = [c_idx, o1_idx, o2_idx]
        for idx in group_indices:
            new_table[idx, 1] = bdc_label  # Atoms belong to BDC side

        # --- Step 3: Duplicate carboxylate atoms to Zr6 side and record new indices ---
        copied_indices = []
        for idx in group_indices:
            new_elements = np.append(new_elements, elements[idx])
            new_coords = np.vstack([new_coords, coords[idx]])
            new_table = np.vstack([new_table, [elements[idx], zr_label]])  # Duplicated atoms belong to Zr6 side
            copied_indices.append(len(new_coords)-1)

        # --- Step 4: Update new carboxylate pairing list ---
        new_carboxylate_indices.append(tuple(copied_indices))
        new_Bridge_carboxylates.append(tuple(copied_indices))
        
        # Track duplicated bridge group
        bridge_group_to_entities[tuple(copied_indices)] = (zr_label, bdc_label)

        # --- Step 5: Duplicate BDC carbon connected to carboxylate ---
        # Find BDC carbon with composite label (C element, label == original composite label, not in carboxylate atoms)
        for idx, lbl in enumerate(element_entity_table[:,1]):
            if lbl == label and elements[idx] == "C" and idx not in group_indices:
                new_table[idx, 1] = bdc_label
                new_elements = np.append(new_elements, elements[idx])
                new_coords = np.vstack([new_coords, coords[idx]])
                new_table = np.vstack([new_table, [elements[idx], zr_label]]) 
                break  # Only duplicate one BDC carbon per carboxylate
    
    # --- Step 6: Detect and handle incomplete entities ---
    # Check entity atom counts after duplication
    invalidated_entities = set()
    entity_atom_counts = {}
    
    for i, label in enumerate(new_table[:, 1]):
        if label == "UNASSIGNED" or not (label.startswith("Zr6_") or label.startswith("Zr12_") or label.startswith("BDC_")):
            continue
        if label not in entity_atom_counts:
            entity_atom_counts[label] = 0
        entity_atom_counts[label] += 1
    
    # Check for incomplete entities
    for entity_label, atom_count in entity_atom_counts.items():
        expected_atoms = None
        if entity_label.startswith("Zr6") or entity_label.startswith("Zr12"):
            expected_atoms = 62
        elif entity_label.startswith("BDC"):
            expected_atoms = 12
        
        if expected_atoms is not None and atom_count != expected_atoms:
            print(f"  Warning: Entity {entity_label} has {atom_count} atoms; expected {expected_atoms} atoms. Marking as incomplete.")
            invalidated_entities.add(entity_label)
    
    # Process invalidated entities BEFORE marking atoms as UNASSIGNED
    # This is critical: we need to check entity labels before they are changed to UNASSIGNED
    # Step 6a: Remove free carboxylates on invalidated entities
    # Step 6b: Remove/convert bridge carboxylates connected to invalidated entities
    # Step 6c: Mark atoms as UNASSIGNED (last step)
    if invalidated_entities:
        # --- Step 6a: Remove free carboxylates on invalidated entities ---
        # Check which entity each free carboxylate belongs to by checking atom labels
        # Note: We check BEFORE marking atoms as UNASSIGNED, so labels are still valid
        if Zr6_carboxylates is not None:
            original_zr6_count = len(Zr6_carboxylates)
            valid_zr6_carboxylates = []
            for g in Zr6_carboxylates:
                # Check if any atom in this carboxylate group belongs to an invalidated entity
                belongs_to_invalidated = False
                entity_labels_found = set()
                for atom_idx in g:
                    if atom_idx < len(new_table):
                        atom_label = new_table[atom_idx, 1]
                        entity_labels_found.add(atom_label)
                        if atom_label in invalidated_entities:
                            belongs_to_invalidated = True
                            break
                
                # Also check if all atoms are UNASSIGNED (which might happen if entity was already invalidated)
                if not belongs_to_invalidated and all(l == "UNASSIGNED" for l in entity_labels_found if l):
                    # If all atoms are UNASSIGNED, this carboxylate is invalid
                    belongs_to_invalidated = True
                
                if not belongs_to_invalidated:
                    valid_zr6_carboxylates.append(g)
                else:
                    # Debug: print which entity this carboxylate belonged to
                    non_unassigned_labels = [l for l in entity_labels_found if l != "UNASSIGNED"]
                    if non_unassigned_labels:
                        print(f"  Removing Zr6 carboxylate {g}: atoms belong to {non_unassigned_labels}")
            
            Zr6_carboxylates[:] = valid_zr6_carboxylates
            removed_zr6 = original_zr6_count - len(Zr6_carboxylates)
            if removed_zr6 > 0:
                print(f"  Removed {removed_zr6} Zr6 free carboxylates from invalidated entities")
        
        if BDC_carboxylates is not None:
            original_bdc_count = len(BDC_carboxylates)
            valid_bdc_carboxylates = []
            for g in BDC_carboxylates:
                # Check if any atom in this carboxylate group belongs to an invalidated entity
                belongs_to_invalidated = False
                entity_labels_found = set()
                for atom_idx in g:
                    if atom_idx < len(new_table):
                        atom_label = new_table[atom_idx, 1]
                        entity_labels_found.add(atom_label)
                        if atom_label in invalidated_entities:
                            belongs_to_invalidated = True
                            break
                
                # Also check if all atoms are UNASSIGNED
                if not belongs_to_invalidated and all(l == "UNASSIGNED" for l in entity_labels_found if l):
                    belongs_to_invalidated = True
                
                if not belongs_to_invalidated:
                    valid_bdc_carboxylates.append(g)
                else:
                    # Debug: print which entity this carboxylate belonged to
                    non_unassigned_labels = [l for l in entity_labels_found if l != "UNASSIGNED"]
                    if non_unassigned_labels:
                        print(f"  Removing BDC carboxylate {g}: atoms belong to {non_unassigned_labels}")
            
            BDC_carboxylates[:] = valid_bdc_carboxylates
            removed_bdc = original_bdc_count - len(BDC_carboxylates)
            if removed_bdc > 0:
                print(f"  Removed {removed_bdc} BDC free carboxylates from invalidated entities")
        
        # --- Step 6b: Remove bridge carboxylates connected to invalidated entities and convert to free carboxylates ---
        # This must be done BEFORE marking atoms as UNASSIGNED, so we can check entity labels
        print(f"  Processing {len(invalidated_entities)} invalidated entities...")
        
        # Helper function to check if two carboxylate groups are spatially superimposed
        def carboxylates_overlap(group1, group2, coords, threshold=0.1):
            """Check if two carboxylate groups (C, O1, O2) are spatially superimposed."""
            threshold_sq = 3 * threshold ** 2
            
            # Compare carbon atoms
            c1_coord = coords[group1[0]]
            c2_coord = coords[group2[0]]
            diff = c1_coord - c2_coord
            if np.dot(diff, diff) > threshold_sq:
                return False
            
            # Compare oxygen atoms
            for o1_idx in group1[1:]:
                for o2_idx in group2[1:]:
                    o1_coord = coords[o1_idx]
                    o2_coord = coords[o2_idx]
                    diff = o1_coord - o2_coord
                    if np.dot(diff, diff) <= threshold_sq:
                        return True
            return False
        
        # Find bridge carboxylates connected to invalidated entities
        # Strategy: For each bridge carboxylate pair, check both sides:
        # - If one side is invalidated and the other is valid, convert the valid side to free carboxylate
        # - If both sides are invalidated, remove both
        # - If both sides are valid, keep as bridge (no action needed)
        bridges_to_remove = set()  # Use set to avoid duplicates
        bridges_to_convert_to_free = {}  # {bridge_group: target_type}
        processed_pairs = set()  # Track processed pairs to avoid double processing
        
        for bridge_group in new_Bridge_carboxylates:
            if bridge_group in processed_pairs:
                continue
            
            # Determine which entity this bridge carboxylate belongs to
            bridge_entity_label = None
            bridge_is_zr = False
            bridge_is_bdc = False
            
            if bridge_group in bridge_group_to_entities:
                zr_label, bdc_label = bridge_group_to_entities[bridge_group]
                # This bridge carboxylate connects Zr6 and BDC, so we need to find which side it belongs to
                # Check atom labels to determine which entity this carboxylate belongs to
                bridge_entity_labels = set()
                for atom_idx in bridge_group:
                    if atom_idx < len(new_table):
                        bridge_entity_labels.add(new_table[atom_idx, 1])
                
                # Determine which entity this carboxylate belongs to
                bridge_is_zr = any(l.startswith("Zr6") or l.startswith("Zr12") for l in bridge_entity_labels if l != "UNASSIGNED")
                bridge_is_bdc = any(l.startswith("BDC") for l in bridge_entity_labels if l != "UNASSIGNED")
                
                # Get the specific entity label for this carboxylate
                if bridge_is_zr:
                    bridge_entity_label = zr_label
                elif bridge_is_bdc:
                    bridge_entity_label = bdc_label
            else:
                # If not in tracking dict, check atom labels directly
                bridge_entity_labels = set()
                for atom_idx in bridge_group:
                    if atom_idx < len(new_table):
                        bridge_entity_labels.add(new_table[atom_idx, 1])
                
                bridge_is_zr = any(l.startswith("Zr6") or l.startswith("Zr12") for l in bridge_entity_labels if l != "UNASSIGNED")
                bridge_is_bdc = any(l.startswith("BDC") for l in bridge_entity_labels if l != "UNASSIGNED")
                
                # Get the entity label (should be consistent within the carboxylate)
                valid_labels = [l for l in bridge_entity_labels if l != "UNASSIGNED" and (l.startswith("Zr6") or l.startswith("Zr12") or l.startswith("BDC"))]
                if valid_labels:
                    bridge_entity_label = valid_labels[0]  # Should be consistent
            
            # Check if this bridge carboxylate belongs to an invalidated entity
            bridge_invalidated = bridge_entity_label in invalidated_entities if bridge_entity_label else False
            
            # Find the paired bridge carboxylate (spatially superimposed)
            paired_group = None
            paired_entity_label = None
            paired_is_zr = False
            paired_is_bdc = False
            
            for other_group in new_Bridge_carboxylates:
                if other_group == bridge_group or other_group in processed_pairs:
                    continue
                
                # Check if they are spatially superimposed (paired)
                if carboxylates_overlap(bridge_group, other_group, new_coords, threshold=0.1):
                    paired_group = other_group
                    
                    # Determine which entity the paired carboxylate belongs to
                    if other_group in bridge_group_to_entities:
                        other_zr, other_bdc = bridge_group_to_entities[other_group]
                        # Check atom labels to determine which side
                        other_entity_labels = set()
                        for atom_idx in other_group:
                            if atom_idx < len(new_table):
                                other_entity_labels.add(new_table[atom_idx, 1])
                        
                        paired_is_zr = any(l.startswith("Zr6") or l.startswith("Zr12") for l in other_entity_labels if l != "UNASSIGNED")
                        paired_is_bdc = any(l.startswith("BDC") for l in other_entity_labels if l != "UNASSIGNED")
                        
                        if paired_is_zr:
                            paired_entity_label = other_zr
                        elif paired_is_bdc:
                            paired_entity_label = other_bdc
                    else:
                        # Check atom labels directly
                        other_entity_labels = set()
                        for atom_idx in other_group:
                            if atom_idx < len(new_table):
                                other_entity_labels.add(new_table[atom_idx, 1])
                        
                        paired_is_zr = any(l.startswith("Zr6") or l.startswith("Zr12") for l in other_entity_labels if l != "UNASSIGNED")
                        paired_is_bdc = any(l.startswith("BDC") for l in other_entity_labels if l != "UNASSIGNED")
                        
                        valid_labels = [l for l in other_entity_labels if l != "UNASSIGNED" and (l.startswith("Zr6") or l.startswith("Zr12") or l.startswith("BDC"))]
                        if valid_labels:
                            paired_entity_label = valid_labels[0]
                    
                    break
            
            # Process the bridge pair based on validity
            if paired_group is not None:
                paired_invalidated = paired_entity_label in invalidated_entities if paired_entity_label else False
                
                # Case 1: Both sides are invalidated -> remove both
                if bridge_invalidated and paired_invalidated:
                    bridges_to_remove.add(bridge_group)
                    bridges_to_remove.add(paired_group)
                    processed_pairs.add(bridge_group)
                    processed_pairs.add(paired_group)
                    print(f"  Removing bridge pair (both invalidated): {bridge_group} <-> {paired_group}")
                
                # Case 2: Bridge side invalidated, paired side valid -> convert paired to free, remove bridge
                elif bridge_invalidated and not paired_invalidated:
                    bridges_to_remove.add(bridge_group)
                    if paired_is_zr:
                        bridges_to_convert_to_free[paired_group] = 'Zr6'
                    elif paired_is_bdc:
                        bridges_to_convert_to_free[paired_group] = 'BDC'
                    processed_pairs.add(bridge_group)
                    processed_pairs.add(paired_group)
                    print(f"  Bridge {bridge_group} invalidated, converting paired {paired_group} to free carboxylate")
                
                # Case 3: Bridge side valid, paired side invalidated -> convert bridge to free, remove paired
                elif not bridge_invalidated and paired_invalidated:
                    bridges_to_remove.add(paired_group)
                    if bridge_is_zr:
                        bridges_to_convert_to_free[bridge_group] = 'Zr6'
                    elif bridge_is_bdc:
                        bridges_to_convert_to_free[bridge_group] = 'BDC'
                    processed_pairs.add(bridge_group)
                    processed_pairs.add(paired_group)
                    print(f"  Paired {paired_group} invalidated, converting bridge {bridge_group} to free carboxylate")
                
                # Case 4: Both sides valid -> keep as bridge (no action)
                else:
                    processed_pairs.add(bridge_group)
                    processed_pairs.add(paired_group)
            
            else:
                # No paired carboxylate found - if invalidated, just remove it
                if bridge_invalidated:
                    bridges_to_remove.add(bridge_group)
                    processed_pairs.add(bridge_group)
                    print(f"  Removing unpaired bridge carboxylate {bridge_group} (invalidated)")
        
        # Remove bridge carboxylates connected to invalidated entities
        for bridge_group in bridges_to_remove:
            if bridge_group in new_Bridge_carboxylates:
                new_Bridge_carboxylates.remove(bridge_group)
        
        # Convert paired bridge carboxylates to free carboxylates
        if Zr6_carboxylates is not None and BDC_carboxylates is not None:
            for bridge_group, target_type in bridges_to_convert_to_free.items():
                if bridge_group in new_Bridge_carboxylates:
                    new_Bridge_carboxylates.remove(bridge_group)
                    if target_type == 'Zr6':
                        Zr6_carboxylates.append(bridge_group)
                        print(f"  Converted bridge carboxylate {bridge_group} to Zr6 free carboxylate")
                    elif target_type == 'BDC':
                        BDC_carboxylates.append(bridge_group)
                        print(f"  Converted bridge carboxylate {bridge_group} to BDC free carboxylate")
        
        # --- Step 6c: Mark all atoms of invalidated entities as UNASSIGNED (LAST STEP) ---
        # This must be done AFTER all checks that rely on entity labels
        for entity_label in invalidated_entities:
            for i, lbl in enumerate(new_table[:, 1]):
                if lbl == entity_label:
                    new_table[i, 1] = "UNASSIGNED"
        
        print(f"  Invalidated {len(invalidated_entities)} entities: {sorted(invalidated_entities)}")

    return new_elements, new_coords, new_table, new_carboxylate_indices, new_Bridge_carboxylates, invalidated_entities

def instantiate_entity_from_global(entity_label, element_entity_table, coordinates, carboxylate_indices):
    """
    Extract a single entity from the global table and convert it into an SBU_Entity instance
    (with locally indexed carboxylates).
    """
    global_indices = [i for i, label in enumerate(element_entity_table[:, 1])
                      if label == entity_label and label != "UNASSIGNED"]
    if len(global_indices) == 0:
        print(f"[Warning] No atoms found for entity {entity_label}, skipped.")
        return None

    local_coords = coordinates[global_indices].copy()
    local_elements = element_entity_table[global_indices, 0].copy()

    global_to_local = {g: l for l, g in enumerate(global_indices)}

    entity_carboxylates = []
    for group in carboxylate_indices:
        try:
            local_group = [global_to_local[idx] for idx in group]
            entity_carboxylates.append(local_group)
        except KeyError:
            continue

    center = np.mean(local_coords, axis=0)
    radius = np.max(np.linalg.norm(local_coords - center, axis=1)) + 3.0

    if entity_label.startswith("Zr6"):
        entity = Zr6_AA()
    elif entity_label.startswith("Zr12"):
        entity = Zr12_AA()
    elif entity_label.startswith("BDC"):
        entity = BDC()
    else:
        raise ValueError(
            f"Unknown entity type for label '{entity_label}'. "
            f"Expected entity labels starting with 'Zr6_', 'Zr12_', or 'BDC_'. "
            f"This may indicate an issue with entity assignment in the structure."
        )

    entity.elements = local_elements
    entity.coordinates = local_coords
    # Ensure atom_indexes are arrays (consistent with Assembly definition)
    entity.carboxylates = [Carboxylate(np.asarray(c), entity) for c in entity_carboxylates]
    entity.center = center
    entity.radius = radius

    return entity


def instantiate_all_entities(element_entity_table, coordinates, carboxylate_indices, 
                             Bridge_carboxylates=None, Zr6_carboxylates=None, BDC_carboxylates=None):
    """
    Instantiate all entities (Zr6_*, Zr12_*, BDC_*) and return entities along with classified carboxylates.
    
    Parameters
    ----------
    element_entity_table : np.ndarray
        Array of shape (N, 2), where each row is [element_symbol, entity_label].
    coordinates : np.ndarray
        Atomic coordinates with shape (N, 3).
    carboxylate_indices : list
        List of carboxylate groups (global indices).
    Bridge_carboxylates : list, optional
        List of bridge carboxylate groups (global indices) that connect entities.
    Zr6_carboxylates : list, optional
        List of free carboxylate groups belonging to Zr6 entities (global indices).
    BDC_carboxylates : list, optional
        List of free carboxylate groups belonging to BDC entities (global indices).
    
    Returns
    -------
    assembly_entities : RandomizedSet
        Set of instantiated SBU_Entity objects.
    free_carboxylates : RandomizedSet
        Set of all free carboxylate objects (with local indices).
    MC_free_carboxylates : RandomizedSet
        Set of free carboxylates on metal cluster entities.
    Linker_free_carboxylates : RandomizedSet
        Set of free carboxylates on linker entities.
    linked_carboxylate_pairs : RandomizedSet
        Set of linked carboxylate pairs (bridge carboxylates).
    pair_index : dict
        Dictionary mapping each carboxylate to its pair (consistent with Assembly definition).
        Keys are Carboxylate objects, values are tuples (carb1, carb2).
    ready_to_connect_carboxylate_pairs : RandomizedSet
        Set of carboxylate pairs that are ready to connect (spatially superimposed)
        but not yet linked. These are free carboxylates from MC and Linker entities.
    """
    # Initialize default values
    if Bridge_carboxylates is None:
        Bridge_carboxylates = []
    if Zr6_carboxylates is None:
        Zr6_carboxylates = []
    if BDC_carboxylates is None:
        BDC_carboxylates = []
    
    print(f"\n=== Starting instantiate_all_entities ===")
    print(f"Input parameters:")
    print(f"  Bridge_carboxylates: {len(Bridge_carboxylates)} groups")
    print(f"  Zr6_carboxylates: {len(Zr6_carboxylates)} groups")
    print(f"  BDC_carboxylates: {len(BDC_carboxylates)} groups")
    print(f"  carboxylate_indices: {len(carboxylate_indices)} groups")
    print(f"  element_entity_table shape: {element_entity_table.shape}")
    
    unique_entities = sorted(set([
        label for label in element_entity_table[:, 1]
        if label != "UNASSIGNED" and (label.startswith("Zr6_") or label.startswith("Zr12_") or label.startswith("BDC_"))
    ]))

    assembly_entities = RandomizedSet()
    free_carboxylates = RandomizedSet()
    MC_free_carboxylates = RandomizedSet()
    Linker_free_carboxylates = RandomizedSet()
    linked_carboxylate_pairs = RandomizedSet()
    pair_index = {}  # Dictionary to map each carboxylate to its pair (consistent with Assembly definition)
    
    print(f"\nFound {len(unique_entities)} unique entities to instantiate.")

    # Step 1: Instantiate all entities and build global/local index mappings
    entity_label_to_entity = {}
    global_idx_to_entity_label = {}
    entity_label_to_global_indices = {}
    entity_label_to_global_to_local = {}
    entity_label_to_carb_global_map = {}  # maps sorted global triple -> Carboxylate
    
    for entity_label in unique_entities:
        entity = instantiate_entity_from_global(entity_label, element_entity_table, coordinates, carboxylate_indices)
        if entity is not None:
            assembly_entities.add(entity)
            entity_label_to_entity[entity_label] = entity
            # Build global index list for this entity
            # After duplicate_composite_units_with_bdc, labels should be exact (e.g., "Zr6_1", "BDC_2")
            # So we use exact matching here, consistent with instantiate_entity_from_global
            global_indices = [i for i, label in enumerate(element_entity_table[:, 1]) 
                            if label == entity_label and label != "UNASSIGNED"]
            if entity_label.startswith("Zr6"):
                expected_atoms = 62
            elif entity_label.startswith("BDC"):
                expected_atoms = 12
            else:
                expected_atoms = None
            if expected_atoms is not None and len(global_indices) != expected_atoms:
                raise ValueError(
                    f"Entity {entity_label} has {len(global_indices)} atoms; expected {expected_atoms} atoms."
                )
            entity_label_to_global_indices[entity_label] = global_indices
            # global->local mapping (based on enumerate index order)
            global_to_local = {int(g): l for l, g in enumerate(global_indices)}
            entity_label_to_global_to_local[entity_label] = global_to_local
            # update reverse mapping for quick entity lookup
            for g_idx in global_indices:
                global_idx_to_entity_label[int(g_idx)] = entity_label
            # build carboxylate global triple -> Carboxylate map
            carb_map = {}
            for carb in entity.carboxylates:
                local_idxs = np.asarray(carb.atom_indexes)
                try:
                    global_triple = tuple(int(global_indices[int(l)]) for l in local_idxs)
                except Exception:
                    continue
                carb_map[tuple(sorted(global_triple))] = carb
            entity_label_to_carb_global_map[entity_label] = carb_map
            
            print(f"Instantiated entity: {entity_label} ({entity.entity_type}), atoms={len(global_indices)}, carboxylates={len(entity.carboxylates)}")

    print(f"\nTotal instantiated entities: {len(assembly_entities)}")
    print(f"Built global_idx_to_entity_label mapping with {len(global_idx_to_entity_label)} entries")
    
    # Step 2: Process bridge carboxylates (linked pairs)
    # After duplication, bridge carboxylates are fully contained in each entity
    # Each bridge carboxylate has a pair: one in Zr6 entity, one in BDC entity
    # We need to pair them based on the original composite labels (e.g., Zr6_1_BDC_2)
    print(f"\n=== Step 2: Processing bridge carboxylates ===")
    print(f"Processing {len(Bridge_carboxylates)} bridge carboxylates...")
    bridge_matched = 0
    
    # Group bridge carboxylates by entity
    bridge_by_entity = {}  # {entity_label: [(bridge_group, carb_obj), ...]}
    for bridge_group in Bridge_carboxylates:
        # Convert to tuple of ints to ensure consistent type matching
        bridge_group_ints = tuple(int(idx) for idx in bridge_group)
        
        # Find which entity this bridge carboxylate belongs to (after duplication, all atoms belong to one entity)
        # Strict check: all three atoms (C, O1, O2) must belong to the same entity
        entity_labels = []
        for atom_idx in bridge_group_ints:
            if atom_idx in global_idx_to_entity_label:
                entity_labels.append(global_idx_to_entity_label[atom_idx])
            else:
                raise ValueError(
                    f"Bridge carboxylate atom {atom_idx} (in group {bridge_group_ints}) not found in any entity. "
                    f"This indicates a problem with entity assignment or bridge carboxylate indexing."
                )
        
        # Check that all atoms belong to the same entity
        if len(set(entity_labels)) != 1:
            raise ValueError(
                f"Bridge carboxylate {bridge_group_ints} atoms belong to different entities: {set(entity_labels)}. "
                f"All three atoms (C, O1, O2) must belong to the same entity after duplication."
            )
        
        entity_label = entity_labels[0]
        
        # Find the carboxylate object in this entity
        entity = entity_label_to_entity.get(entity_label)
        if entity is None:
            raise ValueError(
                f"Entity {entity_label} not found for bridge carboxylate {bridge_group_ints}. "
                f"This indicates an inconsistency between entity_label_to_entity mapping and global_idx_to_entity_label."
            )
        
        global_indices_entity = [i for i, label in enumerate(element_entity_table[:, 1])
                                if label == entity_label and label != "UNASSIGNED"]
        
        # Find the Carboxylate object in this entity that matches bridge_group_ints
        # bridge_group_ints: global indices of bridge carboxylate (C, O1, O2), e.g., (100, 200, 300)
        # entity.carboxylates: Carboxylate objects with local indices, e.g., [0, 1, 2]
        # We need to convert local indices to global indices for comparison
        matched_carb = None
        for local_carb in entity.carboxylates:
            local_indices = np.asarray(local_carb.atom_indexes)
            # Convert local indices to global indices
            global_carb_indices = tuple(global_indices_entity[int(local_idx)] for local_idx in local_indices)
            global_carb_indices_ints = tuple(int(idx) for idx in global_carb_indices)
            # Compare the sets to match the carboxylate (order may vary, but same atoms)
            if set(bridge_group_ints) == set(global_carb_indices_ints):
                matched_carb = local_carb
                break
        
        if matched_carb is None:
            raise ValueError(
                f"Bridge carboxylate {bridge_group_ints} not found in entity {entity_label}'s carboxylates. "
                f"This indicates a mismatch between Bridge_carboxylates and entity carboxylate list."
            )
        
        if entity_label not in bridge_by_entity:
            bridge_by_entity[entity_label] = []
        bridge_by_entity[entity_label].append((bridge_group_ints, matched_carb))
    
    # Now pair bridge carboxylates: Zr6 <-> BDC
    # Strategy: For each Zr6 bridge carboxylate, find a BDC bridge carboxylate
    # We'll match them by trying all combinations and checking if they should be paired
    processed_carbs = set()
    
    for zr_label, zr_bridges in bridge_by_entity.items():
        if not (zr_label.startswith("Zr6") or zr_label.startswith("Zr12")):
            continue
        
        for zr_bridge_group, zr_carb in zr_bridges:
            if zr_carb in processed_carbs:
                continue
            
            # Try to find a matching BDC bridge carboxylate by checking spatial overlap
            # Bridge carboxylates from Zr6 and BDC should have overlapping coordinates
            # because they represent the same physical carboxylate (one duplicated for each side)
            bdc_carb = None
            bdc_label = None
            
            for bdc_entity_label, bdc_bridges in bridge_by_entity.items():
                if not bdc_entity_label.startswith("BDC"):
                    continue
                
                for bdc_bridge_group, bdc_carb_candidate in bdc_bridges:
                    if bdc_carb_candidate in processed_carbs:
                        continue
                    
                    # Check if these two carboxylates are spatially superimposed (overlapping)
                    # This verifies they represent the same physical carboxylate
                    if zr_carb.carboxylates_superimposed(bdc_carb_candidate,threshold=0.1):
                        bdc_carb = bdc_carb_candidate
                        bdc_label = bdc_entity_label
                        break
                
                if bdc_carb is not None:
                    break
            
            if zr_carb is not None and bdc_carb is not None:
                pair = (zr_carb, bdc_carb)
                linked_carboxylate_pairs.add(pair)
                # Update pair_index
                for c in pair:
                    pair_index[c] = pair
                # Update connected_entities
                entity1_obj = zr_carb.belonging_entity
                entity2_obj = bdc_carb.belonging_entity
                if not hasattr(entity1_obj, 'connected_entities') or entity1_obj.connected_entities is None:
                    entity1_obj.connected_entities = []
                if not hasattr(entity2_obj, 'connected_entities') or entity2_obj.connected_entities is None:
                    entity2_obj.connected_entities = []
                if entity2_obj not in entity1_obj.connected_entities:
                    entity1_obj.connected_entities.append(entity2_obj)
                if entity1_obj not in entity2_obj.connected_entities:
                    entity2_obj.connected_entities.append(entity1_obj)
                bridge_matched += 1
                processed_carbs.add(zr_carb)
                processed_carbs.add(bdc_carb)
                print(f"  Linked pair: {zr_label} <-> {bdc_label} (matched by spatial overlap)")
            elif zr_carb is not None:
                # If we couldn't find a matching BDC carboxylate, this is an error
                raise ValueError(
                    f"Could not find matching BDC bridge carboxylate for Zr6 carboxylate "
                    f"({zr_label}, global indices: {zr_bridge_group}). "
                    f"This may indicate a problem with bridge carboxylate duplication or spatial alignment."
                )
    
    print(f"  Matched {bridge_matched} bridge carboxylate pairs")
    print(f"    Grouped into {len(bridge_by_entity)} entities")
    
    # Step 3: Process free carboxylates (Zr6 and BDC)
    print(f"\n=== Step 3: Processing free carboxylates ===")
    print(f"Processing {len(Zr6_carboxylates)} Zr6 free carboxylates...")
    zr6_matched = 0
    
    # Precompute fast lookup: entity -> carb_global_map
    # (already built: entity_label_to_carb_global_map)
    
    for zr6_group in Zr6_carboxylates:
        # Convert to tuple of ints to ensure consistent type matching
        zr6_group_ints = tuple(sorted(int(idx) for idx in zr6_group))
        
        matched = False
        # search across all Zr6/Zr12 entities (more robust than relying on one atom's label)
        for ent_label, carb_map in entity_label_to_carb_global_map.items():
            if not (ent_label.startswith("Zr6") or ent_label.startswith("Zr12")):
                continue
            carb = carb_map.get(zr6_group_ints)
            if carb is not None:
                free_carboxylates.add(carb)
                MC_free_carboxylates.add(carb)
                zr6_matched += 1
                matched = True
                break
        
        if not matched:
            raise ValueError(
                f"Zr6 carboxylate {zr6_group_ints} not matched in any Zr6 entity. "
                f"This indicates a problem with carboxylate indexing or entity assignment."
            )
    
    print(f"  Matched {zr6_matched}/{len(Zr6_carboxylates)} Zr6 free carboxylates")
    
    print(f"\nProcessing {len(BDC_carboxylates)} BDC free carboxylates...")
    bdc_matched = 0
    
    for bdc_group in BDC_carboxylates:
        bdc_group_ints = tuple(sorted(int(idx) for idx in bdc_group))
        matched = False
        for ent_label, carb_map in entity_label_to_carb_global_map.items():
            if not ent_label.startswith("BDC"):
                continue
            carb = carb_map.get(bdc_group_ints)
            if carb is not None:
                free_carboxylates.add(carb)
                Linker_free_carboxylates.add(carb)
                bdc_matched += 1
                matched = True
                break
        
        if not matched:
            raise ValueError(
                f"BDC carboxylate {bdc_group_ints} not matched in any BDC entity. "
                f"This indicates a problem with carboxylate indexing or entity assignment."
            )
    
    print(f"  Matched {bdc_matched}/{len(BDC_carboxylates)} BDC free carboxylates")
    
    print(f"\nSummary:")
    print(f"  Total entities: {len(assembly_entities)}")
    print(f"  Free carboxylates: {len(free_carboxylates)}")
    print(f"  MC free carboxylates: {len(MC_free_carboxylates)}")
    print(f"  Linker free carboxylates: {len(Linker_free_carboxylates)}")
    print(f"  Linked carboxylate pairs: {len(linked_carboxylate_pairs)}")
    print(f"  Pair index entries: {len(pair_index)}")
    
    # Clear KDTree for all entities (KDTree will be rebuilt on demand via lazy loading)
    # This ensures consistency with build_assembly_from_state and prevents serialization issues
    kdtree_cleared = 0
    for entity in assembly_entities:
        if hasattr(entity, 'kdtree'):
            if entity.kdtree is not None:
                kdtree_cleared += 1
            entity.kdtree = None
    if kdtree_cleared > 0:
        print(f"\nCleared {kdtree_cleared} KDTree objects from entities (will be rebuilt on demand)")
    
    # Final verification: Ensure all carboxylates have valid belonging_entity
    # This is a defensive check to catch any unexpected None values
    # All carboxylates should have belonging_entity set when created via Carboxylate(atom_indexes, entity)
    print(f"\n=== Final verification: Checking belonging_entity references ===")
    all_carbs_to_check = []
    all_carbs_to_check.extend(free_carboxylates)
    all_carbs_to_check.extend(MC_free_carboxylates)
    all_carbs_to_check.extend(Linker_free_carboxylates)
    for pair in linked_carboxylate_pairs:
        all_carbs_to_check.extend(pair)
    
    missing_belonging_entity = []
    for carb in all_carbs_to_check:
        if not hasattr(carb, 'belonging_entity') or carb.belonging_entity is None:
            missing_belonging_entity.append(carb)
    
    if missing_belonging_entity:
        raise ValueError(
            f"{len(missing_belonging_entity)} carboxylates have None belonging_entity. "
            f"This indicates a problem in entity instantiation. "
            f"All carboxylates should have belonging_entity set when created."
        )
    
    print(f"  All {len(all_carbs_to_check)} carboxylates have valid belonging_entity references")
    
    # Step 4: Build ready_to_connect_carboxylate_pairs
    # Find all pairs of free carboxylates (MC and Linker) that can be superimposed but are not yet linked
    print(f"\n=== Step 4: Building ready_to_connect_carboxylate_pairs ===")
    ready_to_connect_carboxylate_pairs = RandomizedSet()
    
    # Create set of already linked carboxylates for fast lookup
    linked_carboxylates_set = set()
    for pair in linked_carboxylate_pairs:
        linked_carboxylates_set.update(pair)
    
    # Iterate over all MC free carboxylates and Linker free carboxylates
    # Check if they are spatially superimposed (ready to connect)
    ready_matched = 0
    for mc_carb in MC_free_carboxylates:
        # Skip if already linked
        if mc_carb in linked_carboxylates_set:
            continue
        
        # Try to find a matching Linker carboxylate
        for linker_carb in Linker_free_carboxylates:
            # Skip if already linked
            if linker_carb in linked_carboxylates_set:
                continue
            
            # Skip if same entity
            if mc_carb.belonging_entity is linker_carb.belonging_entity:
                continue
            
            # Check if they are spatially superimposed (ready to connect)
            if mc_carb.carboxylates_superimposed(linker_carb, threshold=0.1):
                pair = (mc_carb, linker_carb)
                ready_to_connect_carboxylate_pairs.add(pair)
                # Update pair_index
                pair_index[mc_carb] = pair
                pair_index[linker_carb] = pair
                ready_matched += 1
                # Mark as used (one carboxylate can only pair with one other)
                linked_carboxylates_set.add(mc_carb)
                linked_carboxylates_set.add(linker_carb)
                break
    
    print(f"  Found {ready_matched} ready-to-connect carboxylate pairs")
    print(f"  (MC free: {len(MC_free_carboxylates)}, Linker free: {len(Linker_free_carboxylates)})")
    
    return (assembly_entities, free_carboxylates, MC_free_carboxylates, 
            Linker_free_carboxylates, linked_carboxylate_pairs, pair_index, ready_to_connect_carboxylate_pairs)


def remove_unconnected_entities(element_entity_table, coordinates, linked_carboxylate_pairs, entities,
                                 free_carboxylates, MC_free_carboxylates, Linker_free_carboxylates,
                                 pair_index, ready_to_connect_carboxylate_pairs, verbose=True):
    """
    Remove entities that have no connections to other entities.
    If all carboxylate groups of an entity are not in linked carboxylate pairs, 
    mark all atoms of that entity as UNASSIGNED and clean up related carboxylate data structures.
    
    Parameters
    ----------
    element_entity_table : np.ndarray
        Array of shape (N, 2), where each row is [element_symbol, entity_label].
    coordinates : np.ndarray
        Atomic coordinates with shape (N, 3).
    linked_carboxylate_pairs : RandomizedSet or set
        Set of linked carboxylate pairs (tuples of Carboxylate objects).
    entities : RandomizedSet or list
        Set or list of all entity objects.
    free_carboxylates : RandomizedSet or set
        Set of all free carboxylate objects.
    MC_free_carboxylates : RandomizedSet or set
        Set of free carboxylates on metal cluster entities.
    Linker_free_carboxylates : RandomizedSet or set
        Set of free carboxylates on linker entities.
    pair_index : dict
        Dictionary mapping each carboxylate to its pair.
    ready_to_connect_carboxylate_pairs : RandomizedSet or set
        Set of carboxylate pairs that are ready to connect.
    verbose : bool
        Whether to print detailed information about removed entities.
    
    Returns
    -------
    new_table : np.ndarray
        Updated element_entity_table with unconnected entities marked as UNASSIGNED.
    updated_entities : RandomizedSet or list
        Updated entities set with unconnected entities removed.
    updated_free_carboxylates : RandomizedSet or set
        Updated free carboxylates with unconnected entity carboxylates removed.
    updated_MC_free_carboxylates : RandomizedSet or set
        Updated MC free carboxylates with unconnected entity carboxylates removed.
    updated_Linker_free_carboxylates : RandomizedSet or set
        Updated Linker free carboxylates with unconnected entity carboxylates removed.
    updated_linked_carboxylate_pairs : RandomizedSet or set
        Updated linked carboxylate pairs with unconnected entity pairs removed.
    updated_pair_index : dict
        Updated pair_index with unconnected entity entries removed.
    updated_ready_to_connect_carboxylate_pairs : RandomizedSet or set
        Updated ready_to_connect pairs with unconnected entity pairs removed.
    removed_entity_labels : set
        Set of entity labels that were removed (marked as UNASSIGNED).
    
    Example
    -------
    Call this function after instantiate_all_entities to remove unconnected entities:
    
    >>> (entities, free_cs, MC_free, Linker_free, linked_pairs, pair_index, ready_pairs) = instantiate_all_entities(...)
    >>> (new_table, updated_entities, updated_free_cs, updated_MC_free, updated_Linker_free,
    ...  updated_linked_pairs, updated_pair_index, updated_ready_pairs, removed_labels) = remove_unconnected_entities(
    ...     new_table, new_coords, linked_pairs, entities, free_cs, MC_free, Linker_free,
    ...     pair_index, ready_pairs
    ... )
    >>> # Then use the updated data structures to rebuild the assembly
    >>> asm = build_assembly_from_state(
    ...     entities=updated_entities,
    ...     free_cs=updated_free_cs,
    ...     MC_free=updated_MC_free,
    ...     Linker_free=updated_Linker_free,
    ...     linked_pairs=updated_linked_pairs,
    ...     pair_index=updated_pair_index,
    ...     ready_pairs=updated_ready_pairs,
    ...     ...
    ... )
    """
    new_table = element_entity_table.copy()
    
    # Convert entities to list if it's a RandomizedSet for easier manipulation
    entities_list = list(entities) if hasattr(entities, 'items') else list(entities)
    
    # Build a set containing all carboxylate groups in linked pairs
    linked_carboxylates_set = set()
    for pair in linked_carboxylate_pairs:
        linked_carboxylates_set.add(pair[0])
        linked_carboxylates_set.add(pair[1])
    
    # Get all unique entity labels
    unique_entity_labels = set([
        label for label in element_entity_table[:, 1]
        if label != "UNASSIGNED" and (label.startswith("Zr6_") or label.startswith("Zr12_") or label.startswith("BDC_"))
    ])
    
    removed_entity_labels = set()
    removed_entity_objs = set()
    
    # Build mapping from entity_label to entity_obj
    # Match by entity center coordinates and atom count (more reliable)
    entity_label_to_obj = {}
    for entity_label in unique_entity_labels:
        entity_atom_indices = [i for i, label in enumerate(element_entity_table[:, 1]) 
                              if label == entity_label]
        if len(entity_atom_indices) == 0:
            continue
        
        # Calculate the center coordinates of this entity
        entity_coords = coordinates[entity_atom_indices]
        entity_center = np.mean(entity_coords, axis=0)
        entity_atom_count = len(entity_atom_indices)
        
        # Find the corresponding entity object by matching center coordinates and atom count
        best_match = None
        best_distance = float('inf')
        for ent in entities_list:
            if not hasattr(ent, 'center') or not hasattr(ent, 'coordinates'):
                continue
            # Check if atom count matches
            if len(ent.coordinates) != entity_atom_count:
                continue
            # Check if center coordinates are close (allow small numerical errors)
            ent_center = np.asarray(ent.center)
            distance = np.linalg.norm(entity_center - ent_center)
            if distance < best_distance and distance < 0.1:  # Threshold 0.1 Angstrom
                best_distance = distance
                best_match = ent
        
        if best_match is not None:
            entity_label_to_obj[entity_label] = best_match
    
    # Iterate over all entity labels and check if their carboxylate groups are in linked pairs
    for entity_label in unique_entity_labels:
        entity_obj = entity_label_to_obj.get(entity_label)
        
        if entity_obj is None:
            if verbose:
                print(f"  Warning: Could not find entity object for {entity_label}, skipping.")
            continue
        
        # Check all carboxylate groups of this entity
        if not hasattr(entity_obj, 'carboxylates') or len(entity_obj.carboxylates) == 0:
            # If entity has no carboxylate groups, also treat as unconnected
            if verbose:
                print(f"  Entity {entity_label} has no carboxylates, marking as UNASSIGNED.")
            removed_entity_labels.add(entity_label)
            removed_entity_objs.add(entity_obj)
            continue
        
        # Check if all carboxylate groups of this entity are in linked pairs
        has_connected_carboxylate = False
        for carb in entity_obj.carboxylates:
            if carb in linked_carboxylates_set:
                has_connected_carboxylate = True
                break
        
        # If all carboxylates of this entity are not in linked pairs, mark as unconnected
        if not has_connected_carboxylate:
            if verbose:
                print(f"  Entity {entity_label} has {len(entity_obj.carboxylates)} carboxylates, "
                      f"none are in linked pairs. Marking as UNASSIGNED.")
            removed_entity_labels.add(entity_label)
            removed_entity_objs.add(entity_obj)
    
    # Mark all atoms of unconnected entities as UNASSIGNED
    for entity_label in removed_entity_labels:
        for i, label in enumerate(new_table[:, 1]):
            if label == entity_label:
                new_table[i, 1] = "UNASSIGNED"
    
    # Collect all carboxylates from removed entities
    removed_carboxylates = set()
    for entity_obj in removed_entity_objs:
        if hasattr(entity_obj, 'carboxylates'):
            removed_carboxylates.update(entity_obj.carboxylates)
    
    # Helper function to create updated set (preserving type)
    def create_updated_set(original_set, filter_func):
        """Create a new set of the same type as original_set, filtered by filter_func."""
        if isinstance(original_set, RandomizedSet):
            new_set = RandomizedSet()
            new_set.update([item for item in original_set if filter_func(item)])
            return new_set
        else:
            return set([item for item in original_set if filter_func(item)])
    
    # Remove carboxylates from removed entities from all carboxylate sets
    updated_free_carboxylates = create_updated_set(
        free_carboxylates, 
        lambda c: c not in removed_carboxylates
    )
    
    updated_MC_free_carboxylates = create_updated_set(
        MC_free_carboxylates,
        lambda c: c not in removed_carboxylates
    )
    
    updated_Linker_free_carboxylates = create_updated_set(
        Linker_free_carboxylates,
        lambda c: c not in removed_carboxylates
    )
    
    # Remove linked pairs that contain carboxylates from removed entities
    updated_linked_carboxylate_pairs = create_updated_set(
        linked_carboxylate_pairs,
        lambda pair: pair[0] not in removed_carboxylates and pair[1] not in removed_carboxylates
    )
    
    # Remove ready_to_connect pairs that contain carboxylates from removed entities
    updated_ready_to_connect_carboxylate_pairs = create_updated_set(
        ready_to_connect_carboxylate_pairs,
        lambda pair: pair[0] not in removed_carboxylates and pair[1] not in removed_carboxylates
    )
    
    # Update pair_index: remove entries for removed carboxylates
    updated_pair_index = {
        carb: pair for carb, pair in pair_index.items() 
        if carb not in removed_carboxylates and pair[0] not in removed_carboxylates and pair[1] not in removed_carboxylates
    }
    
    # Remove removed entities from entities set
    if isinstance(entities, RandomizedSet):
        updated_entities = RandomizedSet()
        updated_entities.update([e for e in entities_list if e not in removed_entity_objs])
    else:
        updated_entities = [e for e in entities_list if e not in removed_entity_objs]
    
    if verbose and removed_entity_labels:
        print(f"\n  Removed {len(removed_entity_labels)} unconnected entities: {sorted(removed_entity_labels)}")
        print(f"  Removed {len(removed_carboxylates)} carboxylates from unconnected entities")
        print(f"  Updated entities: {len(updated_entities)} (removed {len(removed_entity_objs)})")
        print(f"  Updated linked pairs: {len(updated_linked_carboxylate_pairs)} (removed {len(linked_carboxylate_pairs) - len(updated_linked_carboxylate_pairs)})")
        print(f"  Updated free carboxylates: {len(updated_free_carboxylates)} (removed {len(free_carboxylates) - len(updated_free_carboxylates)})")
    
    return (new_table, updated_entities, updated_free_carboxylates, updated_MC_free_carboxylates,
            updated_Linker_free_carboxylates, updated_linked_carboxylate_pairs, updated_pair_index,
            updated_ready_to_connect_carboxylate_pairs, removed_entity_labels)

def compute_roi_range(entities):
    """
    Compute the ROI (Region of Interest) range based on entity radii.
    
    ROI_range is used for spatial indexing to divide the 3D space into cubes.
    Each cube has a size of ROI_range^3, and entities are mapped to cubes based on their center coordinates.
    This enables efficient spatial queries and neighbor finding.
    
    The ROI_range is set to twice the maximum radius of all entities, ensuring that
    entities in adjacent cubes can be efficiently found during spatial searches.
    
    Parameters
    ----------
    entities : list or RandomizedSet
        List or set of entity objects. Each entity should have a 'radius' attribute.
    
    Returns
    -------
    float
        ROI range value, which is 2.0 * max(entity.radius for entity in entities).
        Returns 1.0 if entities is empty.
    
    Example
    -------
    >>> entities = [entity1, entity2, entity3]  # entities with radius attributes
    >>> roi_range = compute_roi_range(entities)
    >>> # roi_range = 2.0 * max(entity.radius for entity in entities)
    >>> # This value is used to divide space into cubes for spatial indexing
    """
    if not entities:
        return 1.0
    return 2.0 * max(getattr(e, 'radius', 0.0) for e in entities)

def rebuild_node_mapping(entities, roi_range=None):
    if roi_range is None:
        roi_range = compute_roi_range(entities)
    buckets = defaultdict(list)
    for ent in entities:
        c = np.asarray(ent.center, dtype=float)
        idx = tuple(np.floor(c / roi_range).astype(int))
        buckets[idx].append(ent)
    return dict(buckets), roi_range

def rebuild_connected_from_pairs(entities, linked_pairs, verbose=False):
    """
    Rebuild connected_entities for all entities based on linked carboxylate pairs.
    
    Parameters
    ----------
    entities : list
        List of all entities
    linked_pairs : list or set
        List of linked carboxylate pairs (tuples of Carboxylate objects)
    verbose : bool
        Whether to print detailed information about missing belonging_entity
    """
    # Clear all connected_entities first
    for e in entities:
        if hasattr(e, 'connected_entities') and e.connected_entities is not None:
            e.connected_entities.clear()
        else:
            e.connected_entities = []
    
    # Rebuild connections from linked pairs
    missing_entity_count = 0
    valid_connections = 0
    
    for c1, c2 in linked_pairs:
        # Check if carboxylates have valid belonging_entity
        if not hasattr(c1, 'belonging_entity') or c1.belonging_entity is None:
            missing_entity_count += 1
            if verbose and missing_entity_count <= 5:
                print(f"  Warning: carboxylate c1 has None belonging_entity in pair")
            continue
        
        if not hasattr(c2, 'belonging_entity') or c2.belonging_entity is None:
            missing_entity_count += 1
            if verbose and missing_entity_count <= 5:
                print(f"  Warning: carboxylate c2 has None belonging_entity in pair")
            continue
        
        e1, e2 = c1.belonging_entity, c2.belonging_entity
        
        # Ensure entities have connected_entities attribute
        if not hasattr(e1, 'connected_entities'):
            e1.connected_entities = []
        if not hasattr(e2, 'connected_entities'):
            e2.connected_entities = []
        
        # Add connection if entities are different
        if e1 is not e2:
            if e2 not in e1.connected_entities:
                e1.connected_entities.append(e2)
            if e1 not in e2.connected_entities:
                e2.connected_entities.append(e1)
            valid_connections += 1
    
    if verbose:
        print(f"  Rebuilt {valid_connections} entity connections from {len(linked_pairs)} linked pairs")
        if missing_entity_count > 0:
            print(f"  Warning: {missing_entity_count} pairs had carboxylates with None belonging_entity")

def rebuild_pair_index(linked_pairs, ready_pairs, old_index=None):
    idx = {} if old_index is None else dict(old_index)
    valid_pairs = set(linked_pairs) | set(ready_pairs)
    idx = {c: p for c, p in idx.items() if (p in valid_pairs and c in p)}
    for p in valid_pairs:
        a, b = p
        idx[a] = p; idx[b] = p
    return idx

def normalize_free_sets(free_cs, mc_set, lk_set):
    all_free = set(free_cs) | set(mc_set) | set(lk_set)
    mc = set(mc_set)
    lk = set(lk_set)
    if not mc or not lk:
        for c in all_free:
            ent = c.belonging_entity
            if ent is None: 
                continue
            if getattr(ent, 'entity_type', None) == 'Zr':
                mc.add(c)
            else:
                lk.add(c)
    return all_free, mc, lk

def filter_free_by_pairs(free_all, mc_set, lk_set, pair_index):
    paired = set(pair_index.keys()) if pair_index else set()
    if paired:
        free_all -= paired
        mc_set   -= paired
        lk_set   -= paired
    return free_all, mc_set, lk_set

def build_assembly_from_state(entities,
                              free_cs, MC_free, Linker_free,
                              linked_pairs, ready_pairs,
                              pair_index,
                              ZR6_PERCENTAGE=0.6, ENTROPY_GAIN=1.0, BUMPING_THRESHOLD=2.0,
                              roi_range=None):
    ents = list(entities)
    if not ents:
        raise ValueError("entities is empty, cannot initialize Assembly")
    anchor = ents[0]
    asm = Assembly(anchor, ZR6_PERCENTAGE, ENTROPY_GAIN, BUMPING_THRESHOLD)

    asm.entities = RandomizedSet()
    asm.entities.update(ents)

    # Rebuild connected_entities from linked pairs
    # Note: All carboxylates should have valid belonging_entity (verified in instantiate_all_entities)
    # This rebuild ensures consistency and handles any edge cases
    rebuild_connected_from_pairs(ents, linked_pairs, verbose=True)

    pair_index = rebuild_pair_index(linked_pairs, ready_pairs, pair_index)

    free_all, mc_set, lk_set = normalize_free_sets(free_cs, MC_free, Linker_free)
    free_all, mc_set, lk_set = filter_free_by_pairs(free_all, mc_set, lk_set, pair_index)

    asm.free_carboxylates = RandomizedSet();        asm.free_carboxylates.update(free_all)
    asm.MC_free_carboxylates = RandomizedSet();     asm.MC_free_carboxylates.update(mc_set)
    asm.Linker_free_carboxylates = RandomizedSet(); asm.Linker_free_carboxylates.update(lk_set)

    asm.linked_carboxylate_pairs = RandomizedSet(); asm.linked_carboxylate_pairs.update(set(linked_pairs))
    asm.ready_to_connect_carboxylate_pairs = RandomizedSet(); asm.ready_to_connect_carboxylate_pairs.update(set(ready_pairs))
    asm.pair_index = pair_index or {}

    asm.ROI_range = roi_range if roi_range is not None else compute_roi_range(ents)
    asm.node_mapping, _ = rebuild_node_mapping(ents, asm.ROI_range)

    # Note: KDTree is already cleared in instantiate_all_entities, no need to clear again
            
    for k in ("time_for_align_new_entity","time_for_creating_new_entity","time_for_find_connection",
              "time_for_finding_ROI","time_for_judge_bumping","time_for_is_too_close",
              "time_for_add_new_entity","time_for_cut_a_bond","time_for_remove_entities"):
        setattr(asm, k, 0.0)

    return asm

def write_assembly_to_mol2(assembly, filepath, molecule_name="UiO-66"):
    """
    Write assembly structure to MOL2 file format with entity labels.
    
    The labels indicate which entity each atom belongs to:
    - Zr6 atoms are labeled as Zr6_{序号}, where 序号 is the sequential number of the Zr6 cluster (1, 2, 3, ...)
    - BDC atoms are labeled as BDC_{序号}, where 序号 is the sequential number of the BDC entity (1, 2, 3, ...)
    Both Zr6 and BDC are numbered independently in the order they appear in the assembly.
    
    Parameters:
        assembly: Assembly object with entities
        filepath: output file path
        molecule_name: molecule name for the header
    """
    # Step 1: Assign sequential numbers to all entities
    # Zr6 and BDC are numbered independently in order of appearance
    zr6_counter = 0
    bdc_counter = 0
    
    # Step 2: Collect all atoms and assign labels
    mol2_elements = []
    mol2_coordinates = []
    mol2_entity_labels = []
    
    for entity in assembly.entities:
        if entity.entity_type == 'Zr':
            # Zr6 entity: assign sequential number
            zr6_counter += 1
            entity_label = f"Zr6_{zr6_counter}"
        elif entity.entity_type == 'Ligand':
            # BDC entity: assign sequential number independently
            bdc_counter += 1
            entity_label = f"BDC_{bdc_counter}"
        else:
            # Other entity types: use entity_subtype as fallback
            entity_label = f"{entity.entity_type}_{entity.entity_subtype}"
        
        # Get atoms from entity
        if hasattr(entity, 'elements') and hasattr(entity, 'coordinates'):
            for elem, coord in zip(entity.elements, entity.coordinates):
                mol2_elements.append(elem)
                mol2_coordinates.append(coord)
                mol2_entity_labels.append(entity_label)
    
    n_atoms = len(mol2_elements)
    
    with open(filepath, 'w') as f:
        # Header
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{molecule_name}\n")
        f.write(f"{n_atoms} 0 0 0 0\n")  # No bonds in output
        f.write("SMALL\n")
        f.write("NO_CHARGES\n\n")
        
        # Atoms
        f.write("@<TRIPOS>ATOM\n")
        for i, (elem, coord, entity_label) in enumerate(zip(mol2_elements, mol2_coordinates, mol2_entity_labels)):
            x, y, z = coord
            atom_name = f"{elem}{i+1}"
            # Format: atom_id atom_name x y z element subst_id subst_name charge
            f.write(f"{i+1:7d} {atom_name:<8s} {x:10.4f} {y:10.4f} {z:10.4f} "
                   f"{elem:>2s} 1 {entity_label} 0.0000\n")
    
    print(f"Written {n_atoms} atoms to {filepath}")
    return n_atoms