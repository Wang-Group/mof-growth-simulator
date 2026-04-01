from mol2pkl import *
import numpy as np
import pickle
from collections import Counter, defaultdict

# Load the data
base_name = "UiO-66_15x15x15_sphere_R1"
file_name = f"data/{base_name}.mol2"
elements, coordinates, connectivity_map,carboxylate_indices = read_mol_file(file_name)

print(elements)
print(coordinates)
print(connectivity_map)
print(carboxylate_indices)
print(elements[list(carboxylate_indices[20])])


# Identify the entities that the atom belongs to
element_entity_table = assign_entities(elements, coordinates, connectivity_map,carboxylate_indices)

# # Visualize the first Zr6 cluster
# visualize_entity("Zr6_1", element_entity_table, coordinates, connectivity_map)
# # Visualize the second BDC fragment
# visualize_entity("BDC_3418", element_entity_table, coordinates, connectivity_map)

element_entity_table_1,Bridge_carboxylates,Zr6_carboxylates,BDC_carboxylates,Unnormal_carboxylates = assign_carboxylates(
    elements,
    coordinates,
    connectivity_map,
    carboxylate_indices,
    element_entity_table
)

# visualize_entities_by_type("Zr6_666", element_entity_table_1, coordinates, connectivity_map)
# visualize_entities_by_type("BDC_3420", element_entity_table_1, coordinates, connectivity_map)


# Replica the Bridge blocks
new_elements, new_coords, new_table, new_carboxylate_indices, new_Bridge_carboxylates, invalidated_entities = duplicate_composite_units_with_bdc(
    elements, coordinates, element_entity_table_1, carboxylate_indices, Bridge_carboxylates,
    Zr6_carboxylates=Zr6_carboxylates, BDC_carboxylates=BDC_carboxylates
)
if invalidated_entities:
    print(f"Invalidated {len(invalidated_entities)} incomplete entities: {sorted(invalidated_entities)}")

print("new_table shape:", new_table.shape)

labels_new = new_table[:, 1].astype(str)
elements_new = new_table[:, 0].astype(str)

is_composite_new = np.array(['_BDC_' in lbl for lbl in labels_new])
num_composite_atoms = int(np.sum(is_composite_new))
print(f"Composite-labeled atoms in new_table: {num_composite_atoms}")

if num_composite_atoms > 0:
    composite_labels = labels_new[is_composite_new]
    composite_elements = elements_new[is_composite_new]
    label_to_count = Counter(composite_labels)
    label_to_elements = defaultdict(list)
    for lbl, elem in zip(composite_labels, composite_elements):
        label_to_elements[lbl].append(elem)

    print(f"Distinct composite labels (new_table): {len(label_to_count)}")
    for lbl, cnt in label_to_count.most_common(10):
        print(f"- {lbl}: count={cnt}, elements={Counter(label_to_elements[lbl])}")

    comp_idx = np.where(is_composite_new)[0]
    print("\nSample atoms (<=10):")
    for i in comp_idx[:10]:
        print(f"idx={i:>7d}, elem={elements_new[i]:>2s}, label={labels_new[i]}")

# visualize_entities_by_type("Zr6_33", new_table, new_coords, connectivity_map,
#                                figsize=(7,6), show_bonds=True)
# visualize_entities_by_type("BDC_100", new_table, new_coords, connectivity_map,
#                                figsize=(7,6), show_bonds=True)


# Rebuild assembly
# 1、Indentify all entities, free carboxylates and linked carboxylate pairs;
(entities, free_cs, MC_free, Linker_free, linked_pairs, pair_index, ready_to_connect_carboxylate_pairs) = instantiate_all_entities(
    new_table, new_coords, new_carboxylate_indices,
    Bridge_carboxylates=new_Bridge_carboxylates,
    Zr6_carboxylates=Zr6_carboxylates,       
    BDC_carboxylates=BDC_carboxylates       
)
print(f"entities: {len(entities)}")
print(f"MC_free(Zr6)  = {len(MC_free)}  (expected = {len(Zr6_carboxylates)})")
print(f"Linker_free   = {len(Linker_free)} (expected = {len(BDC_carboxylates)})")
print(f"linked_pairs  = {len(linked_pairs)} (expected = {len(Bridge_carboxylates)})")
print(f"pair_index    = {len(pair_index)}  (expected = {len(new_Bridge_carboxylates)})")
print(f"ready_to_connect = {len(ready_to_connect_carboxylate_pairs)}")

# 2、Remove free entities
(new_table, updated_entities, updated_free_cs, updated_MC_free, updated_Linker_free,
 updated_linked_pairs, updated_pair_index, updated_ready_to_connect_carboxylate_pairs, removed_labels) = remove_unconnected_entities(
    new_table, new_coords, linked_pairs, entities, free_cs, MC_free, Linker_free,
    pair_index, ready_to_connect_carboxylate_pairs, verbose=True
)

print(f"entities: {len(updated_entities)}")
print(f"MC_free(Zr6)  = {len(updated_MC_free)}  (expected = {len(Zr6_carboxylates)})")
print(f"Linker_free   = {len(updated_Linker_free)} (expected = {len(BDC_carboxylates)})")
print(f"linked_pairs  = {len(updated_linked_pairs)} (expected = {len(Bridge_carboxylates)})")
print(f"pair_index    = {len(updated_pair_index)}  (expected = {len(new_Bridge_carboxylates)})")
print(f"ready_to_connect = {len(updated_ready_to_connect_carboxylate_pairs)}")

# 3、Rebuild assembly
asm = build_assembly_from_state(
    entities=updated_entities,
    free_cs=updated_free_cs,
    MC_free=updated_MC_free,
    Linker_free=updated_Linker_free,
    linked_pairs=updated_linked_pairs,
    ready_pairs=updated_ready_to_connect_carboxylate_pairs,
    pair_index=updated_pair_index,
    ZR6_PERCENTAGE=None,
    ENTROPY_GAIN=None,
    BUMPING_THRESHOLD=None
)

save_ok = safe_pickle_save(
    assembly=asm,
    filepath=f"output/{base_name}.pkl",         
    clean_connected_entities=True,         
    rebuild_after_save=True,               
    protocol=pickle.HIGHEST_PROTOCOL                       
)
print("saved:", save_ok)

asm_loaded = safe_pickle_load(f"output/{base_name}.pkl", rebuild_references=True)
print(type(asm_loaded), "entities:", len(list(asm_loaded.entities)))