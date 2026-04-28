from collections import deque

import numpy as np


def connected_components(assembly):
    entities = assembly.entities.to_list() if hasattr(assembly.entities, "to_list") else list(assembly.entities)
    visited = set()
    components = []

    for entity in entities:
        if entity in visited:
            continue

        component = []
        queue = deque([entity])
        visited.add(entity)

        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in getattr(current, "connected_entities", []) or []:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    components.sort(key=len, reverse=True)
    return components


def remove_entities_from_assembly(assembly, entities_to_remove):
    for entity in entities_to_remove:
        for carboxylate in entity.carboxylates:
            if carboxylate in assembly.free_carboxylates:
                assembly.free_carboxylates.remove(carboxylate)
                if carboxylate.belonging_entity.entity_type == "Zr":
                    assembly.MC_free_carboxylates.remove(carboxylate)
                else:
                    assembly.Linker_free_carboxylates.remove(carboxylate)
            else:
                pair = assembly.pair_index.get(carboxylate, None)
                if pair:
                    other_carboxylate = pair[0] if carboxylate == pair[1] else pair[1]
                    if pair in assembly.ready_to_connect_carboxylate_pairs:
                        assembly.ready_to_connect_carboxylate_pairs.remove(pair)
                    elif pair in assembly.linked_carboxylate_pairs:
                        assembly.linked_carboxylate_pairs.remove(pair)
                    for item in pair:
                        assembly.pair_index.pop(item, None)

                    assembly.free_carboxylates.add(other_carboxylate)
                    if other_carboxylate.belonging_entity.entity_type == "Zr":
                        assembly.MC_free_carboxylates.add(other_carboxylate)
                    else:
                        assembly.Linker_free_carboxylates.add(other_carboxylate)
            carboxylate.belonging_entity = None

        entity.kdtree = None

        for connected_entity in list(getattr(entity, "connected_entities", []) or []):
            connected_entity.connected_entities.remove(entity)
        entity.connected_entities = None

        cube_index = tuple(np.floor(entity.center / assembly.ROI_range).astype(int))
        assembly.node_mapping[cube_index].remove(entity)
        if len(assembly.node_mapping[cube_index]) == 0:
            del assembly.node_mapping[cube_index]

        assembly.entities.remove(entity)


def prune_disconnected_fragments(assembly):
    components = connected_components(assembly)
    if len(components) <= 1:
        return {
            "component_count_before": len(components),
            "removed_component_count": 0,
            "removed_entity_count": 0,
            "kept_component_size": len(components[0]) if components else 0,
        }

    kept_component = components[0]
    removed_components = components[1:]
    entities_to_remove = [entity for component in removed_components for entity in component]
    remove_entities_from_assembly(assembly, entities_to_remove)

    return {
        "component_count_before": len(components),
        "removed_component_count": len(removed_components),
        "removed_entity_count": len(entities_to_remove),
        "kept_component_size": len(kept_component),
    }
