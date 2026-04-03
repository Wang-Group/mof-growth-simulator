import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from UiO66_Assembly_Large_Correction_conc import (
    Assembly,
    Zr6_AA,
    safe_pickle_save,
)


CLUSTER_TO_RATIO = {
    "zr6": 1.0,
    "zr12": 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a connected mixed Zr6_AA/Zr12_AA nucleus by reusing the existing "
            "AA-based alignment and growth logic."
        )
    )
    parser.add_argument(
        "--cluster-sequence",
        default="zr6,zr12,zr6",
        help=(
            "Comma-separated cluster sequence. The first cluster is used as the seed, "
            "and BDC linkers are inserted automatically between consecutive clusters."
        ),
    )
    parser.add_argument("--rng-seed", type=int, default=7, help="Seed for Python and NumPy RNGs.")
    parser.add_argument(
        "--attempts-per-step",
        type=int,
        default=500,
        help="Maximum placement attempts for each new entity.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/mixed_nuclei",
        help="Directory to store the generated nucleus files.",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Optional basename for the output files. A timestamped name is used when omitted.",
    )
    parser.add_argument(
        "--grow-to-entities",
        type=int,
        default=0,
        help=(
            "If > 0, continue stochastic off-lattice growth from the mixed seed until "
            "the total entity count reaches this value."
        ),
    )
    parser.add_argument(
        "--max-growth-steps",
        type=int,
        default=200000,
        help="Maximum stochastic growth iterations when --grow-to-entities is enabled.",
    )
    parser.add_argument(
        "--cluster-add-probability",
        type=float,
        default=0.10,
        help=(
            "Probability of attempting a cluster addition during external growth when "
            "both cluster and linker growth sites are available."
        ),
    )
    parser.add_argument(
        "--internal-link-probability",
        type=float,
        default=0.20,
        help=(
            "Probability of consuming a ready internal linker-cluster pair instead of "
            "adding a new entity during stochastic growth."
        ),
    )
    parser.add_argument(
        "--zr6-growth-percentage",
        type=float,
        default=0.60,
        help="Probability that a newly added cluster is Zr6_AA rather than Zr12_AA.",
    )
    parser.add_argument(
        "--stall-link-burst",
        type=int,
        default=40,
        help=(
            "If this many external growth attempts fail in a row, consume a burst of "
            "ready internal links to unjam the assembly."
        ),
    )
    parser.add_argument(
        "--link-burst-size",
        type=int,
        default=25,
        help="Maximum number of ready internal links to consume during one stall-recovery burst.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print one progress line every N growth iterations when stochastic growth is enabled.",
    )
    parser.add_argument(
        "--max-zr12-coordination",
        type=int,
        default=None,
        help=(
            "If set, fail unless every Zr12_AA cluster has coordination <= this value "
            "(coordination counted as linked carboxylates)."
        ),
    )
    return parser.parse_args()


def normalize_cluster_sequence(raw_sequence):
    cluster_sequence = [token.strip().lower() for token in raw_sequence.split(",") if token.strip()]
    if len(cluster_sequence) < 2:
        raise ValueError("cluster-sequence must contain at least two clusters.")
    invalid = [token for token in cluster_sequence if token not in CLUSTER_TO_RATIO]
    if invalid:
        raise ValueError(f"Unsupported cluster labels in cluster-sequence: {invalid}")
    if len(set(cluster_sequence)) < 2:
        raise ValueError("cluster-sequence must contain both 'zr6' and 'zr12' to produce a mixed nucleus.")
    return cluster_sequence


def compatible_sites(assembly, next_kind):
    if next_kind == "bdc":
        return [carboxylate for carboxylate in assembly.free_carboxylates if carboxylate.carboxylate_type == "formate"]
    if next_kind in CLUSTER_TO_RATIO:
        return [carboxylate for carboxylate in assembly.free_carboxylates if carboxylate.carboxylate_type == "benzoate"]
    raise ValueError(f"Unsupported entity kind: {next_kind}")


def add_entity_with_existing_geometry(assembly, next_kind, attempts_per_step):
    original_entity_count = len(assembly.entities)
    original_ratio = assembly.ZR6_PERCENTAGE
    if next_kind in CLUSTER_TO_RATIO:
        assembly.ZR6_PERCENTAGE = CLUSTER_TO_RATIO[next_kind]

    try:
        for attempt_idx in range(1, attempts_per_step + 1):
            candidate_sites = compatible_sites(assembly, next_kind)
            if not candidate_sites:
                raise RuntimeError(f"No compatible free carboxylate sites found for {next_kind}.")

            selected_carboxylate = random.choice(candidate_sites)
            assembly.grow_one_step(selected_carboxylate)

            if len(assembly.entities) == original_entity_count + 1:
                return {
                    "kind": next_kind,
                    "attempts": attempt_idx,
                    "selected_site_type": selected_carboxylate.carboxylate_type,
                    "selected_site_owner": selected_carboxylate.belonging_entity.entity_type,
                }
        raise RuntimeError(f"Failed to place {next_kind} after {attempts_per_step} attempts.")
    finally:
        assembly.ZR6_PERCENTAGE = original_ratio


def link_all_ready_pairs(assembly, max_links=1000):
    linked_pairs = 0
    while len(assembly.ready_to_connect_carboxylate_pairs) > 0:
        pair = assembly.ready_to_connect_carboxylate_pairs.get_random()
        assembly.link_internal_carboxylate(pair)
        linked_pairs += 1
        if linked_pairs >= max_links:
            break
    return linked_pairs


def link_ready_pairs_burst(assembly, max_links):
    linked_pairs = 0
    while linked_pairs < max_links and len(assembly.ready_to_connect_carboxylate_pairs) > 0:
        pair = assembly.ready_to_connect_carboxylate_pairs.get_random()
        assembly.link_internal_carboxylate(pair)
        linked_pairs += 1
    return linked_pairs


def entity_counts(assembly):
    counts = {
        "Zr6_AA": 0,
        "Zr12_AA": 0,
        "BDC": 0,
    }
    for entity in assembly.entities:
        if entity.entity_type == "Zr" and entity.entity_subtype == 0:
            counts["Zr6_AA"] += 1
        elif entity.entity_type == "Zr" and entity.entity_subtype == 1:
            counts["Zr12_AA"] += 1
        elif entity.entity_type == "Ligand":
            counts["BDC"] += 1
    counts["total_entities"] = len(assembly.entities)
    counts["linked_pairs"] = len(assembly.linked_carboxylate_pairs)
    counts["ready_pairs"] = len(assembly.ready_to_connect_carboxylate_pairs)
    counts["free_carboxylates"] = len(assembly.free_carboxylates)
    return counts


def cluster_capacity(entity):
    if getattr(entity, "entity_type", None) != "Zr":
        return None
    if getattr(entity, "entity_subtype", None) == 0:
        return 12
    if getattr(entity, "entity_subtype", None) == 1:
        return 18
    return len(getattr(entity, "carboxylates", []))


def cluster_kind(entity):
    if getattr(entity, "entity_type", None) != "Zr":
        return None
    if getattr(entity, "entity_subtype", None) == 0:
        return "Zr6_AA"
    if getattr(entity, "entity_subtype", None) == 1:
        return "Zr12_AA"
    return "Zr_unknown"


def cluster_coordination_summary(assembly):
    clusters = [entity for entity in assembly.entities if getattr(entity, "entity_type", None) == "Zr"]
    cluster_ids = {id(entity): idx for idx, entity in enumerate(clusters)}
    linked_counts = {id(entity): 0 for entity in clusters}

    for pair in assembly.linked_carboxylate_pairs:
        for carb in pair:
            entity = getattr(carb, "belonging_entity", None)
            if entity is not None and getattr(entity, "entity_type", None) == "Zr":
                linked_counts[id(entity)] += 1

    summary = []
    for entity in clusters:
        max_sites = cluster_capacity(entity)
        coord = linked_counts[id(entity)]
        free_sites = max_sites - coord if max_sites is not None else None
        summary.append(
            {
                "cluster_id": cluster_ids[id(entity)],
                "kind": cluster_kind(entity),
                "coordination": coord,
                "max_sites": max_sites,
                "free_sites": free_sites,
                "saturation_fraction": (coord / max_sites) if max_sites else None,
                "center": np.asarray(entity.center, dtype=float).round(4).tolist(),
            }
        )
    summary.sort(key=lambda item: (item["kind"], item["cluster_id"]))
    return summary


def zr12_coordination_ok(assembly, max_zr12_coordination):
    if max_zr12_coordination is None:
        return True
    for item in cluster_coordination_summary(assembly):
        if item["kind"] == "Zr12_AA" and item["coordination"] > max_zr12_coordination:
            return False
    return True


def is_connected(assembly):
    entities = assembly.entities.to_list()
    if not entities:
        return False

    visited = {entities[0]}
    stack = [entities[0]]
    while stack:
        current = stack.pop()
        for neighbor in current.connected_entities:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == len(entities)


def build_target_sequence(cluster_sequence, attempts_per_step):
    seed_cluster = cluster_sequence[0]
    if seed_cluster != "zr6":
        raise ValueError(
            "The current builder starts from Zr6_AA to match the existing UiO-66 seed workflow. "
            "Use a sequence beginning with 'zr6'."
        )

    assembly = Assembly(Zr6_AA(), ZR6_PERCENTAGE=0.5, ENTROPY_GAIN=30.9, BUMPING_THRESHOLD=2.0)
    build_log = [{"kind": "zr6", "attempts": 0, "selected_site_type": None, "selected_site_owner": None}]

    for next_cluster in cluster_sequence[1:]:
        build_log.append(add_entity_with_existing_geometry(assembly, "bdc", attempts_per_step))
        build_log.append(add_entity_with_existing_geometry(assembly, next_cluster, attempts_per_step))

    linked_pairs = link_all_ready_pairs(assembly)
    return assembly, build_log, linked_pairs


def choose_cluster_kind(zr6_growth_percentage):
    return "zr6" if random.random() < zr6_growth_percentage else "zr12"


def stochastic_amorphous_growth(
    assembly,
    target_entities,
    max_growth_steps,
    attempts_per_step,
    cluster_add_probability,
    internal_link_probability,
    zr6_growth_percentage,
    stall_link_burst,
    link_burst_size,
    progress_every,
):
    if target_entities <= len(assembly.entities):
        return {
            "growth_steps_used": 0,
            "forced_link_bursts": 0,
            "burst_links_created": 0,
            "stalled_external_attempts": 0,
            "target_reached": True,
            "progress_snapshots": [],
        }

    if not 0.0 <= cluster_add_probability <= 1.0:
        raise ValueError("cluster-add-probability must be in [0, 1].")
    if not 0.0 <= internal_link_probability <= 1.0:
        raise ValueError("internal-link-probability must be in [0, 1].")
    if not 0.0 <= zr6_growth_percentage <= 1.0:
        raise ValueError("zr6-growth-percentage must be in [0, 1].")

    original_ratio = assembly.ZR6_PERCENTAGE
    assembly.ZR6_PERCENTAGE = zr6_growth_percentage

    progress_snapshots = []
    forced_link_bursts = 0
    burst_links_created = 0
    stalled_external_attempts = 0
    growth_steps_used = 0

    try:
        for step_idx in range(1, max_growth_steps + 1):
            growth_steps_used = step_idx
            pre_entities = len(assembly.entities)
            pre_ready = len(assembly.ready_to_connect_carboxylate_pairs)

            formate_sites = compatible_sites(assembly, "bdc")
            benzoate_sites = compatible_sites(assembly, "zr6")

            action = None
            if (
                pre_ready > 0
                and random.random() < internal_link_probability
            ):
                pair = assembly.ready_to_connect_carboxylate_pairs.get_random()
                assembly.link_internal_carboxylate(pair)
                action = "internal_link"
                stalled_external_attempts = 0
            else:
                if benzoate_sites and formate_sites:
                    next_kind = (
                        choose_cluster_kind(zr6_growth_percentage)
                        if random.random() < cluster_add_probability
                        else "bdc"
                    )
                elif benzoate_sites:
                    next_kind = choose_cluster_kind(zr6_growth_percentage)
                elif formate_sites:
                    next_kind = "bdc"
                elif pre_ready > 0:
                    pair = assembly.ready_to_connect_carboxylate_pairs.get_random()
                    assembly.link_internal_carboxylate(pair)
                    action = "internal_link_fallback"
                    stalled_external_attempts = 0
                    next_kind = None
                else:
                    raise RuntimeError("Growth stalled: no compatible free carboxylates and no ready internal links.")

                if action is None and next_kind is not None:
                    add_entity_with_existing_geometry(assembly, next_kind, attempts_per_step)
                    if len(assembly.entities) > pre_entities:
                        action = f"grow_{next_kind}"
                        stalled_external_attempts = 0
                    else:
                        action = f"failed_{next_kind}"
                        stalled_external_attempts += 1

            if (
                stalled_external_attempts >= stall_link_burst
                and len(assembly.ready_to_connect_carboxylate_pairs) > 0
            ):
                linked_now = link_ready_pairs_burst(assembly, link_burst_size)
                forced_link_bursts += 1
                burst_links_created += linked_now
                stalled_external_attempts = 0
                action = f"{action}+burst_{linked_now}"

            if progress_every > 0 and (step_idx == 1 or step_idx % progress_every == 0 or len(assembly.entities) >= target_entities):
                counts = entity_counts(assembly)
                snapshot = {
                    "step": step_idx,
                    "action": action,
                    "total_entities": counts["total_entities"],
                    "Zr6_AA": counts["Zr6_AA"],
                    "Zr12_AA": counts["Zr12_AA"],
                    "BDC": counts["BDC"],
                    "ready_pairs": counts["ready_pairs"],
                    "free_carboxylates": counts["free_carboxylates"],
                }
                progress_snapshots.append(snapshot)
                print(json.dumps(snapshot))

            if len(assembly.entities) >= target_entities:
                break
    finally:
        assembly.ZR6_PERCENTAGE = original_ratio

    return {
        "growth_steps_used": growth_steps_used,
        "forced_link_bursts": forced_link_bursts,
        "burst_links_created": burst_links_created,
        "stalled_external_attempts": stalled_external_attempts,
        "target_reached": len(assembly.entities) >= target_entities,
        "progress_snapshots": progress_snapshots,
    }


def main():
    args = parse_args()
    cluster_sequence = normalize_cluster_sequence(args.cluster_sequence)

    script_dir = Path(__file__).resolve().parent
    output_dir = (script_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    assembly, build_log, linked_pairs = build_target_sequence(cluster_sequence, args.attempts_per_step)
    growth_summary = None
    if args.grow_to_entities > 0:
        growth_summary = stochastic_amorphous_growth(
            assembly=assembly,
            target_entities=args.grow_to_entities,
            max_growth_steps=args.max_growth_steps,
            attempts_per_step=args.attempts_per_step,
            cluster_add_probability=args.cluster_add_probability,
            internal_link_probability=args.internal_link_probability,
            zr6_growth_percentage=args.zr6_growth_percentage,
            stall_link_burst=args.stall_link_burst,
            link_burst_size=args.link_burst_size,
            progress_every=args.progress_every,
        )

    counts = entity_counts(assembly)
    coordination_summary = cluster_coordination_summary(assembly)
    counts["connected"] = is_connected(assembly)
    counts["internal_links_created"] = linked_pairs

    if counts["Zr6_AA"] < 1 or counts["Zr12_AA"] < 1:
        raise RuntimeError(f"Builder finished without both cluster types present: {counts}")
    if not counts["connected"]:
        raise RuntimeError(f"Builder finished with a disconnected assembly: {counts}")
    if growth_summary is not None and not growth_summary["target_reached"]:
        raise RuntimeError(
            f"Growth stopped before reaching target entity count {args.grow_to_entities}. Final counts: {counts}"
        )
    if not zr12_coordination_ok(assembly, args.max_zr12_coordination):
        raise RuntimeError(
            f"Zr12 coordination exceeds requested max {args.max_zr12_coordination}. "
            f"Coordination summary: {coordination_summary}"
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    basename = args.basename or (
        f"mixed_AA_nucleus_{timestamp}_zr6_{counts['Zr6_AA']}_zr12_{counts['Zr12_AA']}_bdc_{counts['BDC']}"
    )

    mol2_path = output_dir / f"{basename}.mol2"
    pkl_path = output_dir / f"{basename}.pkl"
    json_path = output_dir / f"{basename}.json"

    assembly.get_mol2_file(str(mol2_path))
    if not safe_pickle_save(assembly, str(pkl_path), rebuild_after_save=False):
        raise RuntimeError(f"Failed to save pickle file to {pkl_path}")

    summary = {
        "cluster_sequence": cluster_sequence,
        "rng_seed": args.rng_seed,
        "attempts_per_step": args.attempts_per_step,
        "growth_settings": {
            "grow_to_entities": args.grow_to_entities,
            "max_growth_steps": args.max_growth_steps,
            "cluster_add_probability": args.cluster_add_probability,
            "internal_link_probability": args.internal_link_probability,
            "zr6_growth_percentage": args.zr6_growth_percentage,
            "stall_link_burst": args.stall_link_burst,
            "link_burst_size": args.link_burst_size,
            "progress_every": args.progress_every,
        },
        "counts": counts,
        "cluster_coordination": coordination_summary,
        "build_log": build_log,
        "growth_summary": growth_summary,
        "outputs": {
            "mol2": str(mol2_path),
            "pkl": str(pkl_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
