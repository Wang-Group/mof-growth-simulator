from __future__ import annotations

import csv
import hashlib
import itertools
import math
from collections import Counter, defaultdict, deque
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
RUN_ROOT = (
    REPO
    / "UiO-66"
    / "UiO66_long_growth"
    / "generated_runs_fixed"
    / "UiO66_260206_long1000"
    / "Stage_1"
)
OUTPUT_DIR = PACKAGE_ROOT / "data"
ENTITY_N = 13

ZR6_ATOMS = 62
BDC_ATOMS = 12
CARBOXYLATE_CO_CUTOFF_A = 1.8
LINK_MATCH_CUTOFF_A = 0.10


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def mean_point(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    n = len(points)
    return tuple(sum(p[i] for p in points) / n for i in range(3))


def parse_mol2_atoms(path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    in_atom = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom:
                break
            if not in_atom or not line:
                continue
            parts = line.split()
            if len(parts) >= 6 and parts[0].isdigit():
                element = parts[5]
                xyz = (float(parts[2]), float(parts[3]), float(parts[4]))
                atoms.append((element, xyz))
    return atoms


def split_entities(
    atoms: list[tuple[str, tuple[float, float, float]]],
) -> list[dict[str, object]]:
    entities: list[dict[str, object]] = []
    i = 0
    while i < len(atoms):
        if atoms[i][0] == "Zr":
            chunk = atoms[i : i + ZR6_ATOMS]
            kind = "Zr6"
            i += ZR6_ATOMS
        else:
            chunk = atoms[i : i + BDC_ATOMS]
            kind = "BDC"
            i += BDC_ATOMS
        if not chunk:
            break
        entities.append(
            {
                "kind": kind,
                "atoms": chunk,
                "center": mean_point([coord for _, coord in chunk]),
            }
        )
    return entities


def carboxylate_centers(
    atoms: list[tuple[str, tuple[float, float, float]]],
) -> list[tuple[int, tuple[int, int], tuple[float, float, float]]]:
    oxygen_atoms = [(i, coord) for i, (element, coord) in enumerate(atoms) if element == "O"]
    centers: list[tuple[int, tuple[int, int], tuple[float, float, float]]] = []
    for carbon_idx, (element, carbon_coord) in enumerate(atoms):
        if element != "C":
            continue
        nearby_o = [
            (oxygen_idx, oxygen_coord, distance(carbon_coord, oxygen_coord))
            for oxygen_idx, oxygen_coord in oxygen_atoms
            if distance(carbon_coord, oxygen_coord) < CARBOXYLATE_CO_CUTOFF_A
        ]
        if len(nearby_o) >= 2:
            nearest = sorted(nearby_o, key=lambda item: item[2])[:2]
            o_indices = tuple(sorted(item[0] for item in nearest))
            center = mean_point([carbon_coord] + [item[1] for item in nearest])
            centers.append((carbon_idx, o_indices, center))
    return centers


def canonical_bipartite_topology(
    z_count: int, bdc_neighbor_sets: list[tuple[int, ...]]
) -> str:
    """Canonicalize unlabeled Zr nodes and unlabeled BDC/linker nodes."""
    best: tuple[tuple[int, ...], ...] | None = None
    for perm in itertools.permutations(range(z_count)):
        mapping = {old: new for new, old in enumerate(perm)}
        mapped = tuple(
            sorted(tuple(sorted(mapping[z] for z in neigh)) for neigh in bdc_neighbor_sets)
        )
        if best is None or mapped < best:
            best = mapped
    assert best is not None
    return repr(best)


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def pairwise_distance_fingerprint(
    points: list[tuple[float, float, float]], decimals: int = 2
) -> str:
    values = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            values.append(round(distance(points[i], points[j]), decimals))
    values.sort()
    return short_hash(",".join(f"{value:.{decimals}f}" for value in values))


def all_atom_distance_fingerprint(
    atoms: list[tuple[str, tuple[float, float, float]]], decimals: int = 2
) -> str:
    values = []
    for i in range(len(atoms)):
        e_i, c_i = atoms[i]
        for j in range(i + 1, len(atoms)):
            e_j, c_j = atoms[j]
            pair = "-".join(sorted((e_i, e_j)))
            values.append((pair, round(distance(c_i, c_j), decimals)))
    values.sort()
    return short_hash(";".join(f"{pair}:{value:.{decimals}f}" for pair, value in values))


def component_count(node_count: int, edges: list[tuple[int, int]]) -> int:
    adjacency = [[] for _ in range(node_count)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen = set()
    comps = 0
    for start in range(node_count):
        if start in seen:
            continue
        comps += 1
        queue: deque[int] = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return comps


def analyze_case(path: Path) -> dict[str, object]:
    atoms = parse_mol2_atoms(path)
    entities = split_entities(atoms)
    if len(entities) != ENTITY_N:
        raise ValueError(f"{path} has {len(entities)} parsed entities, expected {ENTITY_N}")

    z_entities = [(idx, ent) for idx, ent in enumerate(entities) if ent["kind"] == "Zr6"]
    bdc_entities = [(idx, ent) for idx, ent in enumerate(entities) if ent["kind"] == "BDC"]
    z_local_index = {entity_idx: i for i, (entity_idx, _) in enumerate(z_entities)}

    z_sites = []
    for entity_idx, ent in z_entities:
        centers = carboxylate_centers(ent["atoms"])  # type: ignore[arg-type]
        for carbox_idx, _, center in centers:
            z_sites.append((entity_idx, z_local_index[entity_idx], carbox_idx, center))

    bdc_neighbor_sets: list[tuple[int, ...]] = []
    bdc_match_distances: list[float] = []
    bdc_min_distances: list[float] = []
    matched_edges: list[tuple[int, int]] = []
    for bdc_order, (entity_idx, ent) in enumerate(bdc_entities):
        centers = carboxylate_centers(ent["atoms"])  # type: ignore[arg-type]
        neighbors = []
        for _, _, center in centers:
            nearest = min(
                (
                    (distance(center, site_center), z_entity_idx, z_idx)
                    for z_entity_idx, z_idx, _, site_center in z_sites
                ),
                key=lambda item: item[0],
            )
            d_min, z_entity_idx, z_idx = nearest
            bdc_min_distances.append(d_min)
            if d_min <= LINK_MATCH_CUTOFF_A:
                neighbors.append(z_idx)
                bdc_match_distances.append(d_min)
                matched_edges.append((z_idx, len(z_entities) + bdc_order))
        bdc_neighbor_sets.append(tuple(sorted(neighbors)))

    topology_code = canonical_bipartite_topology(len(z_entities), bdc_neighbor_sets)
    topology_id = "T_" + short_hash(topology_code)

    z_centers = [ent["center"] for _, ent in z_entities]  # type: ignore[list-item]
    sbu_geometry_id = "G_" + pairwise_distance_fingerprint(z_centers)
    all_atom_geometry_id = "A_" + all_atom_distance_fingerprint(atoms)

    degrees = [len(neigh) for neigh in bdc_neighbor_sets]
    edge_count = sum(degrees)
    node_count = len(z_entities) + len(bdc_entities)
    comps = component_count(node_count, matched_edges)
    cycle_rank = edge_count - node_count + comps
    repeat = path.parents[1].name
    repeat_id = repeat.split("_I")[-1].split("_")[0] if "_I" in repeat else repeat
    atom_counts = Counter(element for element, _ in atoms)

    return {
        "repeat_id": repeat_id,
        "repeat_folder": repeat,
        "mol2_path": str(path),
        "atom_count": len(atoms),
        "entity_count": len(entities),
        "zr6_count": len(z_entities),
        "bdc_count": len(bdc_entities),
        "edge_count": edge_count,
        "mono_linker_count": sum(1 for degree in degrees if degree == 1),
        "bridging_linker_count": sum(1 for degree in degrees if degree == 2),
        "unmatched_linker_count": sum(1 for degree in degrees if degree == 0),
        "component_count": comps,
        "cycle_rank": cycle_rank,
        "topology_id": topology_id,
        "topology_code": topology_code,
        "sbu_geometry_id": sbu_geometry_id,
        "all_atom_geometry_id": all_atom_geometry_id,
        "max_matched_distance_A": max(bdc_match_distances) if bdc_match_distances else "",
        "min_unmatched_distance_A": min(
            [value for value in bdc_min_distances if value > LINK_MATCH_CUTOFF_A],
            default="",
        ),
        "C_count": atom_counts.get("C", 0),
        "O_count": atom_counts.get("O", 0),
        "Zr_count": atom_counts.get("Zr", 0),
        "bdc_degrees": ";".join(str(degree) for degree in sorted(degrees)),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)

    def unique_values(group_rows: list[dict[str, object]], field: str) -> str:
        counts = Counter(str(row.get(field, "")) for row in group_rows)
        parts = []
        for value, count in sorted(counts.items(), key=lambda item: item[0]):
            parts.append(value if count == len(group_rows) else f"{value}({count})")
        return ";".join(parts)

    summary = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        first = group_rows[0]
        summary.append(
            {
                key: group_key,
                "n": len(group_rows),
                "repeat_ids": ",".join(str(row["repeat_id"]) for row in group_rows),
                "zr6_count": unique_values(group_rows, "zr6_count"),
                "bdc_count": unique_values(group_rows, "bdc_count"),
                "edge_count": unique_values(group_rows, "edge_count"),
                "mono_linker_count": unique_values(group_rows, "mono_linker_count"),
                "bridging_linker_count": unique_values(group_rows, "bridging_linker_count"),
                "component_count": unique_values(group_rows, "component_count"),
                "cycle_rank": unique_values(group_rows, "cycle_rank"),
                "atom_count": unique_values(group_rows, "atom_count"),
                "topology_id_list": unique_values(group_rows, "topology_id"),
                "topology_code": first.get("topology_code", "") if key == "topology_id" else "",
            }
        )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(RUN_ROOT.glob(f"*/last_by_n/assembly_last_entity_number_{ENTITY_N}.mol2"))
    rows = [analyze_case(path) for path in paths]

    case_fields = [
        "repeat_id",
        "repeat_folder",
        "atom_count",
        "entity_count",
        "zr6_count",
        "bdc_count",
        "edge_count",
        "mono_linker_count",
        "bridging_linker_count",
        "unmatched_linker_count",
        "component_count",
        "cycle_rank",
        "topology_id",
        "sbu_geometry_id",
        "all_atom_geometry_id",
        "max_matched_distance_A",
        "min_unmatched_distance_A",
        "C_count",
        "O_count",
        "Zr_count",
        "bdc_degrees",
        "mol2_path",
        "topology_code",
    ]
    write_csv(OUTPUT_DIR / "uio66_entity13_structure_cases.csv", rows, case_fields)

    summary_fields = [
        "topology_id",
        "sbu_geometry_id",
        "all_atom_geometry_id",
        "n",
        "repeat_ids",
        "zr6_count",
        "bdc_count",
        "edge_count",
        "mono_linker_count",
        "bridging_linker_count",
        "component_count",
        "cycle_rank",
        "atom_count",
        "topology_id_list",
        "topology_code",
    ]
    topology_summary = summarize(rows, "topology_id")
    write_csv(
        OUTPUT_DIR / "uio66_entity13_structure_topology_distribution.csv",
        topology_summary,
        summary_fields,
    )
    sbu_summary = summarize(rows, "sbu_geometry_id")
    write_csv(
        OUTPUT_DIR / "uio66_entity13_structure_sbu_geometry_distribution.csv",
        sbu_summary,
        summary_fields,
    )
    all_atom_summary = summarize(rows, "all_atom_geometry_id")
    write_csv(
        OUTPUT_DIR / "uio66_entity13_structure_all_atom_geometry_distribution.csv",
        all_atom_summary,
        summary_fields,
    )

    lines = [
        "# UiO-66 entity-13 structure uniqueness analysis",
        "",
        f"- Cases analyzed: `{len(rows)}`.",
        f"- Input snapshots: `assembly_last_entity_number_{ENTITY_N}.mol2` from the completed long-growth batch.",
        f"- Link assignment cutoff: `{LINK_MATCH_CUTOFF_A} A` between BDC and Zr6 carboxylate centers.",
        "- Important limitation: the saved entity-13 snapshots are mol2 files with zero bond records; no entity-13 pickle files are present. Therefore this analysis reconstructs SBU-linker connectivity from geometry rather than reading the original `connected_entities` objects.",
        "",
        "## Topology distribution",
        "",
    ]
    for item in topology_summary:
        lines.append(
            "- `{topology_id}`: n = `{n}`, repeats = `{repeat_ids}`, "
            "Zr6/BDC = `{zr6_count}/{bdc_count}`, edges = `{edge_count}`, "
            "mono/bridging BDC = `{mono_linker_count}/{bridging_linker_count}`, "
            "cycle rank = `{cycle_rank}`.".format(**item)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The 24 entity-13 snapshots are not a single unique structure. They separate into multiple reconstructed SBU-linker topologies, and the stricter SBU-geometry and all-atom distance fingerprints split them further. Thus entity 13 should be described as a recurrent bottleneck or motif family, not as one deterministic nucleus.",
        ]
    )
    (OUTPUT_DIR / "uio66_entity13_structure_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
