from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "reports" / "figure_assets" / "prebound_motifs"


@dataclass
class Atom:
    index: int
    element: str
    x: float
    y: float
    z: float


@dataclass
class Bond:
    a1: int
    a2: int
    order: int


def parse_mol(path: Path) -> tuple[list[Atom], list[Bond]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    counts = lines[3].split()
    atom_count = int(counts[0])
    bond_count = int(counts[1])

    atoms: list[Atom] = []
    for idx in range(atom_count):
        line = lines[4 + idx]
        atoms.append(
            Atom(
                index=idx + 1,
                x=float(line[0:10]),
                y=float(line[10:20]),
                z=float(line[20:30]),
                element=line[31:34].strip(),
            )
        )

    bonds: list[Bond] = []
    for idx in range(bond_count):
        line = lines[4 + atom_count + idx]
        bonds.append(
            Bond(
                a1=int(line[0:3]),
                a2=int(line[3:6]),
                order=int(line[6:9]),
            )
        )
    return atoms, bonds


def adjacency_map(bonds: list[Bond]) -> dict[int, list[tuple[int, int]]]:
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for bond in bonds:
        adjacency.setdefault(bond.a1, []).append((bond.a2, bond.order))
        adjacency.setdefault(bond.a2, []).append((bond.a1, bond.order))
    return adjacency


def vec(atom: Atom) -> np.ndarray:
    return np.array([atom.x, atom.y, atom.z], dtype=float)


def centroid(atoms: list[Atom]) -> np.ndarray:
    if not atoms:
        raise RuntimeError("Cannot compute centroid for an empty atom list.")
    coordinates = np.array([vec(atom) for atom in atoms], dtype=float)
    return coordinates.mean(axis=0)


def distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def identify_carboxylates(atoms: list[Atom], bonds: list[Bond]) -> list[tuple[int, int, int]]:
    adjacency = adjacency_map(bonds)
    atom_map = {atom.index: atom for atom in atoms}
    carboxylates: list[tuple[int, int, int]] = []

    for atom in atoms:
        if atom.element != "C":
            continue
        oxygen_neighbors = [
            neighbor_idx
            for neighbor_idx, _ in adjacency.get(atom.index, [])
            if atom_map[neighbor_idx].element == "O"
        ]
        if len(oxygen_neighbors) != 2:
            continue

        single_bond_oxygen = None
        for neighbor_idx, order in adjacency.get(atom.index, []):
            if neighbor_idx in oxygen_neighbors and order == 1:
                single_bond_oxygen = neighbor_idx
                break

        oxygen_a = single_bond_oxygen or oxygen_neighbors[0]
        oxygen_b = oxygen_neighbors[0] if oxygen_neighbors[1] == oxygen_a else oxygen_neighbors[1]
        carboxylates.append((atom.index, oxygen_a, oxygen_b))

    return carboxylates


def choose_outermost_carboxylate(
    atoms: list[Atom],
    carboxylates: list[tuple[int, int, int]],
    scope_atom_ids: set[int] | None = None,
) -> tuple[int, int, int]:
    if not carboxylates:
        raise RuntimeError("Failed to find a carboxylate group.")

    atom_map = {atom.index: atom for atom in atoms}
    scoped_atoms = [atom for atom in atoms if scope_atom_ids is None or atom.index in scope_atom_ids]
    center = centroid(scoped_atoms)
    return max(
        carboxylates,
        key=lambda group: distance(vec(atom_map[group[0]]), center),
    )


def rigid_transform(
    source_points: list[np.ndarray],
    target_points: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    source = np.vstack(source_points)
    target = np.vstack(target_points)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_shifted = source - source_center
    target_shifted = target - target_center
    covariance = source_shifted.T @ target_shifted
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def transform_atoms(atoms: list[Atom], rotation: np.ndarray, translation: np.ndarray) -> list[Atom]:
    transformed: list[Atom] = []
    for atom in atoms:
        coordinates = vec(atom) @ rotation.T + translation
        transformed.append(
            Atom(
                index=atom.index,
                element=atom.element,
                x=float(coordinates[0]),
                y=float(coordinates[1]),
                z=float(coordinates[2]),
            )
        )
    return transformed


def collect_cluster_ligand_atoms(
    atoms: list[Atom],
    bonds: list[Bond],
    carboxylate: tuple[int, int, int],
) -> set[int]:
    adjacency = adjacency_map(bonds)
    atom_map = {atom.index: atom for atom in atoms}
    carbon_idx, oxygen_a, oxygen_b = carboxylate
    removable = {carbon_idx, oxygen_a, oxygen_b}
    blocked = {oxygen_a, oxygen_b}
    stack: list[int] = []

    for neighbor_idx, _ in adjacency.get(carbon_idx, []):
        if neighbor_idx in blocked or atom_map[neighbor_idx].element == "Zr":
            continue
        stack.append(neighbor_idx)

    while stack:
        current = stack.pop()
        if current in removable:
            continue
        if atom_map[current].element == "Zr":
            continue
        removable.add(current)
        for neighbor_idx, _ in adjacency.get(current, []):
            if neighbor_idx in removable or neighbor_idx in blocked:
                continue
            if atom_map[neighbor_idx].element == "Zr":
                continue
            stack.append(neighbor_idx)

    return removable


def minimum_heavy_atom_distance(
    cluster_atoms: list[Atom],
    linker_atoms: list[Atom],
    excluded_linker_atoms: set[int],
) -> float:
    min_distance = float("inf")
    linker_candidates = [
        atom for atom in linker_atoms if atom.element != "H" and atom.index not in excluded_linker_atoms
    ]
    cluster_candidates = [atom for atom in cluster_atoms if atom.element != "H"]

    for linker_atom in linker_candidates:
        linker_position = vec(linker_atom)
        for cluster_atom in cluster_candidates:
            candidate_distance = distance(linker_position, vec(cluster_atom))
            if candidate_distance < min_distance:
                min_distance = candidate_distance

    return 0.0 if min_distance == float("inf") else min_distance


def build_substituted_motif(
    cluster_path: Path,
    linker_path: Path,
    motif_name: str,
) -> tuple[list[Atom], list[Bond], int]:
    cluster_atoms, cluster_bonds = parse_mol(cluster_path)
    linker_atoms, linker_bonds = parse_mol(linker_path)
    cluster_atom_map = {atom.index: atom for atom in cluster_atoms}
    cluster_adjacency = adjacency_map(cluster_bonds)

    cluster_carboxylate = choose_outermost_carboxylate(
        cluster_atoms,
        identify_carboxylates(cluster_atoms, cluster_bonds),
        scope_atom_ids={atom.index for atom in cluster_atoms if atom.element != "H"},
    )
    linker_carboxylate = choose_outermost_carboxylate(
        linker_atoms,
        identify_carboxylates(linker_atoms, linker_bonds),
    )

    removable_cluster_atoms = collect_cluster_ligand_atoms(cluster_atoms, cluster_bonds, cluster_carboxylate)
    retained_cluster_atoms = [
        atom for atom in cluster_atoms if atom.index not in removable_cluster_atoms and atom.element != "H"
    ]

    target_points = [vec(cluster_atom_map[idx]) for idx in cluster_carboxylate]
    linker_atom_map = {atom.index: atom for atom in linker_atoms}
    linker_orders = [
        linker_carboxylate,
        (linker_carboxylate[0], linker_carboxylate[2], linker_carboxylate[1]),
    ]

    best_transformed_linker: list[Atom] | None = None
    best_oxygen_mapping: dict[int, int] | None = None
    best_score: tuple[float, float] | None = None
    cluster_center = centroid([atom for atom in cluster_atoms if atom.element != "H"])
    linker_carboxylate_set = set(linker_carboxylate)

    for linker_order in linker_orders:
        source_points = [vec(linker_atom_map[idx]) for idx in linker_order]
        rotation, translation = rigid_transform(source_points, target_points)
        transformed_linker = [
            atom for atom in transform_atoms(linker_atoms, rotation, translation) if atom.element != "H"
        ]
        transformed_linker_map = {atom.index: atom for atom in transformed_linker}
        linker_body = [
            atom for atom in transformed_linker if atom.index not in linker_carboxylate_set
        ]
        linker_body_center = centroid(linker_body) if linker_body else centroid(transformed_linker)
        clash_score = minimum_heavy_atom_distance(
            retained_cluster_atoms,
            transformed_linker,
            excluded_linker_atoms=linker_carboxylate_set,
        )
        outward_score = distance(linker_body_center, cluster_center)
        score = (clash_score, outward_score)
        if best_score is None or score > best_score:
            best_score = score
            best_transformed_linker = transformed_linker
            best_oxygen_mapping = {
                linker_order[1]: cluster_carboxylate[1],
                linker_order[2]: cluster_carboxylate[2],
            }

    if best_transformed_linker is None or best_oxygen_mapping is None:
        raise RuntimeError(f"Failed to align linker carboxylate for {motif_name}.")

    combined_atoms: list[Atom] = []
    index_map_cluster: dict[int, int] = {}
    index_map_linker: dict[int, int] = {}

    for atom in retained_cluster_atoms:
        new_index = len(combined_atoms) + 1
        index_map_cluster[atom.index] = new_index
        combined_atoms.append(Atom(new_index, atom.element, atom.x, atom.y, atom.z))

    for atom in best_transformed_linker:
        new_index = len(combined_atoms) + 1
        index_map_linker[atom.index] = new_index
        combined_atoms.append(Atom(new_index, atom.element, atom.x, atom.y, atom.z))

    combined_bonds: list[Bond] = []
    retained_cluster_ids = set(index_map_cluster)
    for bond in cluster_bonds:
        if bond.a1 in retained_cluster_ids and bond.a2 in retained_cluster_ids:
            combined_bonds.append(Bond(index_map_cluster[bond.a1], index_map_cluster[bond.a2], bond.order))

    for bond in linker_bonds:
        if bond.a1 in index_map_linker and bond.a2 in index_map_linker:
            combined_bonds.append(Bond(index_map_linker[bond.a1], index_map_linker[bond.a2], bond.order))

    for linker_oxygen_idx, cluster_oxygen_idx in best_oxygen_mapping.items():
        for neighbor_idx, bond_order in cluster_adjacency.get(cluster_oxygen_idx, []):
            if cluster_atom_map[neighbor_idx].element != "Zr":
                continue
            combined_bonds.append(
                Bond(
                    index_map_cluster[neighbor_idx],
                    index_map_linker[linker_oxygen_idx],
                    bond_order,
                )
            )

    combined_center = centroid(combined_atoms)
    recentered_atoms = [
        Atom(
            atom.index,
            atom.element,
            atom.x - float(combined_center[0]),
            atom.y - float(combined_center[1]),
            atom.z - float(combined_center[2]),
        )
        for atom in combined_atoms
    ]
    return recentered_atoms, combined_bonds, len(retained_cluster_atoms)


def atom_name(element: str, counter: int) -> str:
    return f"{element}{counter}"


def atom_type(element: str) -> str:
    mapping = {
        "C": "C",
        "H": "H",
        "O": "O",
        "Zr": "Zr",
    }
    return mapping.get(element, element)


def write_mol2(path: Path, name: str, atoms: list[Atom], bonds: list[Bond], cluster_atom_count: int) -> None:
    lines: list[str] = []
    lines.append("@<TRIPOS>MOLECULE")
    lines.append(name)
    lines.append(f"{len(atoms)} {len(bonds)} 2 0 0")
    lines.append("SMALL")
    lines.append("USER_CHARGES")
    lines.append("@<TRIPOS>ATOM")

    element_counter: dict[str, int] = {}
    for atom in atoms:
        element_counter[atom.element] = element_counter.get(atom.element, 0) + 1
        substructure_id = 1 if atom.index <= cluster_atom_count else 2
        substructure_name = "CORE" if substructure_id == 1 else "LIG"
        lines.append(
            f"{atom.index} {atom_name(atom.element, element_counter[atom.element])} "
            f"{atom.x: .4f} {atom.y: .4f} {atom.z: .4f} {atom_type(atom.element)} "
            f"{substructure_id} {substructure_name} 0.0000"
        )

    lines.append("@<TRIPOS>BOND")
    for bond_index, bond in enumerate(bonds, start=1):
        bond_type = "ar" if bond.order == 4 else str(bond.order)
        lines.append(f"{bond_index} {bond.a1} {bond.a2} {bond_type}")

    lines.append("@<TRIPOS>SUBSTRUCTURE")
    lines.append("1 CORE 1")
    lines.append(f"2 LIG {cluster_atom_count + 1}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_motif(cluster_path: Path, linker_path: Path, output_name: str) -> Path:
    atoms, bonds, cluster_atom_count = build_substituted_motif(cluster_path, linker_path, output_name)
    output_path = OUTPUT_DIR / output_name
    write_mol2(output_path, output_name.replace(".mol2", ""), atoms, bonds, cluster_atom_count)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zr_btb_path = export_motif(
        REPO_ROOT / "MOL-Zr-Ligand" / "input_monomers" / "Zr6.mol",
        REPO_ROOT / "MOL-Zr-Ligand" / "input_monomers" / "BTBa.mol",
        "simplified_Zr-BTB.mol2",
    )
    zr_bdc_path = export_motif(
        REPO_ROOT / "UiO-66" / "Zr-Ligand-ExplicitChannel" / "input_monomers" / "Zr6_AA.mol",
        REPO_ROOT / "UiO-66" / "Zr-Ligand-ExplicitChannel" / "input_monomers" / "BDC.mol",
        "simplified_Zr-BDC.mol2",
    )

    print(f"Wrote {zr_btb_path}")
    print(f"Wrote {zr_bdc_path}")


if __name__ == "__main__":
    main()
