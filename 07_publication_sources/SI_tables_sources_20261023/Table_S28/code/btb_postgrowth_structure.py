"""Post-growth dimensionality metrics for unrestricted Zr6/Hf6--BTB assemblies.

This module does not constrain growth.  It inspects a completed molecular
assembly after all twelve carboxylate directions on each M6 node have been
available to the KMC engine.  Layer formation is therefore an output.

Two complementary descriptions are retained:

* global principal-axis metrics of the metal-node centres; and
* local layer planes defined only when one tripodal BTB is connected to three
  distinct metal nodes.  Compatible local planes are clustered into layer
  domains without assigning a plane during growth.

The raw metrics are the primary output.  The shape label is explicitly
provisional until its tolerances have been compared with the archived
``plane_count`` post-processor and representative structures.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class LocalPlane:
    """Plane supported by one fully connected tripodal linker."""

    linker_index: int
    metal_indices: tuple[int, int, int]
    center: np.ndarray
    normal: np.ndarray
    offset: float


def _canonical_plane(normal: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, float]:
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    pivot = int(np.argmax(np.abs(normal)))
    if normal[pivot] < 0.0:
        normal = -normal
    return normal, float(np.dot(normal, center))


def best_fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centre, unit normal, and descending covariance eigenvalues."""

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("at least three three-dimensional points are required")
    center = np.mean(points, axis=0)
    _, singular_values, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal, _ = _canonical_plane(normal, center)
    eigenvalues = singular_values**2 / max(len(points) - 1, 1)
    eigenvalues = np.pad(eigenvalues, (0, max(0, 3 - len(eigenvalues))))[:3]
    return center, normal, np.sort(eigenvalues)[::-1]


def _entity_lists(assembly: Any) -> tuple[list[Any], list[Any], dict[int, int]]:
    entities = list(assembly.entities)
    metals = [entity for entity in entities if getattr(entity, "entity_type", None) == "Zr"]
    linkers = [
        entity for entity in entities if getattr(entity, "entity_type", None) == "Ligand"
    ]
    metal_index_by_identity = {id(entity): index for index, entity in enumerate(metals)}
    return metals, linkers, metal_index_by_identity


def local_tripodal_planes(assembly: Any) -> list[LocalPlane]:
    """Construct one local plane for every BTB bound to three metal nodes."""

    metals, linkers, metal_index_by_identity = _entity_lists(assembly)
    observations: list[LocalPlane] = []
    for linker_index, linker in enumerate(linkers):
        neighbors: list[Any] = []
        seen: set[int] = set()
        for neighbor in getattr(linker, "connected_entities", ()):
            key = id(neighbor)
            if key in seen or key not in metal_index_by_identity:
                continue
            seen.add(key)
            neighbors.append(neighbor)
        if len(neighbors) != 3:
            continue
        points = np.asarray([neighbor.center for neighbor in neighbors], dtype=float)
        vector_a = points[1] - points[0]
        vector_b = points[2] - points[0]
        normal = np.cross(vector_a, vector_b)
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm < 1.0e-8:
            continue
        center = np.mean(points, axis=0)
        normal, offset = _canonical_plane(normal / norm, center)
        observations.append(
            LocalPlane(
                linker_index=linker_index,
                metal_indices=tuple(
                    sorted(metal_index_by_identity[id(neighbor)] for neighbor in neighbors)
                ),
                center=center,
                normal=normal,
                offset=offset,
            )
        )
    return observations


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _planes_are_compatible(
    left: LocalPlane,
    right: LocalPlane,
    *,
    angle_tolerance_deg: float,
    offset_tolerance_angstrom: float,
) -> bool:
    dot = float(np.clip(np.dot(left.normal, right.normal), -1.0, 1.0))
    sign = 1.0 if dot >= 0.0 else -1.0
    angle = math.degrees(math.acos(abs(dot)))
    aligned_offset = sign * right.offset
    return (
        angle <= angle_tolerance_deg
        and abs(left.offset - aligned_offset) <= offset_tolerance_angstrom
    )


def cluster_local_planes(
    observations: Iterable[LocalPlane],
    *,
    angle_tolerance_deg: float = 7.5,
    offset_tolerance_angstrom: float = 3.0,
) -> list[list[LocalPlane]]:
    """Cluster mutually compatible local planes into post-growth domains."""

    planes = list(observations)
    union_find = _UnionFind(len(planes))
    for left in range(len(planes)):
        for right in range(left + 1, len(planes)):
            if _planes_are_compatible(
                planes[left],
                planes[right],
                angle_tolerance_deg=angle_tolerance_deg,
                offset_tolerance_angstrom=offset_tolerance_angstrom,
            ):
                union_find.union(left, right)
    clusters: dict[int, list[LocalPlane]] = {}
    for index, plane in enumerate(planes):
        clusters.setdefault(union_find.find(index), []).append(plane)
    return sorted(
        clusters.values(),
        key=lambda group: (
            len({index for plane in group for index in plane.metal_indices}),
            len(group),
        ),
        reverse=True,
    )


def _cluster_record(cluster: list[LocalPlane], metal_centers: np.ndarray) -> dict[str, Any]:
    metal_indices = sorted({index for plane in cluster for index in plane.metal_indices})
    points = metal_centers[metal_indices]
    center, normal, eigenvalues = best_fit_plane(points)
    distances = (points - center) @ normal
    return {
        "local_plane_observations": len(cluster),
        "supporting_metal_nodes": len(metal_indices),
        "supporting_metal_indices": metal_indices,
        "center_angstrom": center.tolist(),
        "normal": normal.tolist(),
        "rms_support_thickness_angstrom": float(np.sqrt(np.mean(distances**2))),
        "support_thickness_span_angstrom": float(np.ptp(distances)),
        "support_covariance_eigenvalues_angstrom2": eigenvalues.tolist(),
    }


def analyze_btb_assembly(
    assembly: Any,
    *,
    angle_tolerance_deg: float = 7.5,
    offset_tolerance_angstrom: float = 3.0,
    minimum_local_planes_for_stable_layer: int = 2,
) -> dict[str, Any]:
    """Measure layer emergence after an unrestricted BTB growth trace."""

    metals, linkers, _ = _entity_lists(assembly)
    result: dict[str, Any] = {
        "definition": (
            "Post-growth analysis only; all twelve M6 carboxylate directions were "
            "available during growth. Local planes are defined by tripodal BTB linkers "
            "bound to three distinct metal nodes."
        ),
        "metal_nodes": len(metals),
        "linkers": len(linkers),
        "angle_tolerance_deg": float(angle_tolerance_deg),
        "offset_tolerance_angstrom": float(offset_tolerance_angstrom),
        "minimum_local_planes_for_stable_layer": int(
            minimum_local_planes_for_stable_layer
        ),
    }
    if len(metals) < 3:
        result.update(
            {
                "status": "insufficient_metal_nodes",
                "local_tripodal_planes": 0,
                "layer_domain_count": 0,
                "stable_layer_domain_count": 0,
                "geometric_shape_class_provisional": "insufficient",
            }
        )
        return result

    metal_centers = np.asarray([entity.center for entity in metals], dtype=float)
    global_center, global_normal, eigenvalues = best_fit_plane(metal_centers)
    global_distances = (metal_centers - global_center) @ global_normal
    lambda_1, lambda_2, lambda_3 = [float(value) for value in eigenvalues]
    linearity_ratio = lambda_2 / max(lambda_1, 1.0e-15)
    flatness_ratio = lambda_3 / max(lambda_2, 1.0e-15)

    observations = local_tripodal_planes(assembly)
    clusters = cluster_local_planes(
        observations,
        angle_tolerance_deg=angle_tolerance_deg,
        offset_tolerance_angstrom=offset_tolerance_angstrom,
    )
    cluster_records = [_cluster_record(cluster, metal_centers) for cluster in clusters]
    stable = [
        record
        for record in cluster_records
        if record["local_plane_observations"] >= minimum_local_planes_for_stable_layer
    ]

    if linearity_ratio < 0.05:
        provisional_shape = "one_dimensional_like"
    elif flatness_ratio < 0.05:
        provisional_shape = "two_dimensional_like"
    else:
        provisional_shape = "three_dimensional_like"

    result.update(
        {
            "status": "measured",
            "global_center_angstrom": global_center.tolist(),
            "global_plane_normal": global_normal.tolist(),
            "global_covariance_eigenvalues_angstrom2": eigenvalues.tolist(),
            "global_linearity_ratio_lambda2_over_lambda1": float(linearity_ratio),
            "global_flatness_ratio_lambda3_over_lambda2": float(flatness_ratio),
            "global_rms_thickness_angstrom": float(
                np.sqrt(np.mean(global_distances**2))
            ),
            "global_thickness_span_angstrom": float(np.ptp(global_distances)),
            "local_tripodal_planes": len(observations),
            "layer_domain_count": len(cluster_records),
            "stable_layer_domain_count": len(stable),
            "layer_domains": cluster_records,
            "largest_domain_metal_fraction": (
                cluster_records[0]["supporting_metal_nodes"] / len(metals)
                if cluster_records
                else 0.0
            ),
            "largest_domain_tripodal_linker_fraction": (
                cluster_records[0]["local_plane_observations"] / len(observations)
                if cluster_records and observations
                else 0.0
            ),
            "geometric_shape_class_provisional": provisional_shape,
            "classification_warning": (
                "The provisional label uses raw PCA ratios of 0.05. Manuscript use "
                "requires calibration against the archived plane-count routine and "
                "visual inspection of representative molecular structures."
            ),
        }
    )
    return result

