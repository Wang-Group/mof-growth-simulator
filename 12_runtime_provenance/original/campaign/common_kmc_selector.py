"""Common site-resolved event selector for UiO-66 and BTB-MOL recalculation.

This module deliberately contains no recipe-to-rate chemistry.  It accepts the
four rate coordinates that the chemistry layer must supply and combines them
with the compatible opportunity counts in a molecular assembly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Callable, Mapping, Protocol, Sequence


CLOSURE_ENTROPY_SCALE = 30.9


class RandomPool(Protocol):
    """Minimal protocol implemented by the geometry engines' RandomizedSet."""

    items: Sequence[Any]

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class PhysicalCoordinates:
    """Rate coordinates relative to one declared site-opening reference.

    ``external_activity`` is the sum of productive incoming-node and
    incoming-linker activities. ``node_fraction`` partitions that activity.
    ``off_weight`` is the bond-removal weight on the same reference.
    ``closure_entropy_coefficient`` generates a constant internal-closure
    weight, exp(30.9 * SC), with no entity-number correction.
    """

    external_activity: float
    node_fraction: float
    off_weight: float
    closure_entropy_coefficient: float
    closure_entropy_scale: float = CLOSURE_ENTROPY_SCALE

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite; received {value!r}")
        if self.external_activity < 0.0:
            raise ValueError("external_activity must be non-negative")
        if not 0.0 <= self.node_fraction <= 1.0:
            raise ValueError("node_fraction must lie in [0, 1]")
        if self.off_weight < 0.0:
            raise ValueError("off_weight must be non-negative")
        if self.closure_entropy_scale <= 0.0:
            raise ValueError("closure_entropy_scale must be positive")

    @property
    def node_activity(self) -> float:
        return self.external_activity * self.node_fraction

    @property
    def linker_activity(self) -> float:
        return self.external_activity * (1.0 - self.node_fraction)

    @property
    def closure_weight(self) -> float:
        exponent = self.closure_entropy_scale * self.closure_entropy_coefficient
        if exponent > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            raise OverflowError("closure weight exceeds finite float range")
        return math.exp(exponent)

    @property
    def normalized_off_ratio(self) -> float:
        if self.external_activity == 0.0:
            return math.inf if self.off_weight > 0.0 else math.nan
        return self.off_weight / self.external_activity

    @property
    def normalized_closure_ratio(self) -> float:
        if self.external_activity == 0.0:
            return math.inf
        return self.closure_weight / self.external_activity


@dataclass(frozen=True)
class OpportunityCounts:
    linker_free_sites: int
    metal_free_sites: int
    ready_internal_pairs: int
    existing_bonds: int
    removal_opportunity_weight: float | None = None

    def validate(self) -> None:
        for name in ("linker_free_sites", "metal_free_sites", "ready_internal_pairs", "existing_bonds"):
            value = getattr(self, name)
            if int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.removal_opportunity_weight is not None:
            value = float(self.removal_opportunity_weight)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("removal_opportunity_weight must be finite and non-negative")

    @property
    def effective_removal_opportunities(self) -> float:
        return (
            float(self.existing_bonds)
            if self.removal_opportunity_weight is None
            else float(self.removal_opportunity_weight)
        )


@dataclass(frozen=True)
class EventPropensities:
    add_node: float
    add_linker: float
    close_internal: float
    remove_bond: float

    @property
    def total(self) -> float:
        return self.add_node + self.add_linker + self.close_internal + self.remove_bond

    def normalized(self) -> "EventPropensities":
        if self.total <= 0.0:
            raise ValueError("cannot normalize a zero-propensity state")
        return EventPropensities(
            add_node=self.add_node / self.total,
            add_linker=self.add_linker / self.total,
            close_internal=self.close_internal / self.total,
            remove_bond=self.remove_bond / self.total,
        )


@dataclass(frozen=True)
class SelectedEvent:
    event: str
    selected_site: Any | None
    selected_pair: Any | None
    propensities: EventPropensities


def propensities_from_weights(
    counts: OpportunityCounts,
    *,
    node_activity: float,
    linker_activity: float,
    closure_weight: float,
    off_weight: float,
) -> EventPropensities:
    """Combine compatible opportunity counts with per-opportunity weights."""

    counts.validate()
    weights = {
        "node_activity": node_activity,
        "linker_activity": linker_activity,
        "closure_weight": closure_weight,
        "off_weight": off_weight,
    }
    for name, value in weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    return EventPropensities(
        add_node=counts.linker_free_sites * node_activity,
        add_linker=counts.metal_free_sites * linker_activity,
        close_internal=counts.ready_internal_pairs * closure_weight,
        remove_bond=counts.effective_removal_opportunities * off_weight,
    )


def compute_propensities(
    counts: OpportunityCounts, coordinates: PhysicalCoordinates
) -> EventPropensities:
    coordinates.validate()
    return propensities_from_weights(
        counts,
        node_activity=coordinates.node_activity,
        linker_activity=coordinates.linker_activity,
        closure_weight=coordinates.closure_weight,
        off_weight=coordinates.off_weight,
    )


def counts_from_assembly(assembly: Any) -> OpportunityCounts:
    """Extract the four compatible opportunity counts from either engine."""

    return OpportunityCounts(
        linker_free_sites=len(assembly.Linker_free_carboxylates),
        metal_free_sites=len(assembly.MC_free_carboxylates),
        ready_internal_pairs=len(assembly.ready_to_connect_carboxylate_pairs),
        existing_bonds=len(assembly.linked_carboxylate_pairs),
    )


def bond_removal_pool(
    assembly: Any,
    mode: str = "all_bonds",
    active_site_predicate: Callable[[Any], bool] | None = None,
) -> tuple[list[Any], list[float]]:
    """Return linked bonds and their solution-accessibility weights.

    ``all_bonds`` reproduces the archived assumption.  In
    ``undercoordinated_metal`` mode, a bond on a node with ``d`` occupied sites
    among ``z`` active sites receives weight ``(z-d)/z``.  Thus a fully
    coordinated interior node is protected, while bonds on an undercoordinated
    growth-front node remain exchange-accessible.  ``dangling_only`` is a
    limiting sensitivity model that exposes only bonds attached to an entity
    with graph degree one.
    """

    if mode not in {"all_bonds", "undercoordinated_metal", "dangling_only"}:
        raise ValueError(f"unknown bond accessibility mode {mode!r}")
    bonds = list(assembly.linked_carboxylate_pairs.items)
    if mode == "all_bonds":
        return bonds, [1.0] * len(bonds)
    if mode == "dangling_only":
        return bonds, [
            1.0
            if min(len(pair[0].belonging_entity.connected_entities), len(pair[1].belonging_entity.connected_entities)) == 1
            else 0.0
            for pair in bonds
        ]

    metal_entities: dict[int, Any] = {}
    degree: dict[int, int] = {}
    for pair in bonds:
        metal_sites = [site for site in pair if getattr(site, "carboxylate_type", None) == "formate"]
        if len(metal_sites) != 1:
            raise ValueError("each framework bond must contain exactly one metal carboxylate")
        entity = metal_sites[0].belonging_entity
        key = id(entity)
        metal_entities[key] = entity
        degree[key] = degree.get(key, 0) + 1
    accessibility: dict[int, float] = {}
    for key, entity in metal_entities.items():
        sites = list(entity.carboxylates)
        capacity = sum(
            1 for site in sites if active_site_predicate is None or active_site_predicate(site)
        )
        if capacity <= 0:
            accessibility[key] = 0.0
        else:
            accessibility[key] = max(0.0, min(1.0, (capacity - degree[key]) / capacity))
    return bonds, [
        accessibility[id(next(site for site in pair if getattr(site, "carboxylate_type", None) == "formate").belonging_entity)]
        for pair in bonds
    ]


def choose_weighted(items: Sequence[Any], weights: Sequence[float], rng: random.Random) -> Any:
    if len(items) != len(weights) or not items:
        raise RuntimeError("weighted molecular pool is empty or inconsistent")
    total = float(sum(weights))
    if total <= 0.0:
        raise RuntimeError("weighted molecular pool has zero accessible weight")
    draw = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += float(weight)
        if draw < cumulative:
            return item
    return items[-1]


def _choose(pool: RandomPool, rng: random.Random) -> Any:
    if len(pool) == 0:
        raise RuntimeError("selected event has no compatible molecular opportunity")
    return rng.choice(pool.items)


def select_event(
    assembly: Any,
    coordinates: PhysicalCoordinates,
    rng: random.Random,
    bond_accessibility_mode: str = "all_bonds",
) -> SelectedEvent:
    """Select one molecular event without incompatible-site fallbacks."""

    counts = counts_from_assembly(assembly)
    bonds, bond_weights = bond_removal_pool(assembly, bond_accessibility_mode)
    counts = OpportunityCounts(
        counts.linker_free_sites,
        counts.metal_free_sites,
        counts.ready_internal_pairs,
        counts.existing_bonds,
        sum(bond_weights),
    )
    prop = compute_propensities(counts, coordinates)
    if prop.total <= 0.0:
        raise RuntimeError("no enabled event in the current molecular state")

    draw = rng.random() * prop.total
    if draw < prop.add_node:
        return SelectedEvent(
            "add_node", _choose(assembly.Linker_free_carboxylates, rng), None, prop
        )
    draw -= prop.add_node
    if draw < prop.add_linker:
        return SelectedEvent(
            "add_linker", _choose(assembly.MC_free_carboxylates, rng), None, prop
        )
    draw -= prop.add_linker
    if draw < prop.close_internal:
        return SelectedEvent(
            "close_internal",
            None,
            _choose(assembly.ready_to_connect_carboxylate_pairs, rng),
            prop,
        )
    return SelectedEvent(
        "remove_bond", None, choose_weighted(bonds, bond_weights, rng), prop
    )


def gillespie_waiting_time(total_propensity: float, rng: random.Random) -> float:
    if not math.isfinite(total_propensity) or total_propensity <= 0.0:
        raise ValueError("total_propensity must be finite and positive")
    uniform = max(rng.random(), float.fromhex("0x0.0000000000001p-1022"))
    return -math.log(uniform) / total_propensity


def ideal_node_fraction(atomic_metal_to_linker_ratio: float) -> float:
    """Ideal Zr6/Hf6 entity-activity fraction for equal capture coefficients."""

    q = float(atomic_metal_to_linker_ratio)
    if not math.isfinite(q) or q < 0.0:
        raise ValueError("atomic_metal_to_linker_ratio must be finite and non-negative")
    return q / (q + 6.0)


def chemistry_record(coordinates: PhysicalCoordinates) -> Mapping[str, float]:
    """Serializable record containing physical and derived normalized values."""

    coordinates.validate()
    return {
        **asdict(coordinates),
        "node_activity": coordinates.node_activity,
        "linker_activity": coordinates.linker_activity,
        "closure_weight": coordinates.closure_weight,
        "normalized_off_ratio": coordinates.normalized_off_ratio,
        "normalized_closure_ratio": coordinates.normalized_closure_ratio,
    }
