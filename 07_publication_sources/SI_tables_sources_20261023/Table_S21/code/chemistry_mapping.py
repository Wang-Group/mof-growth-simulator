"""Map solution composition to the reduced molecular-KMC event weights.

All ligand-exchange constants are anchored to a formate-capped Zr/Hf site.
The acid and carboxylate exchange channels are constrained by one
thermodynamic cycle. A single bath-level ratio distinguishes displacement of
solvent by formic acid from displacement by formate. Consequently, each
competitor is specified by its total concentration, pKa, exchange constant
relative to formate, and accessibility; competitor-specific acid-exchange
coefficients are not fitted independently.

The pH-like argument is an operational proton-activity coordinate on the same
scale as the supplied pKa values. Aqueous pKa values are useful for sensitivity
analysis, but are not calibrated constants for hot DMF/DEF mixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class AnchoredCompetitor:
    """One carboxylic species competing for an opened metal site.

    ``carboxylate_exchange_constant`` refers to

        Zr-F + A-  <=>  Zr-A + F-

    where F is formate. Concentrations and activities may be dimensionless
    relative to any declared common standard state, but the same convention
    must be used for linker, modulator, and solvent weights.
    """

    name: str
    total_concentration: float
    pka: float
    carboxylate_exchange_constant: float
    accessibility: float = 1.0

    def validate(self) -> None:
        for field in (
            "total_concentration",
            "pka",
            "carboxylate_exchange_constant",
            "accessibility",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value):
                raise ValueError(f"{self.name}.{field} must be finite")
        if self.total_concentration < 0.0:
            raise ValueError("total concentration must be non-negative")
        if self.carboxylate_exchange_constant <= 0.0:
            raise ValueError("carboxylate exchange constant must be positive")
        if self.accessibility < 0.0:
            raise ValueError("accessibility must be non-negative")


@dataclass(frozen=True)
class CompetitiveBath:
    linker: AnchoredCompetitor
    modulator: AnchoredCompetitor
    solvent_weight: float
    operational_ph: float
    formic_acid_pka: float = 3.75
    acid_to_carboxylate_solvent_exchange_ratio: float = 1.0

    def validate(self) -> None:
        self.linker.validate()
        self.modulator.validate()
        for name in (
            "solvent_weight",
            "operational_ph",
            "formic_acid_pka",
            "acid_to_carboxylate_solvent_exchange_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.solvent_weight < 0.0:
            raise ValueError("solvent weight must be non-negative")
        if self.acid_to_carboxylate_solvent_exchange_ratio < 0.0:
            raise ValueError(
                "acid_to_carboxylate_solvent_exchange_ratio must be non-negative"
            )


@dataclass(frozen=True)
class AnchoredContribution:
    acid_fraction: float
    carboxylate_fraction: float
    acid_exchange_constant: float
    carboxylate_exchange_constant: float
    acid_weight: float
    carboxylate_weight: float
    compact_weight: float
    shared_acid_factor: float

    @property
    def total_weight(self) -> float:
        return self.acid_weight + self.carboxylate_weight


@dataclass(frozen=True)
class PreassociationState:
    """Mass balance for M + L <=> ML in one declared concentration unit.

    ``association_constant`` has reciprocal units to the supplied
    concentrations.  For concentrations in mM, use K in mM^-1.  This state is
    deliberately separate from the formate-anchored site-exchange constants:
    the latter do not determine the solution M6-linker association constant.
    """

    total_node: float
    total_linker: float
    association_constant: float
    free_node: float
    free_linker: float
    paired_node_linker: float


def solve_one_to_one_preassociation(
    total_node: float,
    total_linker: float,
    association_constant: float,
) -> PreassociationState:
    """Solve the exact finite mass balance for M + L <=> ML.

    The positive association constant is defined by

        K = [ML] / ([M][L]).

    A zero constant is accepted as the no-preassociation limit.  The stable
    quadratic root is used so that the result remains bounded when K is large.
    """

    values = {
        "total_node": float(total_node),
        "total_linker": float(total_linker),
        "association_constant": float(association_constant),
    }
    for name, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    node = values["total_node"]
    linker = values["total_linker"]
    constant = values["association_constant"]
    if node == 0.0 or linker == 0.0 or constant == 0.0:
        paired = 0.0
    else:
        b = node + linker + 1.0 / constant
        discriminant = max(b * b - 4.0 * node * linker, 0.0)
        # The algebraically equivalent form avoids subtractive cancellation.
        paired = 2.0 * node * linker / (b + math.sqrt(discriminant))
    paired = min(max(paired, 0.0), node, linker)
    return PreassociationState(
        total_node=node,
        total_linker=linker,
        association_constant=constant,
        free_node=node - paired,
        free_linker=linker - paired,
        paired_node_linker=paired,
    )


def protonation_fractions(operational_ph: float, pka: float) -> tuple[float, float]:
    """Return (acid fraction, carboxylate fraction) stably."""

    delta = float(operational_ph) - float(pka)
    if not math.isfinite(delta):
        raise ValueError("pH and pKa must be finite")
    if delta >= 0.0:
        inverse = 10.0 ** (-min(delta, 300.0))
        acid = inverse / (1.0 + inverse)
        carboxylate = 1.0 / (1.0 + inverse)
    else:
        ratio = 10.0 ** max(delta, -300.0)
        acid = 1.0 / (1.0 + ratio)
        carboxylate = ratio / (1.0 + ratio)
    return acid, carboxylate


def acid_exchange_constant_from_carboxylate_exchange(
    carboxylate_exchange_constant: float,
    *,
    incoming_pka: float,
    leaving_acid_pka: float,
) -> float:
    """Return the acid-channel constant required by the formate cycle.

    For

        Zr-F + A-  <=> Zr-A + F-       (K_A-)
        Zr-F + HA  <=> Zr-A + HF       (K_HA)

    the cycle gives

        K_HA = K_A- * Ka(HA) / Ka(HF)
             = K_A- * 10**(pKa(HF) - pKa(HA)).
    """

    value = float(carboxylate_exchange_constant)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("carboxylate exchange constant must be positive and finite")
    exponent = float(leaving_acid_pka) - float(incoming_pka)
    if not math.isfinite(exponent):
        raise ValueError("pKa values must be finite")
    return value * 10.0**exponent


def anchored_contribution(
    species: AnchoredCompetitor,
    *,
    operational_ph: float,
    formic_acid_pka: float,
    acid_to_carboxylate_solvent_exchange_ratio: float = 1.0,
) -> AnchoredContribution:
    """Evaluate the acid and carboxylate channels from total concentration.

    Algebraically,

        W_i = g_i C_i [K_i- x_i- + chi_S K_Hi x_Hi]
            = g_i C_i K_i- x_i- [1 + chi_S 10**(pKa_HF - pH)].

    Here chi_S is the formic-acid/formate solvent-displacement ratio. The
    bracketed factor is common to every carboxylic competitor referenced to
    formate. It cancels from linker/modulator ratios but not from their
    competition with the solvent weight. Setting chi_S = 1 recovers the
    earlier equal-anchor special case; it is not assumed in sensitivity runs.
    """

    species.validate()
    acid_fraction, carboxylate_fraction = protonation_fractions(
        operational_ph, species.pka
    )
    acid_constant = acid_exchange_constant_from_carboxylate_exchange(
        species.carboxylate_exchange_constant,
        incoming_pka=species.pka,
        leaving_acid_pka=formic_acid_pka,
    )
    acid_weight = (
        species.accessibility
        * species.total_concentration
        * acid_fraction
        * acid_constant
        * acid_to_carboxylate_solvent_exchange_ratio
    )
    carboxylate_weight = (
        species.accessibility
        * species.total_concentration
        * carboxylate_fraction
        * species.carboxylate_exchange_constant
    )
    shared_acid_factor = 1.0 + acid_to_carboxylate_solvent_exchange_ratio * 10.0 ** max(
        min(float(formic_acid_pka) - float(operational_ph), 300.0), -300.0
    )
    compact_weight = carboxylate_weight * shared_acid_factor
    return AnchoredContribution(
        acid_fraction=acid_fraction,
        carboxylate_fraction=carboxylate_fraction,
        acid_exchange_constant=acid_constant,
        carboxylate_exchange_constant=species.carboxylate_exchange_constant,
        acid_weight=acid_weight,
        carboxylate_weight=carboxylate_weight,
        compact_weight=compact_weight,
        shared_acid_factor=shared_acid_factor,
    )


def bath_weights(bath: CompetitiveBath) -> Mapping[str, float]:
    bath.validate()
    linker = anchored_contribution(
        bath.linker,
        operational_ph=bath.operational_ph,
        formic_acid_pka=bath.formic_acid_pka,
        acid_to_carboxylate_solvent_exchange_ratio=(
            bath.acid_to_carboxylate_solvent_exchange_ratio
        ),
    )
    modulator = anchored_contribution(
        bath.modulator,
        operational_ph=bath.operational_ph,
        formic_acid_pka=bath.formic_acid_pka,
        acid_to_carboxylate_solvent_exchange_ratio=(
            bath.acid_to_carboxylate_solvent_exchange_ratio
        ),
    )
    linker_weight = linker.total_weight
    modulator_weight = modulator.total_weight
    denominator = linker_weight + modulator_weight + bath.solvent_weight
    if denominator <= 0.0:
        raise ValueError("competitive bath has zero total weight")
    return {
        "linker_acid_fraction": linker.acid_fraction,
        "linker_carboxylate_fraction": linker.carboxylate_fraction,
        "modulator_acid_fraction": modulator.acid_fraction,
        "modulator_carboxylate_fraction": modulator.carboxylate_fraction,
        "linker_acid_exchange_constant": linker.acid_exchange_constant,
        "modulator_acid_exchange_constant": modulator.acid_exchange_constant,
        "linker_acid_weight": linker.acid_weight,
        "linker_carboxylate_weight": linker.carboxylate_weight,
        "modulator_acid_weight": modulator.acid_weight,
        "modulator_carboxylate_weight": modulator.carboxylate_weight,
        "linker_weight": linker_weight,
        "modulator_weight": modulator_weight,
        "linker_compact_weight": linker.compact_weight,
        "modulator_compact_weight": modulator.compact_weight,
        "shared_acid_factor": linker.shared_acid_factor,
        "acid_to_carboxylate_solvent_exchange_ratio": (
            bath.acid_to_carboxylate_solvent_exchange_ratio
        ),
        "solvent_weight": bath.solvent_weight,
        "total_weight": denominator,
        "linker_capture_fraction": linker_weight / denominator,
        "modulator_capture_fraction": modulator_weight / denominator,
        "solvent_capture_fraction": bath.solvent_weight / denominator,
    }


def external_and_removal_coordinates(
    bath: CompetitiveBath,
    *,
    reactive_node_activity: float,
    node_capture_coefficient: float,
    tethered_linker_weight: float,
    opened_site_rate_ratio: float = 1.0,
    bond_opening_rate_ratio: float = 1.0,
    solvent_displacement_efficiency: float = 1.0,
    free_linker_exchange_fate: str = "topology_loss",
) -> Mapping[str, float]:
    """Construct reduced KMC weights from one complete competitive bath.

    Exposed sites on the tracked assembly are counted by the molecular
    selector. This mapping supplies per-site weights. For node addition, a
    reactive node must capture the tethered growth-front carboxylate before
    free linker, modulator, or solvent occupies the opened node site. For bond
    removal, the original tethered linker can rebind, free linker can replace
    it, or modulator/solvent can cap the opened site.  The graph model cannot
    represent substitution and fragment loss in one elementary event, so two
    declared limits are supported.  ``topology_loss`` counts capture by a
    different free linker as loss of the original graph edge;
    ``protective_replacement`` treats immediate free-linker replacement as
    preserving an occupied framework site.  Both include the local tethered
    linker in the fate denominator.
    """

    weights = bath_weights(bath)
    scalars = {
        "reactive_node_activity": reactive_node_activity,
        "node_capture_coefficient": node_capture_coefficient,
        "tethered_linker_weight": tethered_linker_weight,
        "opened_site_rate_ratio": opened_site_rate_ratio,
        "bond_opening_rate_ratio": bond_opening_rate_ratio,
        "solvent_displacement_efficiency": solvent_displacement_efficiency,
    }
    for name, value in scalars.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if free_linker_exchange_fate not in {"topology_loss", "protective_replacement"}:
        raise ValueError(
            "free_linker_exchange_fate must be 'topology_loss' or "
            "'protective_replacement'"
        )

    site_competition_denominator = (
        tethered_linker_weight
        + weights["linker_weight"]
        + weights["modulator_weight"]
        + weights["solvent_weight"]
    )
    tethered_capture_fraction = (
        tethered_linker_weight / site_competition_denominator
        if site_competition_denominator > 0.0
        else 0.0
    )
    node_activity = (
        opened_site_rate_ratio
        * node_capture_coefficient
        * reactive_node_activity
        * tethered_capture_fraction
    )
    linker_activity = opened_site_rate_ratio * weights["linker_capture_fraction"]

    opened_bond_fate_denominator = (
        tethered_linker_weight
        + weights["linker_weight"]
        + weights["modulator_weight"]
        + weights["solvent_weight"]
    )
    capping_weight = (
        weights["modulator_weight"]
        + solvent_displacement_efficiency * weights["solvent_weight"]
    )
    free_linker_replacement_weight = weights["linker_weight"]
    topology_loss_weight = capping_weight + (
        free_linker_replacement_weight
        if free_linker_exchange_fate == "topology_loss"
        else 0.0
    )
    removal_fraction = (
        topology_loss_weight / opened_bond_fate_denominator
        if opened_bond_fate_denominator > 0.0
        else 0.0
    )
    off_weight = bond_opening_rate_ratio * removal_fraction

    external_activity = node_activity + linker_activity
    node_fraction = node_activity / external_activity if external_activity > 0.0 else math.nan
    normalized_off_ratio = (
        off_weight / external_activity if external_activity > 0.0 else math.inf
    )
    return {
        **weights,
        "site_competition_denominator": site_competition_denominator,
        "opened_bond_fate_denominator": opened_bond_fate_denominator,
        "tethered_capture_fraction": tethered_capture_fraction,
        "node_activity": node_activity,
        "linker_activity": linker_activity,
        "external_activity": external_activity,
        "node_fraction": node_fraction,
        "capping_weight": capping_weight,
        "free_linker_replacement_weight": free_linker_replacement_weight,
        "free_linker_exchange_fate": free_linker_exchange_fate,
        "topology_loss_weight": topology_loss_weight,
        "tethered_rebinding_fraction": (
            tethered_linker_weight / opened_bond_fate_denominator
            if opened_bond_fate_denominator > 0.0
            else 0.0
        ),
        "free_linker_replacement_fraction": (
            free_linker_replacement_weight / opened_bond_fate_denominator
            if opened_bond_fate_denominator > 0.0
            else 0.0
        ),
        "capping_fraction": (
            capping_weight / opened_bond_fate_denominator
            if opened_bond_fate_denominator > 0.0
            else 0.0
        ),
        "removal_fraction_per_bond_opening": removal_fraction,
        "off_weight": off_weight,
        "normalized_off_ratio": normalized_off_ratio,
    }


# Backward-compatible import name for audit scripts. New work should use the
# chemically explicit class name above.
AcidCapture = AnchoredCompetitor
