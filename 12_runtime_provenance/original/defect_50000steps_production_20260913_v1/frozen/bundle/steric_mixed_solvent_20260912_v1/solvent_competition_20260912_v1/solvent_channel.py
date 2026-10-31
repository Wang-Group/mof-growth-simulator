"""Solvent/modulator competition building blocks, NOT an integrated KMC runner.

User-defined geometry: hydrogen-free formate (C,O,O) as a compact solvent proxy.
The proxy is not a chemical water/DMF model and has no assigned charge, energy,
pKa, or atomic scattering factors. Competition follows current SI S13-S19,
not the older formate-anchored chemistry_mapping.py.
"""
from dataclasses import dataclass
import math

import numpy as np


def conditional_contribution(concentration, protonation_ratio, acid_constant,
                             carboxylate_constant):
    """w_X=c_X*(K_HX/S + K_X-/S*r_X)/(1+r_X), r_X=[X-]/[HX].

    Concentration and reciprocal-concentration constants must use the SAME
    declared units and medium. Keep protonated acid and carboxylate terms
    separate; no pKa-only conversion of bound-state equilibria is assumed.
    """
    values = (concentration, protonation_ratio, acid_constant, carboxylate_constant)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Inputs must be finite and non-negative')
    acid_fraction = 1.0 / (1.0 + protonation_ratio)
    anion_fraction = protonation_ratio / (1.0 + protonation_ratio)
    acid = concentration * acid_constant * acid_fraction
    carboxylate = concentration * carboxylate_constant * anion_fraction
    total = acid + carboxylate
    if not math.isfinite(total):
        raise ValueError('Contribution overflow')
    return {'acid': acid, 'carboxylate': carboxylate, 'total': total}


@dataclass(frozen=True)
class SiteCompetition:
    """Dimensionless contributions in one common normalization.

    Current SI chooses solvent=1. from_rho_fraction instead chooses linker=1;
    this harmless change of normalization also permits the solvent-free limit.
    No modulator-dependent accessibility multiplier is included here.
    """
    linker: float
    modulator: float
    solvent: float

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0
               for x in (self.linker, self.modulator, self.solvent)):
            raise ValueError('Contributions must be finite and non-negative')
        if self.linker <= 0 or self.modulator + self.solvent <= 0:
            raise ValueError('Positive productive and non-framework contributions required')
        if not math.isfinite(self.linker + self.modulator + self.solvent):
            raise ValueError('Contribution sum overflow')

    @classmethod
    def from_rho_fraction(cls, rho_off, solvent_fraction):
        """Uncrowded conditional solvent fraction, not imposed surface coverage."""
        if not math.isfinite(rho_off) or rho_off <= 0:
            raise ValueError('rho_off must be finite and positive')
        if not math.isfinite(solvent_fraction) or not 0 <= solvent_fraction <= 1:
            raise ValueError('solvent_fraction must be in [0,1]')
        return cls(1.0, rho_off * (1 - solvent_fraction), rho_off * solvent_fraction)

    @property
    def rho_off(self):
        return (self.modulator + self.solvent) / self.linker

    @property
    def solvent_fraction(self):
        return self.solvent / (self.modulator + self.solvent)

    @property
    def opening_outcomes(self):
        total = self.linker + self.modulator + self.solvent
        return {'linker': self.linker / total, 'modulator': self.modulator / total,
                'solvent': self.solvent / total}

    def cap_exchange_rates(self):
        """Per-site rates in tau=k_open*phi_L*t, COMMON opening-clock model.

        These are supplementary cap-identity events, not another multiplier on
        framework addition/removal. With no steric rejection they obey
        pi_S*r_SM=pi_M*r_MS. Explicit integration into the graph clock is pending.
        """
        return {'solvent_to_modulator': self.modulator / self.linker,
                'modulator_to_solvent': self.solvent / self.linker}

def solvent_proxy(anchor_xyz):
    """Place exactly C,O,O at the site's existing carboxylate anchor positions.

    No methyl group, formyl H or acid proton is present. Coordinates must be
    ordered C,O,O. No geometry optimization or covalent valence is implied.
    """
    xyz = np.array(anchor_xyz, dtype=float, copy=True)
    if xyz.shape != (3, 3) or not np.isfinite(xyz).all():
        raise ValueError('Expected finite C,O,O anchor coordinates of shape (3,3)')
    if np.linalg.matrix_rank(xyz[1:] - xyz[0]) != 2:
        raise ValueError('Anchor points must be non-collinear')
    return {'role': 'solvent_proxy', 'elements': ['C', 'O', 'O'], 'xyz': xyz,
            'geometry_only': True, 'chemical_formula': None, 'formal_charge': None,
            'pxrd_atomic_form_factors_allowed': False}


def on_metal_site(site):
    entity = site.belonging_entity
    if entity.entity_type != 'Zr':
        raise ValueError('Solvent proxy belongs to a metal site, not a linker terminus')
    indices = site.atom_indexes
    if [entity.elements[i] for i in indices] != ['C', 'O', 'O']:
        raise ValueError('Site anchor order must be C,O,O')
    return solvent_proxy(entity.coordinates[indices])
