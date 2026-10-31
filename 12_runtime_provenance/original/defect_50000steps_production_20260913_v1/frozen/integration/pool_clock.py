"""Fixed-volume, first-order precursor clock; no extra encounter multiplier.

The initial sum of incoming entity coefficients defines the fixed time unit.
Each coefficient subsequently scales ONCE with its own free species inventory.
Acetate/solvent reservoirs and the intrinsic closure coefficient remain fixed.
This is a buffered rate-coordinate screen, not a fitted speciation calculation.
"""
from collections import Counter
from dataclasses import dataclass, field
import math

SPECIES = ('Zr', 'Ligand')


@dataclass
class PoolClock:
    initial_free: dict
    initial_tracked: dict
    mode: str
    fragment_policy: str
    rho0: float
    s_close: float = 1.0
    eta_solvent: float = .5
    free: dict = field(init=False)
    tracked: dict = field(init=False)
    detached: dict = field(init=False)
    supplied: dict = field(init=False)
    returned: dict = field(init=False)

    def __post_init__(self):
        if self.mode not in {'fixed', 'finite'}:
            raise ValueError('Declare fixed or finite bath')
        if self.fragment_policy not in {'retain_oligomers', 'redissolve'}:
            raise ValueError('Declare a fragment boundary policy')
        for counts in (self.initial_free, self.initial_tracked):
            if set(counts) != set(SPECIES) or any(type(v) is not int or v < 0 for v in counts.values()):
                raise ValueError('Nonnegative integer counts for both species required')
        if any(v <= 0 for v in self.initial_free.values()):
            raise ValueError('Both initial free inventories must be positive')
        if not (math.isfinite(self.rho0) and self.rho0 > 0 and math.isfinite(self.s_close)
                and 0 <= self.s_close <= 2 and math.isfinite(self.eta_solvent)
                and 0 <= self.eta_solvent <= 1):
            raise ValueError('Invalid rate coordinates')
        self.initial_free = dict(self.initial_free)
        self.initial_tracked = dict(self.initial_tracked)
        self.free, self.tracked = dict(self.initial_free), dict(self.initial_tracked)
        self.detached, self.supplied, self.returned = ({s: 0 for s in SPECIES} for _ in range(3))

    def incorporate(self, species):
        if self.free[species] <= 0:
            raise ValueError('Cannot incorporate exhausted species')
        if self.mode == 'finite':
            self.free[species] -= 1
        else:
            self.supplied[species] += 1
        self.tracked[species] += 1
        self.check()

    def release(self, species_list):
        counts = Counter(species_list)
        if any(s not in SPECIES or n > self.tracked[s] for s, n in counts.items()):
            raise ValueError('Invalid fragment release')
        recycle = len(species_list) == 1 or self.fragment_policy == 'redissolve'
        for s, n in counts.items():
            self.tracked[s] -= n
            if not recycle:
                self.detached[s] += n
            elif self.mode == 'finite':
                self.free[s] += n
            else:
                self.returned[s] += n
        self.check()

    def check(self, actual_tracked=None):
        for s in SPECIES:
            assert all(type(d[s]) is int and d[s] >= 0
                       for d in (self.free, self.tracked, self.detached, self.supplied, self.returned))
            lhs = self.free[s] + self.tracked[s] + self.detached[s]
            rhs = self.initial_free[s] + self.initial_tracked[s] + self.supplied[s] - self.returned[s]
            assert lhs == rhs, (s, lhs, rhs)
            if actual_tracked is not None:
                assert self.tracked[s] == actual_tracked.get(s, 0)

    def coefficients(self):
        initial_total = sum(self.initial_free.values())
        node, linker = self.free['Zr'] / initial_total, self.free['Ligand'] / initial_total
        total = node + linker
        linker_relative = self.free['Ligand'] / self.initial_free['Ligand']
        return {
            'add_node': node, 'add_linker': linker, 'remove_bond': self.rho0,
            'close_internal': math.exp(30.9 * self.s_close),
            'solvent_to_modulator': self.rho0 * (1 - self.eta_solvent),
            'modulator_to_solvent': self.rho0 * self.eta_solvent,
            'p_Zr': node / total if total else None,
            'rho_off_total_external': self.rho0 / total if total else None,
            'rho_off_linker_reference': self.rho0 / linker_relative if linker_relative else None,
            'off_per_linker_addition': self.rho0 / linker if linker else None,
        }

    def record(self):
        return {**{name: dict(getattr(self, name)) for name in
                   ('free', 'tracked', 'detached', 'supplied', 'returned')},
                'mode': self.mode, 'fragment_policy': self.fragment_policy,
                'coefficients': self.coefficients(),
                'clock': 'tau=k_external_total_initial*t; unchanged during the trace'}
