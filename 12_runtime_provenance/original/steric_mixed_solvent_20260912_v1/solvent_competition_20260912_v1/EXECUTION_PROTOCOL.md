# Frozen mixed-solvent test and grid protocol

Author authorization: "please go ahead with the test and grid run".
All trajectories run on 5090. Local work is restricted to implementation,
short algebra/structural tests, manifests, and returned-data analysis.

## Meaning of the six event classes

The existing external-node addition, external-linker addition, internal closure,
and uniform bond-removal coefficients are unchanged. Add solvent-to-modulator
and modulator-to-solvent exchange at genuinely free metal sites. With external
activity normalized to one, their per-site rates are respectively
rho_off*(1-eta_S) and rho_off*eta_S. Their changing opportunity counts enter
the same Gillespie total as the four framework event classes. Rejected geometric
proposals advance time; there is no second phi_L or phi_N multiplier.

The local cap process has product prior weight eta_S for solvent and
(1-eta_S)/n for each of n explicitly tested modulator conformer/orientation
alternatives. A proposed M insertion samples all n alternatives before the
contact test, including blocked alternatives. More library entries do not
increase the total modulator chemical contribution. A blocked M state has
zero allowed-state probability under the existing strict 2.0-A hard contact
criterion. Allowed local S/M transitions obey the corresponding balance relation.
No force-field energy or fitted steric factor enters a rate.

## Mixed initialization and retermination

For fixed framework and neighboring cap geometry, a complete termination
configuration is sampled from the product prior conditioned on nonoverlap.
The sampler first removes candidates that overlap fixed atoms, identifies
connected components of the remaining cap-contact graph, and independently
rejection-samples each whole component with its original relative priors.
This is an exact conditional sampler on the finite candidate library, not a
greedy packing or finite-equilibration MCMC result. It stops with an explicitly
invalid record after 100,000 proposals per component if necessary.

Initialization applies this distribution to all free sites of the N1 or N76
starting framework. Following fragment loss, it applies the same conditional
competition to newly vacated sites while existing terminations retain their
instantaneous coordinates and identities. Framework bond removal is executed
first and never vetoed. S is a candidate from the outset, not a fallback after
a selected modulator cannot be placed. If no defined termination state can be
sampled, the resulting invalid record is preserved.

This conditional retermination is the coarse-grained immediate capture boundary
after the existing fragment-deletion event. It does not add a separately timed
bare-vacancy state. Subsequent S/M exchange is explicitly timed. Adjacent
opened/ready pairs remain cap-free, and their local closure/opening ratio is
unchanged. Global equilibrium of a closed fragment population is not claimed:
the previously approved smaller-fragment deletion remains an open-reservoir
approximation.

## Incoming nodes

An incoming Zr6 node receives a full twelve-site mixed termination draw against
its own inorganic core before testing contact with the growing framework.
Caps displaced by its primary connection and any simultaneous ready pairs are
then removed. Remaining caps participate in the contact test, including contacts
with the receiving framework. The incoming caps are not repeatedly resampled
until an otherwise blocked addition fits. Other all-twelve-site geometry rules
and the 0.5-A match criterion are retained.

This replaces the older pilot's deterministic incoming-cap packing against the
receiving assembly. Comparisons should use the present code consistently across
all identities and solvent preferences, not pool the old pilot with this grid.

## Design and observation windows

- 12 modulator identities; rho_off=0.1,1,3,10; atomic Zr/BDC=0.3,1,3.
- eta_S=0.1,0.5,0.9: uncrowded conditional solvent preferences, not measured
  coverages or imposed integer solvent counts on each node.
- s_close=1, identical assigned chemical constants across identities.
- 432 parameter conditions, two assays, 32 replicates: 27,648 grid records.
- Nucleation: N1->30; model-time observation window 100; 1,000,000-event and
  120-s per-record wall limits.
- Seeded growth: identical reviewed N76->126; model-time window 20;
  5,000,000-event and 600-s per-record wall limits.
- N counts Zr6 nodes plus BDC, not solvent/modulator terminations.
- Pilot: rho_off=1,q=1; all 12 identities and three eta_S values, two assays,
  four replicates, 288 records, 16 workers.
- Grid: maximum 96 workers of 192 host CPUs; numerical-library threads=1.
- Pilot and grid have overlapping random-seed labels; never pool them as
  independent replicates. Grid-only statistics are the production results.

Grid launch requires all 288 pilot records, zero implementation/geometry
failures and byte-identical runtime code. A scheduling guard stops new
submissions after ten invalid/error records while allowing already submitted
records to finish. Designed but unstarted records remain explicitly counted.
No automatic cap escalation or rerun is permitted.

## Analysis and artifact boundaries

Record first-passage times, all termination counts, accepted/rejected event
counts, small entity-number curves and actual total removal/addition ratios.
The fixed rho_off input is not the changing total event ratio. Treat observation
window stops and computational caps as different censoring reasons; invalid
geometry is not physical censoring or zero nucleation/growth. Report physical
event probabilities only with denominators and censoring limits. Successful-only
50/first-passage-time estimates must be labelled conditional.

Replicate zero retains initial/final geometry and separate framework-only XYZ.
Every invalid dynamic record additionally retains a diagnostic geometry. Full
coordinate outputs containing C,O,O solvent proxies are labelled GEOMETRY_ONLY;
the proxy carbon is not a physical carbon scattering atom. Framework-only
scattering excludes ALL terminations, not only solvent, for consistent comparison.

Source, input, manifest and completed-artifact SHA-256 records are retained.
The previous 96-record all-modulator pilot and its exact code remain untouched.
No manuscript is revised on the basis of this launch.
