"""Load frozen reviewed molecular dependencies without editing them."""
import hashlib
import os
from pathlib import Path
import pickle
import random
import socket
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(root, bundle, preflight):
    if socket.gethostname() != 'xmhy-G3261':
        raise RuntimeError('Molecular integration tests and growth run on 5090 only')
    root, bundle, preflight = Path(root), Path(bundle), Path(preflight)
    os.environ['MOL_GROWTH_REVIEW_BUNDLE'] = str(bundle)
    sys.path[:0] = [str(bundle/'code/proposed'), str(root),
                   str(root/'solvent_competition_20260912_v1'), str(preflight)]
    import common_molecular_runner as runner
    engine, path, redirects = runner._load_engine('uio66')
    if sha(path) != '4cda9182267ef4fc29a5537a60c23d0fd592b27c92221c7dc1b7731b291e24e9':
        raise ValueError('Reviewed engine hash changed')
    engine.MAX_DEPTH = 10001
    inputs = {name: {'path': str(p), 'sha256': sha(p)}
              for name, p in runner.ENGINE_SPECS['uio66']['monomer_files'].items()}
    return engine, {'engine': sha(path), 'engine_path': str(path), 'monomer_inputs': inputs,
                    'loader_path': str(Path(runner.__file__)), 'loader_sha256': sha(runner.__file__),
                    'redirects': redirects}


def make_model(engine, root, acetate_source, *, seed=20260912, delta=0., mode='finite',
               fragment_policy='retain_oligomers', rho=1., s_close=1., free=(1000, 6000)):
    from acetate_cap_library import AcetateCapLibrary
    from defect_model import DefectAssembly
    from solvent_channel import SiteCompetition
    from pool_clock import PoolClock
    source = Path(root)/'framework_inputs/UiO66_N76_reviewed.pkl'
    if sha(source) != '9b08834d55e0e1674d6e379eb991594947d000adeea6831d430e1f69b98e48c7':
        raise ValueError('Reviewed seed hash changed')
    random.seed(seed)
    np.random.seed(seed % (2**32-1))
    with source.open('rb') as handle:
        assembly = pickle.load(handle)
    pool = PoolClock({'Zr': free[0], 'Ligand': free[1]}, {'Zr': 16, 'Ligand': 60},
                     mode, fragment_policy, rho, s_close)
    library = AcetateCapLibrary(acetate_source)
    if library.provenance['source_sha256'] != 'eacc7ef08be79274dbf5cc89a61ae023029df32e1e6cc26a883a1bbdec87c4c3':
        raise ValueError('Reviewed explicit-H acetate source changed')
    model = DefectAssembly(assembly, engine, library, SiteCompetition.from_rho_fraction(rho, .5),
                           np.random.default_rng(seed ^ 0xACA7E), pool, delta)
    return model
