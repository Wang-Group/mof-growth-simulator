"""Only restore reviewed checkpoints; no new model is constructed."""
import json
from pathlib import Path
import sys
from campaign_io import load_campaign, setup, sha


def load_environment(base, m):
    parent_path=Path(m['parent_manifest'])
    assert sha(parent_path)==m['parent_manifest_sha256']
    parent, original=load_campaign(parent_path)
    assert original['files']==m['parent_frozen_files']
    engine, root, acetate, provenance=setup(parent, original)
    return parent, original, engine, provenance


def restore_trace(engine, provenance, trace):
    from trajectory import Simulation
    cp=Path(trace['source_checkpoint'])
    assert sha(cp)==trace['source_checkpoint_sha256']
    assert sha(trace['source_result'])==trace['source_result_sha256']
    sim=Simulation.restore(cp,engine,provenance)
    assert sim.event_count==trace['source_events']
    assert len(sim.model.assembly.entities)==trace['source_final_N']
    assert sim.tau==trace['source_tau']
    assert sim.model.delta==trace['delta']
    assert sim.model.ledger.rho0==trace['rho0'] and sim.model.ledger.mode==trace['mode']
    assert sim.model.ledger.fragment_policy==trace['fragment_policy']
    assert sim.model.ledger.s_close==1.
    return sim


def original_wrapper(parent):
    # Use a distinct module name so this continuation entrypoint cannot shadow it.
    import importlib.util
    spec=importlib.util.spec_from_file_location('reviewed_step_wrapper',parent/'run_production_trace.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
