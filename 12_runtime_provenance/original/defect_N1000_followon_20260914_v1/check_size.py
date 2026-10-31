"""Real molecular checkpoint checks, including preserved random streams."""
import json, random
from pathlib import Path
import numpy as np
from campaign_io import load_campaign,sha,write_json
from continuation_runtime import load_environment,restore_trace
import size_wrapper as wrapper

def main():
    base,m=load_campaign(Path(__file__).parent/'manifest.json')
    out=base/'wrapper_checks';out.mkdir(exist_ok=False)
    parent,original,engine,provenance=load_environment(base,m)
    from trajectory import Simulation
    from check_molecular_integration import fingerprint
    def state(sim):
        return (fingerprint(sim.model),sim.tau,sim.event_count,sim.events_rng.getstate(),
                random.getstate(),repr(np.random.get_state()))
    t=m['traces'][0]
    sim=restore_trace(engine,provenance,t)
    initial=sim.event_count; assert initial>=50000
    target=t['source_final_N']+1
    for _ in range(2000):
        assert sim.advance()=='event'
        if len(sim.model.assembly.entities)>=target:break
    else:raise AssertionError('Bounded test did not reach nearby size')
    expected=state(sim)
    sim=restore_trace(engine,provenance,t)
    spec=dict(original,target_N=target,wall_seconds=300,milestones_steps=[],milestones_N=[])
    r=wrapper.run(sim,out/'target',provenance,spec,t)
    assert r['status']=='size_target_reached' and r['events']>50000
    assert state(sim)==expected
    again=Simulation.restore(out/'target/snapshots/terminal/checkpoint.pkl',engine,provenance)
    assert state(again)==expected
    for _ in range(10):again.advance()
    expected2=state(again)
    sim=restore_trace(engine,provenance,t)
    for _ in range(r['events']-initial+10):sim.advance()
    assert state(sim)==expected2
    sim=restore_trace(engine,provenance,t);before=state(sim)
    skipped=wrapper.run(sim,out/'already_large',provenance,dict(spec,target_N=t['source_final_N']),t)
    assert skipped['target_reached'] and state(sim)==before
    sim=restore_trace(engine,provenance,t)
    capped=wrapper.run(sim,out/'wall_cap',provenance,dict(spec,wall_seconds=0),t)
    assert capped['status']=='wall_time_cap' and capped['censored'] and not capped['target_reached']
    assert sha(t['source_checkpoint'])==t['source_checkpoint_sha256']
    assert sha(t['source_result'])==t['source_result_sha256']
    write_json(out/'summary.json',dict(passed=True,manifest_sha256=sha(base/'manifest.json'),
        source=t['id'],initial_N=t['source_final_N'],target_N=target,events_added=r['events']-initial,
        tests=['real_checkpoint_restore','size_first_passage_matches_direct_events_and_all_RNG',
               'no_50000_stop','post_checkpoint_restart_exact','already_large_zero_events',
               'wall_cap_censoring','source_hashes_unchanged']))
    print((out/'summary.json').read_text())

if __name__=='__main__':main()
