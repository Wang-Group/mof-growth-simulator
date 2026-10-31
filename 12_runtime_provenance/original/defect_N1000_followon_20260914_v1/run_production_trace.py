"""Continue one capped trajectory to 50000 TOTAL attempts in a fresh output path."""
import argparse
import json
from pathlib import Path
import signal
from campaign_io import load_campaign, write_json, sha
from continuation_runtime import load_environment, restore_trace, original_wrapper


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--index',type=int,required=True)
    args=ap.parse_args()
    base,m=load_campaign(args.manifest)
    parent,original,engine,provenance=load_environment(base,m)
    trace=m['traces'][args.index]
    sim=restore_trace(engine,provenance,trace)
    import size_wrapper as wrapper
    signal.signal(signal.SIGTERM,wrapper.handle_signal)
    signal.signal(signal.SIGINT,wrapper.handle_signal)
    # Only the new wall budget differs; counters, chemistry, clock and RNG do not.
    spec=dict(original,wall_seconds=m['wall_seconds'],target_N=m['target_N'],
              milestones_steps=[],milestones_N=[])
    result=wrapper.run(sim,base/'traces'/trace['id'],provenance,spec,trace)
    result['continuation']={
        'manifest_sha256':sha(args.manifest),'parent_manifest_sha256':m['parent_manifest_sha256'],
        'source_checkpoint':trace['source_checkpoint'],'source_checkpoint_sha256':trace['source_checkpoint_sha256'],
        'source_result':trace['source_result'],'source_result_sha256':trace['source_result_sha256'],
        'events_before':trace['source_events'],'events_added':sim.event_count-trace['source_events'],
        'source_tau':trace['source_tau'],'wall_cap_this_extension_seconds':m['wall_seconds'],
        'original_wall_seconds':trace['source_wall_seconds'],
        'cumulative_compute_wall_seconds':trace['source_wall_seconds']+result['wall_seconds'],
        'checkpoint_provenance_note':'Molecular provenance stays the verified parent manifest; this sidecar records the continuation budget and source checkpoint.'}
    write_json(base/'traces'/trace['id']/'result.json',result)
    assert sha(trace['source_checkpoint'])==trace['source_checkpoint_sha256']
    assert sha(trace['source_result'])==trace['source_result_sha256']
    print(json.dumps({k:result[k] for k in ('status','final_N','events','tau','continuation')}),flush=True)
    if not result['valid']:raise SystemExit(2)


if __name__=='__main__':main()
