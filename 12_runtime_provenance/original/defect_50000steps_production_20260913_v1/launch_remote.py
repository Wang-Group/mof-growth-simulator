"""Start the verified finite queue detached from SSH; never duplicate a launch."""
import json
import os
from pathlib import Path
import subprocess
import sys
from campaign_io import load_campaign, sha

base, manifest = load_campaign(Path(__file__).parent/'manifest.json')
assert not (base/'launch.json').exists()
checks = json.loads((base/'wrapper_checks/summary.json').read_text())
assert checks['passed'] and checks['manifest_sha256'] == sha(base/'manifest.json')
measurements = json.loads((base/'wrapper_checks/target/filling_analysis/summary.json').read_text())
assert len(measurements['snapshots']) == 6
assert all(r['status'] == 'completed' for r in measurements['snapshots'])
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONHASHSEED='0')
with (base/'supervisor.log').open('xb') as log:
    proc = subprocess.Popen([sys.executable, '-u', str(base/'supervise_campaign.py'),
               '--manifest', str(base/'manifest.json')], stdout=log, stderr=subprocess.STDOUT,
               stdin=subprocess.DEVNULL, cwd=base, env=env, close_fds=True, start_new_session=True)
print(json.dumps({'supervisor_pid': proc.pid, 'manifest_sha256': sha(base/'manifest.json'),
                  'launcher_sha256': sha(__file__)}))
