"""Launch once, only after molecular restart and analyzer tests pass."""
import json,os,subprocess,sys
from pathlib import Path
from campaign_io import load_campaign,sha
base,m=load_campaign(Path(__file__).parent/'manifest.json')
assert not (base/'launch.json').exists()
checks=json.loads((base/'wrapper_checks/summary.json').read_text())
assert checks['passed'] and checks['manifest_sha256']==sha(base/'manifest.json')
analysis=json.loads((base/'wrapper_checks/target/filling_analysis/summary.json').read_text())
assert len(analysis['snapshots'])==1 and all(x['status']=='completed' for x in analysis['snapshots'])
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONHASHSEED='0')
with (base/'supervisor.log').open('xb') as log:
    p=subprocess.Popen([sys.executable,'-u',str(base/'supervise_campaign.py'),'--manifest',str(base/'manifest.json')],
       stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=base,env=env,start_new_session=True,close_fds=True)
print(json.dumps(dict(pid=p.pid,manifest_sha256=sha(base/'manifest.json'))))
