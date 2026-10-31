"""Freeze a passed validation release and create a checksummed small-artifact bundle."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from runtime import configure, make_model, sha


def main():
    ap = argparse.ArgumentParser()
    for name in ('root', 'bundle', 'preflight', 'acetate-source'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    reports = [json.loads((here/'checks_04/integration_checks.json').read_text()),
               json.loads((here/'timed_checks_02/trajectory_checks.json').read_text())]
    for report in reports:
        assert report['status'] == 'passed'
        hashes = report.get('source_hashes', report['provenance'].get('sources'))
        for name, expected in hashes.items():
            assert sha(here/name) == expected, ('source_changed_after_test', name)
    unit = subprocess.run([sys.executable, '-m', 'unittest', 'test_primitives', '-v'],
                          cwd=here, text=True, capture_output=True)
    (here/'unit_tests_5090.log').write_text(unit.stdout+unit.stderr)
    assert unit.returncode == 0
    engine, provenance = configure(args.root, args.bundle, args.preflight)
    model = make_model(engine, args.root, args.acetate_source)
    copied = here/'validated_release'
    copied.mkdir(exist_ok=False)
    source_dir = copied/'integration_source'
    source_dir.mkdir()
    for p in sorted(here.glob('*.py'))+sorted(here.glob('*.md')):
        shutil.copy2(p, source_dir/p.name)
    dependencies = {}
    prefixes = (str(args.bundle.resolve()), str(args.root.resolve()), str(here))
    for module in list(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p and p.endswith('.py') and str(Path(p).resolve()).startswith(prefixes):
            p = Path(p).resolve()
            dependencies[str(p)] = {'sha256': sha(p), 'kind': 'python_source'}
    for item in provenance['monomer_inputs'].values():
        dependencies[item['path']] = {'sha256': item['sha256'], 'kind': 'molecular_input'}
    for p in (args.acetate_source, args.root/'framework_inputs/UiO66_N76_reviewed.pkl'):
        dependencies[str(p)] = {'sha256': sha(p), 'kind': 'molecular_input'}
    dep_dir = copied/'dependencies'
    dep_dir.mkdir()
    for original, item in dependencies.items():
        p = Path(original)
        destination = dep_dir/(item['sha256'][:12]+'_'+p.name)
        if not destination.exists():
            shutil.copy2(p, destination)
        assert sha(destination) == item['sha256']
        item['frozen_copy'] = str(destination.relative_to(here))
    manifest = {'sources': dependencies, 'runtime': {'python': sys.version},
                'validated_reports': ['checks_04/integration_checks.json', 'timed_checks_02/trajectory_checks.json'],
                'model': {'acetate': model.library.provenance, 'seed_N': len(model.assembly.entities),
                          'all_node_sites': 12, 'BDC_aromatic_H': 4},
                'unit_tests': {'count': 13, 'exit_code': unit.returncode},
                'molecular_test_groups': len(reports[0]['tests']),
                'timed_test_groups': len(reports[1]['tests']),
                'production_launched': False,
                'interpretation_gate': 'interior-linker coverage inadequate in the bounded size checks'}
    (here/'VALIDATED_RELEASE.json').write_text(json.dumps(manifest, indent=2))
    # Retain every historical failed check and source version. Exclude large
    # continuation pickles except the reviewed small input seed in dependencies.
    extensions = {'.py', '.json', '.md', '.xyz', '.mol', '.log'}
    files = sorted(p for p in here.rglob('*') if p.is_file() and
                   (p.suffix in extensions or (p.parent == dep_dir and p.suffix == '.pkl')))
    sums = '\n'.join(sha(p)+'  '+p.relative_to(here).as_posix() for p in files)+'\n'
    (here/'ARTIFACT_SHA256SUMS.txt').write_text(sums)
    with tarfile.open(here/'verified_small_artifacts.tgz', 'x:gz') as archive:
        for p in files+[here/'ARTIFACT_SHA256SUMS.txt']:
            archive.add(p, arcname=p.relative_to(here).as_posix(), recursive=False)
    print(json.dumps({'files': len(files), 'bundle_sha256': sha(here/'verified_small_artifacts.tgz'),
                      'bundle_bytes': (here/'verified_small_artifacts.tgz').stat().st_size,
                      'tests': [13, len(reports[0]['tests']), len(reports[1]['tests'])]}))


if __name__ == '__main__':
    main()
