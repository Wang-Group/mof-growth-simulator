"""Offline provenance checks. Never import archived simulators or unpickle seeds."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '12_runtime_provenance'
TABLES = ROOT / '13_table_source_audit'
REMOTE_ROOT = '/home/wave/MOL_growth_recalculation_20260830/'
PARENT = 'defect_50000steps_production_20260913_v1'
FOLLOW = 'defect_N1000_followon_20260914_v1'
PAIRED = 'defect_paired_response_20260916_v2'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def local_path(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), f'Path outside package: {relative}')
    return path


class Sources:
    def __init__(self):
        data = json.loads((RUNTIME / 'source_manifest.json').read_text(encoding='utf-8'))
        self.rows = data['files']
        self.index = {r['source_relative']: r for r in self.rows}
        require(len(self.index) == len(self.rows), 'Duplicate source path')
        self.cache = {}
        self.by_hash = defaultdict(list)
        for row in self.rows:
            self.read(row['source_relative'])
            self.by_hash[row['source_sha256']].append(row['release_path'])
        # Existing byte-identical runtime copies are legitimate hash witnesses.
        for line in (ROOT / 'CHECKSUMS.sha256').read_text(encoding='utf-8').splitlines():
            sha, relative = line.split('  ', 1)
            if relative.startswith(('07_publication_sources/', '09_recovered_runtime/')):
                path = local_path(relative)
                require(digest(path.read_bytes()) == sha, f'Changed existing source: {relative}')
                self.by_hash[sha].append(relative)

    def read(self, name):
        if name not in self.cache:
            row = self.index[name]
            packed = local_path(row['release_path']).read_bytes()
            require(digest(packed) == row['stored_sha256'], f'Stored hash: {name}')
            require(len(packed) == row['stored_bytes'], f'Stored length: {name}')
            require(row['encoding'] in ('identity', 'gzip'), f'Encoding: {name}')
            raw = gzip.decompress(packed) if row['encoding'] == 'gzip' else packed
            require(digest(raw) == row['source_sha256'], f'Source hash: {name}')
            require(len(raw) == row['source_bytes'], f'Source length: {name}')
            self.cache[name] = raw
        return self.cache[name]

    def json(self, name):
        return json.loads(self.read(name).decode('utf-8-sig'))

    def csv(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name).decode('utf-8-sig'))))


def runtime_checks(sources):
    references = []

    def input_hash(row, key):
        declared = row.get(key + '_sha256')
        if declared:
            return declared
        path = row.get(key, '')
        if path.startswith(REMOTE_ROOT):
            recovered = sources.index.get(path[len(REMOTE_ROOT):])
            if recovered:
                return recovered['source_sha256']
        return None

    def reference(origin, role, sha, path=''):
        matches = sorted(set(sources.by_hash.get(sha, [])))
        row = dict(origin=origin, role=role, source_path=path, expected_sha256=sha,
                   status='EXACT_BYTES_PACKAGED' if matches else 'EXPECTED_VERSION_NOT_PACKAGED',
                   packaged_match=matches[0] if matches else '')
        references.append(row)
        return row

    csv_counts = []
    seed_refs = set()
    for name in sorted(sources.index):
        if not name.endswith('.csv'):
            continue
        rows = sources.csv(name)
        csv_counts.append({'source_manifest': name, 'design_rows': len(rows),
                           'rng_column': 'random_seed' if rows and 'random_seed' in rows[0] else
                                         'seed' if rows and 'seed' in rows[0] else '',
                           'evidence': 'design rows, not proof of completed runs'})
        for row in rows:
            for key in ('seed_pickle', 'host_pickle'):
                sha = input_hash(row, key)
                if sha:
                    role = key if row.get(key + '_sha256') else key + '_path_matched_recovered_bytes'
                    seed_refs.add((name, role, row.get(key, ''), sha))
    for name, key, path, sha in sorted(seed_refs):
        reference(name, key, sha, path)

    seed_metadata = []
    for name in sorted(sources.index):
        if name.startswith('frozen_seed_ensemble_v2/') and name.endswith('.json') and '_seed_' in Path(name).name:
            d = sources.json(name)
            if 'provenance' not in d:
                continue
            p = d['provenance']
            for role in ('engine', 'generator', 'pickle'):
                if role + '_sha256' in p:
                    reference(name, role, p[role + '_sha256'], p.get(role + '_path', ''))
            seed_metadata.append({'source': name, 'random_seed': d['random_seed'],
                                  'entities': d['state']['entities'],
                                  'topology_mode': d.get('topology_mode', '')})
    require(len(seed_metadata) == 24, 'Expected 24 frozen seed metadata records')

    manifests = {family: sources.json(f'{family}/manifest.json') for family in (PARENT, FOLLOW, PAIRED)}
    for family, d in manifests.items():
        for path, sha in d['files'].items():
            reference(f'{family}/manifest.json', 'frozen_file', sha, f'{family}/{path}')
    btb_snapshot = 'btb_size_extension_20260916_v1/source_snapshot_sha256.json'
    for path, sha in sources.json(btb_snapshot).items():
        reference(btb_snapshot, 'snapshot_file', sha, path)

    parent, follow, paired = (manifests[f] for f in (PARENT, FOLLOW, PAIRED))
    parent_sha = digest(sources.read(f'{PARENT}/manifest.json'))
    follow_sha = digest(sources.read(f'{FOLLOW}/manifest.json'))
    require(follow['parent_manifest_sha256'] == parent_sha == paired['parent_sha256'], 'Parent manifest hash')
    require(paired['source_manifest_sha256'] == follow_sha, 'Paired source manifest hash')
    require(follow['parent_frozen_files'] == parent['files'], 'Inherited frozen versions differ')
    remote = json.loads((RUNTIME / 'remote_checkpoint_verification.json').read_text(encoding='utf-8'))
    witnesses = {r['path']: r for r in remote['files']}
    require(len(witnesses) == len(remote['files']) == 1950, 'Remote witness count')
    require(remote['read_only'] and not remote['unpickled'] and not remote['payloads_in_package'], 'Audit scope')
    for row in witnesses.values():
        require(row['exists'] and row['expected_match'] is True, f'Remote hash mismatch: {row["path"]}')
    observed_manifests = {r['path']: r['sha256'] for r in remote['followed_manifests']}
    for family in (FOLLOW, PAIRED):
        name = f'{family}/manifest.json'
        require(observed_manifests[name] == digest(sources.read(name)), f'Remote manifest witness: {name}')

    lineage = []
    used = set()
    parent_rows = {r['id']: r for r in parent['traces']}
    follow_rows = {r['id']: r for r in follow['traces']}
    for family in (FOLLOW, PAIRED):
        for row in manifests[family]['traces']:
            source = row if family == FOLLOW else row['source']
            prefix = 'source_' if family == FOLLOW else ''
            source_id = row['id'] if family == FOLLOW else source['id']
            parent_row = (parent_rows if family == FOLLOW else follow_rows)[source_id]
            require(row['replicate'] == parent_row['replicate'], f'Replicate mapping: {row["id"]}')
            if family == FOLLOW:
                for key in ('seed', 'condition', 'delta', 'rho0', 'mode', 'fragment_policy'):
                    require(row[key] == parent_row[key], f'Continuation changed {key}: {row["id"]}')
            values = {}
            for kind in ('checkpoint', 'result'):
                path = source[prefix + kind]
                require(path.startswith(REMOTE_ROOT), f'External source root: {path}')
                relative = path[len(REMOTE_ROOT):]
                witness = witnesses[relative]
                require(witness['sha256'] == source[prefix + kind + '_sha256'], f'Lineage hash: {path}')
                used.add(relative)
                values[kind + '_path'] = relative
                values[kind + '_sha256'] = witness['sha256']
            lineage.append(dict(family=family, run_id=row['id'], replicate=row['replicate'],
                                random_seed=row['seed'], parent_run_id=source_id,
                                relation='continuation_same_rng_seed' if family == FOLLOW else 'paired_branch_shared_particle',
                                code_manifest=f'{family}/manifest.json', remote_status='HASH_VERIFIED_20261031', **values))
    require(used == set(witnesses), 'Unexplained checkpoint witnesses')
    require(len(parent_rows) == 1152 and len(follow_rows) == 943 and len(paired['traces']) == 384, 'Family sizes')
    require(len({r['source']['checkpoint'] for r in paired['traces']}) == 32, 'Paired source particles')

    # Execution records are stronger evidence than a design manifest by itself.
    results = [(name, sources.json(name)) for name in sorted(sources.index)
               if name.endswith('.json') and name.startswith(('production_sc1_anchor_v1/seeded/', 'campaign/competitive_bath_seeded_v2/'))]
    folder = ROOT / '07_publication_sources/SI_tables_sources_20261023/Table_S7/matched_source_files'
    results.extend((p.relative_to(ROOT).as_posix(), json.loads(p.read_text(encoding='utf-8')))
                   for p in sorted(folder.glob('uio66*.json')))
    all_csv_rows = {}
    for name in sorted(sources.index):
        if name.endswith('.csv'):
            for row in sources.csv(name):
                if 'run_id' in row:
                    all_csv_rows.setdefault(row['run_id'], []).append((name, row))
    result_links = []
    for name, result in results:
        p = result['provenance']
        for role in ('engine', 'runner', 'selector'):
            reference(name, role, p[role + '_sha256'], p.get(role + '_path', ''))
        seed = result['seed']['source']
        reference(name, 'executed_seed', seed['sha256'], seed['path'])
        candidates = [(f, row) for f, row in all_csv_rows.get(result['run_id'], [])
                      if input_hash(row, 'seed_pickle') == seed['sha256']
                      and int(row['random_seed']) == result['random_seed']]
        require(candidates, f'No RNG/seed manifest match: {name}')
        for _, row in candidates:
            for column, field in (('external_activity', 'external_activity'), ('node_fraction', 'node_fraction'),
                                  ('off_weight', 'off_weight'), ('SC', 'closure_entropy_coefficient')):
                require(math.isclose(float(row[column]), result['coordinates'][field], rel_tol=1e-12),
                        f'Executed coordinate differs: {name} {column}')
            for field in ('event_cap', 'target_entities', 'wall_time_cap_s'):
                require(float(row[field]) == result['termination'][field], f'Executed cap differs: {name} {field}')
        result_links.append({'result': name, 'run_id': result['run_id'],
                             'random_seed': result['random_seed'], 'seed_sha256': seed['sha256'],
                             'matching_manifests': sorted({f for f, _ in candidates})})
    require(len(result_links) == 10, 'Expected six S7 and four sampled historical result matches')
    gaps = [r for r in references if r['status'] != 'EXACT_BYTES_PACKAGED']
    require(not gaps, f'Recorded code/seed versions are missing: {gaps}')
    report = {
        'source_records': len(sources.rows), 'source_bytes_before_deduplication': sum(r['source_bytes'] for r in sources.rows),
        'csv_manifests': csv_counts, 'frozen_seed_metadata': seed_metadata,
        'code_and_seed_reference_count': len(references), 'unmatched_version_references': gaps,
        'fixed_event_design_rows': len(parent_rows), 'size_continuation_rows': len(follow_rows),
        'paired_branches': len(paired['traces']), 'paired_independent_source_particles': 32,
        'lineage_rows': len(lineage), 'remote_files_hash_verified': len(witnesses),
        'remote_checkpoint_count': 975, 'remote_bytes': sum(r['bytes'] for r in witnesses.values()),
        'remote_checked_utc': remote['checked_utc'], 'executed_result_links': result_links,
        'limits': ['Manifest row count is not a completed-run count.',
                   'Seed/checkpoint pickle contents were never deserialized.',
                   'Remote checkpoint payloads are external and are not rechecked by this offline verifier.',
                   'Fixed-event, size-followon and paired-response datasets are not pooled as independent repeats.',
                   'No full KMC replay, scientific recalibration or final figure adoption is claimed.'],
    }
    return report, references, lineage


def table_checks():
    manifest = json.loads((TABLES / 'source_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        raw = local_path(row['release_path']).read_bytes()
        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], f'Table source changed: {row}')
    with (TABLES / 's1_original/B_seed_studies_original.csv').open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    dose = [r for r in records if r['starting_scope'] == 'dose_or_presence']
    groups = defaultdict(list)
    for row in dose:
        groups[row['publication_group_id']].append(row['doi'])
    counts = Counter(r['family_resolved'] for r in dose)
    saved = json.loads((TABLES / 's1_original/B_family_counts_original.json').read_text(encoding='utf-8'))
    require(len(dose) == len({r['doi'] for r in dose}) == saved['unique_dose_or_presence_papers'] == 69, 'S1 DOI counts')
    require(len(groups) == saved['distinct_dose_or_presence_publication_groups'] == 68, 'S1 publication groups')
    require(dict(counts) == saved['dose_family_counts_by_doi_record'], 'S1 family counts')
    documents = json.loads((TABLES / 'selected_si_tables_20261031.json').read_text(encoding='utf-8'))['documents']
    require(len(documents) == 2, 'Selected SI baselines')
    for a, b in zip(documents[0]['tables'], documents[1]['tables']):
        require(a['caption'] == b['caption'] and a['rows'] == b['rows'], 'Table versions differ')
    tables = {t['caption'].split('.')[0]: t for t in documents[-1]['tables']}
    require(len(tables) == 4, 'Expected S1/S2/S6/S8')
    with (ROOT / '08_entropy_inputs/data/entropy_results.csv').open(encoding='utf-8-sig', newline='') as f:
        entropy = {r['']: r for r in csv.DictReader(f)}
    names = {'SBU': 'SBU', 'BTB': 'BTB', 'MOL1': 'MOL_1', "MOL1'": "MOL_1'", 'MOL2': 'MOL_2', 'MOL3': 'MOL_3'}
    comparison = []
    for row in tables['Table S6']['rows'][1:]:
        original = entropy[names[row[0]]]
        total = float(original['Translational Entropy']) + float(original['Rotational Entropy'])
        require(abs(total - float(original['Entropy'])) < 1e-12, f'Entropy sum: {row[0]}')
        for column, value in zip(('Translational Entropy', 'Rotational Entropy', 'Entropy'), row[1:]):
            raw = float(original[column])
            error = abs(raw - float(value))
            require(error <= 5.0001e-9, f'S6 value mismatch: {row[0]} {column}')
            comparison.append({'species_in_si': row[0], 'species_in_source': names[row[0]],
                               'column': column, 'si_value': value, 'source_value': original[column],
                               'absolute_difference': error, 'status': 'MATCH_AT_DISPLAYED_PRECISION'})
    require(len(comparison) == 18, 'S6 expected 18 values')
    decisions = json.loads((ROOT / 'metadata/publication_decisions_20261031.json').read_text(encoding='utf-8'))
    s1 = decisions['TableS1']
    require(s1['status'] == 'COUNTING_SCOPE_AUTHOR_CONFIRMED', 'S1 must retain its author-confirmation basis')
    require(s1['adopted_total'] == 69 and s1['adopted_zr_hf'] == 53, 'S1 adopted counts must remain 69/53')
    require(s1['additional_count_confirmation_request'] is False, 'S1 must not repeat the closed count request')
    nas = decisions['NAS_followup_20261101']
    s8 = decisions['TableS8']
    require(s8['status'] == 'CLOSED_AUTHOR_CONFIRMED', 'S8 closure must retain its author-confirmation basis')
    require(s8['additional_raw_record_request'] is False, 'S8 must not request another raw record')
    require(s8['observations'] == len(tables['Table S8']['rows']) - 1 == 12 and s8['time_unit'] == 'min',
            'S8 confirmation must refer to the unchanged 12-row minute-scale table')
    report = {
        'source_files_checked': len(manifest['files']),
        'selected_document_hashes': {d['file']: d['sha256'] for d in documents},
        'latest_nas_check_record': nas,
        's1': {'dose_doi_records': len(dose), 'distinct_publication_groups': len(groups),
               'doi_family_counts': dict(counts), 'duplicate_groups': {k:v for k,v in groups.items() if len(v)>1},
               'zr_hf_doi_records': counts['Zr/Hf'],
               'zr_hf_publication_groups': len({r['publication_group_id'] for r in dose if r['family_resolved']=='Zr/Hf'}),
               'status': s1['status'], 'author_confirmation': s1,
               'limit': 'Author-confirmed publication counts are 69/53. Archived grouping statistics are historical provenance, not alternative adopted counts or an open correction request. This check does not re-read all literature or reconstruct final direction labels.'},
        's2': {'historical_packaged_total': sum(int(r[1]) for r in tables['Table S2']['rows'][1:]),
               'current_total': nas['TableS2']['current_total'],
               'status': nas['TableS2']['status'],
               'limit': nas['TableS2']['remaining']},
        's6': {'matched_values': len(comparison), 'maximum_absolute_difference': max(r['absolute_difference'] for r in comparison),
               'status': nas['TableS6']['status'],
               'offline_verification_scope': 'Packaged derived table only. The separately recorded NAS notebook recalculation is not rerun by this verifier.',
               'limit': nas['TableS6']['remaining']},
        's8': {'displayed_observations': len(tables['Table S8']['rows']) - 1,
               'status': s8['status'], 'source_confirmation': s8,
               'limit': '12 observations in minutes, not the single in-situ DLS size trace. Do not infer extra repeats.'},
    }
    return report, comparison


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def verify(write_derived=False):
    sources = Sources()
    runtime, refs, lineage = runtime_checks(sources)
    tables, comparison = table_checks()
    report = {'status': 'PASS_FOR_STATED_PROVENANCE_CHECKS', 'publication_ready': False,
              'runtime': runtime, 'tables': tables}
    outputs = {
        RUNTIME / 'code_seed_version_links.csv': csv_bytes(refs),
        RUNTIME / 'checkpoint_lineage.csv': csv_bytes(lineage),
        TABLES / 's6_value_comparison.csv': csv_bytes(comparison),
        ROOT / 'metadata/runtime_table_audit_20261031.json':
            (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8'),
    }
    for path, raw in outputs.items():
        if write_derived:
            path.write_bytes(raw)
        else:
            require(path.read_bytes() == raw, f'Stale derived audit: {path.name}')
    return report


def materialize():
    sources = Sources()
    destination = ROOT / 'reproduced_local/runtime_source_tree'
    for name in sources.index:
        target = (destination / name).resolve()
        require(target.is_relative_to(destination.resolve()), f'Unsafe materialization path: {name}')
        raw = sources.read(name)
        if target.exists():
            require(target.read_bytes() == raw, f'Will not overwrite differing file: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    print(f'Materialized {len(sources.index)} source files under {destination}; no scripts executed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-derived', action='store_true')
    parser.add_argument('--materialize', action='store_true')
    args = parser.parse_args()
    if args.materialize:
        materialize()
    else:
        result = verify(args.write_derived)
        print(json.dumps({'status': result['status'],
                          'source_files': result['runtime']['source_records'],
                          'unmatched_versions': len(result['runtime']['unmatched_version_references']),
                          'lineage_rows': result['runtime']['lineage_rows'],
                          's6_values': result['tables']['s6']['matched_values']}, indent=2))
