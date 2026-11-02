"""Import the approved November 2 supplement; never modify upstream files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / '17_submission_sources'
IDENTITY_HASHES = {
    'chen_exact_selected.csv': '64ddf96c757e7ab5e387e2391a99ba82df4d4f93c3edd11a2b5986961065e970',
    'mol_fitted_constants_checked.csv': 'fbb275d3928c7bd343e3f2b6cc1c1fcd3dc25616213e3d15db0b0bc5fff9be32',
    'recomputed_correlations.csv': 'a7905dd1acf7d42a4692601a4b0a85e16e74f4c8a8433751c43d6e2e288f8405',
}
DOCUMENTS = {
    'MS': ('MS/manuscript_defect_section_to_SI_20261031.docx',
           '2c553784038b250a96b878ce27b052b35c71ef4d6fd970c8d66ea213e52b2a7a'),
    'SI': ('SI/supporting_information_S2_updated_20261031_v2.docx',
           '5038f6291d95c7b3ad69c9c8a24bd741ac0570ae92a561537c9716cce9478680'),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f'Refusing to replace different existing content: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-project', required=True, type=Path)
    args = parser.parse_args()
    project = args.shared_project.resolve()
    data = project / 'MOF_growth_Data'
    bundle = data / 'MOL_figure_source_bundles_20261022_v4'
    files = []

    def copy(source, relative, purpose, expected=None):
        source = source.resolve()
        if not source.is_relative_to(project):
            raise ValueError(f'Upstream path outside selected project: {source}')
        target = (GROUP / relative).resolve()
        if not target.is_relative_to(GROUP):
            raise ValueError(f'Unsafe package path: {relative}')
        raw = source.read_bytes()
        if len(raw) > 10_000_000:
            raise ValueError(f'Unexpected large supplement: {source}')
        sha = digest(raw)
        if expected is not None and expected != sha:
            raise ValueError(f'Upstream hash differs: {source}')
        write_new(target, raw)
        files.append(dict(source_relative=source.relative_to(project).as_posix(),
                          release_path=target.relative_to(ROOT).as_posix(),
                          bytes=len(raw), sha256=sha, purpose=purpose))
        return target.relative_to(ROOT).as_posix()

    identity = bundle / 'SI_tables_sources_20261023/Section_S7_identity_evidence'
    for name, sha in IDENTITY_HASHES.items():
        copy(identity/name, 'identity/'+name, 'SI S7 identity/regression source', sha)
    copy(identity/'README.md', 'identity/SOURCE_README.md', 'Original source and hash record')

    mapping_folder = data / 'Table_S4_NMR_mapping_20261102'
    mapping_path = mapping_folder / 'Table_S4_acquisition_mapping.json'
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    copy(mapping_path, 'nmr/Table_S4_acquisition_mapping.original.json', 'Author-confirmed original mapping')
    for suffix in ['md', 'html']:
        copy(mapping_folder/('Table_S4_\u9010\u884c\u5bf9\u5e94.'+suffix),
             'nmr/source_mapping.'+suffix, 'Original mapping explanation')
    nmr_rows = []
    for row in mapping['rows']:
        nominal = row['temperature_K_nominal_in_SI']
        acquisition = PurePosixPath(row['acquisition_directory'])
        packed = {}
        for kind, info in row['files'].items():
            path = PurePosixPath(info['path'])
            marker = '/MOF_growth_Data/Experiment/HNMR/'
            if marker not in str(path):
                raise ValueError(f'Unexpected NMR source: {path}')
            tail = str(path).split(marker, 1)[1]
            source = data/'Experiment/HNMR'/tail
            if kind == 'exported_spectrum_image':
                relative = f'nmr/spectra/{nominal}K.jpg'
            else:
                relative = f'nmr/acquisitions/{nominal}K/{row["experiment_number"]}/' + path.relative_to(acquisition).as_posix()
            packed[kind] = copy(source, relative, f'Table S4 entry {row["table_entry"]}: {kind}', info['sha256'])
        nmr_rows.append({**{k: row[k] for k in ['table_entry','temperature_K_nominal_in_SI',
                         'linewidth_Hz_as_reported_in_Table_S4','exchange_rate_s_minus_1_as_reported',
                         'experiment_number','processing_number','linewidth_source','SI_figure']},
                         'files': packed})
    write_new(GROUP/'nmr/portable_mapping.json', (json.dumps(nmr_rows, indent=2)+'\n').encode())

    for extension, folder in [('svg','vector_artwork'),('png','raster_reference_artwork')]:
        copy(bundle/f'MS_Figure_6/{folder}/0001_Figure6C_N2000.{extension}',
             f'figure6c/Figure6C_N2000.{extension}', 'N=2000 four-per-condition reference panel')

    documents = []
    for role, (relative, expected) in DOCUMENTS.items():
        raw = (project/relative).read_bytes()
        if digest(raw) != expected:
            raise ValueError(f'Document changed since reviewed final view: {role}')
        with zipfile.ZipFile(project/relative) as archive:
            xml = ET.fromstring(archive.read('word/document.xml'))
            w = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            record = dict(role=role, filename=Path(relative).name, bytes=len(raw), sha256=expected,
                          insertions_in_document_xml=len(list(xml.iter(w+'ins'))),
                          deletions_in_document_xml=len(list(xml.iter(w+'del'))))
            if role == 'MS':
                image = archive.read('word/media/image10.png')
                record['adopted_figure6'] = dict(docx_member='word/media/image10.png',
                    sha256=digest(image), bytes=len(image), N=2000, repeats_per_condition=4,
                    inspection='Accepted embedded image visually checked; panel E caption present')
            documents.append(record)
    manifest = dict(schema_version=1, checked_on='2026-11-02',
                    source_root_description='Author shared project; paths below are relative to project root',
                    documents=documents, files=files, documents_included_or_edited=False)
    write_new(GROUP/'source_manifest.json', (json.dumps(manifest, indent=2)+'\n').encode())
    print(json.dumps({'files':len(files),'bytes':sum(f['bytes'] for f in files),'documents_included':False}))


if __name__ == '__main__':
    main()
