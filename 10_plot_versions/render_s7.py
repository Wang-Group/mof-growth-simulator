"""Publication S7 entry point: algorithm time u, with canonical S7 filenames."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RENDERER = Path(__file__).resolve().parent / 'shared_20261031/SI_Figure_S7/original_code/export_figure_s7_panels.py'
NAMES = {
    'Figure_S8A_trajectories': 'SI_Figure_S7_A',
    'Figure_S8B_first_passage': 'SI_Figure_S7_B',
    'Figure_S8C_transition_intervals': 'SI_Figure_S7_C',
    'Figure_S8D_residence_map': 'SI_Figure_S7_D',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced_local/S7_publication_u')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'reproduced_local'):
        parser.error('In-repository outputs must be under reproduced_local; preserve source archives')
    spec = importlib.util.spec_from_file_location('_s7_raw_u_renderer', RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_save = module.save

    def publication_save(fig, output_dir, qa_dir, name, source):
        return original_save(fig, output_dir, qa_dir, NAMES[name], source)

    module.save = publication_save
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(RENDERER), '--data-dir',
                    str(ROOT / '07_publication_sources/SI_Figure_S7/content_checked_20261023/source_data'),
                    '--output-dir', str(output), '--qa-dir', str(output / 'qa')]
        module.main()
    finally:
        sys.argv = previous_argv
    report = json.loads((output / 'qa/validation.json').read_text(encoding='utf-8'))
    if report['source_time_factor_removed'] != 0.175 or not report['output_time_coordinate'].startswith('raw algorithmic time u'):
        raise AssertionError('Publication S7 must use algorithm time u')


if __name__ == '__main__':
    main()
