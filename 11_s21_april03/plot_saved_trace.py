"""Run the archived plotting-only helper in a separate output workspace."""
import argparse
from pathlib import Path
import runpy
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced_local/S21_april03')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    allowed = ROOT / 'reproduced_local'
    if output.is_relative_to(ROOT) and not output.is_relative_to(allowed):
        parser.error('In-repository outputs must be under reproduced_local; do not overwrite source archives')
    source = HERE / 'source_archive'
    for relative in ('plot_code/replot_saved_trace.py',
                     'source_data/oneshot_regenerated_seed160_eq2_cap180_zr5000_pruned.summary_trace.json'):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != (source / relative).read_bytes():
            raise ValueError(f'Refusing to overwrite a different source: {destination}')
        shutil.copyfile(source / relative, destination)
    plt.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none'})
    runpy.run_path(str(output / 'plot_code/replot_saved_trace.py'), run_name='__main__')


if __name__ == '__main__':
    main()
