import sys
from pathlib import Path

FIGURE5_DIR = Path(
    r"D:\SYH\课题\MOL生长模拟\论文\student_handoff_20260915_v5\MS-SI\Figure\Figure5"
)
FIGURE_DIR = Path(
    r"D:\SYH\课题\MOL生长模拟\论文\student_handoff_20260915_v5\MS-SI\Figure\Figure3"
)

sys.path.insert(0, str(FIGURE5_DIR))
from plot_entities_broken_axis import process_single_pkl  # noqa: E402


if __name__ == "__main__":
    process_single_pkl(
        pkl_file_path=FIGURE_DIR / "entities_number.pkl",
        save_dir=FIGURE_DIR,
        x_range=(3350, 3435),
        y_range=(0, 30),
        figsize=(9, 6),
        dpi=300,
        x_label="Growth Time",
        y_label="Number of Entities",
        zero_fraction=0.03,
        output_name="entities_number_broken_x_short_gap_v3_matched.png",
        y_ticks=range(0, 31, 5),
    )
