from __future__ import annotations

from pathlib import Path
import importlib.util
import os


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATOR_PATH = SCRIPT_DIR / "UiO66_generate_excel_conc.py"
ANACONDA_ROOT = Path(r"C:\Users\yibinjiang\anaconda3")


def load_generator_module():
    spec = importlib.util.spec_from_file_location("uio66_long_growth_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load generator from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PATH"] = str(ANACONDA_ROOT) + os.pathsep + os.environ.get("PATH", "")

    generator = load_generator_module()
    generator.TEMPLATE_EXCEL_BASE = "UiO66_260206_long1000"
    generator.PARENT_FOLDER = str((SCRIPT_DIR / "generated_runs_fixed").resolve()).replace("\\", "/")
    generator.NUM_CORES = 8
    generator.NUM_STAGES = 1
    generator.DRY_RUN = False
    generator.CLEAN_UP = True
    generator.main(1)


if __name__ == "__main__":
    main()
