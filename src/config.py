from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__import__("os").environ.get(
    "STEEL_DEFECT_DATA", PROJECT_ROOT / "data" / "NEU-DET"
))

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
