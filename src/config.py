from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_FILE = DATA_DIR / "winequality-red.csv"
MODEL_FILE = MODELS_DIR / "wine_quality_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
METRICS_FILE = OUTPUTS_DIR / "metrics.csv"
TRAINING_REPORT_FILE = OUTPUTS_DIR / "training_report.txt"
FEATURE_IMPORTANCE_FILE = OUTPUTS_DIR / "feature_importance.png"

TARGET_COLUMN = "quality"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
