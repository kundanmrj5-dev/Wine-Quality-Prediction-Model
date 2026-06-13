from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_FILE = DATA_DIR / "winequality-red.csv"
MODEL_FILE = MODELS_DIR / "wine_quality_model.joblib"
METRICS_FILE = OUTPUTS_DIR / "metrics.csv"
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

