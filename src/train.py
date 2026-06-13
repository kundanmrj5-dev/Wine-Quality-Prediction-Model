import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_FILE,
    METRICS_FILE,
    MODEL_FILE,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from data_utils import load_wine_data


def build_models() -> dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        random_state=RANDOM_STATE,
                        min_samples_leaf=2,
                    ),
                )
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                (
                    "model",
                    GradientBoostingRegressor(
                        random_state=RANDOM_STATE,
                        learning_rate=0.05,
                        n_estimators=220,
                        max_depth=3,
                    ),
                )
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": rmse,
        "r2": r2_score(y_test, predictions),
    }


def save_feature_importance(best_model: Pipeline) -> None:
    estimator = best_model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance, x="importance", y="feature", color="#59a14f")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_FILE, dpi=150)
    plt.close()


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_wine_data()
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    stratify_target = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    results = []
    trained_models = {}

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        trained_models[name] = model
        results.append({"model": name, **metrics})

    metrics_table = pd.DataFrame(results).sort_values("rmse")
    metrics_table.to_csv(METRICS_FILE, index=False)

    best_name = metrics_table.iloc[0]["model"]
    best_model = trained_models[best_name]
    joblib.dump(best_model, MODEL_FILE)
    save_feature_importance(best_model)

    print("Model evaluation:")
    print(metrics_table.round(4).to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved metrics to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
