import argparse

import joblib
import pandas as pd

from config import FEATURE_COLUMNS, MODEL_FILE


DEFAULT_SAMPLE = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


def add_feature_arguments(parser: argparse.ArgumentParser) -> None:
    for feature, default in DEFAULT_SAMPLE.items():
        argument = "--" + feature.replace(" ", "_")
        parser.add_argument(argument, type=float, default=default)


def parse_sample(args: argparse.Namespace) -> pd.DataFrame:
    values = {
        feature: getattr(args, feature.replace(" ", "_"))
        for feature in FEATURE_COLUMNS
    }
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict wine quality")
    add_feature_arguments(parser)
    args = parser.parse_args()

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_FILE}. Run training first: python src/train.py"
        )

    model_artifact = joblib.load(MODEL_FILE)
    if isinstance(model_artifact, dict) and "model" in model_artifact:
        model = model_artifact["model"]
        feature_columns = model_artifact.get("feature_columns", FEATURE_COLUMNS)
        model_name = model_artifact.get("model_name", "trained model")
    else:
        model = model_artifact
        feature_columns = FEATURE_COLUMNS
        model_name = "trained model"

    sample = parse_sample(args)
    sample = sample[feature_columns]
    prediction = float(model.predict(sample)[0])

    print("Input sample:")
    print(sample.to_string(index=False))
    print(f"\nLoaded model: {model_name}")
    print(f"\nPredicted wine quality: {prediction:.2f}")


if __name__ == "__main__":
    main()
