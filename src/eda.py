import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUTS_DIR, TARGET_COLUMN
from data_utils import load_wine_data


def save_target_distribution(data) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x=TARGET_COLUMN, color="#4c78a8")
    plt.title("Wine Quality Distribution")
    plt.xlabel("Quality Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "quality_distribution.png", dpi=150)
    plt.close()


def save_correlation_heatmap(data) -> None:
    plt.figure(figsize=(11, 8))
    correlation = data.corr(numeric_only=True)
    sns.heatmap(correlation, cmap="vlag", center=0, annot=False, square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def save_feature_relationships(data) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    features = ["alcohol", "volatile acidity", "sulphates"]

    for axis, feature in zip(axes, features):
        sns.scatterplot(data=data, x=feature, y=TARGET_COLUMN, alpha=0.45, ax=axis)
        axis.set_title(f"{feature.title()} vs Quality")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "feature_relationships.png", dpi=150)
    plt.close()


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_wine_data()

    print("Dataset shape:", data.shape)
    print("\nMissing values:")
    print(data.isna().sum())
    print("\nSummary statistics:")
    print(data.describe().round(2))

    save_target_distribution(data)
    save_correlation_heatmap(data)
    save_feature_relationships(data)
    print(f"\nEDA plots saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()

