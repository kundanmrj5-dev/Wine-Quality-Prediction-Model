from urllib.request import urlretrieve

from config import DATA_DIR, DATA_FILE


UCI_RED_WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urlretrieve(UCI_RED_WINE_URL, DATA_FILE)
    print(f"Downloaded official UCI red-wine dataset to: {DATA_FILE}")


if __name__ == "__main__":
    main()
