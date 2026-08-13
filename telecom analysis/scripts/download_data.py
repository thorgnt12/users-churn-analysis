"""Download the IBM Telco Customer Churn dataset used by this project."""

from pathlib import Path
from urllib.request import urlretrieve


DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)
DESTINATION = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {DESTINATION}...")
    urlretrieve(DATA_URL, DESTINATION)
    print("Done.")


if __name__ == "__main__":
    main()
