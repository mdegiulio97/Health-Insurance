"""Idempotent download of the Medical Cost Personal dataset ('insurance.csv').

Portfolio rules:
- Data are NEVER committed (see .gitignore).
- The script is idempotent: if the file already exists, it does not download it again.
- The dataset's source and licence are documented in the README.

Source: the "insurance.csv" dataset from the book "Machine Learning with R" (B. Lantz),
publicly redistributed in the repository stedy/Machine-Learning-with-R-datasets.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_NAME = "insurance.csv"
URL = (
    "https://raw.githubusercontent.com/stedy/"
    "Machine-Learning-with-R-datasets/master/insurance.csv"
)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / CSV_NAME
    if dest.exists():
        print(f"Dataset already present, download skipped: {dest}")
        return
    print(f"Downloading the dataset from: {URL}")
    urllib.request.urlretrieve(URL, dest)  # noqa: S310 - static, trusted URL
    size_kb = dest.stat().st_size / 1024
    print(f"Saved: {dest} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
