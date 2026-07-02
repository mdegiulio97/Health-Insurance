"""Download idempotente del dataset Medical Cost Personal ('insurance.csv').

Regole del portfolio:
- I dati NON vengono mai committati (vedi .gitignore).
- Lo script è idempotente: se il file esiste già, non riscarica.
- Fonte e licenza del dataset sono documentate nel README.

Fonte: dataset "insurance.csv" dal libro "Machine Learning with R" (B. Lantz),
redistribuito pubblicamente nel repository stedy/Machine-Learning-with-R-datasets.
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
        print(f"Dataset già presente, download saltato: {dest}")
        return
    print(f"Scarico il dataset da: {URL}")
    urllib.request.urlretrieve(URL, dest)  # noqa: S310 - URL statico e fidato
    size_kb = dest.stat().st_size / 1024
    print(f"Salvato: {dest} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
