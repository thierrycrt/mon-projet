"""mon_projet: recycling rate relative to collected weight.

Standard library only. Expects the CSV columns:
site, semaine, categorie, collecte_kg, revendu_kg, recycle_kg,
dechetterie_kg, attente_kg.
"""

import csv

CSV_PATH = "data/pesees.csv"


def load_rows(path):
    """Read the CSV file and return a list of dicts (one per line).

    Numeric columns are converted from text to numbers so we can
    do arithmetic on them.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["semaine"] = int(row["semaine"])
        for col in ("collecte_kg", "revendu_kg", "recycle_kg",
                    "dechetterie_kg", "attente_kg"):
            row[col] = float(row[col])
    return rows


def recycling_rate(rows):
    """Return the recycling rate in percent, all categories combined.

    Rule: rate = (recycle_kg / collecte_kg) * 100, rounded to 2 decimals.

    Args:
        rows: list of dicts returned by load_rows().

    Returns:
        The rate as a float (e.g. 57.84).

    Raises:
        ValueError: if the total collected weight is zero, because
            the rate cannot be computed (division by zero).
    """
    # Step 1: add up the collected weight of all categories
    collected = sum(r["collecte_kg"] for r in rows)

    # Step 2: add up the recycled weight of all categories
    recycled = sum(r["recycle_kg"] for r in rows)

    # Step 3: stop with a clear message if we cannot divide
    if collected == 0:
        raise ValueError("Total collected weight is zero: rate undefined.")

    # Step 4: apply the rule and round to 2 decimals
    return round((recycled / collected) * 100, 2)


if __name__ == "__main__":
    data = load_rows(CSV_PATH)

    # Example 1: the whole file
    print("Whole file:", recycling_rate(data), "%")

    # Example 2: only the first 3 lines (Textiles, Papier, Plastique)
    print("First 3 lines:", recycling_rate(data[:3]), "%")
