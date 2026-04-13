# Deterministic sales-data generator. Run by prepare_commands so the
# agent gets a real 1080-row CSV (90 days x 12 SKUs) instead of having
# to trust hand-written comments.
import csv
import os
import random
from datetime import date, timedelta

random.seed(42)
START = date(2026, 1, 8)
DAYS = 90
# baseline daily units; the LAST 30 days for the 3 anomaly SKUs
# are scaled by `anomaly_factor`.
skus = {
    "SKU-A1": (42, 1.00),
    "SKU-A2": (38, 1.00),
    "SKU-B1": (120, 0.57),  # supplier change → -43%
    "SKU-B2": (95, 1.00),
    "SKU-C1": (18, 1.00),
    "SKU-C2": (22, 1.00),
    "SKU-D1": (67, 0.61),  # pack-size mismatch → -39%
    "SKU-D2": (55, 1.00),
    "SKU-E1": (12, 1.00),
    "SKU-E2": (9, 1.00),
    "SKU-F1": (210, 0.64),  # banner pulled → -36%
    "SKU-F2": (180, 1.00),
}

os.makedirs("data", exist_ok=True)
with open("data/sales-90d.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "sku", "units"])
    for d in range(DAYS):
        day = START + timedelta(days=d)
        in_last_30 = d >= DAYS - 30
        for sku, (base, factor) in skus.items():
            target = base * factor if in_last_30 else base
            # ±10% noise so it looks real
            noise = random.uniform(-0.10, 0.10)
            units = max(0, round(target * (1 + noise)))
            w.writerow([day.isoformat(), sku, units])
print(f"wrote data/sales-90d.csv with {DAYS * len(skus)} rows")
