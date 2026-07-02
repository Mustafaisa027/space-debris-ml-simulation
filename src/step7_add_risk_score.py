import csv

INPUT = r"outputs\cpa_with_vrel.csv"
OUTPUT = r"outputs\cpa_with_risk.csv"

rows_out = []

with open(INPUT, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        d_min = float(row["d_min_km_used"])
        v_rel = float(row["v_rel_km_s"])

        # Basit risk skoru: hız / mesafe (birim: 1/s)
        risk_score = v_rel / d_min

        row["risk_score_1_per_s"] = f"{risk_score:.10f}"
        rows_out.append(row)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    fieldnames = list(rows_out[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

print(f"OK -> {OUTPUT} yazildi ({len(rows_out)} satir).")
