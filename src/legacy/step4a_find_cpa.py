import csv
from datetime import datetime, timezone

INPUT = r"outputs\pair_distance.csv"
OUTPUT = r"outputs\cpa_summary.csv"

def parse_utc_iso(s: str) -> datetime:
    # ör: 2026-01-21T13:09:43Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)

rows = []
with open(INPUT, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["minute"] = int(r["minute"])
        r["distance_km"] = float(r["distance_km"])
        r["dt"] = parse_utc_iso(r["utc_iso"])
        rows.append(r)

if not rows:
    raise SystemExit("pair_distance.csv bos!")

# Global minimum (tüm aralıkta)
global_min = min(rows, key=lambda r: r["distance_km"])

# Pencere tespiti: distance < threshold ise aday pencere
THRESHOLD_KM = 10000.0
windows = []
current = []

for r in rows:
    if r["distance_km"] < THRESHOLD_KM:
        current.append(r)
    else:
        if current:
            windows.append(current)
            current = []
if current:
    windows.append(current)

# Her pencere için CPA (min distance)
cpa_rows = []
for i, w in enumerate(windows, start=1):
    cpa = min(w, key=lambda r: r["distance_km"])
    cpa_rows.append({
        "window_id": i,
        "obj1": cpa["obj1"],
        "obj2": cpa["obj2"],
        "tca_utc": cpa["utc_iso"],
        "minute": cpa["minute"],
        "d_min_km": f"{cpa['distance_km']:.3f}",
        "window_start_utc": w[0]["utc_iso"],
        "window_end_utc": w[-1]["utc_iso"],
        "num_points": len(w),
    })

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["window_id","obj1","obj2","window_start_utc","window_end_utc","num_points","tca_utc","minute","d_min_km"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in cpa_rows:
        w.writerow(r)

print("=== GLOBAL CPA (tum aralik) ===")
print(f"obj1={global_min['obj1']} obj2={global_min['obj2']}")
print(f"TCA={global_min['utc_iso']}  minute={global_min['minute']}  d_min={global_min['distance_km']:.3f} km")
print()
print(f"OK -> {OUTPUT} yazildi. (window sayisi: {len(cpa_rows)})")
