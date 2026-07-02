import csv

INPUT = r"outputs\pair_distance.csv"
OUTPUT = r"outputs\conjunction_events.csv"

THRESHOLD_KM = 10000  # 10 km değil, 10.000 km (örnek eşik)

events = []

with open(INPUT, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        d = float(row["distance_km"])
        if d < THRESHOLD_KM:
            events.append(row)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["obj1", "obj2", "minute", "utc_iso", "distance_km"]
    )
    writer.writeheader()
    for e in events:
        writer.writerow(e)

print(f"Conjunction adayi sayisi: {len(events)}")
print(f"OK -> {OUTPUT} yazildi")
