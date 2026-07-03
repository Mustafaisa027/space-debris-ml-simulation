import csv
from datetime import datetime
import matplotlib.pyplot as plt

PAIR_FILE = r"outputs\pair_distance.csv"
CPA_FILE = r"outputs\cpa_summary.csv"

times = []
distances = []

def parse_utc(s):
    if s.endswith("Z"):
        s = s[:-1]
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

# Mesafe-zaman verisini oku
with open(PAIR_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        times.append(parse_utc(row["utc_iso"]))
        distances.append(float(row["distance_km"]))

# CPA noktalarını oku
cpa_times = []
cpa_distances = []

with open(CPA_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cpa_times.append(parse_utc(row["tca_utc"]))
        cpa_distances.append(float(row["d_min_km"]))

# Grafik
plt.figure(figsize=(10, 5))
plt.plot(times, distances, label="ISS - NOAA15 Mesafe (km)")
plt.scatter(cpa_times, cpa_distances, marker="x", s=100, label="CPA Noktaları")

plt.axhline(10000, linestyle="--", label="Eşik (10000 km)")

plt.xlabel("UTC Zaman")
plt.ylabel("Mesafe (km)")
plt.title("ISS – NOAA15 Yakınlaşma Analizi ve CPA")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
