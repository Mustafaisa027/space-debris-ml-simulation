import csv
from datetime import datetime
import matplotlib.pyplot as plt

PAIR_FILE = r"outputs\pair_distance.csv"
CPA_RISK_FILE = r"outputs\cpa_with_risk.csv"

def parse_utc(s):
    if s.endswith("Z"):
        s = s[:-1]
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

# --- Mesafe - zaman verisi ---
times = []
distances = []

with open(PAIR_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        times.append(parse_utc(row["utc_iso"]))
        distances.append(float(row["distance_km"]))

# --- CPA + risk verisi ---
cpa_times = []
cpa_distances = []
labels = []
risk_scores = []

with open(CPA_RISK_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = parse_utc(row["tca_utc_used"])
        d = float(row["d_min_km_used"])
        v = float(row["v_rel_km_s"])
        r = float(row["risk_score_1_per_s"])

        cpa_times.append(t)
        cpa_distances.append(d)
        risk_scores.append(r)

        labels.append(
            f"d_min={d:.0f} km\nv_rel={v:.2f} km/s\nrisk={r:.4e}"
        )

# En riskli CPA’yı bul
max_risk = max(risk_scores)
max_idx = risk_scores.index(max_risk)

# --- Grafik ---
plt.figure(figsize=(11, 6))
plt.plot(times, distances, label="ISS – NOAA15 Mesafe (km)")
plt.axhline(10000, linestyle="--", label="Eşik (10000 km)")

# CPA noktaları
plt.scatter(cpa_times, cpa_distances, s=80, label="CPA Noktaları")

# En riskli CPA’yı kırmızıyla vurgula
plt.scatter(
    cpa_times[max_idx],
    cpa_distances[max_idx],
    s=150,
    marker="X",
    label="En Riskli CPA"
)

# Etiketler
for i, txt in enumerate(labels):
    plt.annotate(
        txt,
        (cpa_times[i], cpa_distances[i]),
        textcoords="offset points",
        xytext=(10, -20),
        fontsize=9
    )

plt.xlabel("UTC Zaman")
plt.ylabel("Mesafe (km)")
plt.title("ISS – NOAA15 Yakınlaşma Analizi\nCPA, Bağıl Hız ve Risk Skoru ile")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
