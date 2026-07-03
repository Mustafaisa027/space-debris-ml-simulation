import csv
import math
from datetime import datetime

from skyfield.api import EarthSatellite, load

# --- TLE'ler (şimdilik sabit; sonra güncel TLE çekeceğiz) ---
iss_l1 = "1 25544U 98067A   24019.54791435  .00016717  00000+0  10270-3 0  9991"
iss_l2 = "2 25544  51.6415  67.7316 0005447  83.1490  28.3976 15.50012378431589"

noaa_l1 = "1 25338U 98030A   24019.54262076  .00000073  00000+0  69177-4 0  9993"
noaa_l2 = "2 25338  98.7314  58.7588 0011420 193.1801 166.9396 14.25933904353136"

iss = EarthSatellite(iss_l1, iss_l2, "ISS")
noaa15 = EarthSatellite(noaa_l1, noaa_l2, "NOAA15")

ts = load.timescale()

# Hangi CPA dosyası varsa onu kullan (refined varsa onu tercih eder)
INPUT_REFINED = r"outputs\cpa_refined.csv"
INPUT_MINUTE = r"outputs\cpa_summary.csv"
OUTPUT = r"outputs\cpa_with_vrel.csv"


def parse_utc_iso_to_components(s: str):
    # "2026-01-21T14:09:43Z"
    if s.endswith("Z"):
        s = s[:-1]
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second


def vrel_km_s_at(utc_iso: str) -> float:
    y, mo, d, h, mi, sec = parse_utc_iso_to_components(utc_iso)
    t = ts.utc(y, mo, d, h, mi, sec)

    v1 = iss.at(t).velocity.km_per_s
    v2 = noaa15.at(t).velocity.km_per_s

    dvx = v1[0] - v2[0]
    dvy = v1[1] - v2[1]
    dvz = v1[2] - v2[2]
    return math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)


# Input seçimi
use_refined = False
try:
    with open(INPUT_REFINED, newline="", encoding="utf-8") as f:
        pass
    input_path = INPUT_REFINED
    use_refined = True
except FileNotFoundError:
    input_path = INPUT_MINUTE

rows_out = []

with open(input_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if use_refined:
            tca = row["tca_refined_utc"]
            dmin = float(row["d_min_refined_km"])
            tca_minute = row["tca_minute_utc"]
            dmin_minute = float(row["d_min_minute_km"])
        else:
            tca = row["tca_utc"]
            dmin = float(row["d_min_km"])
            tca_minute = tca
            dmin_minute = dmin

        vrel = vrel_km_s_at(tca)

        rows_out.append({
            "window_id": row.get("window_id", ""),
            "obj1": row.get("obj1", "ISS"),
            "obj2": row.get("obj2", "NOAA15"),
            "tca_utc_used": tca,
            "d_min_km_used": f"{dmin:.3f}",
            "v_rel_km_s": f"{vrel:.6f}",
            "tca_minute_utc": tca_minute,
            "d_min_minute_km": f"{dmin_minute:.3f}",
            "source": "refined" if use_refined else "minute",
        })

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "window_id", "obj1", "obj2",
        "source",
        "tca_utc_used", "d_min_km_used", "v_rel_km_s",
        "tca_minute_utc", "d_min_minute_km"
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

print(f"OK -> {OUTPUT} yazildi ({len(rows_out)} pencere). Kaynak: {'refined' if use_refined else 'minute'}")
