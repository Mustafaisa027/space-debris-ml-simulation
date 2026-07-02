import csv
import math
from pathlib import Path
from datetime import datetime

from skyfield.api import EarthSatellite, load

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def latest_tle_file(prefix: str) -> Path:
    # Örn: prefix="tle_ISS_25544_"
    files = sorted(DATA_DIR.glob(prefix + "*.txt"))
    if not files:
        raise FileNotFoundError(f"{DATA_DIR} icinde {prefix} ile baslayan TLE yok. Once step9'u calistir.")
    return files[-1]  # alfabetik sıralama timestamp ile uyumlu

def read_tle(path: Path):
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"TLE dosyasi beklenenden kisa: {path}")
    name = lines[0]
    l1 = lines[1]
    l2 = lines[2]
    return name, l1, l2

# En güncel dosyaları bul
iss_path = latest_tle_file("tle_ISS_25544_")
noaa_path = latest_tle_file("tle_NOAA15_25338_")

iss_name, iss_l1, iss_l2 = read_tle(iss_path)
noaa_name, noaa_l1, noaa_l2 = read_tle(noaa_path)

ts = load.timescale()
iss = EarthSatellite(iss_l1, iss_l2, iss_name)
noaa15 = EarthSatellite(noaa_l1, noaa_l2, noaa_name)

# 0–60 dk, 1 dk adım
start = ts.now()
minutes = list(range(0, 61, 1))

pair_path = OUT_DIR / "pair_distance_live.csv"
with open(pair_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["obj1","obj2","minute","utc_iso","distance_km"])
    for m in minutes:
        t = start + (m / (24 * 60))
        p1 = iss.at(t).position.km
        p2 = noaa15.at(t).position.km
        dx = p1[0]-p2[0]; dy = p1[1]-p2[1]; dz = p1[2]-p2[2]
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        w.writerow([iss.name, noaa15.name, m, t.utc_iso(), f"{d:.3f}"])

# CPA (dakika çözünürlüğü)
rows = []
with open(pair_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["minute"] = int(r["minute"])
        r["distance_km"] = float(r["distance_km"])
        rows.append(r)

global_min = min(rows, key=lambda r: r["distance_km"])

# Basit eşik ile pencere tespiti (10000 km)
THRESHOLD_KM = 10000.0
windows = []
cur = []
for r in rows:
    if r["distance_km"] < THRESHOLD_KM:
        cur.append(r)
    else:
        if cur:
            windows.append(cur); cur=[]
if cur:
    windows.append(cur)

cpa_path = OUT_DIR / "cpa_live.csv"
with open(cpa_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["window_id","obj1","obj2","window_start_utc","window_end_utc","tca_utc","minute","d_min_km"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for i, win in enumerate(windows, start=1):
        cpa = min(win, key=lambda r: r["distance_km"])
        w.writerow({
            "window_id": i,
            "obj1": cpa["obj1"],
            "obj2": cpa["obj2"],
            "window_start_utc": win[0]["utc_iso"],
            "window_end_utc": win[-1]["utc_iso"],
            "tca_utc": cpa["utc_iso"],
            "minute": cpa["minute"],
            "d_min_km": f"{cpa['distance_km']:.3f}",
        })

# v_rel + risk
def parse_utc_iso(s: str) -> datetime:
    if s.endswith("Z"): s = s[:-1]
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

risk_path = OUT_DIR / "cpa_with_risk_live.csv"
with open(cpa_path, newline="", encoding="utf-8") as f_in, open(risk_path, "w", newline="", encoding="utf-8") as f_out:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames + ["v_rel_km_s","risk_score_1_per_s"]
    w = csv.DictWriter(f_out, fieldnames=fieldnames)
    w.writeheader()

    for row in reader:
        dt = parse_utc_iso(row["tca_utc"])
        t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

        v1 = iss.at(t).velocity.km_per_s
        v2 = noaa15.at(t).velocity.km_per_s
        dvx = v1[0]-v2[0]; dvy = v1[1]-v2[1]; dvz = v1[2]-v2[2]
        vrel = math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)

        dmin = float(row["d_min_km"])
        risk = vrel / dmin

        row["v_rel_km_s"] = f"{vrel:.6f}"
        row["risk_score_1_per_s"] = f"{risk:.10f}"
        w.writerow(row)

print("=== LIVE PIPELINE OK ===")
print(f"TLE(ISS)  : {iss_path.name}")
print(f"TLE(NOAA) : {noaa_path.name}")
print(f"pair      : {pair_path}")
print(f"cpa       : {cpa_path}")
print(f"risk      : {risk_path}")
print()
print("GLOBAL CPA:")
print(f"TCA={global_min['utc_iso']}  minute={global_min['minute']}  d_min={global_min['distance_km']:.3f} km")
