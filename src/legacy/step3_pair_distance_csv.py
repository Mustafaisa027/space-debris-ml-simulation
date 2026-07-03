from skyfield.api import EarthSatellite, load
import csv
import math

ts = load.timescale()

# Uydu-1: ISS (ZARYA)
iss_l1 = "1 25544U 98067A   24019.54791435  .00016717  00000+0  10270-3 0  9991"
iss_l2 = "2 25544  51.6415  67.7316 0005447  83.1490  28.3976 15.50012378431589"
iss = EarthSatellite(iss_l1, iss_l2, "ISS")

# Uydu-2: NOAA 15 (örnek TLE)  -> eğer bu TLE eski çıkarsa sonra güncelleyeceğiz
noaa_l1 = "1 25338U 98030A   24019.54262076  .00000073  00000+0  69177-4 0  9993"
noaa_l2 = "2 25338  98.7314  58.7588 0011420 193.1801 166.9396 14.25933904353136"
noaa15 = EarthSatellite(noaa_l1, noaa_l2, "NOAA15")

start = ts.now()
minutes = list(range(0, 61, 1))  # 0–60 dk, her 1 dk (daha sık)

out_path = r"outputs\pair_distance.csv"

with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["obj1", "obj2", "minute", "utc_iso", "distance_km"])

    for m in minutes:
        t = start + (m / (24 * 60))

        p1 = iss.at(t).position.km
        p2 = noaa15.at(t).position.km

        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]

        d = math.sqrt(dx*dx + dy*dy + dz*dz)

        w.writerow(["ISS", "NOAA15", m, t.utc_iso(), f"{d:.3f}"])

print(f"OK -> {out_path} yazildi ({len(minutes)} satir).")
