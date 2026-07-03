from skyfield.api import EarthSatellite, load
import csv

# ISS örnek TLE
line1 = "1 25544U 98067A   24019.54791435  .00016717  00000+0  10270-3 0  9991"
line2 = "2 25544  51.6415  67.7316 0005447  83.1490  28.3976 15.50012378431589"

sat = EarthSatellite(line1, line2, "ISS")
ts = load.timescale()

# Simülasyon ayarı: 0–60 dk, her 5 dk
start = ts.now()
minutes = list(range(0, 61, 5))

out_path = r"outputs\positions.csv"

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sat_name", "minute", "utc_iso", "x_km", "y_km", "z_km"])

    for m in minutes:
        # Skyfield zaman ekleme: gün cinsinden
        t = start + (m / (24 * 60))
        pos = sat.at(t)
        x, y, z = pos.position.km

        writer.writerow([
            "ISS",
            m,
            t.utc_iso(),
            f"{x:.2f}",
            f"{y:.2f}",
            f"{z:.2f}"
        ])

print(f"OK -> {out_path} yazildi ({len(minutes)} satir).")
