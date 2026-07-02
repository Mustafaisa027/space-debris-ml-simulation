from skyfield.api import EarthSatellite, load

# ISS örnek TLE (ileride güncel TLE çekeceğiz)
line1 = "1 25544U 98067A   24019.54791435  .00016717  00000+0  10270-3 0  9991"
line2 = "2 25544  51.6415  67.7316 0005447  83.1490  28.3976 15.50012378431589"

sat = EarthSatellite(line1, line2, "ISS")
ts = load.timescale()
t = ts.now()

pos = sat.at(t)
x, y, z = pos.position.km

print("ISS ECI konumu (km):")
print(f"x={x:.2f}, y={y:.2f}, z={z:.2f}")

