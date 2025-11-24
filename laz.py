import laspy
import numpy as np
from pyproj import Transformer
from plot_maps import plot_google_maps

# Load LAZ
laz_path = "vollerup.laz"
las = laspy.read(laz_path)

# Bounding box
min_x, max_x = np.min(las.x), np.max(las.x)  # Easting
min_y, max_y = np.min(las.y), np.max(las.y)  # Northing

print(f"LAZ Extents (UTM32 / ETRS89): Easting[{min_x}, {max_x}], Northing[{min_y}, {max_y}]")

# Transformer: EPSG:25832 → WGS84
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

# Corners in (lat, lon) = (Northing → lat, Easting → lon)
corners = []
for ex, ny in [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]:
    lon, lat = transformer.transform(ex, ny)  # always_xy=True → x=Easting, y=Northing
    corners.append((lat, lon))

print("LAZ corners (lat, lon):")
for lat, lon in corners:
    print(f"Lat: {lat:.7f}, Lon: {lon:.7f}")

# Center
center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2
center_lon, center_lat = transformer.transform(center_x, center_y)

# Plot on Google Maps
plot_google_maps(
    target_gps=(center_lat, center_lon),
    corner_gps=corners
)
