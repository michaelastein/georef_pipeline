import folium
import webbrowser
import os
import json
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

CAD_MAP = "panels_with_row_plaintext_below.geojson"

def plot_cad_map(target_gps, points=None, corner_gps=None, drone_gps=None,
                 geojson_file=CAD_MAP, map_file="target_map_cad.html"):
    """
    Plots the target, drone, corners, and multiple GPS points on a folium map using a GeoJSON CAD file.
    GPS points are color-coded by score.

    Parameters:
        target_gps: tuple (lat, lon) of the main target
        points: list of dicts [{'target_gps': (lat, lon), 'score': float}, ...] (optional)
        corner_gps: list of tuples [(lat, lon), ...] for image corners (optional)
        drone_gps: tuple (lat, lon) for drone position (optional)
        geojson_file: path to GeoJSON CAD file
        map_file: filename to save HTML map
    """
    lat, lon = target_gps

    # --- Load GeoJSON ---
    with open(geojson_file, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    # --- Check CRS and convert if needed ---
    crs_name = geojson_data.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    if "4326" not in crs_name:
        try:
            src_crs = CRS.from_user_input(crs_name)
        except Exception:
            epsg_code = crs_name.split(":")[-1]
            src_crs = CRS.from_epsg(int(epsg_code))
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
        for feature in geojson_data["features"]:
            geom = feature["geometry"]
            if geom["type"] == "Polygon":
                geom["coordinates"] = [[list(transformer.transform(x, y)) for x, y in ring]
                                       for ring in geom["coordinates"]]
            elif geom["type"] == "MultiPolygon":
                geom["coordinates"] = [[[list(transformer.transform(x, y)) for x, y in ring] for ring in poly]
                                       for poly in geom["coordinates"]]

    # --- Base map centered on target ---
    m = folium.Map(
        location=[lat, lon],
        zoom_start=20,
        min_zoom=15,
        max_zoom=24,
        tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg",
        attr="Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors"
    )

    # --- Add GeoJSON layer ---
    folium.GeoJson(
        geojson_data,
        name="CAD Overlay",
        style_function=lambda x: {
            "color": "black",
            "weight": 1,
            "fillColor": "#cccccc",
            "fillOpacity": 0.3,
        },
        tooltip=folium.GeoJsonTooltip(fields=[]),
    ).add_to(m)

    # --- Plot GPS points with color based on score ---
    if points:
        scores = [pt.get('score', 0.5) for pt in points]
        norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
        cmap = cm.get_cmap('YlOrRd')

        for pt in points:
            lat_pt, lon_pt = pt.get('target_gps', (None, None))
            if lat_pt is None or lon_pt is None:
                continue  # skip points without GPS

            score = pt.get('score', 0.5)
            color = mcolors.to_hex(cmap(norm(score)))

            # --- Use folium.Circle with 0.4m radius (40cm) ---
            folium.Circle(
                [lat_pt, lon_pt],
                radius=0.4,  # in meters
                color=None,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Score: {score:.2f}\nLat: {lat_pt:.7f}\nLon: {lon_pt:.7f}"
            ).add_to(m)

    # --- Marker for target pixel ---
    folium.Marker(
        [lat, lon],
        popup=f"Target\nLat: {lat:.7f}\nLon: {lon:.7f}",
        icon=folium.Icon(color="red", icon="crosshairs"),
    ).add_to(m)

    # --- Polygon connecting image corners if provided ---
    if corner_gps:
        folium.Polygon(
            corner_gps + [corner_gps[0]],
            color="#00FF00",
            weight=2,
            fill=False,
            tooltip="Image Corners",
        ).add_to(m)

    # --- Marker for drone GPS if provided ---
    if drone_gps:
        d_lat, d_lon = drone_gps
        folium.Marker(
            [d_lat, d_lon],
            popup=f"Drone\nLat: {d_lat:.7f}\nLon: {d_lon:.7f}",
            icon=folium.Icon(color="blue", icon="plane"),
        ).add_to(m)

    # --- Save and open map ---
    m.save(map_file)
    webbrowser.open("file://" + os.path.abspath(map_file))
