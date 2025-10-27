import folium
import webbrowser
import os
import json
from pyproj import Transformer

CAD_MAP = "panels_with_row_plaintext_below.geojson"

def plot_cad_map(target_gps, corner_gps=None, drone_gps=None, geojson_file=CAD_MAP, map_file="target_map_cad.html"):
    """
    Plots the target, drone, and corners on a folium map using a GeoJSON CAD file as background.
    Automatically reprojects the GeoJSON to WGS84 if needed.

    Parameters:
        target_gps: tuple (lat, lon) of the target pixel
        corner_gps: list of tuples [(lat, lon), ...] for image corners (optional)
        drone_gps: tuple (lat, lon) for the drone position (optional)
        geojson_file: path to GeoJSON CAD file
        map_file: filename to save HTML map
    """

    lat, lon = target_gps

    # --- Load GeoJSON ---
    with open(geojson_file, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    # --- Check CRS and convert if needed ---
    crs_name = geojson_data.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    if "4326" not in crs_name:  # Not WGS84
        #print(f"Reprojecting GeoJSON from {crs_name} to EPSG:4326")
        # Extract EPSG code from string (e.g., "urn:ogc:def:crs:EPSG::25829")
        epsg_code = crs_name.split(":")[-1]
        transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)

        for feature in geojson_data["features"]:
            geom = feature["geometry"]
            if geom["type"] == "Polygon":
                new_coords = []
                for ring in geom["coordinates"]:
                    new_ring = [list(transformer.transform(x, y)) for x, y in ring]
                    new_coords.append(new_ring)
                geom["coordinates"] = new_coords

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
