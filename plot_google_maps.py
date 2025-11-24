import folium
import webbrowser
import os

def plot_google_maps(target_gps, corner_gps=None, drone_gps=None, map_file="target_map.html"):
    """
    Plots the target, drone, and corners on a folium map without the target area polygon.
    
    Parameters:
        target_gps: tuple (lat, lon) of the target pixel
        corner_gps: list of tuples [(lat, lon), ...] for image corners (optional)
        drone_gps: tuple (lat, lon) for the drone position (optional)
        map_file: filename to save HTML map
    """
    lat, lon = target_gps

    # --- Base map centered on target ---
    m = folium.Map(location=[lat, lon], zoom_start=21, tiles=None)
    google_sat = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
    folium.TileLayer(tiles=google_sat, attr='Google', name='Google Satellite',
                     overlay=False, control=False, max_zoom=23).add_to(m)

    # --- Marker for target pixel ---
    folium.Marker([lat, lon], popup=f"Target\nLat: {lat:.7f}\nLon: {lon:.7f}",
                  icon=folium.Icon(color='red', icon='crosshairs')).add_to(m)

    # --- Polygon connecting image corners if provided ---
    if corner_gps:
        folium.Polygon(corner_gps + [corner_gps[0]], color="#00FF00", weight=2, fill=False,
                       tooltip="Image Corners").add_to(m)

    # --- Marker for drone GPS if provided ---
    if drone_gps:
        d_lat, d_lon = drone_gps
        folium.Marker([d_lat, d_lon], popup=f"Drone\nLat: {d_lat:.7f}\nLon: {d_lon:.7f}",
                      icon=folium.Icon(color='blue', icon='plane')).add_to(m)

    # --- Save and open map ---
    m.save(map_file)
    webbrowser.open('file://' + os.path.abspath(map_file))
