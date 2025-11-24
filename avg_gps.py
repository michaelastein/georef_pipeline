import numpy as np
import georeferncing
from plot_cad import plot_cad_map  # Updated CAD map function
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import cv2
import os


# ----------------- Haversine distance -----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ----------------- Score computation -----------------
def compute_image_scores(data_array):
    for item in data_array:
        width, height = item.get('image_size', (1, 1))
        cx, cy = width / 2, height / 2

        pitch = item.get('pitch', 0.0) or 0.0
        roll = item.get('roll', 0.0) or 0.0
        nadir_score = np.exp(-(pitch**2 + roll**2) / (2 * 30**2))

        u, v = item.get('pixel_x', cx), item.get('pixel_y', cy)
        du = (u - cx) / cx
        dv = (v - cy) / cy
        dist = np.sqrt(du**2 + dv**2) / np.sqrt(2)
        center_score = 1 - dist

        combined_score = (nadir_score + center_score) / 2
        item['score'] = float(np.clip(combined_score, 0, 1))

# ----------------- Flexible GPS-based filtering -----------------
def filter_points_by_distance(data_array, max_distance=50.0, gps_type='drone'):
    """
    Filter points based on distance from the first valid point.

    Parameters:
        data_array: list of dicts with GPS info
        max_distance: maximum allowed distance in meters
        gps_type: 'drone' (uses gps) or 'target' (uses target_gps)
    
    Returns:
        filtered list of points (always at least the first valid point)
    """
    if not data_array:
        return []

    # Determine reference GPS
    ref_point = None
    ref_item = None
    for item in data_array:
        if gps_type == 'drone' and item.get('gps') is not None:
            ref_point = item['gps'][:2]  # lat, lon
            ref_item = item
            break
        elif gps_type == 'target' and item.get('target_gps') is not None:
            ref_point = item['target_gps']
            ref_item = item
            break

    if ref_point is None:
        print(f"No valid {gps_type} GPS found for filtering.")
        return data_array

    ref_lat, ref_lon = ref_point
    filtered = []
    for item in data_array:
        if gps_type == 'drone':
            gps = item.get('gps')
            if gps is None:
                continue
            lat, lon = gps[:2]
        else:  # target
            tgt = item.get('target_gps')
            if tgt is None:
                continue
            lat, lon = tgt

        distance = haversine(ref_lat, ref_lon, lat, lon)
        if distance <= max_distance:
            filtered.append(item)

    # If all points got discarded, keep the reference point
    if not filtered:
        filtered.append(ref_item)

    return filtered


# ----------------- Weighted average GPS -----------------
def weighted_average_gps(data_array, z_thresh=2.0, max_distance_m=10.0):
    """
    Compute weighted average GPS while ignoring outliers.
    - Removes points farther than max_distance_m from the first valid target GPS.
    - Removes outliers beyond z_thresh standard deviations.
    - Returns robust fallback even if filtering is too strict.
    """
    if not data_array:
        return None, None

    # Collect valid target_gps and scores
    lats, lons, weights, indices = [], [], [], []
    for idx, item in enumerate(data_array):
        target = item.get('target_gps')
        score = item.get('score', 0.0)
        if not target or score <= 0:
            continue

        try:
            lat, lon = float(target[0]), float(target[1])
        except Exception as e:
            continue

        lats.append(lat)
        lons.append(lon)
        weights.append(score)
        indices.append(idx)

    if not lats:
        return None, None

    lats = np.array(lats)
    lons = np.array(lons)
    weights = np.array(weights)
    indices = np.array(indices)

    # --- Step 1: Filter by max_distance from reference point ---
    ref_lat, ref_lon = lats[0], lons[0]
    distances = np.array([haversine(ref_lat, ref_lon, la, lo) for la, lo in zip(lats, lons)])
    mask_distance = distances <= max_distance_m

    if not np.any(mask_distance):
        mask_distance = np.ones_like(distances, dtype=bool)

    lats = lats[mask_distance]
    lons = lons[mask_distance]
    weights = weights[mask_distance]
    indices = indices[mask_distance]

    # --- Step 2: Compute weighted average and standard deviation ---
    lat_avg = np.average(lats, weights=weights)
    lon_avg = np.average(lons, weights=weights)

    lat_std = np.sqrt(np.average((lats - lat_avg) ** 2, weights=weights))
    lon_std = np.sqrt(np.average((lons - lon_avg) ** 2, weights=weights))

    # --- Step 3: Remove statistical outliers ---
    mask_stats = (
        (np.abs(lats - lat_avg) <= z_thresh * lat_std) &
        (np.abs(lons - lon_avg) <= z_thresh * lon_std)
    )

    if not np.any(mask_stats):
        return lat_avg, lon_avg

    to_keep = indices[mask_stats]
    data_array[:] = [data_array[i] for i in to_keep]

    # --- Step 4: Recompute final average ---
    lat_avg = np.average(lats[mask_stats], weights=weights[mask_stats])
    lon_avg = np.average(lons[mask_stats], weights=weights[mask_stats])

    return lat_avg, lon_avg




# ----------------- GUI to display thumbnails with points -----------------

def show_images_with_points(points_data, max_thumb_size=150, master=None):
    """
    Display images with marked points in a Tkinter window.
    
    points_data: list of dicts, each must have:
        - 'image_name' (str)
        - 'pixel_x', 'pixel_y' (float, optional)
        - 'image_size' (tuple, optional)
        - 'score' (float)
    max_thumb_size: max width/height for thumbnails
    master: existing Tk root or Toplevel parent
    """
    valid_points = [p for p in points_data if p.get('score', 0) > 0 and p.get('image_name')]
    if not valid_points:
        print("No images with valid points to show.")
        return

    # Use existing master if given, otherwise create new Tk root
    if master is None:
        root = tk.Tk()
        root.title("Images with Points")
        created_root = True
    else:
        root = tk.Toplevel(master)
        root.title("Images with Points")
        created_root = False

    # Canvas + scrollbar
    canvas = tk.Canvas(root, width=1100, height=600)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    photo_refs = []  # keep references to avoid garbage collection
    cols = 5

    for i, item in enumerate(valid_points):
        img_path = item['image_name']
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = min(max_thumb_size / h, max_thumb_size / w)
        disp_w, disp_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (disp_w, disp_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)

        # Draw point if available
        if 'pixel_x' in item and 'pixel_y' in item and 'image_size' in item:
            orig_w, orig_h = item['image_size']
            px = item['pixel_x'] * (disp_w / orig_w)
            py = item['pixel_y'] * (disp_h / orig_h)
            radius = max(3, int(scale * 5))
            draw = ImageDraw.Draw(im_pil)
            draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill='red')

        im_tk = ImageTk.PhotoImage(im_pil)
        photo_refs.append(im_tk)

        # Frame for image + label
        img_frame = tk.Frame(frame, bd=1, relief="solid")
        img_frame.grid(row=i // cols, column=i % cols, padx=5, pady=5)

        lbl = tk.Label(img_frame, image=im_tk)
        lbl.pack()
        name_lbl = tk.Label(img_frame, text=os.path.basename(img_path), font=("Arial", 8))
        name_lbl.pack()

    frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    if created_root:
        root.mainloop()


# ----------------- Main pipeline -----------------
def main(data, max_drone_distance=40.0, max_target_distance=20.0, show_gui=False, dem_path = None, lidar_path = None, cad_path= None):
    """
    Process image data: filter by distance, compute scores, average GPS, and optionally display.

    Returns:
        processed_data: list of dicts with fields including 'pixel_x', 'pixel_y', 'score', 'target_gps'
        avg_lat, avg_lon: weighted average GPS coordinates of cluster
    """
    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    if not data:
        print("No data provided.")
        return [], None, None

    # --- Filter images with valid drone GPS ---
    data = [item for item in data if item.get('gps') is not None]
    if not data:
        print("No items with valid drone GPS.")
        return [], None, None

    # --- Filter by distance from first drone GPS ---
    data = filter_points_by_distance(data, max_distance=max_drone_distance, gps_type='drone')

    # --- Initialize DroneMapper and fill target_gps ---
    mapper = georeferncing.DroneMapper(lidar_path = lidar_path, dem_path= dem_path)
    data = mapper.get_target_gps_array(data)

    # --- Filter items with valid target GPS ---
    data = [item for item in data if item.get('target_gps') is not None]
    if not data:
        print("No items with valid target GPS.")
        return [], None, None

    # --- Filter by distance from first target GPS ---
    data = filter_points_by_distance(data, max_distance=max_target_distance, gps_type='target')




    # --- Compute image scores ---
    compute_image_scores(data)

    # --- Compute weighted average GPS ---
    avg_lat, avg_lon = weighted_average_gps(data)
    if avg_lat is None or avg_lon is None:
        print("No valid GPS points to compute cluster average.")
        return data, None, None

    print(f"Target GPS: {avg_lat:.6f}, {avg_lon:.6f}")

    # --- Optional CAD map plotting ---
    if cad_path:
        plot_cad_map(target_gps=(avg_lat, avg_lon), points=data, cad_path= cad_path)

    # --- Optional GUI thumbnail display ---
    if show_gui:
        show_images_with_points(data)

    return data, avg_lat, avg_lon
