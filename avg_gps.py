import numpy as np
import georef_new
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
    Filter points based on distance from the first point.
    
    Parameters:
        data_array: list of dicts with GPS info
        max_distance: maximum allowed distance in meters
        gps_type: 'drone' (uses gps_lat/gps_lon) or 'target' (uses target_gps)
    
    Returns:
        filtered list of points
    """
    if not data_array:
        return []

    # Determine reference GPS
    ref_point = None
    for item in data_array:
        if gps_type == 'drone' and item.get('gps_lat') is not None and item.get('gps_lon') is not None:
            ref_point = (item['gps_lat'], item['gps_lon'])
            break
        elif gps_type == 'target' and item.get('target_gps') is not None:
            ref_point = item['target_gps']
            break

    if ref_point is None:
        print(f"No valid {gps_type} GPS found for filtering.")
        return data_array

    ref_lat, ref_lon = ref_point
    filtered = []
    for item in data_array:
        if gps_type == 'drone':
            lat = item.get('gps_lat')
            lon = item.get('gps_lon')
        else:  # target
            tgt = item.get('target_gps')
            if tgt is None:
                continue
            lat, lon = tgt

        if lat is None or lon is None:
            continue

        distance = haversine(ref_lat, ref_lon, lat, lon)
        if distance <= max_distance:
            filtered.append(item)

    return filtered

# ----------------- Weighted average GPS -----------------
def weighted_average_gps(data_array, z_thresh=2.0, max_distance_m=10.0):
    """
    Compute weighted average GPS while ignoring outliers.
    Removes points farther than max_distance_m meters from the first target GPS.
    Outliers above z_thresh are also removed from the data_array itself.
    """
    if not data_array:
        return None, None

    # Collect valid target points and scores, and keep track of original indices
    lats, lons, weights, indices = [], [], [], []
    for idx, item in enumerate(data_array):
        target = item.get('target_gps')
        score = item.get('score', 0.0)
        if target is None or score <= 0:
            continue
        lat, lon = target
        if lat is None or lon is None:
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

    # --- Remove points too far from first target GPS ---
    ref_lat, ref_lon = lats[0], lons[0]
    distances = np.array([haversine(ref_lat, ref_lon, la, lo) for la, lo in zip(lats, lons)])
    mask_distance = distances <= max_distance_m
    if not np.any(mask_distance):
        return None, None  # if all points are too far, return None

    # Keep only points within max distance
    lats = lats[mask_distance]
    lons = lons[mask_distance]
    weights = weights[mask_distance]
    indices = indices[mask_distance]

    # --- Compute initial weighted averages ---
    lat_avg = np.average(lats, weights=weights)
    lon_avg = np.average(lons, weights=weights)

    # --- Weighted standard deviations ---
    lat_std = np.sqrt(np.average((lats - lat_avg)**2, weights=weights))
    lon_std = np.sqrt(np.average((lons - lon_avg)**2, weights=weights))

    # --- Remove statistical outliers ---
    mask = ((np.abs(lats - lat_avg) <= z_thresh * lat_std) &
            (np.abs(lons - lon_avg) <= z_thresh * lon_std))
    if not np.any(mask):
        # fallback if all filtered out
        return lat_avg, lon_avg

    # --- Remove outliers from original data array ---
    to_keep_indices = indices[mask]
    data_array[:] = [data_array[i] for i in to_keep_indices]

    # Final weighted average
    lat_avg = np.average(lats[mask], weights=weights[mask])
    lon_avg = np.average(lons[mask], weights=weights[mask])

    return lat_avg, lon_avg


# ----------------- GUI to display thumbnails with points -----------------

def show_images_with_points(points_data, max_thumb_size=150, master=None):
    """
    Display images with marked points in a Tkinter window.
    
    points_data: list of dicts, each must have:
        - 'image_path' (str)
        - 'pixel_x', 'pixel_y' (float, optional)
        - 'image_size' (tuple, optional)
        - 'score' (float)
    max_thumb_size: max width/height for thumbnails
    master: existing Tk root or Toplevel parent
    """
    valid_points = [p for p in points_data if p.get('score', 0) > 0 and p.get('image_path')]
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
        img_path = item['image_path']
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
def main(data):
    # Filter images too far from first drone GPS
    data = filter_points_by_distance(data, max_distance=70.0, gps_type='drone')

    # Initialize DroneMapper and fill target_gps
    mapper = georef_new.DroneMapper()
    data = mapper.get_target_gps_array(data)

    # Filter points too far from first target GPS
    data = filter_points_by_distance(data, max_distance=10.0, gps_type='target')

    # Compute image scores
    compute_image_scores(data)

    # Compute weighted average GPS as target
    avg_lat, avg_lon = weighted_average_gps(data)
    if avg_lat is not None and avg_lon is not None:
        print(f"Target GPS: {avg_lat:.6f}, {avg_lon:.6f}")
    else:
        print("No valid GPS points in cluster near first image.")
        return

    # Visualize on CAD map directly using data
    plot_cad_map(target_gps=(avg_lat, avg_lon), points=data)

    # Show thumbnails of images with points
    show_images_with_points(data)
