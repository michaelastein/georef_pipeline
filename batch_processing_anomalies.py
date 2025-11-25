import os
import csv
import traceback
import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import Counter
import avg_gps
import anomaly_matching
import plot_cad
import time

# ---------------- Globals ----------------
output_array = []
csv_anomaly_array = []

# ---------------- Helpers ----------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# ---------------- CSV Helpers ----------------
def read_csv(csv_path, image_data_list):
    """Read CSV and filter anomalies matching images in image_data_list."""
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    first_image_name = image_data_list[0]["image_name"]
    file_col = "wiris_image" if first_image_name.strip().lower().endswith(".tiff") else "pi_image"
    if file_col not in df.columns:
        raise ValueError(f"CSV does not contain expected column: {file_col}")

    # Precompute normalized image names
    image_name_map = {os.path.basename(img["image_name"]).strip().lower(): img for img in image_data_list}

    df[file_col] = df[file_col].astype(str).str.strip().str.lower()
    df_filtered = df[df[file_col].isin(image_name_map.keys())].copy()

    csv_anomalies = []
    for _, row in df_filtered.iterrows():
        img_name = row[file_col]
        entry = image_name_map.get(img_name)
        if entry is None:
            continue
        cx = safe_float(row.get("center_x"))
        cy = safe_float(row.get("center_y"))
        csv_anomalies.append({
            "image_name": img_name,
            "center_x": cx,
            "center_y": cy,
            "anomaly": row.get("anomaly")
        })
    return csv_anomalies

def write_csv(csv_path, rows):
    if not rows:
        print("[DEBUG] No rows to save.")
        return None
    base, ext = os.path.splitext(csv_path)
    new_csv_path = f"{base}_georeferenced{ext}"
    fieldnames = ["anomaly_type", "latitude", "longitude", "example_image", "example_pixel_x", "example_pixel_y"]
    with open(new_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Georeferenced CSV saved to: {new_csv_path}")
    return new_csv_path

def add_anomaly_to_csv(output_array, anomaly_type, latitude, longitude, example_image=None, example_pixel_x=None, example_pixel_y=None):
    output_array.append({
        "anomaly_type": anomaly_type,
        "latitude": latitude,
        "longitude": longitude,
        "example_image": example_image,
        "example_pixel_x": example_pixel_x,
        "example_pixel_y": example_pixel_y
    })

# ---------------- Process single anomaly ----------------
def process_next_anomaly(row, image_data_list, build_corr_func, csv_path, original_image_size=None, dem_path=None, lidar_path=None):
    global csv_anomaly_array
    try:
        center_x = row.get("center_x")
        center_y = row.get("center_y")
        image_name = row.get("image_name")

        # Precompute a lookup dictionary for image_data_list
        image_map = {os.path.basename(img.get("image_name", "")).strip().lower(): img for img in image_data_list}
        img_entry = image_map.get(image_name.lower())
        if img_entry is None:
            raise ValueError(f"Image '{image_name}' not found in image_data_list")

        img_w, img_h = img_entry.get("image_size", (512, 512))
        if center_x is None or center_y is None:
            center_x, center_y = img_w / 2, img_h / 2

        # Use scale_coordinates from anomaly_matching.py
        scaled_x, scaled_y = anomaly_matching.scale_coordinates(center_x, center_y, from_size=original_image_size, to_size=(img_w, img_h))

        # Build correspondence array
        correspondence_array = build_corr_func(image_data_list.index(img_entry), scaled_x, scaled_y, image_data_list=image_data_list)

        # Compute weighted average GPS
        processed_data, avg_lat, avg_lon = avg_gps.main(correspondence_array, dem_path=dem_path, lidar_path=lidar_path)
        if avg_lat is None or avg_lon is None:
            print("[DEBUG] Average GPS computation failed for this anomaly.")
            return None

        # Get matched anomalies
        matched_items = anomaly_matching.main(correspondence_array, csv_path=csv_path, original_image_size=original_image_size)
        if not matched_items:
            return None

        # Determine majority anomaly_type
        types = [mi.get("anomaly", "unknown") for mi in matched_items if mi.get("anomaly")]
        anomaly_type = Counter(types).most_common(1)[0][0] if types else "unknown"

        # Find best-centered image
        best_item = None
        min_dist = float("inf")
        for mi in matched_items:
            px, py = mi.get("pixel_x"), mi.get("pixel_y")
            img_name = mi.get("image_name")
            if px is None or py is None or not img_name:
                continue
            entry2 = image_map.get(img_name.lower())
            if entry2:
                w, h = entry2.get("image_size", (512, 512))
                if original_image_size:
                    px, py = anomaly_matching.scale_coordinates(px, py, from_size=original_image_size, to_size=(w, h))
                dist = np.hypot(px - w/2, py - h/2)
                if dist < min_dist:
                    min_dist = dist
                    best_item = {
                        "example_image": os.path.basename(img_name),
                        "example_pixel_x": px,
                        "example_pixel_y": py
                    }

        add_anomaly_to_csv(
            output_array,
            anomaly_type,
            avg_lat,
            avg_lon,
            example_image=best_item.get("example_image") if best_item else None,
            example_pixel_x=best_item.get("example_pixel_x") if best_item else None,
            example_pixel_y=best_item.get("example_pixel_y") if best_item else None
        )

        remove_matched_anomalies(matched_items)

    except Exception as e:
        print(f"[ERROR] Processing anomaly: {e}")
        traceback.print_exc()
        return None

# ---------------- Remove matched anomalies ----------------
def remove_matched_anomalies(matched_items, tol=2.0):
    global csv_anomaly_array
    if not csv_anomaly_array or not matched_items:
        return
    matched_set = set((os.path.basename(mi.get("image_name","")).strip().lower(), mi.get("pixel_x"), mi.get("pixel_y")) for mi in matched_items)
    csv_anomaly_array = [row for row in csv_anomaly_array
                         if (os.path.basename(row.get("image_name","")).strip().lower(), row.get("center_x"), row.get("center_y")) not in matched_set]

# ---------------- Main pipeline ----------------
def main(image_data_list, build_corr_func, print_progress_func, original_image_size=None, 
         output_file="batch_targets_map.html", dem_path=None, lidar_path=None, cad_path=None):
    global csv_anomaly_array
    try:
        Tk().withdraw()
        csv_path = askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            print("No CSV file selected. Exiting.")
            return

        csv_anomaly_array = read_csv(csv_path, image_data_list)
        if not csv_anomaly_array:
            print("No matching anomalies found in CSV. Exiting.")
            return

        total_anomalies = len(csv_anomaly_array)
        print(f"Processing {total_anomalies} anomalies...")
        start_time = time.time()

        for idx, row in enumerate(csv_anomaly_array.copy(), start=1):
            process_next_anomaly(
                row, image_data_list, build_corr_func, csv_path,
                original_image_size=original_image_size,
                dem_path=dem_path, lidar_path=lidar_path
            )
            print_progress_func(current=idx, total=total_anomalies, stage_name="Processing anomalies",
                                start_time=start_time, bar_length=40, update_every_percent=1)

        print("All anomalies processed.")

        georef_csv_path = write_csv(csv_path, output_array)

        points = [{"target_gps": (r["latitude"], r["longitude"]), "score": 1.0} 
                  for r in output_array if r["latitude"] is not None]
        if points and cad_path:
            central = points[0]["target_gps"]
            plot_cad.plot_cad_map(target_gps=central, points=points, cad_path=cad_path, output_file=output_file)

        return georef_csv_path

    except Exception as e:
        print(f"[ERROR] Exception in main_pipeline(): {e}")
        traceback.print_exc()
        return None
