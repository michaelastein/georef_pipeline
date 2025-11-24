import os
import csv
import traceback
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import Counter
import avg_gps
import anomaly_matching
import plot_cad

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
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    first_image_name = image_data_list[0]["image_name"]
    file_col = "wiris_image" if first_image_name.strip().lower().endswith(".tiff") else "pi_image"
    if file_col not in df.columns:
        raise ValueError(f"CSV does not contain expected column: {file_col}")

    image_names_set = {os.path.basename(e["image_name"]).strip().lower() for e in image_data_list}

    csv_anomaly_array_local = []
    for _, row in df.iterrows():
        csv_name = str(row[file_col]).strip().lower()
        if csv_name in image_names_set:
            entry = next((e for e in image_data_list if os.path.basename(e["image_name"]).strip().lower() == csv_name), None)
            if entry is None:
                continue
            current_size = entry.get("image_size", (512, 512))
            cx = safe_float(row.get("center_x"))
            cy = safe_float(row.get("center_y"))

            csv_anomaly_array_local.append({
                "image_name": csv_name,
                "center_x": cx,
                "center_y": cy,
                "anomaly": row.get("anomaly")
            })

    if not csv_anomaly_array_local:
        print("[DEBUG] No matching CSV rows found.")
    else:
        print(f"[DEBUG] Loaded {len(csv_anomaly_array_local)} anomalies from CSV.")


    return csv_anomaly_array_local

def write_csv(csv_path, rows):
    if not rows:
        print("[DEBUG] No rows to save.")
        return None
    base, ext = os.path.splitext(csv_path)
    new_csv_path = f"{base}_georeferenced{ext}"
    with open(new_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "anomaly_type", "latitude", "longitude",
            "example_image", "example_pixel_x", "example_pixel_y"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Georeferenced CSV saved to: {new_csv_path}")
    return new_csv_path

def add_anomaly_to_csv(output_array, anomaly_type, latitude, longitude,
                       example_image=None, example_pixel_x=None, example_pixel_y=None):
    output_array.append({
        "anomaly_type": anomaly_type,
        "latitude": latitude,
        "longitude": longitude,
        "example_image": example_image,
        "example_pixel_x": example_pixel_x,
        "example_pixel_y": example_pixel_y
    })
    print(f"[DEBUG] Added anomaly: type={anomaly_type}, lat={latitude}, lon={longitude}, example_image={example_image}")

# ---------------- Process single anomaly ----------------
def process_next_anomaly(row, image_data_list, build_corr_func, csv_path, original_image_size=None, dem_path=None, lidar_path=None):
    global csv_anomaly_array
    try:
        # original size
        center_x = row.get("center_x")
        center_y = row.get("center_y")

        image_name = row.get("image_name")
        # Find the index of the image in image_data_list
        idx = next((i for i, img in enumerate(image_data_list) if os.path.basename(img.get("image_name", "")).strip().lower() == image_name.strip().lower()), None)
        print(idx)
        print("idx")

        if idx is None:
            raise ValueError(f"Image '{image_name}' not found in image_data_list")
        

        # current image size
        img_w, img_h = image_data_list[idx].get("image_size", (512, 512))
        if center_x is None or center_y is None:
            center_x = img_w / 2
            center_y = img_h / 2


        #scale down to current image size for correspondance array
        scaled_x, scaled_y = anomaly_matching.scale_coordinates(center_x, center_y, from_size=original_image_size,to_size=(img_w, img_h))
        print(center_x, center_y)
        print(scaled_x, scaled_y)

        # Build correspondences
        correspondance_array = build_corr_func(idx, scaled_x, scaled_y, image_data_list=image_data_list)



        # Compute weighted average GPS
        processed_data, avg_lat, avg_lon = avg_gps.main(correspondance_array, dem_path=dem_path, lidar_path=lidar_path)
        if avg_lat is None or avg_lon is None:
            print("[DEBUG] Average GPS computation failed for this anomaly.")
            return None

        # Get matched anomalies
        matched_items = anomaly_matching.main(correspondance_array, csv_path=csv_path, original_image_size=original_image_size)
        print(f"[DEBUG] Found {len(matched_items)} matched items.")

        if not matched_items:
            return None

        # Determine majority anomaly_type
        types = [mi.get("anomaly", "unknown") for mi in matched_items if mi.get("anomaly")]
        anomaly_type = Counter(types).most_common(1)[0][0] if types else "unknown"

        # Find best-centered image
        best_item = None
        min_dist = float("inf")
        for mi in matched_items:
            img_name = mi.get("image_name")
            px = mi.get("pixel_x")
            py = mi.get("pixel_y")
            if img_name and px is not None and py is not None:
                entry = next((e for e in image_data_list if os.path.basename(e["image_name"]).strip().lower() == img_name), None)
                if entry:
                    w, h = entry.get("image_size", (512, 512))
                    if original_image_size:
                        px *= w / original_image_size[0]
                        py *= h / original_image_size[1]
                    cx, cy = w / 2, h / 2
                    dist = ((px - cx)**2 + (py - cy)**2) ** 0.5
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

        # Remove matched anomalies using distance-based removal
        remove_matched_anomalies(matched_items, tol=2.0)

    except Exception as e:
        print(f"[ERROR] Processing anomaly: {e}")
        traceback.print_exc()
        return None

# ---------------- Remove matched anomalies ----------------
def remove_matched_anomalies(matched_items, tol=2.0):
    global csv_anomaly_array
    new_array = []
    for row in csv_anomaly_array:
        row_x = row.get("center_x")
        row_y = row.get("center_y")
        img_name = os.path.basename(row.get("image_name", "")).strip().lower() if row.get("image_name") else None
        matched = False
        for item in matched_items:
            item_name = os.path.basename(item.get("image_name", "")).strip().lower()
            px = item.get("pixel_x")
            py = item.get("pixel_y")
            if img_name == item_name and row_x is not None and row_y is not None:
                dist = ((row_x - px) ** 2 + (row_y - py) ** 2) ** 0.5
                if dist <= tol:
                    matched = True
                    break
        if not matched:
            new_array.append(row)
    csv_anomaly_array = new_array
    print(f"[DEBUG] Remaining rows after removal: {len(csv_anomaly_array)}")

# ---------------- Main pipeline ----------------
def main(image_data_list, build_corr_func, original_image_size=None, output_file="batch_targets_map.html",
         dem_path=None, lidar_path=None, cad_path=None):
    global csv_anomaly_array
    try:
        Tk().withdraw()
        csv_path = askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            print("No CSV file selected. Exiting.")
            return

        csv_anomaly_array = read_csv(csv_path, image_data_list)
        if not csv_anomaly_array:
            return
        

        while csv_anomaly_array:
            row = csv_anomaly_array.pop(0)
            process_next_anomaly(row, image_data_list, build_corr_func, csv_path,
                                 original_image_size=original_image_size,
                                 dem_path=dem_path, lidar_path=lidar_path)

        print("All anomalies processed.")
        write_csv(csv_path, output_array)

        # --- Plot CAD map ---
        points = [{"target_gps": (r["latitude"], r["longitude"]), "score": 1.0}
                  for r in output_array if r["latitude"] is not None]
        if points and cad_path:
            central = points[0]["target_gps"]
            plot_cad.plot_cad_map(target_gps=central, points=points, cad_path=cad_path, output_file=output_file)

    except Exception as e:
        print(f"[ERROR] Exception in main_pipeline(): {e}")
        traceback.print_exc()
        return None
