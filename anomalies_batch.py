import os
import csv
import traceback
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import Counter
import avg_gps
import matching_anomalies  
import plot_cad

# ---------------- Globals ----------------
output_array = []
csv_anomaly_array = []




# ---------------- CSV Helpers ----------------
def read_csv(csv_path, image_data_list, original_image_size=None):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # Pick CSV column based on first image
    first_image_name = image_data_list[0]["image_name"]
    file_col = "wiris_image" if first_image_name.strip().lower().endswith(".tiff") else "pi_image"
    if file_col not in df.columns:
        raise ValueError(f"CSV does not contain expected column: {file_col}")

    image_names_set = {os.path.basename(e["image_name"]).strip().lower() for e in image_data_list}
    csv_anomaly_array_local = []

    for _, row in df.iterrows():
        csv_name = str(row[file_col]).strip().lower()
        if csv_name in image_names_set:
            row_dict = {}
            if "center_x" in row and "center_y" in row:
                cx = matching_anomalies.safe_float(row["center_x"])
                cy = matching_anomalies.safe_float(row["center_y"])
                # Optionally scale to original image size if current image size exists
                entry = next((e for e in image_data_list if os.path.basename(e["image_name"]).strip().lower() == csv_name), None)
                if entry:
                    current_size = entry.get("image_size", (512, 512))
                    if original_image_size:
                        cx, cy = matching_anomalies.scale_coordinates(cx, cy, current_size, original_image_size)
                row_dict["center_x"] = cx
                row_dict["center_y"] = cy
            row_dict["image_name"] = csv_name
            row_dict["anomaly"] = row.get("anomaly", None)
            csv_anomaly_array_local.append(row_dict)

    if not csv_anomaly_array_local:
        print(" No matching CSV rows found.")

    return csv_anomaly_array_local

def write_csv(csv_path, rows):
    """Write final CSV without file_name, including example-image info."""
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
            row_to_write = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(row_to_write)
    print(f"Georeferenced CSV saved to: {new_csv_path}")
    return new_csv_path

def add_anomaly_to_csv(output_array, anomaly_type, latitude, longitude,
                       example_image=None, example_pixel_x=None, example_pixel_y=None):
    entry = {
        "anomaly_type": anomaly_type,
        "latitude": latitude,
        "longitude": longitude,
        "example_image": example_image,
        "example_pixel_x": example_pixel_x,
        "example_pixel_y": example_pixel_y
    }
    output_array.append(entry)

# ---------------- Process single anomaly ----------------
def process_next_anomaly(row, image_data_list, build_corr_func, csv_path, dem_path=None, lidar_path=None, original_image_size=None):
    global csv_anomaly_array
    try:
        center_x = row.get("center_x")
        center_y = row.get("center_y")
        idx = 0  # always use first image

        img_w, img_h = image_data_list[idx].get("image_size", (512, 512))
        if center_x is None or center_y is None:
            center_x, center_y = img_w / 2, img_h / 2

        # Scale coordinates to original image size
        current_size = image_data_list[idx].get("image_size", (img_w, img_h))
        center_x, center_y = matching_anomalies.scale_coordinates(center_x, center_y, current_size, original_image_size)

        # Build correspondences
        correspondance_array = build_corr_func(
            idx,
            center_x,
            center_y,
            image_data_list=image_data_list
        )

        # Compute weighted average GPS
        processed_data, avg_lat, avg_lon = avg_gps.main(correspondance_array, dem_path=dem_path, lidar_path=lidar_path)
        if avg_lat is None or avg_lon is None:
            return None

        # Get matched anomalies
        matched_items = matching_anomalies.main(correspondance_array, csv_path=csv_path, original_image_size=original_image_size)
        if not matched_items:
            return None

        # Compute majority anomaly_type
        types = [mi.get("anomaly", "unknown") for mi in matched_items if mi.get("anomaly")]
        anomaly_type = Counter(types).most_common(1)[0][0] if types else "unknown"

        # Find image where anomaly is most centered
        best_item = None
        min_dist = float("inf")
        for mi in matched_items:
            img_name = mi.get("image_name")
            px = mi.get("pixel_x")
            py = mi.get("pixel_y")
            if img_name and px is not None and py is not None:
                entry = next(
                    (e for e in image_data_list
                     if os.path.basename(e["image_name"]).strip().lower()
                        == os.path.basename(img_name).strip().lower()),
                    None
                )
                if entry:
                    w, h = entry.get("image_size", (512, 512))
                    # Scale pixel coordinates to current image size
                    px_scaled, py_scaled = px, py
                    if original_image_size is not None:
                        px_scaled, py_scaled = matching_anomalies.scale_coordinates(px, py, original_image_size, (w, h))
                    cx, cy = w / 2, h / 2
                    dist = ((px_scaled - cx) ** 2 + (py_scaled - cy) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_item = {
                            "example_image": os.path.basename(img_name),
                            "example_pixel_x": px_scaled,
                            "example_pixel_y": py_scaled
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

        # Remove matched anomalies from CSV array
        remove_matched_anomalies(matched_items, original_image_size=original_image_size)

    except Exception as e:
        print(f"[ERROR] Processing anomaly: {e}")
        traceback.print_exc()
        return None

# ---------------- Remove processed anomalies ----------------
def remove_matched_anomalies(matched_items, original_image_size=None):
    global csv_anomaly_array
    to_remove = set()
    for item in matched_items:
        px = item.get("pixel_x")
        py = item.get("pixel_y")
        img_name = os.path.basename(item.get("image_name", "unknown")).strip().lower()
        if px is not None and py is not None:
            if original_image_size is not None and item.get("current_image_size"):
                px, py = matching_anomalies.scale_coordinates(px, py, original_image_size, item["current_image_size"])
            key = (img_name, int(round(px)), int(round(py)))
            to_remove.add(key)

    new_array = []
    for row in csv_anomaly_array:
        row_x = row.get("center_x")
        row_y = row.get("center_y")
        key = (None, int(round(row_x)) if row_x is not None else None, int(round(row_y)) if row_y is not None else None)
        if key not in to_remove:
            new_array.append(row)

    csv_anomaly_array = new_array
    print(f" Remaining rows: {len(csv_anomaly_array)}")

# ---------------- Main pipeline ----------------
def main(image_data_list, build_corr_func, dem_path=None, lidar_path=None, original_image_size=None):
    global csv_anomaly_array
    try:
        Tk().withdraw()
        csv_path = askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            print("No CSV file selected. Exiting.")
            return

        csv_anomaly_array = read_csv(csv_path, image_data_list, original_image_size=original_image_size)
        if not csv_anomaly_array:
            return

        while csv_anomaly_array:
            row = csv_anomaly_array.pop(0)
            process_next_anomaly(row, image_data_list, build_corr_func, csv_path,
                                 dem_path=dem_path, lidar_path=lidar_path, original_image_size=original_image_size)

        print(" All anomalies processed.")
        write_csv(csv_path, output_array)

        # --- Plot CAD map ---
        points = [{"target_gps": (r["latitude"], r["longitude"]), "score": 1.0}
                  for r in output_array if r["latitude"] is not None]
        if points:
            central = points[0]["target_gps"]
            plot_cad.plot_cad_map(
                target_gps=central,
                points=points,
                geojson_file="panels_with_row_plaintext_below.geojson",
                map_file="batch_targets_map.html"
            )

    except Exception as e:
        print(f"[ERROR] Exception in main_pipeline(): {e}")
        traceback.print_exc()
        return None
