import os
import csv
import traceback
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import avg_gps
import matching_anomalies  # should return flat matched_items
from collections import Counter
import plot_cad

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

    # Pick CSV column based on first image
    first_image_name = image_data_list[0]["image_name"]
    if first_image_name.strip().lower().endswith(".tiff"):
        file_col = "wiris_image"
    else:
        file_col = "pi_image"

    if file_col not in df.columns:
        raise ValueError(f"CSV does not contain expected column: {file_col}")

    print(f"[DEBUG] Using CSV column: {file_col}")

    image_names_set = {os.path.basename(e["image_name"]).strip().lower() for e in image_data_list}

    csv_anomaly_array_local = []
    for _, row in df.iterrows():
        csv_name = str(row[file_col]).strip().lower()
        if csv_name in image_names_set:
            row_dict = {"file_name": csv_name}
            if "center_x" in row: row_dict["center_x"] = safe_float(row["center_x"])
            if "center_y" in row: row_dict["center_y"] = safe_float(row["center_y"])
            csv_anomaly_array_local.append(row_dict)

    if not csv_anomaly_array_local:
        print("[DEBUG] No matching CSV rows found.")

    return csv_anomaly_array_local

def write_csv(csv_path, rows):
    if not rows:
        print("[DEBUG] No rows to save.")
        return None
    base, ext = os.path.splitext(csv_path)
    new_csv_path = f"{base}_georeferenced{ext}"
    with open(new_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "anomaly_type", "latitude", "longitude"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[DEBUG] Georeferenced CSV saved to: {new_csv_path}")
    return new_csv_path

def add_anomaly_to_csv(output_array, file_name, anomaly_type, latitude, longitude):
    output_array.append({
        "file_name": file_name,
        "anomaly_type": anomaly_type,
        "latitude": latitude,
        "longitude": longitude
    })

# ---------------- Process single anomaly ----------------
def process_next_anomaly(row, image_data_list, build_corr_func, csv_path):
    global csv_anomaly_array
    try:
        file_name = row["file_name"]

        # Find the index of the image in image_data_list
        idx = next((i for i, e in enumerate(image_data_list)
                    if os.path.basename(e["image_name"]).strip().lower() == file_name.strip().lower()), None)
        if idx is None:
            print(f"[DEBUG] Skipping {file_name}: image not found in image_data_list")
            return None

        # Get pixel coordinates from CSV row
        center_x = row.get("center_x")
        center_y = row.get("center_y")
        if center_x is None or center_y is None:
            img_w, img_h = image_data_list[idx].get("image_size", (512, 512))
            center_x = img_w / 2
            center_y = img_h / 2
            print(f"[DEBUG] Missing center coordinates for {file_name}, using image center: ({center_x}, {center_y})")
        else:
            print(f"[DEBUG] Using CSV pixel coordinates for {file_name}: ({center_x}, {center_y})")

        # Build correspondences
        correspondance_array = build_corr_func(
            idx,
            center_x,
            center_y,
            image_data_list=image_data_list
        )

        # Compute weighted average GPS
        processed_data, avg_lat, avg_lon = avg_gps.main(correspondance_array)
        if avg_lat is None or avg_lon is None:
            print(f"[DEBUG] No valid GPS for {file_name}")
            return None

        # Get matched anomalies
        matched_items = matching_anomalies.main(correspondance_array, csv_path=csv_path)
        for mi in matched_items:
            print(f"[DEBUG] Matched item: {mi.get('image_name')}, pixel=({mi.get('pixel_x')},{mi.get('pixel_y')})")

        if matched_items:
            # Compute majority anomaly_type
            types = [mi.get("anomaly", "unknown") for mi in matched_items if mi.get("anomaly")]
            anomaly_type = Counter(types).most_common(1)[0][0] if types else "unknown"

            add_anomaly_to_csv(output_array, row["file_name"], anomaly_type, avg_lat, avg_lon)

            # Remove matched anomalies from CSV array
            remove_matched_anomalies(matched_items)
        else:
            print(f"[DEBUG] Skipping failed row: {row['file_name']}")

    except Exception as e:
        print(f"[ERROR] Processing anomaly {row.get('file_name', 'unknown')}: {e}")
        traceback.print_exc()
        return None

# ---------------- Remove processed anomalies ----------------
def remove_matched_anomalies(matched_items):
    global csv_anomaly_array
    print(f"[DEBUG] Number of matched items: {len(matched_items)}")

    # Remove only exact matches by (file_name, pixel_x, pixel_y)
    to_remove = set()
    for item in matched_items:
        px = item.get("pixel_x")
        py = item.get("pixel_y")
        img_name = os.path.basename(item.get("image_name", "unknown")).strip().lower()
        if px is not None and py is not None:
            key = (img_name, int(round(px)), int(round(py)))
            to_remove.add(key)

    new_array = []
    for row in csv_anomaly_array:
        row_x = row.get("center_x")
        row_y = row.get("center_y")
        row_name = row.get("file_name")
        key = (
            row_name.lower(),
            int(round(row_x)) if row_x is not None else None,
            int(round(row_y)) if row_y is not None else None
        )

        if key in to_remove:
            print(f"[DEBUG] Removing row: {row_name}, center_x={row_x}, center_y={row_y}, key={key}")
        else:
            new_array.append(row)

    csv_anomaly_array = new_array
    print(f"[DEBUG] Remaining rows after removal: {len(csv_anomaly_array)}")

# ---------------- Main pipeline ----------------
def main(image_data_list, build_corr_func):
    global csv_anomaly_array
    try:
        Tk().withdraw()
        csv_path = askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            print("[DEBUG] No CSV file selected. Exiting.")
            return

        csv_anomaly_array = read_csv(csv_path, image_data_list)
        if not csv_anomaly_array:
            return

        while csv_anomaly_array:
            row = csv_anomaly_array.pop(0)  # remove the row being processed
            print(f"[DEBUG] Processing row: {row}")
            process_next_anomaly(row, image_data_list, build_corr_func, csv_path)

        print("[DEBUG] All anomalies processed.")
        write_csv(csv_path, output_array)

        
        # --- Plot CAD map ---
        points = [{"target_gps": (r["latitude"], r["longitude"]), "score": 1.0} 
                  for r in output_array if r["latitude"] is not None]
        if points:
            central = points[0]["target_gps"]
            print(f"[DEBUG] Plotting CAD map at central target GPS: {central}")
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
