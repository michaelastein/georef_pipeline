import os
import csv
import traceback
import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import avg_gps
import anomaly_matching
import plot_cad

# ---------------- Globals ----------------
output_array = []
csv_anomaly_array = []

# ---------------- Helpers ----------------
def safe_float(x):
    """
    Convert x to float safely. Returns None if conversion fails.
    Useful for reading CSV numeric columns that may contain missing/invalid data.
    """
    try:
        return float(x)
    except Exception:
        return None


# ---------------- CSV Helpers ----------------
def read_csv(csv_path: str, image_data_list: list) -> list:
    """
    Read CSV and extract anomalies that correspond to images in `image_data_list`.
    
    Args:
        csv_path: Path to CSV file.
        image_data_list: List of dicts containing image names and metadata.
    
    Returns:
        List of dicts with keys:
            - image_name
            - center_x, center_y
            - anomaly
            - normalized_name (lowercased basename)
    """
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # Determine which CSV column contains image filenames
    first_image_name = image_data_list[0]["image_name"]
    file_col = "wiris_image" if first_image_name.strip().lower().endswith(".tiff") else "pi_image"
    if file_col not in df.columns:
        raise ValueError(f"CSV does not contain expected column: {file_col}")

    # Map normalized basenames to image entries for quick lookup
    image_name_map = {os.path.basename(img["image_name"]).strip().lower(): img for img in image_data_list}

    # Normalize CSV filenames and filter to only those present in image_data_list
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
            "anomaly": row.get("anomaly"),
            "normalized_name": img_name
        })
    return csv_anomalies

def write_csv(csv_path: str, rows: list) -> str:
    """
    Write georeferenced anomaly rows to a new CSV file.

    Args:
        csv_path: Original CSV path (used to derive output filename)
        rows: List of dicts with anomaly information

    Returns:
        Path to the saved CSV, or None if no rows.
    """
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


def add_anomaly_to_csv(output_array: list, anomaly_type: str, latitude: float, longitude: float, example_image=None, example_pixel_x=None, example_pixel_y=None):
    """
    Add a single anomaly record to the output array (to later write CSV).

    Args:
        output_array: list to append to
        anomaly_type: string describing the anomaly
        latitude, longitude: georeferenced coordinates
        example_image: optional image filename
        example_pixel_x, example_pixel_y: optional pixel location in image
    """
    output_array.append({
        "anomaly_type": anomaly_type,
        "latitude": latitude,
        "longitude": longitude,
        "example_image": example_image,
        "example_pixel_x": example_pixel_x,
        "example_pixel_y": example_pixel_y
    })


# ---------------- Remove matched anomalies ----------------
def remove_matched_anomalies(matched_items: list, tol=2.0):
    """
    Remove anomalies from the global csv_anomaly_array that have already been matched.
    Matching is done by image basename and pixel coordinates.

    Args:
        matched_items: List of dicts with matched anomalies
        tol: tolerance (currently unused, could be for fuzzy matching)
    """
    global csv_anomaly_array
    if not csv_anomaly_array or not matched_items:
        return

    # Build set of matched keys: (basename, x, y)
    matched_set = set(
        (os.path.basename(mi.get("image_name", "")).strip().lower(), mi.get("pixel_x"), mi.get("pixel_y"))
        for mi in matched_items
    )

    # Filter out matched anomalies from global array
    csv_anomaly_array = [
        row for row in csv_anomaly_array
        if (os.path.basename(row.get("image_name", "")).strip().lower(), row.get("center_x"), row.get("center_y")) not in matched_set
    ]


# ---------------- Process single anomaly ----------------
def process_next_anomaly(row: dict, image_data_list: list, build_corr_func, csv_path: str, image_map: dict,
                         original_image_size=None, dem_path=None, lidar_path=None):
    """
    Process one anomaly row: scale pixel coordinates, build correspondence array,
    compute averaged GPS, determine anomaly type, select best example image, 
    and append result to output_array.

    Args:
        row: CSV row or anomaly dict
        image_data_list: list of all images with metadata
        build_corr_func: function to compute correspondence array for the anomaly
        csv_path: path to CSV for matching anomalies
        image_map: dict mapping lowercase basenames -> image entries
        original_image_size: optional (width, height) for scaling
        dem_path/lidar_path: optional paths for elevation data
    """
    global output_array
    try:
        # --- Step 1: Extract and scale pixel coordinates ---
        center_x = row.get("center_x")
        center_y = row.get("center_y")
        image_name = row.get("image_name")

        img_entry = image_map.get(image_name.lower())
        if img_entry is None:
            raise ValueError(f"Image '{image_name}' not found in image_data_list")

        img_w, img_h = img_entry.get("image_size", (512, 512))
        if center_x is None or center_y is None:
            # Default to image center
            center_x, center_y = img_w / 2, img_h / 2

        scaled_x, scaled_y = anomaly_matching.scale_coordinates(
            center_x, center_y, from_size=original_image_size, to_size=(img_w, img_h)
        )

        # --- Step 2: Build correspondence array ---
        idx_img = image_data_list.index(img_entry)
        correspondence_array = build_corr_func(idx_img, scaled_x, scaled_y, image_data_list=image_data_list)

        # --- Step 3: Compute weighted average GPS ---
        processed_data, avg_lat, avg_lon = avg_gps.main(correspondence_array, dem_path=dem_path, lidar_path=lidar_path)
        if avg_lat is None or avg_lon is None:
            return None

        # --- Step 4: Match anomalies in CSV ---
        matched_items = anomaly_matching.main(correspondence_array, csv_path=csv_path, original_image_size=original_image_size)
        if not matched_items:
            return None

        # --- Step 5: Determine majority anomaly type ---
        types = [mi.get("anomaly", "unknown") for mi in matched_items if mi.get("anomaly")]
        anomaly_type = Counter(types).most_common(1)[0][0] if types else "unknown"

        # --- Step 6: Find best-centered example image ---
        valid_items = [
            mi for mi in matched_items
            if mi.get("pixel_x") is not None and mi.get("pixel_y") is not None and mi.get("image_name")
        ]
        best_item = None
        if valid_items:
            px = np.array([mi["pixel_x"] for mi in valid_items], dtype=float)
            py = np.array([mi["pixel_y"] for mi in valid_items], dtype=float)
            img_names = [mi["image_name"] for mi in valid_items]

            dists = []
            for i, name in enumerate(img_names):
                entry2 = image_map.get(name.lower())
                w, h = entry2.get("image_size", (512, 512))
                if original_image_size:
                    px_scaled, py_scaled = anomaly_matching.scale_coordinates(px[i], py[i], from_size=original_image_size, to_size=(w, h))
                else:
                    px_scaled, py_scaled = px[i], py[i]
                # Euclidean distance from center
                dists.append(np.hypot(px_scaled - w/2, py_scaled - h/2))

            min_idx = np.argmin(dists)
            best_item = {
                "example_image": os.path.basename(img_names[min_idx]),
                "example_pixel_x": px[min_idx],
                "example_pixel_y": py[min_idx]
            }

        # --- Step 7: Append georeferenced anomaly to global output ---
        add_anomaly_to_csv(
            output_array,
            anomaly_type,
            avg_lat,
            avg_lon,
            example_image=best_item.get("example_image") if best_item else None,
            example_pixel_x=best_item.get("example_pixel_x") if best_item else None,
            example_pixel_y=best_item.get("example_pixel_y") if best_item else None
        )

        # --- Step 8: Remove matched anomalies from CSV anomaly array ---
        remove_matched_anomalies(matched_items)

    except Exception as e:
        print(f"[ERROR] Processing anomaly: {e}")
        traceback.print_exc()
        return None
    



def main(image_data_list, build_corr_func, print_progress_func, original_image_size=None, 
         output_file="batch_targets_map.html", dem_path=None, lidar_path=None, cad_path=None,
         max_workers=8):
    """
    Main pipeline to georeference anomalies from images, using parallel processing
    and real-time progress updates.

    Steps:
        1. Prompt user to select a CSV file containing anomalies.
        2. Filter anomalies to only those corresponding to loaded images.
        3. Precompute image name -> metadata mapping.
        4. Process anomalies in parallel using ThreadPoolExecutor:
            - Scale pixel coordinates
            - Build correspondence array
            - Compute weighted average GPS
            - Match anomalies in CSV
            - Determine majority anomaly type
            - Select best example image
        5. Update progress via print_progress_func as each anomaly finishes.
        6. Write georeferenced results to a new CSV.
        7. Optionally, plot anomalies on a CAD map.

    Args:
        image_data_list (list[dict]): List of image metadata dictionaries.
        build_corr_func (callable): Function to build correspondence array for an anomaly.
        print_progress_func (callable): Function to display progress. Called as:
            print_progress_func(current, total, stage_name, start_time, bar_length, update_every_percent)
        original_image_size (tuple, optional): Original image size for scaling (width, height).
        output_file (str): Output HTML filename for plotting CAD map.
        dem_path (str, optional): Path to DEM file for GPS computation.
        lidar_path (str, optional): Path to LiDAR data for GPS computation.
        cad_path (str, optional): Path to CAD file for overlay.
        max_workers (int): Number of threads for parallel processing.

    Returns:
        str | None: Path to saved georeferenced CSV file, or None if no results.
    """
    global csv_anomaly_array
    try:
        # Hide Tkinter main window and prompt for CSV selection
        Tk().withdraw()
        csv_path = askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            print("No CSV file selected. Exiting.")
            return

        # --- Step 1: Read CSV and filter anomalies ---
        csv_anomaly_array = read_csv(csv_path, image_data_list)
        if not csv_anomaly_array:
            print("No matching anomalies found in CSV. Exiting.")
            return

        total_anomalies = len(csv_anomaly_array)
        print(f"Processing {total_anomalies} anomalies...")
        start_time = time.time()

        # --- Step 2: Precompute image name -> metadata mapping ---
        image_map = {os.path.basename(img["image_name"]).strip().lower(): img for img in image_data_list}

        # --- Step 3: Define worker function for parallel processing ---
        def worker(row):
            process_next_anomaly(
                row,
                image_data_list,
                build_corr_func,
                csv_path,
                image_map,
                original_image_size=original_image_size,
                dem_path=dem_path,
                lidar_path=lidar_path
            )
            return 1
            # return 1 to increment progress counter

        # --- Step 4: Run parallel processing with progress updates ---
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, row) for row in csv_anomaly_array]
            for future in as_completed(futures):
                completed += future.result()
                if print_progress_func:
                    print_progress_func(
                        current=completed,
                        total=total_anomalies,
                        stage_name="Processing anomalies",
                        start_time=start_time,
                        bar_length=40,
                        update_every_percent=1
                    )

        print("All anomalies processed.")

        # --- Step 5: Write georeferenced results to CSV ---
        georef_csv_path = write_csv(csv_path, output_array)

        # --- Step 6: Optionally, plot anomalies on CAD map ---
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
