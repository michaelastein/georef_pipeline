# batch_anomalies.py
import os
import csv
from collections import Counter
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

import avg_gps
import matching_anomalies

def main(image_data_list, output_csv_path=None):
    """
    Batch process anomalies using a CSV selected by the user, with progress updates:
    - image_data_list: list of dicts, each with at least "image_path"
    - output_csv_path: optional path to save output CSV
    """
    # Ask user for the anomaly CSV
    root = tk.Tk()
    root.withdraw()  # Hide main window
    csv_path = filedialog.askopenfilename(
        title="Select anomaly CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not csv_path:
        print("No CSV selected. Batch processing canceled.")
        return

    try:
        df_anom = pd.read_csv(csv_path, dtype=str, low_memory=False)
        processed_anomalies = set()
        output_rows = []

        # Quick lookup for images
        data_by_name = {os.path.basename(e["image_path"]): e for e in image_data_list}

        total_rows = len(df_anom)
        print(f"Starting batch processing for {total_rows} anomalies...")

        for idx, row in enumerate(df_anom, start=1):
            image_name = str(row["image"]).strip()
            anomaly_type = str(row.get("anomaly", "unknown")).strip()
            key = (image_name, anomaly_type)
            if key in processed_anomalies:
                continue

            # Skip if image not in data
            if image_name not in data_by_name:
                continue

            # Progress update
            print(f"[{idx}/{total_rows}] Processing anomaly '{anomaly_type}' in image '{image_name}'")

            # Max deviation from bbox
            max_dev = matching_anomalies.compute_max_deviation_from_bbox(row)

            # Find all matching anomalies
            data_point = {
                image_name: {
                    "pixel_x": float(row["center_x"]),
                    "pixel_y": float(row["center_y"])
                }
            }
            matching_rows = matching_anomalies.get_matching_rows_for_image(
                df_anom, image_name, max_dev, "image", data_point
            )
            if not matching_rows:
                matching_rows = [row]

            # Prepare data for avg_gps
            data_for_avg = []
            for r in matching_rows:
                img_path = next(
                    (e["image_path"] for e in image_data_list if os.path.basename(e["image_path"]) == image_name),
                    None
                )
                if img_path:
                    data_for_avg.append({
                        "image_path": img_path,
                        "pixel_x": float(r["center_x"]),
                        "pixel_y": float(r["center_y"])
                    })

            # Call avg_gps
            try:
                avg_result = avg_gps.main(data_for_avg)
            except Exception as e:
                print(f"avg_gps error for {image_name}: {e}")
                avg_result = {"lat": None, "lon": None}

            # Majority anomaly type
            types = [str(r.get("anomaly", "unknown")) for r in matching_rows]
            majority_type = Counter(types).most_common(1)[0][0]

            # Example image (most centered)
            center_row = min(
                matching_rows,
                key=lambda r: ((float(r["center_x"]) - float(row["center_x"]))**2 +
                               (float(r["center_y"]) - float(row["center_y"]))**2)
            )
            example_img_path = next(
                (e["image_path"] for e in image_data_list if os.path.basename(e["image_path"]) == image_name),
                None
            )
            example_pixel_x = float(center_row["center_x"])
            example_pixel_y = float(center_row["center_y"])

            # Append to output
            output_rows.append({
                "anomaly_type": majority_type,
                "avg_lat": avg_result.get("lat"),
                "avg_lon": avg_result.get("lon"),
                "example_image": example_img_path,
                "pixel_x": example_pixel_x,
                "pixel_y": example_pixel_y
            })

            processed_anomalies.add(key)

        # Determine output CSV path
        if output_csv_path is None:
            output_csv_path = os.path.splitext(csv_path)[0] + "_georeferenced.csv"

        # Save results
        keys = ["anomaly_type", "avg_lat", "avg_lon", "example_image", "pixel_x", "pixel_y"]
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(output_rows)

        print(f"Batch processing done. Results saved to: {output_csv_path}")
        messagebox.showinfo("Batch Processing Done", f"Results saved to: {output_csv_path}")

    except Exception as e:
        print(f"Error during batch processing: {e}")
        messagebox.showerror("Batch Processing Error", str(e))
