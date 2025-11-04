import os
import csv
from collections import Counter
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import matching_anomalies
import avg_gps
import georef_new
import plot_cad

MAX_DISTANCE_M = 15.0

def main(image_data_list, output_csv_path=None):
    """
    Batch processing of anomalies with distance filtering, CSV export, and CAD map.
    """
    root = tk.Tk()
    root.withdraw()
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

        data_by_name = {os.path.basename(e["image_path"]): e for e in image_data_list}
        mapper = georef_new.DroneMapper()
        total_rows = len(df_anom)
        print(f"Starting batch processing for {total_rows} anomalies...")

        for idx, row in df_anom.iterrows():
            image_name = str(row["image"]).strip()
            anomaly_type = str(row.get("anomaly", "unknown")).strip()
            key = (image_name, anomaly_type)
            if key in processed_anomalies or image_name not in data_by_name:
                continue

            print(f"[{idx+1}/{total_rows}] Processing anomaly '{anomaly_type}' in image '{image_name}'")

            max_dev = matching_anomalies.compute_max_deviation_from_bbox(row)

            data_point = {image_name: {"pixel_x": float(row["center_x"]), "pixel_y": float(row["center_y"])}}
            matching_rows = matching_anomalies.get_matching_rows_for_image(df_anom, image_name, max_dev, "image", data_point)
            if not matching_rows:
                matching_rows = [row]

            # avg_gps calculation
            data_for_avg = []
            for r in matching_rows:
                img_path = next((e["image_path"] for e in image_data_list if os.path.basename(e["image_path"]) == image_name), None)
                if img_path:
                    data_for_avg.append({"image_path": img_path, "pixel_x": float(r["center_x"]), "pixel_y": float(r["center_y"])})

            avg_result = avg_gps.main(data_for_avg)

            ref_gps = None
            for d in data_for_avg:
                if 'target_gps' in d:
                    ref_gps = d['target_gps']
                    break
            if ref_gps is None and avg_result.get("lat") is not None:
                ref_gps = (avg_result["lat"], avg_result["lon"])

            if ref_gps is None:
                filtered_rows = matching_rows
            else:
                filtered_rows = matching_anomalies.filter_rows_by_geodistance(
                    matching_rows, image_data_list, image_name, ref_gps, mapper, MAX_DISTANCE_M, avg_gps.haversine
                )

            if not filtered_rows:
                print(f"No valid matches within {MAX_DISTANCE_M} m for {image_name}, skipping")
                continue

            # Majority anomaly type
            types = [str(r.get("anomaly", "unknown")) for r in filtered_rows]
            majority_type = Counter(types).most_common(1)[0][0]

            center_row = min(
                filtered_rows,
                key=lambda r: ((float(r["center_x"]) - float(row["center_x"]))**2 + (float(r["center_y"]) - float(row["center_y"]))**2)
            )
            example_img_path = next((e["image_path"] for e in image_data_list if os.path.basename(e["image_path"]) == image_name), None)
            example_pixel_x = float(center_row["center_x"])
            example_pixel_y = float(center_row["center_y"])

            output_rows.append({
                "anomaly_type": majority_type,
                "avg_lat": avg_result.get("lat"),
                "avg_lon": avg_result.get("lon"),
                "example_image": example_img_path,
                "pixel_x": example_pixel_x,
                "pixel_y": example_pixel_y
            })
            processed_anomalies.add(key)

        if output_csv_path is None:
            output_csv_path = os.path.splitext(csv_path)[0] + "_georeferenced.csv"
        keys = ["anomaly_type", "avg_lat", "avg_lon", "example_image", "pixel_x", "pixel_y"]
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(output_rows)

        print(f"Batch processing done. Results saved to: {output_csv_path}")
        messagebox.showinfo("Batch Processing Done", f"Results saved to: {output_csv_path}")

        # CAD Map
        target_points = [{"target_gps": (row["avg_lat"], row["avg_lon"]), "score": 1.0} for row in output_rows if row["avg_lat"] is not None]
        if target_points:
            central_target = (float(target_points[0]["target_gps"][0]), float(target_points[0]["target_gps"][1]))
            plot_cad.plot_cad_map(
                target_gps=central_target,
                points=target_points,
                geojson_file="panels_with_row_plaintext_below.geojson",
                map_file="batch_targets_map.html"
            )

    except Exception as e:
        print(f"Error during batch processing: {e}")
        messagebox.showerror("Batch Processing Error", str(e))
