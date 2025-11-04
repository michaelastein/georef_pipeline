import os
import tkinter as tk
from tkinter import filedialog, messagebox
import traceback
import matching_anomalies
import avg_gps
import plot_cad
import csv

MAX_DISTANCE_M = 15.0

def main(image_data_list, output_csv_path=None):
    """
    Batch processing of anomalies using matching_anomalies.main()
    - Computes average GPS per anomaly
    - Exports results to CSV
    - Generates CAD map
    """

    if not image_data_list:
        return

    root = tk.Tk()
    root.withdraw()

    # --- Select anomaly CSV ---
    csv_path = filedialog.askopenfilename(
        title="Select anomaly CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not csv_path:
        return

    try:
        # --- Get all matched items using matching_anomalies.main() ---
        matched_items = matching_anomalies.main(
            image_data_list=image_data_list,
            csv_path=csv_path,
            max_distance_m=MAX_DISTANCE_M,
            show_gui=False
        )

        if not matched_items:
            messagebox.showinfo("Batch Processing", "No matches found in CSV.")
            return

        # --- Compute average GPS per anomaly and prepare output rows ---
        output_rows = []
        for item in matched_items:
            for row in item.get("rows", []):
                result = avg_gps.main([row])  # returns (data, avg_lat, avg_lon)


                _, lat, lon = result  # unpack correctly

                output_rows.append({
                    "anomaly_type": row.get("anomaly", "unknown"),
                    "avg_lat": lat,
                    "avg_lon": lon,
                    "example_image": item["path"],
                    "pixel_x": row.get("pixel_x"),
                    "pixel_y": row.get("pixel_y")
                })

        if not output_rows:
            messagebox.showinfo("Batch Processing", "No valid GPS points found.")
            return

        # --- Write output CSV ---
        if output_csv_path is None:
            output_csv_path = os.path.splitext(csv_path)[0] + "_georeferenced.csv"

        keys = ["anomaly_type", "avg_lat", "avg_lon", "example_image", "pixel_x", "pixel_y"]
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(output_rows)

        messagebox.showinfo("Batch Processing Done", f"Results saved to: {output_csv_path}")

        # --- Plot CAD map ---
        points = [{"target_gps": (r["avg_lat"], r["avg_lon"]), "score": 1.0} 
                  for r in output_rows if r["avg_lat"] is not None]
        if points:
            central = points[0]["target_gps"]
            plot_cad.plot_cad_map(
                target_gps=central,
                points=points,
                geojson_file="panels_with_row_plaintext_below.geojson",
                map_file="batch_targets_map.html"
            )

    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        traceback.print_exc()
        messagebox.showerror("Batch Processing Error", str(e))

