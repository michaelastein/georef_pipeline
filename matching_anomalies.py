import os
import pandas as pd

# ==== Helpers ====
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def find_exact_row_for_first_image(df, filename_col, filename, pixel_x, pixel_y):
    candidate_rows = df[df[filename_col].astype(str).str.strip() == filename]
    if candidate_rows.empty:
        return None
    tol = 30
    for _, r in candidate_rows.iterrows():
        cx = safe_float(r.get("center_x"))
        cy = safe_float(r.get("center_y"))
        if cx is None or cy is None:
            continue
        if abs(cx - pixel_x) <= tol and abs(cy - pixel_y) <= tol:
            return r
    return None

def compute_max_deviation_from_bbox(row, fraction=1/2.0):
    xmin = safe_float(row.get("xmin"))
    xmax = safe_float(row.get("xmax"))
    ymin = safe_float(row.get("ymin"))
    ymax = safe_float(row.get("ymax"))
    if None in (xmin, xmax, ymin, ymax):
        return 10.0
    w = abs(xmax - xmin)
    h = abs(ymax - ymin)
    return max(w, h) * fraction if max(w, h) > 0 else 10.0

def get_matching_rows_for_image(df, image_name, max_dev, filename_col, correspondence_array):
    matches = []
    if filename_col not in df.columns:
        return matches

    # Filter only rows for the given image name
    candidate_rows = df[df[filename_col].astype(str).str.strip() == image_name]
    if candidate_rows.empty:
        return matches

    # Get the target entry from correspondence_array
    target_entry = next((e for e in correspondence_array if os.path.basename(e["image_name"]) == image_name), None)
    if target_entry is None:
        print(f"[DEBUG] No target entry found for {image_name}")
        return matches

    target_x = target_entry["pixel_x"]
    target_y = target_entry["pixel_y"]

    for _, row in candidate_rows.iterrows():
        cx = safe_float(row.get("center_x"))
        cy = safe_float(row.get("center_y"))
        if cx is None or cy is None:
            continue
        if abs(cx - target_x) <= max_dev and abs(cy - target_y) <= max_dev:
            matches.append({
                "image_name": image_name,
                "pixel_x": cx,
                "pixel_y": cy,
                "anomaly": row.get("anomaly")
            })
        

    return matches

# ==== Main function ====
def main(correspondence_array, csv_path):
    """
    correspondence_array: list of dicts, each dict has keys:
        - image_name
        - pixel_x
        - pixel_y
    csv_path: path to CSV file with anomaly data
    Returns: list of dicts with image_name, pixel_x, pixel_y, anomaly
    """



    if not correspondence_array:
        return []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    first_entry = correspondence_array[0]
    first_filename = os.path.basename(first_entry["image_name"])
    first_pixel_x = float(first_entry["pixel_x"])
    first_pixel_y = float(first_entry["pixel_y"])

    # Determine filename column
    ext = os.path.splitext(first_filename)[1].lower()
    filename_col = "wiris_image" if ext == ".tiff" else "pi_image"

    first_row = find_exact_row_for_first_image(df, filename_col, first_filename, first_pixel_x, first_pixel_y)
    if first_row is None:
        raise ValueError(f"No matching row found for first image '{first_filename}' with pixel_x/pixel_y.")

    max_dev = compute_max_deviation_from_bbox(first_row)
    results = []

    for entry in correspondence_array[1:]:
        bname = os.path.basename(entry["image_name"])
        matches = get_matching_rows_for_image(df, bname, max_dev, filename_col, correspondence_array)
        if matches:
            results.extend(matches)


    return results
