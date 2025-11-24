import os
import pandas as pd

# ==== Helpers ====
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def scale_coordinates(x, y, from_size, to_size):
    """
    Scale coordinates between two image sizes.
    
    from_size: (width, height) of the original coordinate space
    to_size: (width, height) of the target coordinate space
    """
    if from_size is None or to_size is None:
        return x, y

    from_w, from_h = from_size
    to_w, to_h = to_size

    if from_w == 0 or from_h == 0:
        return x, y

    scale_x = to_w / from_w
    scale_y = to_h / from_h

    return int(round(x * scale_x)), int(round(y * scale_y))

def max_deviation(df, filename_col, correspondence_array, fraction=0.5):
    """
    Finds the first image from correspondence_array that exists in the CSV
    and computes the max deviation from its bounding box, ignoring pixel coordinates.

    Parameters:
        df: pandas DataFrame containing CSV data
        filename_col: column name in CSV for image filenames
        correspondence_array: list of dicts with 'image_name'
        fraction: fraction of bounding box size to use as max deviation

    Returns:
        float: max deviation (default 10.0 if no matching image found)
    """
    for entry in correspondence_array:
        filename = os.path.basename(entry["image_name"])
        candidate_rows = df[df[filename_col].astype(str).str.strip() == filename]
        if candidate_rows.empty:
            continue  # image not in CSV, try next one

        # Take the first row for this image
        r = candidate_rows.iloc[0]

        xmin = safe_float(r.get("xmin"))
        xmax = safe_float(r.get("xmax"))
        ymin = safe_float(r.get("ymin"))
        ymax = safe_float(r.get("ymax"))
        if None in (xmin, xmax, ymin, ymax):
            return 10.0
        w = abs(xmax - xmin)
        h = abs(ymax - ymin)
        return max(w, h) * fraction if max(w, h) > 0 else 10.0

    print("use default deviation")
    return 20.0  # default if no image found in CSV

   

def get_matching_rows_for_image(df, entry, max_dev, filename_col, original_image_size=None):
    """
    Return CSV rows matching the given correspondence entry (image + coordinates)
    """
    matches = []
    image_name = os.path.basename(entry["image_name"])

    if filename_col not in df.columns:
        return matches

    candidate_rows = df[df[filename_col].astype(str).str.strip() == image_name]
    if candidate_rows.empty:
        return matches

    # target coordinates
    orig_x = entry["pixel_x"]
    orig_y = entry["pixel_y"]
    current_size = entry.get("image_size")
    target_x, target_y = scale_coordinates(orig_x, orig_y, from_size=current_size, to_size=original_image_size)

    seen_coords = set()
    for idx, row in candidate_rows.iterrows():
        cx = safe_float(row.get("center_x"))
        cy = safe_float(row.get("center_y"))
        if cx is None or cy is None:
            continue

        dx = abs(cx - target_x)
        dy = abs(cy - target_y)
        coord_key = (cx, cy)
        if coord_key in seen_coords:
            continue  # skip repeated debug logs
        seen_coords.add(coord_key)

        print(f"[DEBUG] Comparing CSV coords ({cx}, {cy}) to target ({target_x:.2f}, {target_y:.2f}) -> dx={dx:.2f}, dy={dy:.2f}")

        if dx <= max_dev and dy <= max_dev:
            matches.append({
                "image_name": image_name,
                "pixel_x": cx,
                "pixel_y": cy,
                "anomaly": row.get("anomaly")
            })

    return matches



# ==== Main function ====
def main(correspondence_array, csv_path, original_image_size=None):
    """
    Returns list of dicts with image_name, pixel_x, pixel_y, anomaly
    """
    if not correspondence_array:
        return []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    first_entry = correspondence_array[0]
    ext = os.path.splitext(os.path.basename(first_entry["image_name"]))[1].lower()
    filename_col = "wiris_image" if ext == ".tiff" else "pi_image"

    max_dev = max_deviation(df, filename_col, correspondence_array, fraction=0.5)
    if max_dev is None:
        max_dev = 20.0

    results = []
    for entry in correspondence_array:
        matches = get_matching_rows_for_image(
            df,
            entry,
            max_dev,
            filename_col,
            original_image_size
        )
        results.extend(matches)

    # Deduplicate results
    unique_results = { (r['image_name'], r['pixel_x'], r['pixel_y']): r for r in results }
    return list(unique_results.values())