import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==== Helpers ====
def safe_float(x):
    """
    Convert x to float safely.
    Returns None if conversion fails.
    """
    try:
        return float(x)
    except Exception:
        return None


def scale_coordinates(xs, ys, from_size, to_size):
    """
    Scale coordinates (single values or arrays) from one image size to another.

    Parameters:
        xs, ys: coordinate(s) in 'from_size'
        from_size: tuple (width, height) of source image
        to_size: tuple (width, height) of target image

    Returns:
        Scaled coordinates as float arrays
    """
    if from_size is None or to_size is None:
        return xs, ys

    from_w, from_h = from_size
    to_w, to_h = to_size
    if from_w == 0 or from_h == 0:
        return xs, ys

    scale_x = to_w / from_w
    scale_y = to_h / from_h
    # Round to nearest pixel, keep as float for further processing
    return np.round(xs * scale_x).astype(float), np.round(ys * scale_y).astype(float)


def compute_max_deviation(df, filename_col, correspondence_array, fraction=0.5):
    """
    Compute a maximum deviation threshold for matching anomalies.
    
    Parameters:
        df: DataFrame containing CSV data
        filename_col: name of column in df containing image filenames
        correspondence_array: list of image entries with 'image_name'
        fraction: fraction of bounding box dimension to use as max deviation
    
    Returns:
        max_dev: float, threshold in pixels
    """
    df[filename_col] = df[filename_col].astype(str).str.strip()
    
    for entry in correspondence_array:
        filename = os.path.basename(entry["image_name"]).strip()
        candidate_rows = df[df[filename_col] == filename]
        if candidate_rows.empty:
            continue

        r = candidate_rows.iloc[0]  # take first row
        xmin = safe_float(r.get("xmin"))
        xmax = safe_float(r.get("xmax"))
        ymin = safe_float(r.get("ymin"))
        ymax = safe_float(r.get("ymax"))

        # fallback if any value is invalid
        if None in (xmin, xmax, ymin, ymax):
            return 10.0

        w = abs(xmax - xmin)
        h = abs(ymax - ymin)
        return max(w, h) * fraction if max(w, h) > 0 else 10.0

    print("use default deviation")
    return 20.0


def get_matching_rows(df, entry, max_dev, filename_col, original_image_size=None):
    """
    Vectorized matching of anomaly rows from CSV for a given image entry.

    Parameters:
        df: DataFrame containing CSV anomaly data
        entry: dict containing 'image_name', 'pixel_x', 'pixel_y', 'image_size'
        max_dev: float, maximum allowed pixel deviation
        filename_col: column name in df storing image filename
        original_image_size: optional, scale coordinates to original size

    Returns:
        List of dicts: matching rows with keys 'image_name', 'pixel_x', 'pixel_y', 'anomaly'
    """
    image_name = os.path.basename(entry["image_name"]).strip()
    candidate_rows = df[df[filename_col].astype(str).str.strip() == image_name]
    if candidate_rows.empty:
        return []

    # Scale entry coordinates to match CSV/original image size
    target_x, target_y = scale_coordinates(
        np.array([entry["pixel_x"]]),
        np.array([entry["pixel_y"]]),
        from_size=entry.get("image_size"),
        to_size=original_image_size
    )
    target_x, target_y = target_x[0], target_y[0]

    # Convert CSV columns to float arrays
    cx_arr = candidate_rows["center_x"].astype(float).to_numpy()
    cy_arr = candidate_rows["center_y"].astype(float).to_numpy()

    # Compute deviations
    dx = np.abs(cx_arr - target_x)
    dy = np.abs(cy_arr - target_y)

    mask = (dx <= max_dev) & (dy <= max_dev)
    matched_rows = candidate_rows[mask]

    results = []
    for _, row in matched_rows.iterrows():
        results.append({
            "image_name": image_name,
            "pixel_x": row["center_x"],
            "pixel_y": row["center_y"],
            "anomaly": row.get("anomaly")  # optional field
        })
    return results

# ==== Main function ====
def main(correspondence_array, csv_path, original_image_size=None, max_workers=10):
    """
    Optimized threaded version to match anomalies from a CSV to a batch of images.
    
    Parameters:
        correspondence_array: list of dicts with keys 'image_name', 'pixel_x', 'pixel_y', 'image_size'
        csv_path: path to CSV containing anomaly data
        original_image_size: optional, used to scale coordinates
        max_workers: number of threads for parallel processing

    Returns:
        List of unique matching rows from CSV with keys: 'image_name', 'pixel_x', 'pixel_y', 'anomaly'
    """
    if not correspondence_array:
        return []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Determine filename column based on image type
    first_entry = correspondence_array[0]
    ext = os.path.splitext(os.path.basename(first_entry["image_name"]))[1].lower()
    filename_col = "wiris_image" if ext == ".tiff" else "pi_image"

    # Compute maximum deviation once (avoid repeating per image)
    max_dev = compute_max_deviation(df, filename_col, correspondence_array, fraction=0.5)

    # Precompute basenames for faster comparisons
    for entry in correspondence_array:
        entry["basename"] = os.path.basename(entry["image_name"]).strip()

    results = []

    # Worker function for each image entry
    def worker(entry):
        return get_matching_rows(
            df,
            entry,
            max_dev,
            filename_col,
            original_image_size
        )

    # Multithreaded processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, entry) for entry in correspondence_array]
        for future in as_completed(futures):
            results.extend(future.result())

    # Deduplicate results based on image_name + pixel coordinates
    unique_results = { (r['image_name'], r['pixel_x'], r['pixel_y']): r for r in results }
    return list(unique_results.values())
