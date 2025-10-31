import os
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import pandas as pd
import cv2

# ==== Configuration ====
MAX_DEVIATION_FRACTION = 1 / 2.0  # Fraction of bounding box size allowed for deviation when matching
GRID_COLUMNS = 4  # Number of columns in the displayed image grid
THUMBNAIL_MAX_SIZE = 200  # Maximum size (pixels) for thumbnail display

images_refs = []

# ==== Helpers ====  

def ask_csv_file():
    """Open a file dialog to select a CSV file and return its path."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return path

def safe_float(x):
    """Convert input to float safely; return None on failure."""
    try:
        return float(x)
    except Exception:
        return None

def find_exact_row_for_first_image(df, filename_col, filename, pixel_x, pixel_y):
    """
    Find the row in the DataFrame matching the given filename and pixel coordinates
    within a tolerance.
    """
    candidate_rows = df[df[filename_col].astype(str).str.strip() == filename]
    if candidate_rows.empty:
        return None

    tol = 20  # pixel tolerance
    for _, r in candidate_rows.iterrows():
        cx = safe_float(r.get("center_x"))
        cy = safe_float(r.get("center_y"))
        if cx is None or cy is None:
            continue
        # check if the center coordinates are close enough
        if abs(cx - pixel_x) <= tol and abs(cy - pixel_y) <= tol:
            return r
    return None

def compute_max_deviation_from_bbox(row, fraction=MAX_DEVIATION_FRACTION):
    """
    Compute the maximum allowed deviation from the bounding box for matching.
    If bbox is invalid, return a default value of 10.
    """
    xmin = safe_float(row.get("xmin"))
    xmax = safe_float(row.get("xmax"))
    ymin = safe_float(row.get("ymin"))
    ymax = safe_float(row.get("ymax"))
    if None in (xmin, xmax, ymin, ymax):
        return 10.0
    w = abs(xmax - xmin)
    h = abs(ymax - ymin)
    return max(w, h) * fraction if max(w, h) > 0 else 10.0

def get_matching_rows_for_image(df, image_name, max_dev, filename_col, data_by_name):
    """
    Find rows in the DataFrame that match the given image name and are within
    the allowed deviation from the target coordinates.
    """
    matches = []
    if filename_col not in df.columns:
        return matches

    candidate_rows = df[df[filename_col].astype(str).str.strip() == image_name]
    if candidate_rows.empty:
        return matches

    # Identify center coordinate columns
    cx_col = None
    cy_col = None
    for c in df.columns:
        if c.lower() in ("center_x", "centerx"):
            cx_col = c
        elif c.lower() in ("center_y", "centery"):
            cy_col = c
    if not cx_col or not cy_col:
        return matches

    # Target pixel coordinates
    target_x = data_by_name[image_name]["pixel_x"]
    target_y = data_by_name[image_name]["pixel_y"]

    for _, row in candidate_rows.iterrows():
        cx = safe_float(row.get(cx_col))
        cy = safe_float(row.get(cy_col))
        if cx is None or cy is None:
            continue
        # check if coordinates are within allowed deviation
        if abs(cx - target_x) <= max_dev and abs(cy - target_y) <= max_dev:
            matches.append(row)
    return matches

def draw_bboxes_on_image(pil_img, rows_for_image, display_scale):
    """
    Draw red bounding boxes on a PIL image based on the DataFrame rows.
    """
    draw = ImageDraw.Draw(pil_img)
    for r in rows_for_image:
        xmin = safe_float(r.get("xmin"))
        xmax = safe_float(r.get("xmax"))
        ymin = safe_float(r.get("ymin"))
        ymax = safe_float(r.get("ymax"))
        if None in (xmin, xmax, ymin, ymax):
            continue
        box = (
            xmin * display_scale,
            ymin * display_scale,
            xmax * display_scale,
            ymax * display_scale
        )
        line_w = max(2, int(max(1, pil_img.size[0] // 200)))
        draw.rectangle(box, outline="red", width=line_w)
    return pil_img

def open_and_prepare_image(path, max_size=THUMBNAIL_MAX_SIZE):
    """
    Open an image with PIL, resize it proportionally so the maximum dimension
    does not exceed max_size, and return the resized image and scale factor.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    pil = Image.open(path).convert("RGB")
    orig_size = pil.size
    w_ratio = max_size[0] / orig_size[0]
    h_ratio = max_size[1] / orig_size[1]
    scale = min(1.0, w_ratio, h_ratio)
    new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
    pil_resized = pil.resize(new_size, Image.LANCZOS)
    return pil_resized, scale

def show_images_grid(image_items):
    root = tk.Tk()
    root.title("Matched Images Grid")

    canvas = tk.Canvas(root, width=1200, height=800, bg="black")
    canvas.pack(fill="both", expand=True)

    # attach list to root so GC never deletes the images
    root.image_refs = []

    x, y = 10, 10
    max_row_height = 0

    for idx, item in enumerate(image_items):
        path = item["path"]
        try:
            bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if bgr is None:
                print(f"Skipping unreadable image {path}")
                continue

            # ensure 3-channel RGB
            if len(bgr.shape) == 2:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(rgb)

            scale = THUMBNAIL_MAX_SIZE / max(pil_img.size)
            w, h = int(pil_img.width * scale), int(pil_img.height * scale)
            pil_img = pil_img.resize((w, h), Image.LANCZOS)

            tk_img = ImageTk.PhotoImage(pil_img, master=root)
            root.image_refs.append(tk_img)   # keep reference tied to Tk root

            canvas.create_image(x, y, anchor="nw", image=tk_img)
            name = item.get("title", os.path.basename(path))
            canvas.create_text(x + w//2, y + h + 10, text=name, fill="white", anchor="n")

            # force Tk to register image before loop continues
            root.update_idletasks()

            max_row_height = max(max_row_height, h + 20)
            x += w + 20
            if (idx + 1) % GRID_COLUMNS == 0:
                x = 10
                y += max_row_height
                max_row_height = 0

        except Exception as e:
            print(f"Error displaying {path}: {e}")

    root.mainloop()



# ==== Main ====
def main(data, csv_path):
    """
    Main function to process images and CSV data:
    - Reads CSV file
    - Matches images based on pixel coordinates and bounding boxes
    - Displays matched images in a Tkinter grid
    """
    try:
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            raise ValueError("`data` must be a non-empty list of image entry dicts.")

        df = pd.read_csv(csv_path, dtype={"wiris_image": str, "pi_image": str}, low_memory=False)

        first_entry = data[0]
        first_filename = os.path.basename(first_entry["image_path"])
        first_pixel_x = float(first_entry["pixel_x"])
        first_pixel_y = float(first_entry["pixel_y"])

        # --- determine filename column based on extension ---
        ext = os.path.splitext(first_filename)[1].lower()
        if ext == ".tiff":
            filename_col = "wiris_image"
        elif ext == ".jpg":
            filename_col = "pi_image"
        else:
            messagebox.showerror("Error", f"Unsupported first image extension '{ext}'.")
            return

        if filename_col not in df.columns or first_filename not in df[filename_col].astype(str).values:
            messagebox.showerror("Error", f"Filename '{first_filename}' not found in column '{filename_col}'.")
            return

        # Find matching row for first image
        first_row = find_exact_row_for_first_image(df, filename_col, first_filename, first_pixel_x, first_pixel_y)
        if first_row is None:
            messagebox.showerror("Error", f"No matching row found for {first_filename} with center_x/center_y = pixel_x/pixel_y.")
            return

        max_deviation = compute_max_deviation_from_bbox(first_row)
        data_by_name = {os.path.basename(e["image_path"]): e for e in data}

        # --- collect matched items ---
        matched_items = []
        first_matching_rows = []
        candidate_rows = df[df[filename_col].astype(str).str.strip() == first_filename]
        for _, r in candidate_rows.iterrows():
            cx = safe_float(r.get("center_x"))
            cy = safe_float(r.get("center_y"))
            if cx is None or cy is None:
                continue
            if abs(cx - first_pixel_x) <= max_deviation and abs(cy - first_pixel_y) <= max_deviation:
                first_matching_rows.append(r)
        if not first_matching_rows:
            first_matching_rows = [first_row]

        matched_items.append({
            "title": f"(first) {first_filename}",
            "path": first_entry["image_path"],
            "rows": first_matching_rows
        })

        # Process remaining images
        for entry in data[1:]:
            bname = os.path.basename(entry["image_path"])
            rows = get_matching_rows_for_image(df, bname, max_deviation, filename_col, data_by_name)
            if rows:
                matched_items.append({
                    "title": bname,
                    "path": entry["image_path"],
                    "rows": rows
                })

        if len(matched_items) == 1:
            messagebox.showinfo("Result", "No other images found within the allowed deviation; showing only the first image.")

        # Show all matched images
        show_images_grid(matched_items)

    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Unexpected error", f"{e}")
