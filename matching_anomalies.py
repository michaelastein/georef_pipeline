import os
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import pandas as pd
import cv2

# ==== Configuration ====
MAX_DEVIATION_FRACTION = 0.5  # Fraction of bounding box size allowed for deviation
GRID_COLUMNS = 4               # Number of columns in the displayed image grid
THUMBNAIL_MAX_SIZE = 200       # Maximum size (pixels) for thumbnail display

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

    tol = 20
    for _, r in candidate_rows.iterrows():
        cx = safe_float(r.get("center_x"))
        cy = safe_float(r.get("center_y"))
        if cx is None or cy is None:
            continue
        if abs(cx - pixel_x) <= tol and abs(cy - pixel_y) <= tol:
            return r
    return None

def compute_max_deviation_from_bbox(row, fraction=MAX_DEVIATION_FRACTION):
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

    target_x = data_by_name[image_name]["pixel_x"]
    target_y = data_by_name[image_name]["pixel_y"]

    for _, row in candidate_rows.iterrows():
        cx = safe_float(row.get(cx_col))
        cy = safe_float(row.get(cy_col))
        if cx is None or cy is None:
            continue
        if abs(cx - target_x) <= max_dev and abs(cy - target_y) <= max_dev:
            matches.append(row)
    return matches

def draw_bboxes_on_image(pil_img, rows_for_image, display_scale=1.0):
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

def show_images_grid(image_items):
    root = tk.Tk()
    root.title("Matched Images Grid")

    # --- Canvas and scrollbars ---
    canvas = tk.Canvas(root, bg="white")
    h_scroll = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
    v_scroll = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

    h_scroll.pack(side="bottom", fill="x")
    v_scroll.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    frame = tk.Frame(canvas, bg="white")
    canvas.create_window((0, 0), window=frame, anchor="nw")

    root.image_refs = []

    x, y = 10, 10
    max_row_height = 0

    for idx, item in enumerate(image_items):
        path = item["path"]
        rows_for_image = item.get("rows", [])

        try:
            pil_img = Image.open(path).convert("RGB")

            # Draw bboxes if any
            if rows_for_image:
                scale = min(1.0, THUMBNAIL_MAX_SIZE / max(pil_img.size))
                pil_img = draw_bboxes_on_image(pil_img, rows_for_image, display_scale=scale)
            else:
                scale = min(1.0, THUMBNAIL_MAX_SIZE / max(pil_img.size))

            pil_img.thumbnail((THUMBNAIL_MAX_SIZE, THUMBNAIL_MAX_SIZE), Image.LANCZOS)

            tk_img = ImageTk.PhotoImage(pil_img, master=root)
            root.image_refs.append(tk_img)

            lbl_img = tk.Label(frame, image=tk_img, text=item.get("title", os.path.basename(path)),
                               compound="top", bg="white", fg="black")
            lbl_img.place(x=x, y=y)

            w, h = pil_img.size
            max_row_height = max(max_row_height, h + 40)
            x += w + 20
            if (idx + 1) % GRID_COLUMNS == 0:
                x = 10
                y += max_row_height
                max_row_height = 0

        except Exception as e:
            print(f"Error displaying {path}: {e}")

    frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    root.mainloop()

# ==== Main Function ====
def main(data, csv_path):
    try:
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            raise ValueError("`data` must be a non-empty list of image entry dicts.")

        df = pd.read_csv(csv_path, dtype=str, low_memory=False)

        first_entry = data[0]
        first_filename = os.path.basename(first_entry["image_path"])
        first_pixel_x = float(first_entry["pixel_x"])
        first_pixel_y = float(first_entry["pixel_y"])

        # Determine filename column
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

        first_row = find_exact_row_for_first_image(df, filename_col, first_filename, first_pixel_x, first_pixel_y)
        if first_row is None:
            messagebox.showerror("Error", f"No matching row found for {first_filename}.")
            return

        max_deviation = compute_max_deviation_from_bbox(first_row)
        data_by_name = {os.path.basename(e["image_path"]): e for e in data}

        matched_items = []

        # --- First image ---
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

        # --- Remaining images ---
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
            messagebox.showinfo("Result", "Only the first image found; no other matches.")

        #show_images_grid(matched_items)

        # Print filenames of all matched items
        for item in matched_items:
            print(os.path.basename(item["path"]))


    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Unexpected error", str(e))

