

import os
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import pandas as pd

# ==== Configuration ====
MAX_DEVIATION_FRACTION = 1 / 3.0
THUMBNAIL_MAX_SIZE = (600, 400)
GRID_COLUMNS = 3

# ==== Helpers ====
def ask_csv_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return path

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def find_exact_row_for_first_image(df, filename_col, filename, pixel_x, pixel_y):
    candidate_rows = df[df[filename_col].astype(str).str.strip() == filename]
    if candidate_rows.empty:
        return None
    tol = 1e-3
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
    # identify center columns
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

def draw_bboxes_on_image(pil_img, rows_for_image, display_scale):
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
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(frame)
    vsb = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    hsb = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
    canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    hsb.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    inner = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor='nw')
    photos = []
    row_idx = 0
    col_idx = 0
    for item in image_items:
        try:
            pil_img, scale = open_and_prepare_image(item["path"])
            orig = Image.open(item["path"]).convert("RGB")
            display_scale = pil_img.size[0] / orig.size[0]
            pil_with_boxes = draw_bboxes_on_image(pil_img.copy(), item.get("rows", []), display_scale)
            tk_img = ImageTk.PhotoImage(pil_with_boxes)
            photos.append(tk_img)
            panel = tk.Frame(inner, bd=2, relief=tk.RIDGE)
            lbl = tk.Label(panel, image=tk_img)
            lbl.pack()
            caption = tk.Label(panel, text=item.get("title", os.path.basename(item["path"])))
            caption.pack()
            panel.grid(row=row_idx, column=col_idx, padx=8, pady=8)
            col_idx += 1
            if col_idx >= GRID_COLUMNS:
                col_idx = 0
                row_idx += 1
        except Exception as e:
            print(f"Could not open image {item['path']}: {e}")
    inner.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
    root.mainloop()

# ==== Main ====
def main(data):
    try:
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            raise ValueError("`data` must be a non-empty list of image entry dicts.")

        csv_path = ask_csv_file()
        if not csv_path:
            print("No CSV selected, aborting.")
            return

        df = pd.read_csv(csv_path)
        first_entry = data[0]
        first_filename = os.path.basename(first_entry["image_path"])
        first_pixel_x = float(first_entry["pixel_x"])
        first_pixel_y = float(first_entry["pixel_y"])

        # --- determine column based on extension ---
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
            return  # END PROGRAM

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

        show_images_grid(matched_items)

    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Unexpected error", f"{e}")
