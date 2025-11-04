import os
import pandas as pd
from collections import Counter
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import traceback
import avg_gps
import georef_new

MAX_DEVIATION_FRACTION = 0.5
THUMBNAIL_MAX_SIZE = 200
GRID_COLUMNS = 4
MAX_DISTANCE_M = 15.0


# ---- Helpers ----
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def compute_max_deviation_from_bbox(row, fraction=MAX_DEVIATION_FRACTION):
    xmin = safe_float(row.get("xmin"))
    xmax = safe_float(row.get("xmax"))
    ymin = safe_float(row.get("ymin"))
    ymax = safe_float(row.get("ymax"))
    if None in (xmin, xmax, ymin, ymax):
        print("[DEBUG] Invalid bbox values, returning fallback deviation=10.0")
        return 10.0
    w = abs(xmax - xmin)
    h = abs(ymax - ymin)
    deviation = max(w, h) * fraction if max(w, h) > 0 else 10.0
    print(f"[DEBUG] BBox deviation: w={w:.2f}, h={h:.2f}, max_dev={deviation:.2f}")
    return deviation


def get_matching_rows_for_image(df, image_name, max_dev, filename_col, image_data_list):
    """
    Returns full image_data_list entries with pixel_x/pixel_y added if matching anomaly exists in CSV.
    Deduplicates entries by (pixel_x, pixel_y).
    """
    matches = []
    seen = set()  # keep track of unique pixel positions

    candidate_rows = df[df[filename_col].astype(str).str.strip() == image_name]
    if candidate_rows.empty:
        return matches

    cx_col = next((c for c in df.columns if c.lower() in ("center_x", "centerx")), None)
    cy_col = next((c for c in df.columns if c.lower() in ("center_y", "centery")), None)
    if not cx_col or not cy_col:
        return matches

    img_entry = next((e.copy() for e in image_data_list if os.path.basename(e["image_path"]) == image_name), None)
    if not img_entry:
        return matches

    for idx, row in candidate_rows.iterrows():
        cx = safe_float(row.get(cx_col))
        cy = safe_float(row.get(cy_col))
        if cx is None or cy is None:
            continue

        key = (int(round(cx)), int(round(cy)))
        if key in seen:
            continue  # skip duplicate
        seen.add(key)

        entry_with_pixel = img_entry.copy()
        entry_with_pixel["pixel_x"] = int(round(cx))
        entry_with_pixel["pixel_y"] = int(round(cy))
        entry_with_pixel.update(row.to_dict())
        matches.append(entry_with_pixel)

    return matches



def filter_rows_by_geodistance(rows, ref_gps, max_distance=MAX_DISTANCE_M):
    """
    Filters rows by geodistance to reference GPS.
    """
    mapper = georef_new.DroneMapper()
    filtered = []
    for i, entry in enumerate(rows):
        try:
            if 'gps' not in entry or 'image_size' not in entry:
                continue
            target_gps = mapper.get_target_gps(
                u=entry['pixel_x'],
                v=entry['pixel_y'],
                gps=entry['gps'],
                angles=(entry['yaw'], entry['pitch'], entry['roll']),
                image_size=entry['image_size']
            )
            dist = avg_gps.haversine(ref_gps[0], ref_gps[1], target_gps[0], target_gps[1])
            if dist <= max_distance:
                filtered.append(entry)
        
        except Exception as e:
            print(f"[ERROR] #{i} Error computing GPS for {entry.get('image_path', 'unknown')}: {e}")
    return filtered


def draw_bboxes_on_image(pil_img, rows_for_image, display_scale=1.0):
    draw = ImageDraw.Draw(pil_img)
    for i, r in enumerate(rows_for_image):
        xmin = safe_float(r.get("xmin"))
        xmax = safe_float(r.get("xmax"))
        ymin = safe_float(r.get("ymin"))
        ymax = safe_float(r.get("ymax"))

        box = (xmin*display_scale, ymin*display_scale, xmax*display_scale, ymax*display_scale)
        line_w = max(2, int(max(1, pil_img.size[0] // 200)))
        draw.rectangle(box, outline="red", width=line_w)
    return pil_img


def show_images_grid(image_items):
    root = tk.Tk()
    root.title("Matched Images Grid")
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
            scale = min(1.0, THUMBNAIL_MAX_SIZE / max(pil_img.size))
            if rows_for_image:
                pil_img = draw_bboxes_on_image(pil_img, rows_for_image, display_scale=scale)
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
            print(f"[ERROR] Error displaying {path}: {e}")

    frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))
    root.mainloop()


# ---- Main Function ----
def main(image_data_list, csv_path, max_distance_m=MAX_DISTANCE_M, show_gui=False):
    try:
        if not image_data_list or not csv_path:
            return []

        df = pd.read_csv(csv_path, dtype=str, low_memory=False)

        matched_items = []
        ref_gps = image_data_list[0].get("target_gps")

        for i, entry in enumerate(image_data_list):
            image_name = os.path.basename(entry["image_path"])

            filename_col = "wiris_image" if image_name.lower().endswith(".tiff") else "pi_image"

            candidate_row = df[df[filename_col].astype(str).str.strip() == image_name]
            if not candidate_row.empty:
                max_dev = compute_max_deviation_from_bbox(candidate_row.iloc[0])
            else:
                max_dev = 10.0

            matching_entries = get_matching_rows_for_image(df, image_name, max_dev, filename_col, image_data_list)


            if ref_gps:
                matching_entries = filter_rows_by_geodistance(matching_entries, ref_gps, max_distance=max_distance_m)


            matched_items.append({"title": image_name, "path": entry["image_path"], "rows": matching_entries})


        if show_gui and matched_items:
            show_images_grid(matched_items)

        return matched_items

    except Exception as e:
        print(f"[ERROR] Exception in main(): {e}")
        traceback.print_exc()
        return []
