# gui_canvas.py
import threading
import queue
from collections import OrderedDict
from tkinter import Tk, Canvas, Scrollbar, LEFT, RIGHT, Y, NW
from PIL import Image, ImageTk, ImageDraw
import cv2
import avg_gps
import numpy as np
from feature_matching import build_correspondences_from_pixels

def launch_canvas_gui(img_paths, orig_sizes, image_data_list, H_dict, node_connected_component, shortest_path):
    # ----------------- Tk root -----------------
    root = Tk()
    root.title("Images Grid (Canvas + Lazy Loading)")

    # ----------------- Canvas setup -----------------
    canvas_w, canvas_h = 1400, 800
    canvas_side = Canvas(root, width=canvas_w, height=canvas_h, bg="black")
    scrollbar = Scrollbar(root, orient="vertical", command=canvas_side.yview)
    canvas_side.configure(yscrollcommand=scrollbar.set)
    canvas_side.pack(side=LEFT, fill="both", expand=1)
    scrollbar.pack(side=RIGHT, fill=Y)

    # Layout parameters
    cols = 6
    thumb_size = 200
    padding = 10
    text_height = 18
    row_height = thumb_size + text_height + padding

    # caches
    pil_cache = OrderedDict()  # index -> PIL.Image
    photo_cache = {}           # index -> ImageTk.PhotoImage
    MAX_CACHE_ITEMS = 200
    cache_lock = threading.Lock()

    # loader queues
    load_queue = queue.Queue()
    result_queue = queue.Queue()
    stop_event = threading.Event()

    # calculate positions
    img_positions = []
    for i, (w, h) in enumerate(orig_sizes):
        scale = min(thumb_size / h, thumb_size / w)
        disp_w, disp_h = int(w * scale), int(h * scale)
        col = i % cols
        row = i // cols
        x = padding + col * (thumb_size + padding)
        y = padding + row * row_height
        img_positions.append((x, y, scale, disp_w, disp_h))

    n_rows = (len(img_paths) + cols - 1) // cols
    total_height = padding + n_rows * row_height
    total_width = padding + cols * (thumb_size + padding)
    canvas_side.config(scrollregion=(0, 0, total_width, total_height))

    # ----------------- Helpers -----------------
    def get_visible_indices():
        y0 = canvas_side.canvasy(0)
        y1 = canvas_side.canvasy(canvas_side.winfo_height())
        margin = row_height * 2
        top_row = max(0, int((y0 - margin) // row_height))
        bottom_row = min(n_rows - 1, int((y1 + margin) // row_height))
        indices = []
        for r in range(top_row, bottom_row + 1):
            start = r * cols
            end = min(len(img_paths), start + cols)
            indices.extend(range(start, end))
        return set(indices)

    # ----------------- Background loader -----------------
    def loader_worker():
        while not stop_event.is_set():
            try:
                idx = load_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with cache_lock:
                if idx in pil_cache:
                    load_queue.task_done()
                    continue
            path = img_paths[idx]
            try:
                bgr = cv2.imread(path)
                if bgr is None:
                    load_queue.task_done()
                    continue
                _, _, scale, disp_w, disp_h = img_positions[idx]
                resized = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                result_queue.put((idx, pil_img))
            except Exception as e:
                print(f"Loader error for index {idx}: {e}")
            finally:
                load_queue.task_done()

    NUM_LOADER_THREADS = max(2, min(8, len(img_paths)))
    loader_threads = []
    for _ in range(NUM_LOADER_THREADS):
        t = threading.Thread(target=loader_worker, daemon=True)
        t.start()
        loader_threads.append(t)

    def enqueue_visible_and_neighbors():
        vis = get_visible_indices()
        extra = set()
        for idx in vis:
            row = idx // cols
            for r in range(max(0, row - 1), min(n_rows, row + 2)):
                start = r * cols
                extra.update(range(start, min(start + cols, len(img_paths))))
        to_load = list(extra)
        for idx in to_load:
            with cache_lock:
                if idx in pil_cache:
                    continue
                try:
                    load_queue.put_nowait(idx)
                except queue.Full:
                    pass

    drawn_image_items = {}
    drawn_text_items = {}
    marker_items = []

    def draw_canvas_image(idx, photo):
        if idx in drawn_image_items:
            try:
                canvas_side.delete(drawn_image_items[idx])
            except Exception:
                pass
        x, y, scale, disp_w, disp_h = img_positions[idx]
        img_id = canvas_side.create_image(x, y, image=photo, anchor=NW)
        drawn_image_items[idx] = img_id
        if idx in drawn_text_items:
            canvas_side.delete(drawn_text_items[idx])
        name = img_paths[idx].split("/")[-1].rsplit(".", 1)[0]
        tid = canvas_side.create_text(x + disp_w / 2, y + disp_h + 12, text=name, fill="white")
        drawn_text_items[idx] = tid

    def create_marker_canvas(x, y, color="red", size=6):
        half = size / 2
        return canvas_side.create_rectangle(x - half, y - half, x + half, y + half, fill=color, outline=color)

    def process_loader_results():
        while True:
            try:
                idx, pil_img = result_queue.get_nowait()
            except queue.Empty:
                break
            with cache_lock:
                pil_cache[idx] = pil_img
                pil_cache.move_to_end(idx)
                while len(pil_cache) > MAX_CACHE_ITEMS:
                    old_idx, _ = pil_cache.popitem(last=False)
                    if old_idx in photo_cache:
                        del photo_cache[old_idx]
                    if old_idx in drawn_image_items:
                        canvas_side.delete(drawn_image_items[old_idx])
                        del drawn_image_items[old_idx]
                    if old_idx in drawn_text_items:
                        canvas_side.delete(drawn_text_items[old_idx])
                        del drawn_text_items[old_idx]
                photo = ImageTk.PhotoImage(pil_img)
                photo_cache[idx] = photo
                if idx in get_visible_indices():
                    draw_canvas_image(idx, photo)
            result_queue.task_done()

    # ----------------- Click handling -----------------
    def on_canvas_click(event):
        x_c = canvas_side.canvasx(event.x)
        y_c = canvas_side.canvasy(event.y)
        for idx, (x, y, scale, disp_w, disp_h) in enumerate(img_positions):
            if x <= x_c <= x + disp_w and y <= y_c <= y + disp_h:
                local_x_scaled = (x_c - x)
                local_y_scaled = (y_c - y)
                x_orig = local_x_scaled / scale
                y_orig = local_y_scaled / scale
                handle_image_click(idx, x_orig, y_orig, x, y, scale)
                break

    def handle_image_click(idx, x_orig, y_orig, x_canvas, y_canvas, scale):
        for mid in marker_items:
            canvas_side.delete(mid)
        marker_items.clear()
        m_self = create_marker_canvas(x_canvas + x_orig * scale, y_canvas + y_orig * scale, color="red")
        marker_items.append(m_self)
        # build correspondences
        data = build_correspondences_from_pixels(idx, x_orig, y_orig,
                                               image_data_list=image_data_list,
                                               H_dict=H_dict,
                                               node_connected_component=node_connected_component,
                                               shortest_path=shortest_path)
        # draw yellow markers for correspondences
        for entry in data:
            other_idx = entry["image_index"]
            x_other = entry["pixel_x"]
            y_other = entry["pixel_y"]
            if other_idx == idx:
                continue
            x_img, y_img, sc, disp_w, disp_h = img_positions[other_idx]
            x_disp, y_disp = x_other * sc, y_other * sc
            if 0 <= x_disp <= disp_w and 0 <= y_disp <= disp_h:
                mx = create_marker_canvas(x_img + x_disp, y_img + y_disp, color="yellow")
                marker_items.append(mx)
        # call avg_gps safely
        root.after(0, lambda d=data: safe_avg_gps_call(d))

    def safe_avg_gps_call(data):
        def worker():
            try:
                avg_gps.main(data)
            except Exception as e:
                print(f"avg_gps.main error: {e}")
        threading.Thread(target=worker, daemon=True).start()


    def periodic_update():
        enqueue_visible_and_neighbors()
        process_loader_results()
        vis = get_visible_indices()
        for idx in vis:
            with cache_lock:
                if idx in photo_cache and idx not in drawn_image_items:
                    draw_canvas_image(idx, photo_cache[idx])
        root.after(300, periodic_update)

    enqueue_visible_and_neighbors()
    canvas_side.bind("<Button-1>", on_canvas_click)
    root.after(100, periodic_update)

    # ----------------- Clean exit -----------------
    def on_close():
        stop_event.set()
        for t in loader_threads:
            t.join(timeout=0.5)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


