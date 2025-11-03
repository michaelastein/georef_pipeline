# gui_canvas.py

import threading
import queue
from collections import OrderedDict
from tkinter import Tk, Canvas, Scrollbar, LEFT, RIGHT, Y, NW
from PIL import Image, ImageTk
import cv2


def launch_canvas_gui(image_data_list, orig_sizes):
    """
    Displays images in a scrollable Tkinter canvas grid.
    Lets user click a pixel and returns (image_index, x, y) in original coordinates.

    Args:
        image_data_list: list of dictionaries containing image metadata (including paths)
        orig_sizes: list of tuples (width, height) for each image

    Returns:
        idx: index of the clicked image
        x, y: coordinates in original image size
    """
    # Store results to return after GUI closes
    result = {"idx": None, "x": None, "y": None}

    # ----------------- Tk root window -----------------
    root = Tk()
    root.title("Images Grid (Canvas Viewer)")

    # ----------------- Canvas setup -----------------
    canvas_w, canvas_h = 1400, 800
    canvas_side = Canvas(root, width=canvas_w, height=canvas_h, bg="black")
    scrollbar = Scrollbar(root, orient="vertical", command=canvas_side.yview)
    canvas_side.configure(yscrollcommand=scrollbar.set)
    canvas_side.pack(side=LEFT, fill="both", expand=1)
    scrollbar.pack(side=RIGHT, fill=Y)

    # ----------------- Layout parameters -----------------
    cols = 6  # number of columns in grid
    thumb_size = 200  # maximum thumbnail width/height
    padding = 10  # spacing between images
    text_height = 18  # height of text labels
    row_height = thumb_size + text_height + padding  # total row height

    # ----------------- Caches -----------------
    pil_cache = OrderedDict()  # stores loaded PIL images
    photo_cache = {}  # stores ImageTk.PhotoImage objects
    MAX_CACHE_ITEMS = 200  # limit for cache
    cache_lock = threading.Lock()  # for thread-safe access

    # ----------------- Loader queues -----------------
    load_queue = queue.Queue()
    result_queue = queue.Queue()
    stop_event = threading.Event()  # signal threads to stop

    # ----------------- Precompute image positions -----------------
    img_positions = []
    for i, (w, h) in enumerate(orig_sizes):
        scale = min(thumb_size / h, thumb_size / w)  # scale to fit thumbnail
        disp_w, disp_h = int(w * scale), int(h * scale)
        col = i % cols
        row = i // cols
        x = padding + col * (thumb_size + padding)
        y = padding + row * row_height
        img_positions.append((x, y, scale, disp_w, disp_h))

    n_rows = (len(image_data_list) + cols - 1) // cols
    total_height = padding + n_rows * row_height
    total_width = padding + cols * (thumb_size + padding)
    canvas_side.config(scrollregion=(0, 0, total_width, total_height))

    # ----------------- Helper: visible indices -----------------
    def get_visible_indices():
        """
        Return indices of images currently visible on the canvas + margin.
        Used to decide which images to load in background.
        """
        y0 = canvas_side.canvasy(0)
        y1 = canvas_side.canvasy(canvas_side.winfo_height())
        margin = row_height * 2  # extra rows above/below
        top_row = max(0, int((y0 - margin) // row_height))
        bottom_row = min(n_rows - 1, int((y1 + margin) // row_height))
        indices = []
        for r in range(top_row, bottom_row + 1):
            start = r * cols
            end = min(len(image_data_list), start + cols)
            indices.extend(range(start, end))
        return set(indices)

    # ----------------- Background image loader -----------------
    def loader_worker():
        """
        Thread function: load images from disk, resize, convert to PIL.
        Results put in result_queue.
        """
        while not stop_event.is_set():
            try:
                idx = load_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with cache_lock:
                if idx in pil_cache:
                    load_queue.task_done()
                    continue
            path = image_data_list[idx]["image_path"]
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

    # Start multiple loader threads
    for _ in range(max(2, min(8, len(image_data_list)))):
        t = threading.Thread(target=loader_worker, daemon=True)
        t.start()

    # ----------------- Enqueue images to load -----------------
    def enqueue_visible_and_neighbors():
        """
        Enqueue images currently visible or in neighboring rows for loading.
        """
        vis = get_visible_indices()
        extra = set()
        for idx in vis:
            row = idx // cols
            for r in range(max(0, row - 1), min(n_rows, row + 2)):
                start = r * cols
                extra.update(range(start, min(start + cols, len(image_data_list))))
        for idx in extra:
            with cache_lock:
                if idx not in pil_cache:
                    try:
                        load_queue.put_nowait(idx)
                    except queue.Full:
                        pass

    # ----------------- Drawing -----------------
    drawn_image_items = {}  # canvas IDs for images
    drawn_text_items = {}  # canvas IDs for labels
    marker_items = []  # red marker rectangles

    def draw_canvas_image(idx, photo):
        """Draw a single image thumbnail and label on the canvas."""
        if idx in drawn_image_items:
            canvas_side.delete(drawn_image_items[idx])
        x, y, scale, disp_w, disp_h = img_positions[idx]
        img_id = canvas_side.create_image(x, y, image=photo, anchor=NW)
        drawn_image_items[idx] = img_id
        name = image_data_list[idx]["image_path"].split("/")[-1].rsplit(".", 1)[0]
        tid = canvas_side.create_text(x + disp_w / 2, y + disp_h + 12, text=name, fill="white")
        drawn_text_items[idx] = tid

    def create_marker_canvas(x, y, color="red", size=6):
        """Draw a small square marker at (x,y)."""
        half = size / 2
        return canvas_side.create_rectangle(x - half, y - half, x + half, y + half, fill=color, outline=color)

    # ----------------- Process loaded images -----------------
    def process_loader_results():
        """Move images from result_queue into caches and draw if visible."""
        while True:
            try:
                idx, pil_img = result_queue.get_nowait()
            except queue.Empty:
                break
            with cache_lock:
                pil_cache[idx] = pil_img
                pil_cache.move_to_end(idx)
                while len(pil_cache) > MAX_CACHE_ITEMS:
                    pil_cache.popitem(last=False)
                photo = ImageTk.PhotoImage(pil_img)
                photo_cache[idx] = photo
                if idx in get_visible_indices():
                    draw_canvas_image(idx, photo)
            result_queue.task_done()

    # ----------------- Click handling -----------------
    def on_canvas_click(event):
        """Detect which image was clicked, compute original coordinates, save result."""
        x_c = canvas_side.canvasx(event.x)
        y_c = canvas_side.canvasy(event.y)
        for idx, (x, y, scale, disp_w, disp_h) in enumerate(img_positions):
            if x <= x_c <= x + disp_w and y <= y_c <= y + disp_h:
                local_x_scaled = (x_c - x)
                local_y_scaled = (y_c - y)
                x_orig = local_x_scaled / scale
                y_orig = local_y_scaled / scale

                # Save result
                result["idx"], result["x"], result["y"] = idx, x_orig, y_orig

                # Show a red marker
                for m in marker_items:
                    canvas_side.delete(m)
                marker_items.clear()
                marker_items.append(create_marker_canvas(x, y, color="red"))

                root.after(200, root.destroy)
                break

    # ----------------- Periodic update -----------------
    def periodic_update():
        """Enqueue visible images and process loader results periodically."""
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

    # ----------------- Window close -----------------
    def on_close():
        """Stop loader threads and close window."""
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    # Return result
    if result["idx"] is not None:
        return result["idx"], result["x"], result["y"]
    else:
        return None, None, None
