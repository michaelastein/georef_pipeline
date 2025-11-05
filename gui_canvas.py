import threading
import queue
from collections import OrderedDict
from tkinter import Tk, Canvas, Scrollbar, LEFT, RIGHT, Y, NW
from PIL import Image, ImageTk
import cv2

class CanvasGUI:
    def __init__(self, image_data_list, orig_sizes, click_callback=None):
        self.image_data_list = image_data_list
        self.orig_sizes = orig_sizes
        self.click_callback = click_callback

        # Markers
        self.red_marker_items = []
        self.yellow_marker_items = []

        # Last click result
        self.result = {"idx": None, "x": None, "y": None}

        # Tkinter setup
        self.root = Tk()
        self.root.title("Images Grid (Canvas Viewer)")
        self.canvas_w, self.canvas_h = 1400, 800
        self.canvas = Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="black")
        scrollbar = Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=LEFT, fill="both", expand=1)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Layout
        self.cols = 6
        self.thumb_size = 200
        self.padding = 10
        self.text_height = 18
        self.row_height = self.thumb_size + self.text_height + self.padding

        # Caches
        self.pil_cache = OrderedDict()
        self.photo_cache = {}
        self.MAX_CACHE_ITEMS = 200
        self.cache_lock = threading.Lock()

        # Loader queues
        self.load_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        # Compute positions
        self.img_positions = []
        for i, (w, h) in enumerate(orig_sizes):
            scale = min(self.thumb_size / h, self.thumb_size / w)
            disp_w, disp_h = int(w * scale), int(h * scale)
            col = i % self.cols
            row = i // self.cols
            x = self.padding + col * (self.thumb_size + self.padding)
            y = self.padding + row * self.row_height
            self.img_positions.append((x, y, scale, disp_w, disp_h))

        n_rows = (len(image_data_list) + self.cols - 1) // self.cols
        total_height = self.padding + n_rows * self.row_height
        total_width = self.padding + self.cols * (self.thumb_size + self.padding)
        self.canvas.config(scrollregion=(0, 0, total_width, total_height))

        # Drawn items
        self.drawn_image_items = {}
        self.drawn_text_items = {}

        # Start loader threads
        for _ in range(max(2, min(8, len(image_data_list)))):
            t = threading.Thread(target=self.loader_worker, daemon=True)
            t.start()

        # Bind click
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.enqueue_visible_and_neighbors()
        self.root.after(100, self.periodic_update)

    # ----------------- Loader -----------------
    def loader_worker(self):
        while not self.stop_event.is_set():
            try:
                idx = self.load_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self.cache_lock:
                if idx in self.pil_cache:
                    self.load_queue.task_done()
                    continue
            path = self.image_data_list[idx]["image_name"]
            try:
                bgr = cv2.imread(path)
                if bgr is None:
                    self.load_queue.task_done()
                    continue
                _, _, scale, disp_w, disp_h = self.img_positions[idx]
                resized = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                self.result_queue.put((idx, pil_img))
            except Exception as e:
                print(f"Loader error for index {idx}: {e}")
            finally:
                self.load_queue.task_done()

    # ----------------- Click -----------------
    def on_canvas_click(self, event):
        x_c = self.canvas.canvasx(event.x)
        y_c = self.canvas.canvasy(event.y)
        for idx, (x, y, scale, disp_w, disp_h) in enumerate(self.img_positions):
            if x <= x_c <= x + disp_w and y <= y_c <= y + disp_h:
                x_orig = (x_c - x) / scale
                y_orig = (y_c - y) / scale
                self.handle_click(idx, x_orig, y_orig)
                break

    def handle_click(self, idx, x_orig, y_orig):
        # Remove old red markers
        for m in self.red_marker_items:
            self.canvas.delete(m)
        self.red_marker_items.clear()

        # Draw new red marker
        x_canvas, y_canvas, scale, _, _ = self.img_positions[idx]
        m_red = self.create_marker_canvas(x_canvas + x_orig * scale,
                                          y_canvas + y_orig * scale, color="red")
        self.red_marker_items.append(m_red)

        # Save result
        self.result["idx"], self.result["x"], self.result["y"] = idx, x_orig, y_orig

        # Call click callback if provided
        if self.click_callback:
            def worker():
                try:
                    data = self.click_callback(idx, x_orig, y_orig, self)
                    self.draw_correspondences(data)
                except Exception as e:
                    print(f"click_callback error: {e}")
            threading.Thread(target=worker, daemon=True).start()

    # ----------------- Draw helpers -----------------
    def create_marker_canvas(self, x, y, color="red", size=6):
        half = size / 2
        return self.canvas.create_rectangle(x - half, y - half, x + half, y + half, fill=color, outline=color)

    def draw_canvas_image(self, idx, photo):
        if idx in self.drawn_image_items:
            self.canvas.delete(self.drawn_image_items[idx])
        x, y, scale, disp_w, disp_h = self.img_positions[idx]
        img_id = self.canvas.create_image(x, y, image=photo, anchor=NW)
        self.drawn_image_items[idx] = img_id
        name = self.image_data_list[idx]["image_name"].split("/")[-1].rsplit(".", 1)[0]
        if idx in self.drawn_text_items:
            self.canvas.delete(self.drawn_text_items[idx])
        tid = self.canvas.create_text(x + disp_w / 2, y + disp_h + 12, text=name, fill="white")
        self.drawn_text_items[idx] = tid

    def draw_correspondences(self, data):
        # Clear old yellow markers
        for m in self.yellow_marker_items:
            self.canvas.delete(m)
        self.yellow_marker_items.clear()

        # Draw yellow markers
        for entry in data:
            other_idx = entry["image_index"]
            x_other = entry["pixel_x"]
            y_other = entry["pixel_y"]
            if other_idx == self.result["idx"]:
                continue
            x_img, y_img, sc, disp_w, disp_h = self.img_positions[other_idx]
            x_disp, y_disp = x_other * sc, y_other * sc
            if 0 <= x_disp <= disp_w and 0 <= y_disp <= disp_h:
                mx = self.create_marker_canvas(x_img + x_disp, y_img + y_disp, color="yellow")
                self.yellow_marker_items.append(mx)

    # ----------------- Visible images -----------------
    def get_visible_indices(self):
        y0 = self.canvas.canvasy(0)
        y1 = self.canvas.canvasy(self.canvas.winfo_height())
        margin = self.row_height * 2
        top_row = max(0, int((y0 - margin) // self.row_height))
        bottom_row = min((len(self.image_data_list) + self.cols - 1) // self.cols - 1,
                         int((y1 + margin) // self.row_height))
        indices = []
        for r in range(top_row, bottom_row + 1):
            start = r * self.cols
            end = min(len(self.image_data_list), start + self.cols)
            indices.extend(range(start, end))
        return set(indices)

    def enqueue_visible_and_neighbors(self):
        vis = self.get_visible_indices()
        extra = set()
        n_rows = (len(self.image_data_list) + self.cols - 1) // self.cols
        for idx in vis:
            row = idx // self.cols
            for r in range(max(0, row - 1), min(n_rows, row + 2)):
                start = r * self.cols
                extra.update(range(start, min(start + self.cols, len(self.image_data_list))))
        for idx in extra:
            with self.cache_lock:
                if idx not in self.pil_cache:
                    try:
                        self.load_queue.put_nowait(idx)
                    except queue.Full:
                        pass

    def process_loader_results(self):
        while True:
            try:
                idx, pil_img = self.result_queue.get_nowait()
            except queue.Empty:
                break
            with self.cache_lock:
                self.pil_cache[idx] = pil_img
                self.pil_cache.move_to_end(idx)
                while len(self.pil_cache) > self.MAX_CACHE_ITEMS:
                    self.pil_cache.popitem(last=False)
                photo = ImageTk.PhotoImage(pil_img)
                self.photo_cache[idx] = photo
                if idx in self.get_visible_indices():
                    self.draw_canvas_image(idx, photo)
            self.result_queue.task_done()

    def periodic_update(self):
        self.enqueue_visible_and_neighbors()
        self.process_loader_results()
        vis = self.get_visible_indices()
        for idx in vis:
            with self.cache_lock:
                if idx in self.photo_cache and idx not in self.drawn_image_items:
                    self.draw_canvas_image(idx, self.photo_cache[idx])
        self.root.after(300, self.periodic_update)

    # ----------------- Run -----------------
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.root.after(100, self.periodic_update)
        self.root.mainloop()
