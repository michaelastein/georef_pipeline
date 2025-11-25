import threading
import queue
from collections import OrderedDict
from tkinter import Tk, Canvas, Scrollbar, LEFT, RIGHT, Y, NW
from PIL import Image, ImageTk
import cv2

class CanvasGUI:
    """
    Canvas-based GUI to display a grid of images with clickable markers.
    Features:
        - Efficient thumbnail display with scrollable canvas.
        - Multi-threaded image loading and caching.
        - Click on image to mark a pixel (red marker).
        - Supports displaying correspondences in other images (yellow markers).
    """

    def __init__(self, image_data_list, orig_sizes, click_callback=None):
        """
        Initialize GUI.

        Parameters:
            image_data_list (list of dict): Each dict must have "image_name" (filepath).
            orig_sizes (list of tuples): Original image sizes (width, height).
            click_callback (function, optional): Function called on image click with args (idx, x, y, gui_instance).
        """
        self.image_data_list = image_data_list
        self.orig_sizes = orig_sizes
        self.click_callback = click_callback

        # Marker management
        self.red_marker_items = []      # For user-selected points
        self.yellow_marker_items = []   # For correspondences

        # Last clicked point info
        self.result = {"idx": None, "x": None, "y": None}

        # ---------------- Tkinter Setup ----------------
        self.root = Tk()
        self.root.title("Images Grid (Canvas Viewer)")
        self.canvas_w, self.canvas_h = 1400, 800
        self.canvas = Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="black")

        # Scrollbar
        scrollbar = Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=LEFT, fill="both", expand=1)
        scrollbar.pack(side=RIGHT, fill=Y)

        # ---------------- Layout ----------------
        self.cols = 6                  # Number of columns
        self.thumb_size = 200          # Thumbnail size in pixels
        self.padding = 10              # Padding between images
        self.text_height = 18          # Space for filename
        self.row_height = self.thumb_size + self.text_height + self.padding

        # ---------------- Caches ----------------
        self.pil_cache = OrderedDict()  # Keep recent loaded images as PIL.Image
        self.photo_cache = {}           # Keep ImageTk.PhotoImage for canvas
        self.MAX_CACHE_ITEMS = 200
        self.cache_lock = threading.Lock()

        # ---------------- Loader Queues ----------------
        self.load_queue = queue.Queue()    # Indices of images to load
        self.result_queue = queue.Queue()  # Loaded PIL.Image results
        self.stop_event = threading.Event()

        # ---------------- Compute positions ----------------
        self.img_positions = []
        for i, (w, h) in enumerate(orig_sizes):
            scale = min(self.thumb_size / h, self.thumb_size / w)
            disp_w, disp_h = int(w * scale), int(h * scale)
            col = i % self.cols
            row = i // self.cols
            x = self.padding + col * (self.thumb_size + self.padding)
            y = self.padding + row * self.row_height
            self.img_positions.append((x, y, scale, disp_w, disp_h))

        # Configure canvas scroll region
        n_rows = (len(image_data_list) + self.cols - 1) // self.cols
        total_height = self.padding + n_rows * self.row_height
        total_width = self.padding + self.cols * (self.thumb_size + self.padding)
        self.canvas.config(scrollregion=(0, 0, total_width, total_height))

        # Track drawn items on canvas
        self.drawn_image_items = {}
        self.drawn_text_items = {}

        # ---------------- Start loader threads ----------------
        for _ in range(max(2, min(8, len(image_data_list)))):
            t = threading.Thread(target=self.loader_worker, daemon=True)
            t.start()

        # Bind click event
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Preload visible images
        self.enqueue_visible_and_neighbors()
        self.root.after(100, self.periodic_update)

    # ----------------- Loader Worker -----------------
    def loader_worker(self):
        """Thread function to load images in the background."""
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

    # ----------------- Canvas Click -----------------
    def on_canvas_click(self, event):
        """Convert canvas coordinates to image coordinates and handle click."""
        x_c = self.canvas.canvasx(event.x)
        y_c = self.canvas.canvasy(event.y)
        for idx, (x, y, scale, disp_w, disp_h) in enumerate(self.img_positions):
            if x <= x_c <= x + disp_w and y <= y_c <= y + disp_h:
                x_orig = (x_c - x) / scale
                y_orig = (y_c - y) / scale
                self.handle_click(idx, x_orig, y_orig)
                break

    def handle_click(self, idx, x_orig, y_orig):
        """Update red marker, store click result, call optional callback."""
        # Clear old red markers
        for m in self.red_marker_items:
            self.canvas.delete(m)
        self.red_marker_items.clear()

        # Draw new red marker
        x_canvas, y_canvas, scale, _, _ = self.img_positions[idx]
        m_red = self.create_marker_canvas(x_canvas + x_orig * scale,
                                          y_canvas + y_orig * scale, color="red")
        self.red_marker_items.append(m_red)

        # Save last click
        self.result["idx"], self.result["x"], self.result["y"] = idx, x_orig, y_orig

        # Call click callback asynchronously
        if self.click_callback:
            def worker():
                try:
                    data = self.click_callback(idx, x_orig, y_orig, self)
                    self.draw_correspondences(data)
                except Exception as e:
                    print(f"click_callback error: {e}")
            threading.Thread(target=worker, daemon=True).start()

    # ----------------- Draw Helpers -----------------
    def create_marker_canvas(self, x, y, color="red", size=6):
        """Draw a small rectangle marker at canvas coordinates."""
        half = size / 2
        return self.canvas.create_rectangle(x - half, y - half, x + half, y + half, fill=color, outline=color)

    def draw_canvas_image(self, idx, photo):
        """Draw image and filename text on canvas."""
        if idx in self.drawn_image_items:
            self.canvas.delete(self.drawn_image_items[idx])
        x, y, scale, disp_w, disp_h = self.img_positions[idx]
        img_id = self.canvas.create_image(x, y, image=photo, anchor=NW)
        self.drawn_image_items[idx] = img_id
        # Draw filename
        name = self.image_data_list[idx]["image_name"].split("/")[-1].rsplit(".", 1)[0]
        if idx in self.drawn_text_items:
            self.canvas.delete(self.drawn_text_items[idx])
        tid = self.canvas.create_text(x + disp_w / 2, y + disp_h + 12, text=name, fill="white")
        self.drawn_text_items[idx] = tid

    def draw_correspondences(self, data):
        """Draw yellow markers for related points in other images."""
        for m in self.yellow_marker_items:
            self.canvas.delete(m)
        self.yellow_marker_items.clear()
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

    # ----------------- Visible Images -----------------
    def get_visible_indices(self):
        """Return indices of images currently visible in the canvas viewport."""
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
        """Enqueue visible images and their neighbors for loading."""
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

    # ----------------- Loader Results -----------------
    def process_loader_results(self):
        """Process loaded PIL images and update canvas."""
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
        """Update loop: enqueue images, process loaded results, and redraw visible images."""
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
        """Start Tkinter main loop."""
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.root.after(100, self.periodic_update)
        self.root.mainloop()
