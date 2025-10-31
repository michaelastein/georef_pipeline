# gui_anomalies.py
from tkinter import Tk, Toplevel, Label, Entry, Button, StringVar, END
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk, ImageDraw
import os
import avg_gps
import matching_anomalies
from feature_matching import build_correspondences_from_pixels

def launch_anomaly_gui(img_paths, orig_sizes, image_data_list, H_dict,
                       node_connected_component, shortest_path):
    """
    Launch the anomalies GUI:
    - Select one image from the batch
    - Click or enter pixel coordinates
    - Optionally select a CSV file
    - Compute correspondences and call avg_gps / matching_anomalies
    """
    root = Tk()
    root.title("Anomalies GUI")
    
    win = Toplevel(root)
    win.title("Select Image, Pixel, and CSV")

    # ---------------- UI Elements ----------------
    lbl_img = Label(win)
    lbl_img.grid(row=0, column=0, columnspan=3)

    Label(win, text="X:").grid(row=1, column=0)
    entry_x = Entry(win)
    entry_x.grid(row=1, column=1)

    Label(win, text="Y:").grid(row=2, column=0)
    entry_y = Entry(win)
    entry_y.grid(row=2, column=1)

    csv_strvar = StringVar(value="No file selected")
    Label(win, textvariable=csv_strvar, wraplength=300, fg="gray").grid(row=3, column=1, columnspan=2)

    error_var = StringVar(value="")
    error_label = Label(win, textvariable=error_var, fg="red", wraplength=400)
    error_label.grid(row=6, column=0, columnspan=3)

    csv_path = None
    selected_img_path = None
    tk_img = None
    display_img = None
    img = None
    idx = None
    w = h = 0

    # ---------------- Functions ----------------
    def select_image():
        nonlocal selected_img_path, img, display_img, tk_img, idx, w, h
        error_var.set("")
        path = askopenfilename(
            title="Select one image from the same folder",
            initialdir=os.path.dirname(img_paths[0]),
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            error_var.set("No image selected.")
            return
        if path not in img_paths:
            error_var.set("Selected image was not in the originally selected batch.")
            return

        selected_img_path = path
        idx = img_paths.index(path)
        w, h = orig_sizes[idx]

        img = Image.open(path).convert("RGB")
        display_img = img.copy()
        tk_img = ImageTk.PhotoImage(display_img)
        lbl_img.config(image=tk_img)
        lbl_img.image = tk_img
        win.title(f"Select Pixel and CSV for {os.path.basename(path)}")
        entry_x.delete(0, END)
        entry_y.delete(0, END)
        print(f"Loaded {path}")

    def draw_marker(x, y):
        nonlocal tk_img, display_img
        display_img = img.copy()
        draw = ImageDraw.Draw(display_img)
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill="red")
        tk_img = ImageTk.PhotoImage(display_img)
        lbl_img.config(image=tk_img)
        lbl_img.image = tk_img

    def update_marker_from_entry():
        try:
            x = int(entry_x.get())
            y = int(entry_y.get())
            if 0 <= x < img.width and 0 <= y < img.height:
                draw_marker(x, y)
        except ValueError:
            pass

    entry_x.bind("<KeyRelease>", lambda e: update_marker_from_entry())
    entry_y.bind("<KeyRelease>", lambda e: update_marker_from_entry())

    def on_click(event):
        if img is None:
            return
        x, y = event.x, event.y
        if lbl_img.winfo_width() != img.width or lbl_img.winfo_height() != img.height:
            x = int(x * img.width / lbl_img.winfo_width())
            y = int(y * img.height / lbl_img.winfo_height())
        entry_x.delete(0, END)
        entry_x.insert(0, str(x))
        entry_y.delete(0, END)
        entry_y.insert(0, str(y))
        draw_marker(x, y)

    lbl_img.bind("<Button-1>", on_click)

    def select_csv():
        nonlocal csv_path
        error_var.set("")
        path = askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            csv_path = path
            csv_strvar.set(path)
            print(f"Selected CSV: {path}")

    def on_submit():
        nonlocal csv_path, selected_img_path, idx, w, h
        error_var.set("")
        if selected_img_path is None:
            error_var.set("No image selected.")
            return

        try:
            x = float(entry_x.get())
            y = float(entry_y.get())
        except ValueError:
            error_var.set("Invalid coordinates.")
            return

        if not (0 <= x < w) or not (0 <= y < h):
            error_var.set(f"Coordinates ({x}, {y}) are out of bounds ({w}x{h}).")
            return

        # Build correspondences
        data = build_correspondences_from_pixels(
            idx, x, y, image_data_list=image_data_list, H_dict=H_dict,
            node_connected_component=node_connected_component,
            shortest_path=shortest_path
        )

        # Call avg_gps
        try:
            avg_gps.main(data)
        except Exception as e:
            error_var.set(f"avg_gps.main error: {e}")

        # Call matching_anomalies if CSV provided
        if csv_path:
            try:
                matching_anomalies.main(data, csv_path)
            except Exception as e:
                error_var.set(f"matching_anomalies.main error: {e}")

        print("Processing done. Ready for next image.")
        entry_x.delete(0, END)
        entry_y.delete(0, END)
        lbl_img.config(image="")
        lbl_img.image = None
        selected_img_path = None
        win.title("Select Image, Pixel, and CSV")

    # ---------------- Buttons ----------------
    Button(win, text="Select Image", command=select_image).grid(row=4, column=0)
    Button(win, text="Select CSV", command=select_csv).grid(row=4, column=1)
    Button(win, text="Submit", command=on_submit).grid(row=5, column=0, columnspan=3, pady=5)

    def on_close():
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
