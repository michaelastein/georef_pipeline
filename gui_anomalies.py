from tkinter import Tk, Toplevel, Label, Entry, Button, StringVar, END
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk, ImageDraw
from anomaly_matching import scale_coordinates  # centralized scaling function

def launch_anomaly_gui(image_data_list, original_image_size=None):
    """
    GUI for selecting an image from a batch, picking a pixel, and optionally selecting a CSV file.

    Features:
        - Displays images from a batch.
        - Allows user to click on a pixel to select a target point.
        - Provides entries for manual X/Y input.
        - Optional CSV selection.
        - Coordinates are scaled to original image size if provided.

    Parameters:
        image_data_list (list): List of dicts containing image info, e.g., [{"image_name": ..., "image_size": ...}, ...].
        original_image_size (tuple, optional): Reference size (width, height) to scale clicked coordinates.

    Returns:
        tuple: (idx, x, y, csv_path)
            idx (int): index of the selected image in image_data_list.
            x, y (int): pixel coordinates relative to displayed image (or scaled if original_image_size provided).
            csv_path (str or None): path to selected CSV file, or None if not selected.
    """
    # --- Main Tkinter windows ---
    root = Tk()
    root.title("Anomalies GUI")
    win = Toplevel(root)
    win.title("Select Image, Pixel, and CSV")

    # --- GUI Widgets ---
    lbl_img = Label(win)  # display image
    lbl_img.grid(row=0, column=0, columnspan=3)

    # X/Y entries for manual input
    Label(win, text="X:").grid(row=1, column=0)
    entry_x = Entry(win)
    entry_x.grid(row=1, column=1)

    Label(win, text="Y:").grid(row=2, column=0)
    entry_y = Entry(win)
    entry_y.grid(row=2, column=1)

    # CSV selection display
    csv_strvar = StringVar(value="No file selected")
    Label(win, textvariable=csv_strvar, wraplength=300, fg="gray").grid(row=3, column=1, columnspan=2)

    # Error messages
    error_var = StringVar(value="")
    Label(win, textvariable=error_var, fg="red", wraplength=400).grid(row=6, column=0, columnspan=3)

    # --- Internal state ---
    selected_img_entry = {"clicked": False}  # stores info about selected image & click
    img = display_img = tk_img = None
    idx = None
    csv_path = None
    x = y = None

    # --- Functions ---

    def select_image():
        """Let user select an image from the batch and display it in the GUI."""
        nonlocal selected_img_entry, idx, img, display_img, tk_img
        path = askopenfilename(
            title="Select one image from the batch",
            initialdir=image_data_list[0]["image_name"] if image_data_list else ".",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            error_var.set("No image selected.")
            return

        # Match selection to batch
        for i, entry in enumerate(image_data_list):
            if entry["image_name"] == path:
                selected_img_entry.update(entry)
                selected_img_entry["clicked"] = False  # reset click flag
                idx = i
                break
        else:
            error_var.set("Selected image not in the batch.")
            return

        # Load and display image
        img = Image.open(selected_img_entry["image_name"]).convert("RGB")
        display_img = img.copy()
        tk_img = ImageTk.PhotoImage(display_img)
        lbl_img.config(image=tk_img)
        lbl_img.image = tk_img
        win.title(f"Select Pixel and CSV for {selected_img_entry['image_name'].split('/')[-1]}")

    def draw_marker(px, py):
        """Draw a red dot on the clicked pixel in the displayed image."""
        nonlocal tk_img, display_img
        display_img = img.copy()
        r = 5  # radius of dot
        d = ImageDraw.Draw(display_img)
        d.ellipse((px - r, py - r, px + r, py + r), fill="red")
        tk_img = ImageTk.PhotoImage(display_img)
        lbl_img.config(image=tk_img)
        lbl_img.image = tk_img

    def on_click(event):
        """
        Handle click on the image.
        Scales coordinates relative to original image if original_image_size is provided.
        """
        if img is None:
            return
        # Coordinates relative to displayed image
        px = int(event.x * img.width / lbl_img.winfo_width())
        py = int(event.y * img.height / lbl_img.winfo_height())

        # Scale to original image if requested
        if original_image_size:
            current_size = selected_img_entry.get("image_size", (img.width, img.height))
            display_x, display_y = scale_coordinates(px, py, current_size, original_image_size)
        else:
            display_x, display_y = px, py

        # Update entry boxes
        entry_x.delete(0, END)
        entry_x.insert(0, str(int(display_x)))
        entry_y.delete(0, END)
        entry_y.insert(0, str(int(display_y)))

        draw_marker(px, py)
        selected_img_entry["clicked"] = True
        selected_img_entry["last_click_px"] = px
        selected_img_entry["last_click_py"] = py

    lbl_img.bind("<Button-1>", on_click)  # bind left click to image

    def select_csv():
        """Let user select a CSV file and store path."""
        nonlocal csv_path
        path = askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
        if path:
            csv_path = path
            csv_strvar.set(path)

    def on_submit():
        """Submit selection, read coordinates, scale if needed, and close GUI."""
        nonlocal x, y
        if "image_name" not in selected_img_entry:
            error_var.set("No image selected.")
            return
        try:
            if selected_img_entry.get("clicked"):
                # Return coordinates relative to displayed image
                x = selected_img_entry["last_click_px"]
                y = selected_img_entry["last_click_py"]
            else:
                # Manual input
                px = float(entry_x.get())
                py = float(entry_y.get())
                if original_image_size:
                    current_size = selected_img_entry.get("image_size", (img.width, img.height))
                    x, y = scale_coordinates(px, py, original_image_size, current_size)
                else:
                    x, y = px, py
        except ValueError:
            error_var.set("Invalid coordinates.")
            return

        root.destroy()  # close GUI

    # --- Buttons ---
    Button(win, text="Select Image", command=select_image).grid(row=4, column=0)
    Button(win, text="Select CSV", command=select_csv).grid(row=4, column=1)
    Button(win, text="Submit", command=on_submit).grid(row=5, column=0, columnspan=3, pady=5)

    root.mainloop()

    return idx, x, y, csv_path
