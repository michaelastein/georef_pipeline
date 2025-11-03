# gui_anomalies.py

from tkinter import Tk, Toplevel, Label, Entry, Button, StringVar, END
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk, ImageDraw


def launch_anomaly_gui(image_data_list):
    """
    GUI to select an image from a batch, pick a pixel, optionally select a CSV file.
    Returns:
        idx: index of selected image in image_data_list
        x, y: pixel coordinates selected by the user
        csv_path: path to CSV file (or None if not selected)
    """
    root = Tk()
    root.title("Anomalies GUI")

    # Create a secondary window for controls and image display
    win = Toplevel(root)
    win.title("Select Image, Pixel, and CSV")

    # Label to display the selected image
    lbl_img = Label(win)
    lbl_img.grid(row=0, column=0, columnspan=3)

    # Entry boxes for X and Y coordinates
    Label(win, text="X:").grid(row=1, column=0)
    entry_x = Entry(win)
    entry_x.grid(row=1, column=1)

    Label(win, text="Y:").grid(row=2, column=0)
    entry_y = Entry(win)
    entry_y.grid(row=2, column=1)

    # Label to show selected CSV file
    csv_strvar = StringVar(value="No file selected")
    Label(win, textvariable=csv_strvar, wraplength=300, fg="gray").grid(row=3, column=1, columnspan=2)

    # Label for error messages
    error_var = StringVar(value="")
    Label(win, textvariable=error_var, fg="red", wraplength=400).grid(row=6, column=0, columnspan=3)

    # Internal state variables
    selected_img_entry = None  # stores selected image metadata
    img = display_img = tk_img = None  # PIL images and Tkinter image
    idx = None  # index in image_data_list
    csv_path = None
    x = y = None  # pixel coordinates

    # ------------------- Image selection -------------------
    def select_image():
        """Let user select an image from file system, must exist in image_data_list."""
        nonlocal selected_img_entry, idx, img, display_img, tk_img
        path = askopenfilename(
            title="Select one image from the batch",
            initialdir=image_data_list[0]["image_path"] if image_data_list else ".",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            error_var.set("No image selected.")
            return

        # Find the index of the selected image in image_data_list
        for i, entry in enumerate(image_data_list):
            if entry["image_path"] == path:
                selected_img_entry = entry
                idx = i
                break
        else:
            error_var.set("Selected image not in the batch.")
            return

        # Load and display image
        img = Image.open(selected_img_entry["image_path"]).convert("RGB")
        display_img = img.copy()
        tk_img = ImageTk.PhotoImage(display_img)
        lbl_img.config(image=tk_img)
        lbl_img.image = tk_img
        win.title(f"Select Pixel and CSV for {selected_img_entry['image_path'].split('/')[-1]}")

    # ------------------- Draw marker on image -------------------
    def draw_marker(px, py):
        """
        Draw a red circle at (px, py) on the image.
        Updates the display_img and Tkinter label.
        """
        nonlocal tk_img, display_img
        display_img = img.copy()
        r = 5  # radius of marker
        d = ImageDraw.Draw(display_img)
        d.ellipse((px - r, py - r, px + r, py + r), fill="red")
        tk_img = ImageTk.PhotoImage(display_img)
        lbl_img.config(image=tk_img)
        lbl_img.image = tk_img

    # ------------------- Mouse click on image -------------------
    def on_click(event):
        """Handle user click on the displayed image, update coordinate entries."""
        if img is None:
            return
        px = int(event.x * img.width / lbl_img.winfo_width())
        py = int(event.y * img.height / lbl_img.winfo_height())
        entry_x.delete(0, END)
        entry_x.insert(0, str(px))
        entry_y.delete(0, END)
        entry_y.insert(0, str(py))
        draw_marker(px, py)

    lbl_img.bind("<Button-1>", on_click)

    # ------------------- CSV file selection -------------------
    def select_csv():
        """Let user select a CSV file and update label."""
        nonlocal csv_path
        path = askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
        if path:
            csv_path = path
            csv_strvar.set(path)

    # ------------------- Submit button -------------------
    def on_submit():
        """Validate inputs and close GUI, saving results."""
        nonlocal x, y
        if selected_img_entry is None:
            error_var.set("No image selected.")
            return
        try:
            x = float(entry_x.get())
            y = float(entry_y.get())
        except ValueError:
            error_var.set("Invalid coordinates.")
            return
        root.destroy()

    # ------------------- Buttons -------------------
    Button(win, text="Select Image", command=select_image).grid(row=4, column=0)
    Button(win, text="Select CSV", command=select_csv).grid(row=4, column=1)
    Button(win, text="Submit", command=on_submit).grid(row=5, column=0, columnspan=3, pady=5)

    # Start Tkinter main loop
    root.mainloop()

    return idx, x, y, csv_path
