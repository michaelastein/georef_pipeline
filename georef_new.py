import numpy as np
from camproject import Camera, Extrinsics
import cv2
import tkinter as tk
from tkinter import filedialog
import sys
import os
from PIL import Image, ImageTk
import pyproj
import csv
from pathlib import Path
from plot_maps import plot_google_maps
from plot_cad import plot_cad_map  # optional

# ---------------- User Parameters ----------------
DRONE_OFFSET_NORTH = 0.0
DRONE_OFFSET_EAST  = 0.0
DRONE_OFFSET_UP    = 0.0
PANEL_HEIGHT_CORRECTION = 0.0

# ---------------- Utility Functions ----------------

def extract_metadata_from_csv(img_paths):
    """Extract GPS and orientation metadata from CSV, mapped to image filenames."""
    folder = Path(img_paths[0]).parent
    csv_files = list(folder.glob("*.csv"))
    
    if len(csv_files) != 1:
        from tkinter.filedialog import askopenfilename
        csv_path = askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            raise FileNotFoundError("No CSV file selected")
    else:
        csv_path = str(csv_files[0])
    
    print(f"Using CSV file: {csv_path}")
    
    metadata_dict = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_file = row.get("wiris_image", "").strip()
            if not img_file:
                continue
            metadata_dict[img_file] = {
                "lat": float(row.get("Latitude", "nan")),
                "lon": float(row.get("Longitude", "nan")),
                "alt": float(row.get("alt", "nan")),
                "yaw": float(row.get("GimbalYawE", "nan")),
                "pitch": float(row.get("pitch_agisoft", "nan")),
                "roll": float(row.get("roll", "nan")),
                "rel_alt": float(row.get("CHeight", "nan"))
            }

    data_entries = []
    for idx, path in enumerate(img_paths):
        fname = os.path.basename(path)
        meta = metadata_dict.get(fname)
        width, height = Image.open(path).size
        if meta:
            gps_tuple = (meta["lat"], meta["lon"], meta["alt"], meta["rel_alt"])
            angles = (meta["yaw"], meta["pitch"], meta["roll"])
        else:
            print(f"Warning: No CSV metadata for {fname}")
            gps_tuple = (None, None, None, None)
            angles = (None, None, None)
        data_entries.append({
            "image_index": idx,
            "image_path": path,
            "pixel_x": width / 2,
            "pixel_y": height / 2,
            "gps": gps_tuple,
            "yaw": angles[0],
            "pitch": angles[1],
            "roll": angles[2],
            "image_size": (width, height)
        })
    return data_entries

def enu_to_gps(x, y, origin_lat, origin_lon):
    zone = int((origin_lon + 180)/6)+1
    epsg_code = 32600 + zone if origin_lat >= 0 else 32700 + zone
    utm_crs = pyproj.CRS.from_epsg(epsg_code)

    t_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    t_from_utm = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    utm_x0, utm_y0 = t_to_utm.transform(origin_lon, origin_lat)
    utm_x = utm_x0 + x
    utm_y = utm_y0 + y

    lon, lat = t_from_utm.transform(utm_x, utm_y)
    return lat, lon

def pixel_to_camproject(u, v, width, height):
    return np.array([width - 1 - u, height - 1 - v])

def load_image_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), ("All files", "*.*")]
    )
    if not file_path:
        print("No image selected. Exiting program.")
        sys.exit(0)
    img = Image.open(file_path)
    print("Loaded:", file_path)
    return img, file_path

def select_pixel_gui(img_array):
    clicked_point = {}
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point['u'] = x
            clicked_point['v'] = y
            cv2.destroyAllWindows()
    cv2.imshow("Click on target pixel (press ESC to skip)", img_array)
    cv2.setMouseCallback("Click on target pixel (press ESC to skip)", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    width, height = img_array.shape[1], img_array.shape[0]
    u = clicked_point.get('u', width // 2)
    v = clicked_point.get('v', height // 2)
    return u, v

def show_image_with_buttons(img_array, u, v, filename):
    img_with_dot = img_array.copy()
    cv2.circle(img_with_dot, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
    pil_img = Image.fromarray(cv2.cvtColor(img_with_dot, cv2.COLOR_BGR2RGB))

    root = tk.Tk()
    root.title(os.path.basename(filename))
    state = {"img": pil_img}
    canvas = tk.Label(root)
    canvas.pack()

    def update_image():
        tk_img = ImageTk.PhotoImage(state["img"], master=root)
        canvas.configure(image=tk_img)
        canvas.image = tk_img

    def rotate_left():
        state["img"] = state["img"].rotate(90, expand=True)
        update_image()

    def rotate_right():
        state["img"] = state["img"].rotate(-90, expand=True)
        update_image()

    def on_close():
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="⟲ Rotate Left", command=rotate_left).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="⟳ Rotate Right", command=rotate_right).pack(side=tk.LEFT, padx=5)
    update_image()
    root.mainloop()

# ---------------- Main Mapper Class ----------------
class DroneMapper:
    def __init__(self, drone_offset_north=DRONE_OFFSET_NORTH, drone_offset_east=DRONE_OFFSET_EAST,
                 drone_offset_up=DRONE_OFFSET_UP, panel_height=PANEL_HEIGHT_CORRECTION):
        self.DRONE_OFFSET_NORTH = drone_offset_north
        self.DRONE_OFFSET_EAST = drone_offset_east
        self.DRONE_OFFSET_UP = drone_offset_up
        self.PANEL_HEIGHT_CORRECTION = panel_height

    def get_target_gps(self, u, v, gps, angles, image_size):
        drone_lat, drone_lon, drone_alt, rel_alt = gps
        yaw, pitch, roll = angles
        width, height = image_size

        cam = Camera()
        f_px = 13 * width / 10.88
        cx = width / 2
        cy = height / 2
        cam.intrinsics(width, height, f_px, cx, cy)

        ext = Extrinsics()
        corrected_altitude = rel_alt - self.PANEL_HEIGHT_CORRECTION + self.DRONE_OFFSET_UP
        ext.setPose(
            X=self.DRONE_OFFSET_EAST,
            Y=self.DRONE_OFFSET_NORTH,
            Z=corrected_altitude,
            roll=-roll,
            pitch=90 - pitch,
            yaw=-yaw - 90,
            order="ZYX"
        )
        ext.setGimbal(roll=0, pitch=0, yaw=0, order="ZYX")
        cam.attitudeMat(ext.transform())

        plane = np.array([0, 0, 1, 0])
        target_3D = cam.reprojectToPlane(pixel_to_camproject(u, v, width, height), plane)
        target_lat, target_lon = enu_to_gps(target_3D[0], target_3D[1], drone_lat, drone_lon)
        return target_lat, target_lon

    def get_target_gps_array(self, data_array):
        """
        Returns a list of dicts with 'target_gps' added.
        """
        for data in data_array:
            u = data.get('pixel_x', data['image_size'][0] / 2)
            v = data.get('pixel_y', data['image_size'][1] / 2)
            yaw, pitch, roll = data['yaw'], data['pitch'], data['roll']

            target_lat, target_lon = self.get_target_gps(
                u, v, gps=data['gps'], angles=(yaw, pitch, roll), image_size=data['image_size']
            )
            data['target_gps'] = (target_lat, target_lon)
        return data_array

    def process_images_gui(self, img_paths):
        data_array = extract_metadata_from_csv(img_paths)

        for entry in data_array:
            # Load image
            img = Image.open(entry['image_path'])
            img_array = np.array(img)
            if img_array.dtype == np.uint16:
                img_array = (img_array / 256).astype(np.uint8)
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            else:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Select pixel
            u, v = select_pixel_gui(img_array)
            entry['pixel_x'] = u
            entry['pixel_y'] = v

        # Compute target GPS for all images
        data_array = self.get_target_gps_array(data_array)

        # Plot and show
        # Inside DroneMapper.process_images_gui after computing target GPS
        for entry in data_array:


            # Schedule Google Maps plot in main thread
            root.after(0, lambda e=entry: plot_google_maps(
                target_gps=e['target_gps'],
                corner_gps=None,
                drone_gps=(e['gps'][0] + self.DRONE_OFFSET_NORTH,
                        e['gps'][1] + self.DRONE_OFFSET_EAST)
            ))

            # Schedule CAD map plot in main thread
            root.after(0, lambda e=entry: plot_cad_map(
                target_gps=e['target_gps'],
                corner_gps=None,
                drone_gps=(e['gps'][0] + self.DRONE_OFFSET_NORTH,
                        e['gps'][1] + self.DRONE_OFFSET_EAST)
            ))


            show_image_with_buttons(img_array, u, v, filename=entry['image_path'])


# ---------------- CLI ----------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    img_paths = filedialog.askopenfilenames(
        title="Select images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), ("All files", "*.*")]
    )
    if not img_paths:
        print("No images selected. Exiting.")
        sys.exit(0)
    
    mapper = DroneMapper()
    mapper.process_images_gui(img_paths)
