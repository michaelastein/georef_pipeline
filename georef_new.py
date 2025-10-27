import numpy as np
import camproject
from camproject import Camera, Extrinsics
import cv2
import tkinter as tk
from tkinter import filedialog
import sys
import os
from PIL import Image, ImageTk
import piexif
import pyproj
from plot_maps import plot_google_maps
from plot_cad import plot_cad_map  # optional

# ---------------- User Parameters ----------------
# These allow manual correction of the drone’s position and panel height
DRONE_OFFSET_NORTH = 0.0   # meters: positive moves drone north
DRONE_OFFSET_EAST  = 0.5    # meters: positive moves drone east
DRONE_OFFSET_UP    = 0.0    # meters: positive moves drone up (altitude)
PANEL_HEIGHT_CORRECTION = 2.0  # meters: height of the solar panel above ground

# ---------------- Utility Functions ----------------
def rational_to_float(r):
    try:
        return r[0] / r[1]
    except Exception:
        return float(r)

def gps_to_decimal(coord, ref):
    deg = rational_to_float(coord[0])
    minute = rational_to_float(coord[1])
    sec = rational_to_float(coord[2])
    val = deg + minute / 60.0 + sec / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode(errors='ignore')
    if ref in ['S', 's', 'W', 'w']:
        val = -val
    return val

def parse_description_from_exif(exif_dict):
    desc = exif_dict.get('0th', {}).get(piexif.ImageIFD.ImageDescription, b'')
    if isinstance(desc, bytes):
        desc = desc.decode(errors='ignore')
    yaw = pitch = roll = rel_alt = None
    if desc:
        for part in str(desc).split(","):
            kv = part.strip().split("=")
            if len(kv) == 2:
                key, value = kv
                key_lower = key.strip().lower()
                try:
                    if key_lower == "yaw":
                        yaw = float(value)
                    elif key_lower == "pitch":
                        pitch = float(value)
                    elif key_lower == "roll":
                        roll = float(value)
                    elif key_lower in ["relativealt", "rel_alt"]:
                        rel_alt = float(value)
                except ValueError:
                    pass
    if yaw is None or pitch is None or roll is None:
        raise ValueError("Missing yaw, pitch, or roll in image description.")
    return yaw, pitch, roll, rel_alt

def extract_gps_from_exif(exif_dict):
    gps_ifd = exif_dict.get("GPS", {})
    lat_tag = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)
    lon_tag = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)
    alt_tag = gps_ifd.get(piexif.GPSIFD.GPSAltitude)
    alt_ref = gps_ifd.get(piexif.GPSIFD.GPSAltitudeRef, 0)
    if not (lat_tag and lat_ref and lon_tag and lon_ref and alt_tag is not None):
        raise ValueError("Missing GPS fields in EXIF.")
    lat = gps_to_decimal(lat_tag, lat_ref)
    lon = gps_to_decimal(lon_tag, lon_ref)
    alt = rational_to_float(alt_tag)
    if isinstance(alt_ref, (bytes, bytearray)):
        alt_ref_val = int(alt_ref[0])
    else:
        alt_ref_val = int(alt_ref)
    if alt_ref_val == 1:
        alt = -alt
    return lat, lon, alt

def load_image():
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

def select_pixel(img_array):
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

# ---------------- ENU to GPS ----------------
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

# ---------------- Pixel mapping helper ----------------
def pixel_to_camproject(u, v, width, height):
    """Map image (u,v) to camproject camera frame."""
    return np.array([width - 1 - u, height - 1 - v])

# ---------------- Main ----------------
if __name__ == "__main__":
    img, file_path = load_image()
    width, height = img.size

    # Load EXIF
    try:
        exif_dict = piexif.load(img.info['exif']) if 'exif' in img.info else piexif.load(file_path)
    except Exception:
        exif_dict = piexif.load(file_path)

    yaw, pitch, roll, rel_alt = parse_description_from_exif(exif_dict)
    drone_lat, drone_lon, drone_alt = extract_gps_from_exif(exif_dict)

    print("Drone GPS:", drone_lat, drone_lon, drone_alt)
    print("Yaw/Pitch/Roll from EXIF:", yaw, pitch, roll)
    print("Relative altitude:", rel_alt)

    img_array = np.array(img)
    if img_array.dtype == np.uint16:
        img_array = (img_array / 256).astype(np.uint8)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    # ---------------- Pixel Selection ----------------
    u, v = select_pixel(img_array)

    # ---------------- Camera Setup ----------------
    cam = Camera()
    f_px = 13 * width / 10.88  # 13mm focal length, 10.88mm sensor width
    cx = width / 2
    cy = height / 2
    cam.intrinsics(width, height, f_px, cx, cy)

    # ---------------- Extrinsics ----------------
    ext = Extrinsics()
    corrected_altitude = rel_alt - PANEL_HEIGHT_CORRECTION + DRONE_OFFSET_UP
    ext.setPose(
        X=DRONE_OFFSET_EAST,
        Y=DRONE_OFFSET_NORTH,
        Z=corrected_altitude,
        roll=-roll,
        pitch=90 - pitch,
        yaw=-yaw - 90,
        order="ZYX"
    )
    ext.setGimbal(roll=0, pitch=0, yaw=0, order="ZYX")  # gimbal angles from EXIF are used
    cam.attitudeMat(ext.transform())

    # ---------------- Reproject to Ground Plane ----------------
    plane = np.array([0, 0, 1, 0])  # Z=0
    target_3D = cam.reprojectToPlane(pixel_to_camproject(u, v, width, height), plane)
    corners_px = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    corners_3D = np.array([cam.reprojectToPlane(pixel_to_camproject(c[0], c[1], width, height), plane)[0:3] for c in corners_px])

    # ---------------- ENU -> GPS ----------------
    target_lat, target_lon = enu_to_gps(target_3D[0], target_3D[1], drone_lat, drone_lon)
    corner_gps = [enu_to_gps(c[0], c[1], drone_lat, drone_lon) for c in corners_3D]

    # ---------------- Corrected Drone GPS ----------------
    corrected_drone_lat, corrected_drone_lon = enu_to_gps(DRONE_OFFSET_EAST, DRONE_OFFSET_NORTH, drone_lat, drone_lon)

    # ---------------- Plot ----------------
    plot_google_maps(
        target_gps=(target_lat, target_lon),
        corner_gps=corner_gps,
        drone_gps=(corrected_drone_lat, corrected_drone_lon)
    )
    plot_cad_map(
        target_gps=(target_lat, target_lon),
        corner_gps=corner_gps,
        drone_gps=(corrected_drone_lat, corrected_drone_lon)
    )

    # ---------------- Show Image ----------------
    show_image_with_buttons(img_array, u, v, filename=file_path)
