import numpy as np
from camproject import Camera, Extrinsics  
import cv2
import tkinter as tk
from tkinter import filedialog
import sys
import os
from PIL import Image, ImageTk
from pyproj import Transformer, CRS
import csv
import rasterio
from pathlib import Path
from plot_google_maps import plot_google_maps 
from plot_cad import plot_cad_map       
import laspy
from scipy.spatial import cKDTree

# ---------------- User Parameters ----------------
DRONE_OFFSET_NORTH = 0.0
DRONE_OFFSET_EAST  = 0.0
DRONE_OFFSET_UP    = 0.0
UTM_EPSG = 25832  # UTM32 / ETRS89


# Camera parameters
FOCAL_LENGTH_MM = 13.0   
SENSOR_WIDTH_MM = 10.88

# ---------------- Utility Functions ----------------


# ---------------------- CSV Metadata Extraction ----------------------
def extract_metadata_from_csv(img_paths, image_columns=None):
    """
    Extract image metadata from a CSV file in the same folder as the images.

    Parameters:
        img_paths (list of str): List of image file paths.
        image_columns (list of str, optional): Possible column names in the CSV for image file names.
                                               Defaults to ["wiris_image", "pi_image", "image_name"].

    Returns:
        list of dict: Each dictionary contains metadata for one image, including:
            - 'image_index': Index of image in img_paths
            - 'image_name': Full path to image
            - 'gps': tuple of (lat, lon, alt, rel_alt) or (None, None, None, None) if missing
            - 'yaw', 'pitch', 'roll': Orientation angles (floats or None)
            - 'image_size': (width, height)
    """
    if image_columns is None:
        image_columns = ["wiris_image", "pi_image", "image_name"]

    # Locate CSV file in the same folder as the first image
    folder = Path(img_paths[0]).parent
    csv_files = list(folder.glob("*.csv"))

    if len(csv_files) != 1:
        # Ask user to select CSV if multiple or none are found
        csv_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv")]
        )
        if not csv_path:
            raise FileNotFoundError("No CSV file selected")
    else:
        csv_path = str(csv_files[0])

    print(f"Using CSV file: {csv_path}")

    # Read CSV into a dictionary keyed by image file name
    metadata_dict = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_file = None
            for col in image_columns:
                if col in row and row[col].strip():
                    img_file = row[col].strip()
                    break
            if not img_file:
                continue
            try:
                cheight_col = "CHeight" if "CHeight" in row else "Cheight" if "Cheight" in row else None
                metadata_dict[img_file] = {
                    "lat": float(row.get("Latitude", "nan")),
                    "lon": float(row.get("Longitude", "nan")),
                    "alt": float(row.get("alt", "nan")),
                    "yaw": float(row.get("GimbalYawE", "nan")),
                    "pitch": float(row.get("pitch_agisoft", "nan")),
                    "roll": float(row.get("roll", "nan")),
                    "rel_alt": float(row.get(cheight_col, "nan")) if cheight_col else float("nan")
                }
            except ValueError:
                print(f"Warning: Invalid CSV values for {img_file}, skipping metadata.")

    # Create output list with combined image paths and metadata
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
            "image_name": path,
            "gps": gps_tuple,
            "yaw": angles[0],
            "pitch": angles[1],
            "roll": angles[2],
            "image_size": (width, height)
        })

    return data_entries


# ---------------------- ENU to GPS Conversion ----------------------
def enu_to_gps(x, y, origin_lat, origin_lon):
    """
    Convert local ENU (East-North-Up) coordinates in meters to GPS coordinates.

    Parameters:
        x (float): East offset in meters.
        y (float): North offset in meters.
        origin_lat (float): Latitude of the ENU origin in degrees.
        origin_lon (float): Longitude of the ENU origin in degrees.

    Returns:
        tuple: (latitude, longitude) in degrees.
    """
    # Transformer from WGS84 (lat/lon) to UTM32
    transformer_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    transformer_from_utm = Transformer.from_crs(f"EPSG:{UTM_EPSG}", "EPSG:4326", always_xy=True)

    # Convert origin to UTM
    utm_x0, utm_y0 = transformer_to_utm.transform(origin_lon, origin_lat)
    # Apply offsets
    utm_x = utm_x0 + x
    utm_y = utm_y0 + y
    # Convert back to lat/lon
    lon, lat = transformer_from_utm.transform(utm_x, utm_y)
    return lat, lon


# ---------------------- Pixel to Camera Coordinates ----------------------
def pixel_to_camproject(u, v, width, height):
    """
    Convert image pixel coordinates to camera-projected coordinates.
    Flips horizontal axis to match camera convention.

    Parameters:
        u (float): Pixel x-coordinate.
        v (float): Pixel y-coordinate.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        numpy.ndarray: 2-element array [x_cam, y_cam].
    """
    return np.array([width - 1 - u, height - 1 - v])



# ---------------------- Pixel Selection GUI ----------------------
def select_pixel_gui(img_array):
    """
    Open a simple OpenCV window to let the user click on a pixel.

    Parameters:
        img_array (numpy.ndarray): Image array in BGR format.

    Returns:
        tuple: (u, v) pixel coordinates of the clicked point. If the user does not click,
               returns the image center coordinates.
    """
    clicked_point = {}

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point['u'] = x
            clicked_point['v'] = y
            cv2.destroyAllWindows()  # Close window after click

    # Display the image and set mouse callback
    cv2.imshow("Click on target pixel (press ESC to skip)", img_array)
    cv2.setMouseCallback("Click on target pixel (press ESC to skip)", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Default to center if no click
    width, height = img_array.shape[1], img_array.shape[0]
    u = clicked_point.get('u', width // 2)
    v = clicked_point.get('v', height // 2)
    return u, v


# ---------------------- Image Viewer with Rotate Buttons ----------------------
def show_image_with_buttons(img_array, u, v, filename):
    """
    Display an image in a Tkinter window with a marked point and rotate buttons.

    Parameters:
        img_array (numpy.ndarray): Image array in BGR format.
        u (float or int): x-coordinate of the point to mark.
        v (float or int): y-coordinate of the point to mark.
        filename (str): Image filename (used as window title).

    Returns:
        None
    """
    img_with_dot = img_array.copy()
    # Draw red dot at specified pixel
    if u is not None and v is not None:
        cv2.circle(img_with_dot, (int(u), int(v)), radius=5, color=(0, 0, 255), thickness=-1)

    pil_img = Image.fromarray(cv2.cvtColor(img_with_dot, cv2.COLOR_BGR2RGB))
    root = tk.Tk()
    root.title(os.path.basename(filename))
    state = {"img": pil_img}

    canvas = tk.Label(root)
    canvas.pack()

    def update_image():
        """Update the displayed image after rotation or other changes."""
        tk_img = ImageTk.PhotoImage(state["img"], master=root)
        canvas.configure(image=tk_img)
        canvas.image = tk_img

    def rotate_left():
        """Rotate image 90 degrees counter-clockwise."""
        state["img"] = state["img"].rotate(90, expand=True)
        update_image()

    def rotate_right():
        """Rotate image 90 degrees clockwise."""
        state["img"] = state["img"].rotate(-90, expand=True)
        update_image()

    def on_close():
        """Handle window close event."""
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Buttons for rotation
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="⟲ Rotate Left", command=rotate_left).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="⟳ Rotate Right", command=rotate_right).pack(side=tk.LEFT, padx=5)

    update_image()
    root.mainloop()


# ---------------------- DEM Elevation Query ----------------------
def get_elevation_from_dem(dem_path: str, lat: float, lon: float) -> float:
    """
    Query the elevation (Z) from a DEM (Digital Elevation Model) at a given GPS coordinate.

    Parameters:
        dem_path (str): File path to DEM raster (e.g., GeoTIFF).
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.

    Returns:
        float: Elevation value at the specified coordinates.
    """
    with rasterio.open(dem_path) as dem:
        # Transform lat/lon to DEM CRS
        transformer = Transformer.from_crs("EPSG:4326", dem.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        # Sample DEM at transformed coordinates
        elevation = list(dem.sample([(x, y)]))[0][0]
        return float(elevation)


# ----------------- KD-Tree LAZ Utilities -----------------
def laz_to_kdtree(laz_path, utm_epsg=UTM_EPSG):
    """
    Load a LAZ/LAS LiDAR file and build a KD-Tree for fast nearest neighbor queries.

    Parameters:
        laz_path (str): Path to the .laz or .las LiDAR file.
        utm_epsg (int, optional): EPSG code of the LAZ file coordinates (default uses UTM_EPSG).

    Returns:
        tuple:
            - tree (scipy.spatial.cKDTree): KD-Tree built from the LiDAR points (lon, lat).
            - points_z (numpy.ndarray): Array of z-values (elevation) corresponding to the points.
    """
    # Read LiDAR file (.las or .laz)
    las = laspy.read(laz_path)

    # Transform UTM coordinates to WGS84 lat/lon
    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(las.x, las.y)

    # Stack coordinates for KD-Tree (longitude, latitude)
    points_xy = np.vstack((lons, lats)).T

    # Extract elevations
    points_z = np.array(las.z)

    # Build KD-Tree for spatial queries
    tree = cKDTree(points_xy)
    return tree, points_z


def get_height_from_laser_kdtree(tree, z_values, lon, lat, k=5):
    """
    Query the average elevation near a given GPS coordinate using a KD-Tree.

    Parameters:
        tree (scipy.spatial.cKDTree): KD-Tree built from LiDAR points (lon, lat).
        z_values (numpy.ndarray): Array of elevations corresponding to the KD-Tree points.
        lon (float): Longitude of query point in degrees.
        lat (float): Latitude of query point in degrees.
        k (int, optional): Number of nearest neighbors to consider (default=5).

    Returns:
        float: Average elevation (z) of the k nearest LiDAR points.
    """
    # Query k nearest neighbors
    dists, idxs = tree.query([lon, lat], k=k)

    # Extract corresponding z-values
    z_nearest = z_values[idxs]

    # Return mean elevation
    return float(np.mean(z_nearest))




# ----------------- Main Mapper Class -----------------
class DroneMapper:
    """
    Class to handle drone-based georeferencing of image points.

    Features:
    - Converts selected image pixels to GPS coordinates using drone position, attitude, and camera intrinsics.
    - Supports DEM or LiDAR (LAZ) based elevation for more accurate georeferencing.
    - Provides GUI-based selection of pixels on images.
    - Optional plotting with Google Maps or CAD maps.

    Attributes:
        DRONE_OFFSET_NORTH (float): Drone sensor offset in meters along north.
        DRONE_OFFSET_EAST (float): Drone sensor offset in meters along east.
        DRONE_OFFSET_UP (float): Drone sensor offset in meters upward.
        dem_path (str): Path to DEM file (optional).
        tree (cKDTree): KD-Tree of LiDAR points (if laz_path provided).
        points_z (np.ndarray): LiDAR elevation values.
        focal_length_mm (float): Camera focal length in mm.
        sensor_width_mm (float): Camera sensor width in mm.
    """

    def __init__(self, drone_offset_north=DRONE_OFFSET_NORTH, drone_offset_east=DRONE_OFFSET_EAST,
                 drone_offset_up=DRONE_OFFSET_UP, lidar_path=None, dem_path=None, focal_length_mm=FOCAL_LENGTH_MM,   
                 sensor_width_mm=SENSOR_WIDTH_MM):
        """
        Initialize DroneMapper.

        Parameters:
            drone_offset_north, drone_offset_east, drone_offset_up: offsets of the drone sensor.
            lidar_path (str, optional): Path to LiDAR (.laz) file for height info.
            dem_path (str, optional): Path to DEM file for elevation.
            focal_length_mm (float): Camera focal length in mm.
            sensor_width_mm (float): Camera sensor width in mm.
        """
        self.DRONE_OFFSET_NORTH = drone_offset_north
        self.DRONE_OFFSET_EAST = drone_offset_east
        self.DRONE_OFFSET_UP = drone_offset_up
        self.dem_path = dem_path
        self.tree = None
        self.points_z = None
        self.focal_length_mm = focal_length_mm
        self.sensor_width_mm = sensor_width_mm
        if lidar_path:
            self.tree, self.points_z = laz_to_kdtree(lidar_path, utm_epsg=UTM_EPSG)

    def get_target_gps(self, u, v, gps, angles, image_size):
        """
        Convert a pixel (u,v) in an image to target GPS coordinates.

        Parameters:
            u, v (float): Pixel coordinates in the image.
            gps (tuple): Drone GPS as (lat, lon, alt, rel_alt).
            angles (tuple): Drone orientation (yaw, pitch, roll) in degrees.
            image_size (tuple): Width, height of image in pixels.

        Returns:
            tuple: (lat, lon) GPS coordinates of the target.
        """
        if gps is None or any(a is None for a in angles):
            return None, None

        drone_lat, drone_lon, drone_alt, rel_alt = gps
        width, height = image_size
        yaw, pitch, roll = angles

        # Initialize camera model and intrinsics
        cam = Camera()
        f_px = self.focal_length_mm * width / self.sensor_width_mm
        cam.intrinsics(width, height, f_px, width / 2, height / 2)

        # Initialize extrinsics for drone pose
        ext = Extrinsics()
        ext.setPose(
            X=self.DRONE_OFFSET_EAST,
            Y=self.DRONE_OFFSET_NORTH,
            Z=(drone_alt + self.DRONE_OFFSET_UP),
            roll=-roll,
            pitch=-90 - pitch,
            yaw=-yaw - 90,
            order="ZYX"
        )
        ext.setGimbal(roll=0, pitch=0, yaw=0, order="ZYX")
        cam.attitudeMat(ext.transform())

        # Ray from camera through pixel
        plane_near = np.array([0, 0, 1, -0.1])
        plane_far = np.array([0, 0, 1, -1000])
        p_near = cam.reprojectToPlane(pixel_to_camproject(u, v, width, height), plane_near)[:3]
        p_far = cam.reprojectToPlane(pixel_to_camproject(u, v, width, height), plane_far)[:3]
        ray_dir = p_far - p_near
        ray_dir /= np.linalg.norm(ray_dir)
        R = ext.transform()[:3, :3]
        ray_dir_world = R @ ray_dir

        # If ray points upward or no LiDAR/DEM → fallback to flat plane
        if ray_dir_world[2] >= 0 or (self.tree is None and self.dem_path is None):
            ground_height = (drone_alt + self.DRONE_OFFSET_UP) - rel_alt
            plane = np.array([0, 0, 1, -ground_height])
            target_3D = cam.reprojectToPlane(pixel_to_camproject(u, v, width, height), plane)
            lat, lon = enu_to_gps(target_3D[0], target_3D[1], drone_lat, drone_lon)
            return lat, lon

        # Ray-marching to intersect with ground (LiDAR/DEM)
        step = 1.0
        max_distance = 200.0
        Z_cam = drone_alt + self.DRONE_OFFSET_UP
        drone_xyz = np.array([0.0, 0.0, Z_cam])

        for t in np.arange(0, max_distance, step):
            pos = drone_xyz + t * ray_dir_world
            lat, lon = enu_to_gps(pos[0], pos[1], drone_lat, drone_lon)
            Z_surface = None

            # Query LiDAR or DEM height
            if self.tree is not None:
                try:
                    Z_surface = get_height_from_laser_kdtree(self.tree, self.points_z, lon, lat)
                except Exception:
                    Z_surface = None
            elif self.dem_path:
                try:
                    Z_surface = get_elevation_from_dem(self.dem_path, lat, lon)
                except Exception:
                    Z_surface = None

            # Stop if ray hits the surface
            if Z_surface is not None and pos[2] <= Z_surface:
                return lat, lon

        # Fallback to flat plane
        ground_height = (drone_alt + self.DRONE_OFFSET_UP) - rel_alt
        plane = np.array([0, 0, 1, -ground_height])
        target_3D = cam.reprojectToPlane(pixel_to_camproject(u, v, width, height), plane)
        lat, lon = enu_to_gps(target_3D[0], target_3D[1], drone_lat, drone_lon)
        return lat, lon

    def get_target_gps_array(self, data_array):
        """
        Convert pixel coordinates to GPS for all images in data_array.

        Parameters:
            data_array (list): List of dictionaries with keys 'pixel_x', 'pixel_y', 'gps', 'yaw', 'pitch', 'roll', 'image_size'.

        Returns:
            list: data_array updated with 'target_gps' field as (lat, lon).
        """
        for data in data_array:
            u = data.get('pixel_x', data['image_size'][0] / 2)
            v = data.get('pixel_y', data['image_size'][1] / 2)
            yaw, pitch, roll = data['yaw'], data['pitch'], data['roll']
            lat_lon = self.get_target_gps(u, v, gps=data['gps'], angles=(yaw, pitch, roll), image_size=data['image_size'])
            data['target_gps'] = lat_lon if lat_lon != (None, None) else None
        return data_array

    def process_images_gui(self, img_paths, cad_path=None):
        """
        Launch GUI for user to select target pixels on images and compute GPS coordinates.

        Parameters:
            img_paths (list): List of image file paths.
            cad_path (str, optional): Path to CAD map for optional plotting.

        Returns:
            None. Updates data_array with pixel coordinates and target GPS.
        """
        data_array = extract_metadata_from_csv(img_paths)
        for entry in data_array:
            # Load and normalize image
            img = Image.open(entry['image_name'])
            img_array = np.array(img)
            if img_array.dtype == np.uint16:
                img_array = (img_array / 256).astype(np.uint8)
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            else:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Let user select pixel
            u, v = select_pixel_gui(img_array)
            entry['pixel_x'] = u
            entry['pixel_y'] = v

        # Compute GPS for all selected pixels
        data_array = self.get_target_gps_array(data_array)

        # Optional plotting
        for entry in data_array:
            if entry.get('target_gps') is not None:
                lat, lon = entry['target_gps']
                drone_gps = (
                    entry['gps'][0] + self.DRONE_OFFSET_NORTH,
                    entry['gps'][1] + self.DRONE_OFFSET_EAST
                )
                plot_google_maps(target_gps=(lat, lon),
                                 corner_gps=None,
                                 drone_gps=drone_gps)
                if cad_path:
                    plot_cad_map(target_gps=(lat, lon),
                                 corner_gps=None,
                                 drone_gps=drone_gps,
                                 cad_path=cad_path)
            # Show image with point and rotation buttons
            show_image_with_buttons(img_array, u, v, filename=entry['image_name'])

if __name__ == "__main__":
    import argparse

    root = tk.Tk()
    root.withdraw()

    parser = argparse.ArgumentParser(description="Drone image mapping with optional CAD overlay.")
    parser.add_argument(
        "-c", "--cad",
        type=str,
        default=None,
        help="Path to the CAD file of the solar plant (GeoJSON) for plotting"
    )
    args = parser.parse_args()
    cad_path = args.cad

    img_paths = filedialog.askopenfilenames(
        title="Select images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), ("All files", "*.*")]
    )
    if not img_paths:
        print("No images selected. Exiting.")
        sys.exit(0)

    mapper = DroneMapper()
    mapper.process_images_gui(img_paths, cad_path=cad_path)