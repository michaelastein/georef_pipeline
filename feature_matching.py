import cv2
import numpy as np
import csv
from pathlib import Path
import math
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import anomalies_batch
import gui_canvas
from gui_anomalies import launch_anomaly_gui
import avg_gps
from tkinter.filedialog import askopenfilename, askopenfilenames
from collections import deque
import pickle
import georef_new

import json
# ---------------------- Utility Functions ----------------------

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the Haversine distance between two GPS coordinates in meters.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def preprocess_for_features(img):
    """
    Convert an image to grayscale and apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to enhance features for keypoint detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def print_progress(current, total, stage_name, start_time=None, last_print=[-1], lock=None, bar_length=30, update_every_percent=1):
    """
    Print progress updates with a progress bar and ETA every `update_every_percent` percent.
    """
    percent = int((current / total) * 100) if total > 0 else 100
    if percent // update_every_percent != last_print[0] or current == total:
        bar_filled = int(bar_length * percent / 100)
        bar = "=" * bar_filled + "-" * (bar_length - bar_filled)
        eta_str = ""
        if start_time and percent > 0:
            elapsed = time.time() - start_time
            eta = elapsed * (100 - percent) / percent
            mins, secs = divmod(int(eta), 60)
            eta_str = f" | ETA: {mins:02d}m {secs:02d}s"
        msg = f"[{stage_name}] [{bar}] {percent:3d}% ({current}/{total}){eta_str}"
        if lock:
            with lock:
                print(msg, end='\r', flush=True)
                if current == total:
                    print()
        else:
            print(msg, end='\r', flush=True)
            if current == total:
                print()
        last_print[0] = percent // update_every_percent




def save_homographies(H_dict, image_data_list, filename="homographies.pkl"):
    """
    Save computed homographies and associated image metadata to a file using pickle.
    """
    data_to_save = {
        "H_dict": H_dict,
        "image_data_list": image_data_list
    }
    with open(filename, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Saved homographies and image data to {filename}")


def load_homographies(filename="homographies.pkl"):
    """
    Load previously saved homographies and image metadata from a pickle file.
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded homographies and image data from {filename}")



    return data["H_dict"], data["image_data_list"]


# ---------------------- Metadata / CSV ----------------------

def extract_metadata_from_csv():
    """
    Prompt user to select images, then extract metadata using georef_new.extract_metadata_from_csv.
    Returns a list of image metadata dictionaries.
    """
    # Prompt user to select image files
    img_paths = askopenfilenames(
        title="Select images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if not img_paths:
        print("No images selected.")
        return []

    # Delegate all metadata extraction to georef_new
    return georef_new.extract_metadata_from_csv(img_paths)






# ---------------------- Graph helpers ----------------------

def node_connected_component(start, adj):
    """
    Return all nodes connected to 'start' in the homography graph.
    """
    if start not in adj:
        return {start}
    seen = {start}
    q = [start]
    while q:
        u = q.pop(0)
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def shortest_path(u, v, adj):
    """
    Compute shortest path from node u to v using BFS.
    Returns a list of nodes along the path.
    """
    if u == v:
        return [u]
    if u not in adj:
        return None
    q = deque([u])
    parent = {u: None}
    while q:
        cur = q.popleft()
        for nb in adj.get(cur, ()):
            if nb not in parent:
                parent[nb] = cur
                if nb == v:
                    # Reconstruct path from u to v
                    path = [v]
                    p = v
                    while parent[p] is not None:
                        p = parent[p]
                        path.append(p)
                    path.reverse()
                    return path
                q.append(nb)
    return None



# ---------------------- Main Function ----------------------

def main(algorithm=None, anomalies=None, homographies_path=None):
    """
    Main pipeline:
    1. Load images and metadata (or load precomputed homographies)
    2. Detect and compute features
    3. Compute neighbors based on GPS threshold
    4. Match features between neighbors and compute homographies
    5. Build a graph of homographies
    6. Launch GUI or anomalies processing
    """
    threshold_meters = 40.0
    ratio_test = 0.7
    ransac_thresh = 4.0
    min_inliers = 20
    dist_consistency_thresh = 40.0
    max_workers = 8
    progress_lock = threading.Lock()
    start_time = time.time()
    match_cache, H_dict, H_inliers = {}, {}, {}

    



    # ---------------------- Load or compute homographies ----------------------
    if homographies_path:
        H_dict, image_data_list = load_homographies(homographies_path)
        images = [cv2.imread(entry["image_name"]) for entry in image_data_list]
        orig_sizes = [entry["image_size"] for entry in image_data_list]
        gps_positions = [entry["gps"] for entry in image_data_list]
        print("Using loaded homographies and image data. Skipping image selection.")

        # Print first and last image names
        first_image_name = image_data_list[0]["image_name"]
        last_image_name = image_data_list[-1]["image_name"]
        print(f"First image: {first_image_name}")
        print(f"Last image: {last_image_name}")

  


    else:
        image_data_list = extract_metadata_from_csv()
        if not image_data_list:
            return
        
        print(f"Number of images loaded: {len(image_data_list)}")
        images, orig_sizes, gps_positions = [], [], []
        for entry in image_data_list:
            img = cv2.imread(entry["image_name"])
            if img is None:
                print(f"Warning: could not read {entry['image_name']}")
                continue
            images.append(img)
            orig_sizes.append((img.shape[1], img.shape[0]))
            gps_positions.append(entry["gps"])
        if not images:
            print("No valid images loaded.")
            return

        # ---------------------- Feature Detector Setup ----------------------
        if algorithm is None:
            # Default: BRISK for TIFF images, otherwise SIFT
            algorithm = "BRISK" if image_data_list[0]["image_name"].lower().endswith((".tif", ".tiff")) else "SIFT"

        if algorithm == "SIFT":
            detector = cv2.SIFT_create(); descriptor_type = "float"
        elif algorithm == "AKAZE":
            detector = cv2.AKAZE_create(); descriptor_type = "binary"
        elif algorithm == "ORB":
            detector = cv2.ORB_create(nfeatures=3000); descriptor_type = "binary"
        elif algorithm == "BRISK":
            detector = cv2.BRISK_create(); descriptor_type = "binary"
        elif algorithm == "KAZE":
            detector = cv2.KAZE_create(); descriptor_type = "float"
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        print(f"Using feature algorithm: {algorithm}")

        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)) if descriptor_type == "float" else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # ---------------------- Feature Computation ----------------------
        kp_list, des_list = [None]*len(images), [None]*len(images)

        def ensure_flann_dtype(des):
            """Ensure descriptors have correct type for FLANN matcher"""
            if des is None: return None
            if descriptor_type == "float" and des.dtype != np.float32: return des.astype(np.float32)
            if descriptor_type != "float" and des.dtype != np.uint8: return des.astype(np.uint8)
            return des

        def compute_features_for_index(idx):
            """Compute keypoints and descriptors for a single image."""
            gray = preprocess_for_features(images[idx])
            kp, des = detector.detectAndCompute(gray, None)
            kp_list[idx] = kp
            des_list[idx] = des.copy() if des is not None else None

        # Multi-threaded feature computation
        with ThreadPoolExecutor(max_workers=min(max_workers, len(images))) as ex:
            futures = [ex.submit(compute_features_for_index, i) for i in range(len(images))]
            done_count = 0
            for f in as_completed(futures):
                done_count += 1
                print_progress(done_count,len(futures),"Feature extraction",start_time=start_time,lock=progress_lock)

        # ---------------------- GPS Neighbor Prefilter ----------------------
        neighbors = []
        for i, (lat_i, lon_i, _, _) in enumerate(gps_positions):
            if lat_i is None: continue
            for j, (lat_j, lon_j, _, _) in enumerate(gps_positions):
                if i >= j or lat_j is None: continue
                if haversine(lat_i, lon_i, lat_j, lon_j) <= threshold_meters:
                    neighbors.append((i, j))
        # If no neighbors found, fall back to all pairs
        if not neighbors and len(images) > 1:
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    neighbors.append((i, j))
        print(f"Total pairs to attempt: {len(neighbors)}")

        # ---------------------- Matching & Homography ----------------------

        def match_and_filter_pairs(i, j):
            """
            Match keypoints between two images i and j and filter matches for consistency.
            Returns a list of filtered mutual matches.
            """
            key = (i, j)
            
            # Return cached matches if already computed
            if key in match_cache:
                return match_cache[key]

            des_i, des_j = des_list[i], des_list[j]
            kp_i, kp_j = kp_list[i], kp_list[j]

            # Skip if either image has no descriptors or keypoints
            if des_i is None or des_j is None or kp_i is None or kp_j is None:
                match_cache[key] = []
                return []

            # Ensure correct descriptor type for FLANN or BF matcher
            des_i_q, des_j_q = ensure_flann_dtype(des_i), ensure_flann_dtype(des_j)

            # Try matching with the main matcher
            try:
                knn_j_i = matcher.knnMatch(des_j_q, des_i_q, k=2)  # from j to i
                knn_i_j = matcher.knnMatch(des_i_q, des_j_q, k=2)  # from i to j
            except cv2.error:
                # Fallback to brute-force matcher if FLANN fails
                bf = cv2.BFMatcher(cv2.NORM_L2 if descriptor_type == "float" else cv2.NORM_HAMMING, crossCheck=False)
                knn_j_i = bf.knnMatch(des_j_q, des_i_q, k=2)
                knn_i_j = bf.knnMatch(des_i_q, des_j_q, k=2)

            def filter_good(knn):
                """
                Apply Lowe's ratio test to filter good matches.
                """
                good = [m_n[0] for m_n in knn if len(m_n) == 2 and m_n[0].distance < ratio_test * m_n[1].distance]
                return good

            # Filter matches with ratio test
            good_j_i, good_i_j = filter_good(knn_j_i), filter_good(knn_i_j)

            # Keep only mutual matches (cross-check)
            best_j_to_i = {m.queryIdx: m.trainIdx for m in good_j_i}
            best_i_to_j = {m.queryIdx: m.trainIdx for m in good_i_j}
            mutual = [(q, t) for q, t in best_j_to_i.items() if t in best_i_to_j and best_i_to_j[t] == q]

            if not mutual:
                match_cache[key] = []
                return []

            # Compute geometric consistency (distance vector check)
            pts_j = np.array([kp_j[q].pt for q, _ in mutual])
            pts_i = np.array([kp_i[t].pt for _, t in mutual])
            vecs = pts_i - pts_j
            mean_vec = np.mean(vecs, axis=0)
            dists = np.linalg.norm(vecs - mean_vec, axis=1)
            
            # Keep matches consistent within distance threshold
            filtered = [mutual[idx] for idx, k in enumerate(dists <= dist_consistency_thresh) if k]

            # Cache and return filtered matches
            match_cache[key] = filtered
            return filtered


        def compute_homography_for_pair(pair):
            """
            Compute homography between a pair of images using filtered matches.
            Returns homography matrix and inlier points if sufficient matches exist.
            """
            i, j = pair
            matches = match_and_filter_pairs(i, j)

            if len(matches) < min_inliers:  # skip pairs with too few matches
                return None

            # Prepare source (j) and destination (i) points
            src_pts = np.float32([kp_list[j][q].pt for q, _ in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_list[i][t].pt for _, t in matches]).reshape(-1, 1, 2)

            # Compute homography using RANSAC
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
            if H is None or mask is None or int(np.sum(mask)) < min_inliers:
                return None

            # Extract inlier points
            in_src = src_pts[mask.ravel() == 1].reshape(-1, 2)
            in_dst = dst_pts[mask.ravel() == 1].reshape(-1, 2)

            return i, j, H, in_src.copy(), in_dst.copy()


        # Compute homographies for all neighbor pairs in parallel
        if neighbors:
            done_pairs = 0
            with ThreadPoolExecutor(max_workers=min(max_workers, len(neighbors))) as ex:
                futures = {ex.submit(compute_homography_for_pair, p): p for p in neighbors}
                for fut in as_completed(futures):
                    res = fut.result()
                    done_pairs += 1
                    print_progress(done_pairs,len(futures),"Homography",start_time=start_time,lock=progress_lock)
                    if res is None:
                        continue
                    i, j, H, in_src, in_dst = res
                    # Store homography and inliers
                    H_dict[(j, i)] = H
                    H_inliers[(j, i)] = in_dst.copy()
                    try:
                        H_inv = np.linalg.inv(H)
                        H_dict[(i, j)] = H_inv
                        H_inliers[(i, j)] = in_src.copy()
                    except np.linalg.LinAlgError:
                        pass  # singular matrix, skip inverse

        print(f"Feature extraction + matching + homography runtime: {(time.time() - start_time)/60:.2f} min")
        print(f"Computed {len(H_dict)//2} good homography pairs.")


        
        save_homographies(H_dict, image_data_list)

    # Build adjacency list for homography graph
    adj = {}
    for (a, b) in H_dict.keys():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # ---------------------- Correspondences ----------------------

    def build_correspondences_from_pixels(idx, x, y, image_data_list, pixel_tol=1e-3):
        """
        Given a pixel location in one image (idx), compute its corresponding pixel locations
        in all connected images using the homography graph.
        Duplicate correspondences are removed by rounding pixels to avoid near-duplicate entries.
        The source image is always the first element in the returned list.
        Correspondences that fall outside image bounds are skipped.
        Final pixel coordinates are rounded to integers.
        """
        pt = np.array([[x], [y], [1.0]])  # Homogeneous coordinate
        correspondences_set = set()

        # Add source pixel (rounded to int)
        px_rounded = int(round(x))
        py_rounded = int(round(y))
        correspondences_set.add((idx, px_rounded, py_rounded))

        comp = node_connected_component(idx, adj)  # Get all connected images

        for other in comp:
            if other == idx:
                continue

            path = shortest_path(idx, other, adj)
            if path is None:
                continue

            cur_pt = pt.copy()
            ok = True

            # Apply homographies along the path
            for k in range(len(path) - 1):
                a, b = path[k], path[k + 1]
                H = H_dict.get((a, b))
                if H is None:
                    ok = False
                    break
                try:
                    cur_pt = H @ cur_pt
                    if abs(cur_pt[2, 0]) < 1e-8:
                        ok = False
                        break
                    cur_pt /= cur_pt[2, 0]  # normalize after each step
                except Exception:
                    ok = False
                    break

            # Only add if within image bounds
            if ok:
                w, h = image_data_list[other]["image_size"]
                px, py = cur_pt[0, 0], cur_pt[1, 0]
                if 0 <= px < w and 0 <= py < h:
                    px_int = int(round(px))
                    py_int = int(round(py))
                    correspondences_set.add((other, px_int, py_int))

        # Build final data with pixel info, ensuring source image is first
        result = []

        # Add source image first
        entry = image_data_list[idx].copy()
        entry.update({"pixel_x": int(round(x)),
                    "pixel_y": int(round(y))})
        result.append(entry)

        # Add other correspondences
        for img_idx, px, py in correspondences_set:
            if img_idx == idx:
                continue
            entry = image_data_list[img_idx].copy()
            entry.update({"pixel_x": px, "pixel_y": py})
            result.append(entry)

        return result




            



    # ---------------------- GUI / anomalies mode ----------------------

    if anomalies == "single":
        # Launch anomaly GUI for single point selection
        idx, x, y, csv_path = launch_anomaly_gui(
            image_data_list
        )
        correspondence_data = build_correspondences_from_pixels(
            idx, x, y,
            image_data_list=image_data_list
        )
        avg_gps.main(correspondence_data)
        if csv_path:
            import matching_anomalies

            results = matching_anomalies.main(correspondence_data, csv_path)

            # --- Print all image_name values ---
            if results:
                print("Matched image files:")
                for r in results:
                    print(os.path.basename(r["image_name"]))
            else:
                print("No matching anomalies found.")

    elif anomalies == "batch":
        # Launch batch anomaly processing
        anomalies_batch.main(image_data_list, build_corr_func=build_correspondences_from_pixels)

    else:
        def click_callback(idx, x, y, gui):
            
            data = build_correspondences_from_pixels(
                idx, x, y,
                image_data_list=image_data_list
            )

            avg_gps.main(data)
            return data  # returned data will be drawn as yellow markers

        gui = gui_canvas.CanvasGUI(image_data_list, orig_sizes, click_callback=click_callback)
        gui.run()



# ---------------------- CLI ----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image feature matcher with optional GUI.")
    parser.add_argument(
        "-a", "--algorithm",
        choices=["SIFT", "AKAZE", "ORB", "BRISK", "KAZE"],
        default=None,
        help="Feature algorithm"
    )
    parser.add_argument(
        "--anomalies",
        choices=["none", "single", "batch"],
        default="none",
        help="Anomalies mode: 'none' (default), 'single', or 'batch'"
    )

    parser.add_argument(
        "-l", "--load-homographies",
        type=str,
        default=None,
        help="Load homographies and image data from a file instead of computing them"
    )

    args = parser.parse_args()
    algorithm = args.algorithm.upper() if args.algorithm else None
    anomalies_mode = args.anomalies
    homographies_path = args.load_homographies

    main(algorithm, anomalies=anomalies_mode, homographies_path=homographies_path)
