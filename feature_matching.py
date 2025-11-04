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

# ---------------------- Utility Functions ----------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def preprocess_for_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def print_progress(current, total, stage_name, last_print=[-1], lock=None):
    percent = int((current / total) * 100) if total > 0 else 100
    if lock:
        with lock:
            if percent // 5 != last_print[0] or current == total:
                print(f"[{stage_name}] Progress: {percent}% ({current}/{total})")
                last_print[0] = percent // 5
    else:
        if percent // 5 != last_print[0] or current == total:
            print(f"[{stage_name}] Progress: {percent}% ({current}/{total})")
            last_print[0] = percent // 5


def save_homographies(H_dict, image_data_list, filename="homographies.pkl"):
    data_to_save = {
        "H_dict": H_dict,
        "image_data_list": image_data_list
    }
    with open(filename, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Saved homographies and image data to {filename}")


def load_homographies(filename="homographies.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded homographies and image data from {filename}")
    return data["H_dict"], data["image_data_list"]


# ---------------------- Metadata / CSV ----------------------

def extract_metadata_from_csv():
    img_paths = askopenfilenames(
        title="Select images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if not img_paths:
        print("No images selected.")
        return []
    return georef_new.extract_metadata_from_csv(img_paths)


# ---------------------- Graph Helpers ----------------------

def node_connected_component(start, adj):
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
                    path = [v]
                    p = v
                    while parent[p] is not None:
                        p = parent[p]
                        path.append(p)
                    path.reverse()
                    return path
                q.append(nb)
    return None


# ---------------------- Correspondences ----------------------

def build_correspondences_from_pixels(idx, x, y, image_data_list, H_dict, adj, shortest_path, node_connected_component):
    pt = np.array([[x], [y], [1.0]])
    correspondences = [(idx, x, y)]
    comp = node_connected_component(idx, adj)
    for other in comp:
        if other == idx:
            continue
        path = shortest_path(idx, other, adj)
        if path is None:
            continue
        cur_pt = pt.copy()
        ok = True
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
                cur_pt = cur_pt / cur_pt[2, 0]
            except Exception:
                ok = False
                break
        if ok:
            correspondences.append((other, cur_pt[0, 0], cur_pt[1, 0]))

    result = []
    for img_idx, px, py in correspondences:
        entry = image_data_list[img_idx].copy()
        entry.update({"pixel_x": float(px), "pixel_y": float(py)})
        result.append(entry)
    return result


# ---------------------- Main Function ----------------------

def main(algorithm=None, anomalies=None, homographies_path=None):
    threshold_meters = 40.0
    ratio_test = 0.7
    ransac_thresh = 4.0
    min_inliers = 20
    dist_consistency_thresh = 40.0
    max_workers = 8
    progress_lock = threading.Lock()
    start_time = time.time()
    match_cache, H_dict, H_inliers = {}, {}, {}

    cv2.setNumThreads(1)  # Prevent over-threading when GPU is used

    # ---------------------- Load or compute homographies ----------------------
    if homographies_path:
        H_dict, image_data_list = load_homographies(homographies_path)
        images = [cv2.imread(entry["image_path"]) for entry in image_data_list]
        orig_sizes = [entry["image_size"] for entry in image_data_list]
        gps_positions = [entry["gps"] for entry in image_data_list]
        print(f"Using loaded homographies and image data. Total images: {len(image_data_list)}")
        print(f"First image: {image_data_list[0]['image_path']}")
        print(f"Last image: {image_data_list[-1]['image_path']}")

    else:
        image_data_list = extract_metadata_from_csv()
        if not image_data_list:
            return
        images, orig_sizes, gps_positions = [], [], []
        print(f"Total images selected: {len(image_data_list)}") 
        for entry in image_data_list:
            img = cv2.imread(entry["image_path"])
            if img is None:
                print(f"Warning: could not read {entry['image_path']}")
                continue
            images.append(img)
            orig_sizes.append((img.shape[1], img.shape[0]))
            gps_positions.append(entry["gps"])
        if not images:
            print("No valid images loaded.")
            return

        # ---------------------- Feature Detector Setup ----------------------
        use_cuda = False
        if algorithm is None:
            algorithm = "BRISK" if image_data_list[0]["image_path"].lower().endswith((".tif", ".tiff")) else "SIFT"

        cuda_available = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        if cuda_available:
            print(f"✅ CUDA available: {cv2.cuda.getCudaEnabledDeviceCount()} GPU(s) detected")
        else:
            print("⚠️ CUDA not available — using CPU-based features")

        if algorithm == "SIFT":
            if cuda_available:
                detector = cv2.cuda.SIFT_create()
                use_cuda = True
                descriptor_type = "float"
                print("Using GPU-accelerated SIFT")
            else:
                detector = cv2.SIFT_create()
                descriptor_type = "float"
                print("Using CPU SIFT")

        elif algorithm == "ORB":
            if cuda_available:
                detector = cv2.cuda_ORB_create(nfeatures=3000)
                use_cuda = True
                descriptor_type = "binary"
                print("Using GPU-accelerated ORB")
            else:
                detector = cv2.ORB_create(nfeatures=3000)
                descriptor_type = "binary"
                print("Using CPU ORB")

        elif algorithm == "AKAZE":
            detector = cv2.AKAZE_create(); descriptor_type = "binary"
            print("Using AKAZE (CPU only)")

        elif algorithm == "BRISK":
            detector = cv2.BRISK_create(); descriptor_type = "binary"
            print("Using BRISK (CPU only)")

        elif algorithm == "KAZE":
            detector = cv2.KAZE_create(); descriptor_type = "float"
            print("Using KAZE (CPU only)")

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # ---------------------- Feature Computation ----------------------
        kp_list, des_list = [None]*len(images), [None]*len(images)

        def compute_features_for_index(idx):
            img = preprocess_for_features(images[idx])
            if use_cuda:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                kp, des = detector.detectAndComputeAsync(gpu_img, None)
                if des is not None:
                    des = des.download()
                gpu_img.release()
            else:
                kp, des = detector.detectAndCompute(img, None)
            kp_list[idx], des_list[idx] = kp, des.copy() if des is not None else None

        with ThreadPoolExecutor(max_workers=min(max_workers, len(images))) as ex:
            futures = [ex.submit(compute_features_for_index, i) for i in range(len(images))]
            done_count = 0
            for f in as_completed(futures):
                done_count += 1
                print_progress(done_count, len(futures), "Feature extraction", lock=progress_lock)

        # ---------------------- Matcher Setup ----------------------
        if use_cuda:
            print("Using GPU-accelerated BFMatcher")
            norm = cv2.NORM_L2 if descriptor_type == "float" else cv2.NORM_HAMMING
            matcher = cv2.cuda_BFMatcher_create(norm)
        else:
            print("Using CPU matcher")
            if descriptor_type == "float":
                matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            else:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # ---------------------- GPS Neighbor Prefilter ----------------------
        neighbors = []
        for i, (lat_i, lon_i, _, _) in enumerate(gps_positions):
            if lat_i is None: continue
            for j, (lat_j, lon_j, _, _) in enumerate(gps_positions):
                if i >= j or lat_j is None: continue
                if haversine(lat_i, lon_i, lat_j, lon_j) <= threshold_meters:
                    neighbors.append((i, j))
        if not neighbors and len(images) > 1:
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    neighbors.append((i, j))
        print(f"Total pairs to attempt: {len(neighbors)}")

        # ---------------------- Matching & Homography ----------------------
        def match_and_filter_pairs(i, j):
            key = (i, j)
            if key in match_cache:
                return match_cache[key]
            des_i, des_j = des_list[i], des_list[j]
            kp_i, kp_j = kp_list[i], kp_list[j]
            if des_i is None or des_j is None:
                match_cache[key] = []
                return []
            try:
                if use_cuda:
                    matches = matcher.knnMatch(des_i, des_j, k=2)
                else:
                    matches = matcher.knnMatch(des_i, des_j, k=2)
            except cv2.error:
                match_cache[key] = []
                return []
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < ratio_test * m[1].distance]
            match_cache[key] = good
            return good

        def compute_homography_for_pair(pair):
            i, j = pair
            matches = match_and_filter_pairs(i, j)
            if len(matches) < min_inliers:
                return None
            src_pts = np.float32([kp_list[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_list[j][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
            if H is None or mask is None or int(np.sum(mask)) < min_inliers:
                return None
            return i, j, H

        if neighbors:
            done_pairs = 0
            with ThreadPoolExecutor(max_workers=min(max_workers, len(neighbors))) as ex:
                futures = {ex.submit(compute_homography_for_pair, p): p for p in neighbors}
                for fut in as_completed(futures):
                    res = fut.result()
                    done_pairs += 1
                    print_progress(done_pairs, len(futures), "Homography", lock=progress_lock)
                    if res:
                        i, j, H = res
                        H_dict[(i, j)] = H
                        try:
                            H_inv = np.linalg.inv(H)
                            H_dict[(j, i)] = H_inv
                        except np.linalg.LinAlgError:
                            pass

        print(f"Runtime: {(time.time() - start_time)/60:.2f} min")
        print(f"Computed {len(H_dict)//2} homography pairs.")
        save_homographies(H_dict, image_data_list)

    adj = {}
    for (a, b) in H_dict.keys():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # ---------------------- GUI / anomalies mode ----------------------
    if anomalies == "single":
        idx, x, y, csv_path = launch_anomaly_gui(image_data_list)
        data = build_correspondences_from_pixels(idx, x, y, image_data_list, H_dict, adj, shortest_path, node_connected_component)
        avg_gps.main(data)
        if csv_path:
            import matching_anomalies

            matching_anomalies.main(data, csv_path)
    elif anomalies == "batch":
        anomalies_batch.main(image_data_list)
    else:
        def click_callback(idx, x, y, gui):
            data = build_correspondences_from_pixels(idx, x, y, image_data_list, H_dict, adj, shortest_path, node_connected_component)
            avg_gps.main(data)
            return data
        gui = gui_canvas.CanvasGUI(image_data_list, orig_sizes, click_callback=click_callback)
        gui.run()


# ---------------------- CLI ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Image feature matcher with optional GPU acceleration and GUI.")
    parser.add_argument("-a", "--algorithm", choices=["SIFT", "AKAZE", "ORB", "BRISK", "KAZE"], default=None)
    parser.add_argument("--anomalies", choices=["none", "single", "batch"], default="none")
    parser.add_argument("-l", "--load-homographies", type=str, default=None)
    args = parser.parse_args()
    algorithm = args.algorithm.upper() if args.algorithm else None
    anomalies_mode = args.anomalies
    homographies_path = args.load_homographies
    main(algorithm, anomalies=anomalies_mode, homographies_path=homographies_path)
