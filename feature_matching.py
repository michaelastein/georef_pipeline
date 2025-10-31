import cv2
import numpy as np
import csv
from pathlib import Path


from tkinter.filedialog import askopenfilenames, askopenfilename
import math
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import anomalies_batch





# ---------------------- Utility Functions ----------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
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


# ---------------------- Metadata / CSV ----------------------
def extract_metadata_from_csv(img_paths):
    folder = Path(img_paths[0]).parent
    csv_files = list(folder.glob("*.csv"))
    if len(csv_files) != 1:
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

    image_data_list = []
    for idx, path in enumerate(img_paths):
        fname = os.path.basename(path)
        meta = metadata_dict.get(fname)
        w, h = cv2.imread(path).shape[1], cv2.imread(path).shape[0]
        entry = {
            "image_index": idx,
            "image_path": os.path.abspath(path),
            "gps": (meta["lat"], meta["lon"], meta["alt"], meta["rel_alt"]) if meta else (None, None, None, None),
            "yaw": meta["yaw"] if meta else None,
            "pitch": meta["pitch"] if meta else None,
            "roll": meta["roll"] if meta else None,
            "image_size": (w, h)
        }
        image_data_list.append(entry)
        if meta is None:
            print(f"Warning: No CSV metadata for {fname}")
    return image_data_list


# ---------------------- Correspondences ----------------------
def build_correspondences_from_pixels(idx, x, y, image_data_list, H_dict, node_connected_component, shortest_path):
    pt = np.array([[x], [y], [1.0]])
    correspondences = [(idx, x, y)]
    comp = node_connected_component(idx)
    for other in comp:
        if other == idx:
            continue
        path = shortest_path(idx, other)
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
def main(algorithm=None, anomalies= None):
    threshold_meters = 40.0
    ratio_test = 0.7
    ransac_thresh = 4.0
    min_inliers = 20
    dist_consistency_thresh = 40.0
    max_workers = 8
    progress_lock = threading.Lock()
    start_time = time.time()

    # ---------------------- GUI File Selection ----------------------
    
    img_paths = askopenfilenames(title="Select images", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
    if not img_paths:
        print("No images selected.")
        return
    print(f"Selected {len(img_paths)} images.")

    image_data_list = extract_metadata_from_csv(img_paths)
    images, orig_sizes, gps_positions = [], [], []
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

    # ---------------------- Detector ----------------------
    if algorithm is None:
        algorithm = "BRISK" if img_paths[0].lower().endswith((".tif", ".tiff")) else "SIFT"

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
        if des is None: return None
        if descriptor_type == "float" and des.dtype != np.float32: return des.astype(np.float32)
        if descriptor_type != "float" and des.dtype != np.uint8: return des.astype(np.uint8)
        return des

    def compute_features_for_index(idx):
        gray = preprocess_for_features(images[idx])
        kp, des = detector.detectAndCompute(gray, None)
        kp_list[idx] = kp
        des_list[idx] = des.copy() if des is not None else None

    with ThreadPoolExecutor(max_workers=min(max_workers, len(images))) as ex:
        futures = [ex.submit(compute_features_for_index, i) for i in range(len(images))]
        done_count = 0
        for f in as_completed(futures):
            done_count += 1
            print_progress(done_count, len(futures), "Feature extraction", lock=progress_lock)

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
    match_cache, H_dict, H_inliers = {}, {}, {}

    def match_and_filter_pairs(i, j):
        key = (i, j)
        if key in match_cache: return match_cache[key]
        des_i, des_j = des_list[i], des_list[j]
        kp_i, kp_j = kp_list[i], kp_list[j]
        if des_i is None or des_j is None or kp_i is None or kp_j is None:
            match_cache[key] = []; return []
        des_i_q, des_j_q = ensure_flann_dtype(des_i), ensure_flann_dtype(des_j)
        try:
            knn_j_i = matcher.knnMatch(des_j_q, des_i_q, k=2)
            knn_i_j = matcher.knnMatch(des_i_q, des_j_q, k=2)
        except cv2.error:
            bf = cv2.BFMatcher(cv2.NORM_L2 if descriptor_type == "float" else cv2.NORM_HAMMING, crossCheck=False)
            knn_j_i = bf.knnMatch(des_j_q, des_i_q, k=2)
            knn_i_j = bf.knnMatch(des_i_q, des_j_q, k=2)

        def filter_good(knn):
            good = [m_n[0] for m_n in knn if len(m_n) == 2 and m_n[0].distance < ratio_test*m_n[1].distance]
            return good

        good_j_i, good_i_j = filter_good(knn_j_i), filter_good(knn_i_j)
        best_j_to_i = {m.queryIdx: m.trainIdx for m in good_j_i}
        best_i_to_j = {m.queryIdx: m.trainIdx for m in good_i_j}
        mutual = [(q, t) for q, t in best_j_to_i.items() if t in best_i_to_j and best_i_to_j[t] == q]
        if not mutual:
            match_cache[key] = []; return []
        pts_j = np.array([kp_j[q].pt for q, _ in mutual])
        pts_i = np.array([kp_i[t].pt for _, t in mutual])
        vecs = pts_i - pts_j
        mean_vec = np.mean(vecs, axis=0)
        dists = np.linalg.norm(vecs - mean_vec, axis=1)
        filtered = [mutual[idx] for idx, k in enumerate(dists <= dist_consistency_thresh) if k]
        match_cache[key] = filtered
        return filtered

    def compute_homography_for_pair(pair):
        i, j = pair
        matches = match_and_filter_pairs(i, j)
        if len(matches) < min_inliers: return None
        src_pts = np.float32([kp_list[j][q].pt for q,_ in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp_list[i][t].pt for _,t in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if H is None or mask is None or int(np.sum(mask)) < min_inliers: return None
        in_src = src_pts[mask.ravel()==1].reshape(-1,2)
        in_dst = dst_pts[mask.ravel()==1].reshape(-1,2)
        return i, j, H, in_src.copy(), in_dst.copy()

    if neighbors:
        done_pairs = 0
        with ThreadPoolExecutor(max_workers=min(max_workers,len(neighbors))) as ex:
            futures = {ex.submit(compute_homography_for_pair, p): p for p in neighbors}
            for fut in as_completed(futures):
                res = fut.result()
                done_pairs += 1
                print_progress(done_pairs, len(futures), "Homography", lock=progress_lock)
                if res is None: continue
                i, j, H, in_src, in_dst = res
                H_dict[(j,i)] = H; H_inliers[(j,i)] = in_dst.copy()
                try: H_inv = np.linalg.inv(H); H_dict[(i,j)] = H_inv; H_inliers[(i,j)] = in_src.copy()
                except np.linalg.LinAlgError: pass

    print(f"Feature extraction + matching + homography runtime: {(time.time()-start_time)/60:.2f} min")
    print(f"Computed {len(H_dict)//2} good homography pairs.")

    # ---------------------- Graph helpers ----------------------
    adj = {}
    for (a,b) in H_dict.keys():
        adj.setdefault(a,set()).add(b)
        adj.setdefault(b,set()).add(a)

    def node_connected_component(start):
        if start not in adj: return {start}
        seen = {start}; q=[start]
        while q:
            u = q.pop(0)
            for v in adj.get(u,()):
                if v not in seen:
                    seen.add(v); q.append(v)
        return seen

    def shortest_path(u,v):
        if u==v: return [u]
        if u not in adj: return None
        from collections import deque
        q = deque([u]); parent={u:None}
        while q:
            cur = q.popleft()
            for nb in adj.get(cur,()):
                if nb not in parent:
                    parent[nb]=cur
                    if nb==v:
                        path=[v]; p=v
                        while parent[p] is not None:
                            p=parent[p]; path.append(p)
                        path.reverse(); return path
                    q.append(nb)
        return None

    # ---------------------- GUI / anomalies mode ----------------------
    if anomalies== "single":
        # Launch anomaly GUI
        from gui_anomalies import launch_anomaly_gui
        launch_anomaly_gui( img_paths,orig_sizes, image_data_list, H_dict, node_connected_component, shortest_path)
    
    elif anomalies == "batch":
        anomalies_batch.main(image_data_list)
    

    else:
        # Launch canvas GUI
        from gui_canvas import launch_canvas_gui
        launch_canvas_gui( img_paths, orig_sizes, image_data_list, H_dict, node_connected_component, shortest_path)

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

    args = parser.parse_args()
    algorithm = args.algorithm.upper() if args.algorithm else None
    anomalies_mode = args.anomalies  # will be 'none', 'single', or 'batch'

    main(algorithm, anomalies=anomalies_mode)

