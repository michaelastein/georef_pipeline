import cv2
import numpy as np
import piexif
from tkinter import Tk, Canvas, Scrollbar, LEFT, RIGHT, Y, NW, Toplevel, Label, Entry, Button, Checkbutton, BooleanVar
from PIL import Image, ImageTk, ImageDraw
from tkinter.filedialog import askopenfilenames, askopenfilename
import math
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import avg_gps
import queue
from collections import OrderedDict
import matching_anomalies


def main(algorithm=None, no_gui=False):
    # ----------------- Parameters -----------------

    
    threshold_meters = 40.0
    ratio_test = 0.7
    ransac_thresh = 4.0
    min_inliers = 20
    dist_consistency_thresh = 40.0
    max_workers = 8
    
    feature_algo = algorithm
    start_time = time.time()
    progress_lock = threading.Lock()

    def print_progress(current, total, stage_name, last_print=[-1]):
        percent = int((current / total) * 100) if total > 0 else 100
        with progress_lock:
            if percent // 5 != last_print[0] or current == total:
                print(f"[{stage_name}] Progress: {percent}% ({current}/{total})")
                last_print[0] = percent // 5

    # ----------------- Helper functions -----------------
    def rational_to_float(r):
        if isinstance(r, (tuple, list)):
            return r[0] / r[1]
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
        return yaw, pitch, roll, rel_alt

    def collect_correspondence_data(idx, x_click, y_click, correspondences):
        result = []
        for img_idx, x, y in correspondences:
            lat, lon, alt = gps_positions[img_idx] if gps_positions[img_idx] != (None, None, None) else (None, None, None)

            yaw = pitch = roll = rel_alt = None
            try:
                exif_dict = piexif.load(img_paths[img_idx])
                yaw, pitch, roll, rel_alt = parse_description_from_exif(exif_dict)
            except Exception:
                pass

            entry = {
                "image_index": img_idx,
                "image_path": os.path.abspath(img_paths[img_idx]),
                "pixel_x": float(x),
                "pixel_y": float(y),
                "gps_lat": lat,
                "gps_lon": lon,
                "gps_alt": alt,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "rel_alt": rel_alt,
                "image_size": orig_sizes[img_idx]
            }
            result.append(entry)
        return result

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    def preprocess_for_features(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    # ----------------- GUI File Selection -----------------
    root = Tk()
    root.withdraw()
    img_paths = askopenfilenames(
        title="Select images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if len(img_paths) < 1:
        print("No images selected.")
        return
    print(f"Selected {len(img_paths)} images.")

    # ----------------- Load images and GPS -----------------
    images = []
    gps_positions = []
    orig_sizes = []
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: could not read {path}")
            continue
        images.append(img)
        h, w = img.shape[:2]
        orig_sizes.append((w, h))
        try:
            exif_dict = piexif.load(path)
            gps = extract_gps_from_exif(exif_dict)
            gps_positions.append(gps)
        except Exception as e:
            print(f"Warning: no GPS in {path}, {e}")
            gps_positions.append((None, None, None))
    if len(images) < 1:
        print("No valid images loaded.")
        return
    print(f"Loaded {len(images)} images with GPS info.")

    

    # ----------------- Detector/descriptor -----------------
    # ----------------- Auto-select feature algorithm if None -----------------
    if feature_algo is None:
        first_path = img_paths[0]
        if first_path.lower().endswith(".tiff") or first_path.lower().endswith(".tif"):
            feature_algo = "BRISK"
        else:
            feature_algo = "SIFT"
     

    if feature_algo == "SIFT":
        detector = cv2.SIFT_create()
        descriptor_type = 'float'
    elif feature_algo == "AKAZE":
        detector = cv2.AKAZE_create()
        descriptor_type = 'binary'
    elif feature_algo == "ORB":
        detector = cv2.ORB_create(nfeatures=3000)
        descriptor_type = 'binary'
    elif feature_algo == "BRISK":
        detector = cv2.BRISK_create()
        descriptor_type = 'binary'
    elif feature_algo == "KAZE":
        detector = cv2.KAZE_create()
        descriptor_type = 'float'
    else:
        raise ValueError(f"Unsupported feature algorithm: {feature_algo}")
    
    print(f"Using feature algorithm: {feature_algo}")


    # ----------------- Matcher -----------------
    if descriptor_type == 'float':
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ----------------- Compute features -----------------
    kp_list = [None] * len(images)
    des_list = [None] * len(images)

    def ensure_flann_dtype(des):
        if des is None:
            return None
        if descriptor_type == 'float' and des.dtype != np.float32:
            return des.astype(np.float32)
        if descriptor_type != 'float' and des.dtype != np.uint8:
            return des.astype(np.uint8)
        return des

    def compute_features_for_index(idx):
        img = images[idx]
        gray = preprocess_for_features(img)
        kp, des = detector.detectAndCompute(gray, None)
        kp_list[idx] = kp
        des_list[idx] = None if des is None else des.copy()

    if len(images) > 0:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(images))) as ex:
            futures = [ex.submit(compute_features_for_index, i) for i in range(len(images))]
            done_count = 0
            for f in as_completed(futures):
                done_count += 1
                print_progress(done_count, len(futures), "Feature extraction")

    # ----------------- GPS neighbor prefiltering -----------------
    neighbors = []
    for i, (lat_i, lon_i, _) in enumerate(gps_positions):
        if lat_i is None:
            continue
        for j, (lat_j, lon_j, _) in enumerate(gps_positions):
            if i >= j or lat_j is None:
                continue
            if haversine(lat_i, lon_i, lat_j, lon_j) <= threshold_meters:
                neighbors.append((i, j))
    print(f"Found {len(neighbors)} likely neighbor pairs based on GPS.")
    if len(neighbors) == 0 and len(images) > 1:
        print("No GPS neighbors found — falling back to all pairs (this may be slow).")
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                neighbors.append((i, j))
    print(f"Total pairs to attempt: {len(neighbors)}")

    # ----------------- Matching + Homography -----------------
    match_cache = {}
    H_dict = {}
    H_inliers = {}

    def match_and_filter_pairs(i, j):
        key = (i, j)
        if key in match_cache:
            return match_cache[key]

        des_i = des_list[i]
        des_j = des_list[j]
        kp_i = kp_list[i]
        kp_j = kp_list[j]
        if des_i is None or des_j is None or kp_i is None or kp_j is None:
            match_cache[key] = []
            return []

        des_i_q = ensure_flann_dtype(des_i)
        des_j_q = ensure_flann_dtype(des_j)
        try:
            knn_j_i = matcher.knnMatch(des_j_q, des_i_q, k=2)
            knn_i_j = matcher.knnMatch(des_i_q, des_j_q, k=2)
        except cv2.error:
            bf = cv2.BFMatcher(cv2.NORM_L2 if descriptor_type == 'float' else cv2.NORM_HAMMING, crossCheck=False)
            knn_j_i = bf.knnMatch(des_j_q, des_i_q, k=2)
            knn_i_j = bf.knnMatch(des_i_q, des_j_q, k=2)

        def filter_good(knn):
            good = []
            for m_n in knn:
                if len(m_n) == 2 and m_n[0].distance < ratio_test * m_n[1].distance:
                    good.append(m_n[0])
            return good

        good_j_i = filter_good(knn_j_i)
        good_i_j = filter_good(knn_i_j)

        best_j_to_i = {m.queryIdx: m.trainIdx for m in good_j_i}
        best_i_to_j = {m.queryIdx: m.trainIdx for m in good_i_j}
        mutual = [(q, t) for q, t in best_j_to_i.items() if t in best_i_to_j and best_i_to_j[t] == q]

        if not mutual:
            match_cache[key] = []
            return []

        pts_j = np.array([kp_j[q].pt for q, _ in mutual])
        pts_i = np.array([kp_i[t].pt for _, t in mutual])
        vecs = pts_i - pts_j
        mean_vec = np.mean(vecs, axis=0)
        dists = np.linalg.norm(vecs - mean_vec, axis=1)
        keep_mask = dists <= dist_consistency_thresh
        filtered = [mutual[idx] for idx, k in enumerate(keep_mask) if k]
        match_cache[key] = filtered
        return filtered

    def compute_homography_for_pair(pair):
        i, j = pair
        matches = match_and_filter_pairs(i, j)
        kp_i = kp_list[i]
        kp_j = kp_list[j]
        if len(matches) < min_inliers:
            return None
        src_pts = np.float32([kp_j[q].pt for q, _ in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_i[t].pt for _, t in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if H is None or mask is None or int(np.sum(mask)) < min_inliers:
            return None
        in_src = src_pts[mask.ravel() == 1].reshape(-1, 2)
        in_dst = dst_pts[mask.ravel() == 1].reshape(-1, 2)
        return (i, j, H, in_src.copy(), in_dst.copy())

    pairs_to_process = neighbors.copy()
    print(f"Computing homographies for {len(pairs_to_process)} pairs (parallel)...")
    if len(pairs_to_process) > 0:
        done_pairs = 0
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pairs_to_process))) as ex:
            futures = {ex.submit(compute_homography_for_pair, p): p for p in pairs_to_process}
            for fut in as_completed(futures):
                res = fut.result()
                done_pairs += 1
                print_progress(done_pairs, len(pairs_to_process), "Homography")
                if res is None:
                    continue
                i, j, H, in_src, in_dst = res
                H_dict[(j, i)] = H
                H_inliers[(j, i)] = in_dst.copy()
                try:
                    H_inv = np.linalg.inv(H)
                    H_dict[(i, j)] = H_inv
                    H_inliers[(i, j)] = in_src.copy()
                except np.linalg.LinAlgError:
                    pass
    else:
        print("No image pairs to process — skipping homography computation.")

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Feature extraction + matching + homography runtime: {elapsed_minutes:.2f} minutes "
          f"for {len(img_paths)} images.")
    print(f"Computed {len(H_dict) // 2} good homography pairs.")

    # ----------------- Graph helpers -----------------
    adj = {}
    for (a, b) in H_dict.keys():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def node_connected_component(start):
        if start not in adj:
            return {start}
        seen = {start}
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        return seen

    def shortest_path(u, v):
        if u == v:
            return [u]
        if u not in adj:
            return None
        from collections import deque
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
    
    # No-GUI mode
    # ----------------- No-GUI mode -----------------
        # ----------------- No-GUI mode -----------------
    if no_gui:
        root.deiconify()  # make sure Tk root exists
        while True:
            # Ask user to select one image from the same folder
            single_path = askopenfilename(
                title="Select one image from the same folder",
                initialdir=os.path.dirname(img_paths[0]),
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
            )
            if not single_path:
                print("No image selected. Exiting.")
                break  # exit the loop and program

            if os.path.dirname(single_path) != os.path.dirname(img_paths[0]):
                print("Selected image is not from the same folder. Try again.")
                continue  # loop again

            if single_path not in img_paths:
                print("Selected image was not in the originally selected batch. Try again.")
                continue  # loop again

            idx = img_paths.index(single_path)

            # Ask for X and Y coordinates
            x = simpledialog.askfloat("Input", f"Enter X pixel coordinate for {os.path.basename(single_path)}")
            y = simpledialog.askfloat("Input", f"Enter Y pixel coordinate for {os.path.basename(single_path)}")
            if x is None or y is None:
                print("No coordinates entered. Exiting.")
                break  # exit the loop and program

            w, h = orig_sizes[idx]
            if not (0 <= x < w) or not (0 <= y < h):
                print(f"Coordinates ({x}, {y}) are outside image bounds ({w}x{h}). Try again.")
                continue  # loop again

            # Compute correspondences and call avg_gps
            pt = np.array([[x], [y], [1.0]])
            comp = list(node_connected_component(idx))
            correspondences = [(idx, x, y)]
            for other in comp:
                if other == idx:
                    continue
                path = shortest_path(idx, other)
                if path is None:
                    continue
                cur_pt = pt.copy()
                ok = True
                for k in range(len(path) - 1):
                    a = path[k]
                    b = path[k + 1]
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
                if not ok:
                    continue
                correspondences.append((other, cur_pt[0, 0], cur_pt[1, 0]))

                data = collect_correspondence_data(idx, x, y, correspondences)

                # Always run avg_gps
                try:
                    avg_gps.main(data)
                except Exception as e:
                    print(f"avg_gps.main error: {e}")

            # Show pixel on image
            try:
                img = Image.open(single_path).convert("RGB")
                draw = Image.new("RGB", img.size)
                draw.paste(img)
                from PIL import ImageDraw
                draw_img = ImageDraw.Draw(draw)
                r = 5
                draw_img.ellipse((x-r, y-r, x+r, y+r), outline="red", width=2)
                viz_win = Toplevel()
                viz_win.title(f"Selected Pixel in {os.path.basename(single_path)}")
                tk_img = ImageTk.PhotoImage(draw)
                lbl = Label(viz_win, image=tk_img)
                lbl.image = tk_img
                lbl.pack()
                viz_win.mainloop()
            except Exception as e:
                print(f"Error visualizing the selected pixel: {e}")


    
    else:


        # ----------------- Canvas-based GUI mit Lazy Loading -----------------
        root.deiconify()
        root.title("Images Grid (Canvas + Lazy Loading)")
        canvas_w, canvas_h = 1400, 800
        canvas_side = Canvas(root, width=canvas_w, height=canvas_h, bg="black")
        scrollbar = Scrollbar(root, orient="vertical", command=canvas_side.yview)
        canvas_side.configure(yscrollcommand=scrollbar.set)
        canvas_side.pack(side=LEFT, fill="both", expand=1)
        scrollbar.pack(side=RIGHT, fill=Y)



        # Layout-Parameter
        cols = 6
        thumb_size = 200  # Basisgröße (kann dynamisch angepasst werden)
        padding = 10
        text_height = 18
        row_height = thumb_size + text_height + padding

        # caches
        # pil_cache holds PIL.Image objects produced in background threads
        pil_cache = OrderedDict()  # index -> PIL.Image
        # photo_cache holds ImageTk.PhotoImage objects created in main thread (to draw on Canvas)
        photo_cache = {}  # index -> PhotoImage
        MAX_CACHE_ITEMS = 200  # Gesamtzahl von Thumbnails behalten (anpassen je nach RAM)
        cache_lock = threading.Lock()

        # Queues für Background Loader
        load_queue = queue.Queue()
        result_queue = queue.Queue()
        stop_event = threading.Event()

        # Berechne Positionen bereits (so wissen wir immer bbox auch wenn noch nicht geladen)
        img_positions = []  # tuple: (x_canvas, y_canvas, scale, disp_w, disp_h)
        for i, (w, h) in enumerate(orig_sizes):
            scale = min(thumb_size / h, thumb_size / w)
            disp_w, disp_h = int(w * scale), int(h * scale)
            col = i % cols
            row = i // cols
            x = padding + col * (thumb_size + padding)
            y = padding + row * row_height
            img_positions.append((x, y, scale, disp_w, disp_h))

        # canvas total size
        n_rows = (len(img_paths) + cols - 1) // cols
        total_height = padding + n_rows * row_height
        total_width = padding + cols * (thumb_size + padding)
        canvas_side.config(scrollregion=(0, 0, total_width, total_height))

        # visible indices helpers
        def get_visible_indices():
            y0 = canvas_side.canvasy(0)
            y1 = canvas_side.canvasy(canvas_side.winfo_height())
            # consider a margin to preload just-offscreen tiles
            margin = row_height * 2
            top_row = max(0, int((y0 - margin) // row_height))
            bottom_row = min(n_rows - 1, int((y1 + margin) // row_height))
            indices = []
            for r in range(top_row, bottom_row + 1):
                start = r * cols
                end = min(len(img_paths), start + cols)
                indices.extend(range(start, end))
            return set(indices)

        # ----------------- Background loader thread function -----------------
        def loader_worker():
            while not stop_event.is_set():
                try:
                    idx = load_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                with cache_lock:
                    if idx in pil_cache:
                        load_queue.task_done()
                        continue
                path = img_paths[idx]
                try:
                    bgr = cv2.imread(path)
                    if bgr is None:
                        load_queue.task_done()
                        continue
                    _, _, scale, disp_w, disp_h = img_positions[idx]
                    resized = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    # nur PIL.Image ins result_queue, kein PhotoImage im Thread
                    result_queue.put((idx, pil_img))
                except Exception as e:
                    print(f"Loader error for index {idx}: {e}")
                finally:
                    load_queue.task_done()

        # Start a pool of loader threads (gleichzeitiges Lesen + Resize)
        NUM_LOADER_THREADS = max(2, min(8, os.cpu_count() or 4))
        loader_threads = []
        for _ in range(NUM_LOADER_THREADS):
            t = threading.Thread(target=loader_worker, daemon=True)
            t.start()
            loader_threads.append(t)

        # Preloader: enqueued indices near visible area
        def enqueue_visible_and_neighbors():
            vis = get_visible_indices()
            # add neighbors +/- one row for smoother scrolling
            extra = set()
            for idx in vis:
                row = idx // cols
                for r in range(max(0, row - 1), min(n_rows, row + 2)):
                    start = r * cols
                    extra.update(range(start, min(start + cols, len(img_paths))))
            to_load = list(extra)
            # push into queue if not cached/pending
            for idx in to_load:
                with cache_lock:
                    if idx in pil_cache:
                        continue
                # avoid duplicate queueing via membership check - cheap check by trying to enqueue (no direct way to inspect queue)
                try:
                    load_queue.put_nowait(idx)
                except queue.Full:
                    pass

        # Drawing helpers
        drawn_image_items = {}  # index -> canvas_image_id
        drawn_text_items = {}   # index -> canvas_text_id
        marker_items = []       # list of marker canvas ids

        def draw_canvas_image(idx, photo):
            # If already drawn update? For simplicity: delete old and create new.
            if idx in drawn_image_items:
                try:
                    canvas_side.delete(drawn_image_items[idx])
                except Exception:
                    pass
            x, y, scale, disp_w, disp_h = img_positions[idx]
            img_id = canvas_side.create_image(x, y, image=photo, anchor=NW)
            drawn_image_items[idx] = img_id
            # draw name text below
            if idx in drawn_text_items:
                canvas_side.delete(drawn_text_items[idx])
            name = os.path.splitext(os.path.basename(img_paths[idx]))[0]
            tid = canvas_side.create_text(x + disp_w / 2, y + disp_h + 12, text=name, fill="white")
            drawn_text_items[idx] = tid

        def create_marker_canvas(x, y, color="red", size=6):
            half = size / 2
            return canvas_side.create_rectangle(x - half, y - half, x + half, y + half, fill=color, outline=color)

        # ----------------- Main-thread Verarbeitung -----------------
        def process_loader_results():
            processed = 0
            while True:
                try:
                    idx, pil_img = result_queue.get_nowait()
                except queue.Empty:
                    break
                with cache_lock:
                    pil_cache[idx] = pil_img
                    pil_cache.move_to_end(idx)
                    while len(pil_cache) > MAX_CACHE_ITEMS:
                        old_idx, _ = pil_cache.popitem(last=False)
                        if old_idx in photo_cache:
                            del photo_cache[old_idx]
                        if old_idx in drawn_image_items:
                            canvas_side.delete(drawn_image_items[old_idx])
                            del drawn_image_items[old_idx]
                        if old_idx in drawn_text_items:
                            canvas_side.delete(drawn_text_items[old_idx])
                            del drawn_text_items[old_idx]

                # Tkinter PhotoImage erstellen **im Main-Thread**
                photo = ImageTk.PhotoImage(pil_img)
                photo_cache[idx] = photo
                if idx in get_visible_indices():
                    draw_canvas_image(idx, photo)
                result_queue.task_done()
                processed += 1
            return processed

        # Click handling: map canvas coords to image index and pixel coordinates
        def on_canvas_click(event):
            x_c = canvas_side.canvasx(event.x)
            y_c = canvas_side.canvasy(event.y)
            # find clicked image by bounds
            for idx, (x, y, scale, disp_w, disp_h) in enumerate(img_positions):
                if x <= x_c <= x + disp_w and y <= y_c <= y + disp_h:
                    # compute original image coordinates (in pixel space of original)
                    local_x_scaled = (x_c - x)
                    local_y_scaled = (y_c - y)
                    x_orig = local_x_scaled / scale
                    y_orig = local_y_scaled / scale
                    handle_image_click(idx, x_orig, y_orig, x, y, scale)
                    break

        # ----------------- Click-Handler -----------------
        def handle_image_click(idx, x_orig, y_orig, x_canvas, y_canvas, scale):
            # Marker löschen
            for mid in marker_items:
                canvas_side.delete(mid)
            marker_items.clear()

            # Marker für eigenes Bild
            m_self = create_marker_canvas(x_canvas + x_orig * scale, y_canvas + y_orig * scale, color="red")
            marker_items.append(m_self)

            # Homography-Korrespondenzen
            pt = np.array([[x_orig], [y_orig], [1.0]])
            comp = list(node_connected_component(idx))
            correspondences = [(idx, x_orig, y_orig)]
            for other in comp:
                if other == idx:
                    continue
                path = shortest_path(idx, other)
                if path is None:
                    continue
                cur_pt = pt.copy()
                ok = True
                for k in range(len(path) - 1):
                    a = path[k]
                    b = path[k + 1]
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
                if not ok:
                    continue
                x_img, y_img, sc, disp_w, disp_h = img_positions[other]
                x_disp, y_disp = cur_pt[0, 0] * sc, cur_pt[1, 0] * sc
                if 0 <= x_disp <= disp_w and 0 <= y_disp <= disp_h:
                    mx = create_marker_canvas(x_img + x_disp, y_img + y_disp, color="yellow")
                    marker_items.append(mx)
                    correspondences.append((other, cur_pt[0, 0], cur_pt[1, 0]))

            # avg_gps.main 
            data = collect_correspondence_data(idx, x_orig, y_orig, correspondences)
            root.after(0, lambda d=data: safe_avg_gps_call(d))


        def safe_avg_gps_call(data):
            try:
                avg_gps.main(data)
            except Exception as e:
                print(f"avg_gps.main error: {e}")

        # Periodic UI update function
        def periodic_update():
            # 1) Enqueue visible indices and neighbors
            enqueue_visible_and_neighbors()
            # 2) Process result queue (create PhotoImages + draw)
            processed = process_loader_results()
            # 3) Make sure visible indices are drawn if we already have PhotoImage
            vis = get_visible_indices()
            for idx in vis:
                with cache_lock:
                    if idx in photo_cache and idx not in drawn_image_items:
                        draw_canvas_image(idx, photo_cache[idx])
            # Schedule next call
            root.after(100, periodic_update)

        # initial filling (enqueue a few)
        enqueue_visible_and_neighbors()
        canvas_side.bind("<Button-1>", on_canvas_click)
        root.after(100, periodic_update)

        # Run Tk mainloop; on close, stop threads cleanly
        try:
            root.mainloop()
        finally:
            stop_event.set()
            # wait for loader threads to finish
            for t in loader_threads:
                t.join(timeout=0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image feature matcher with optional GUI or headless mode.\n"
                    "Supports SIFT, AKAZE, ORB, BRISK, KAZE feature algorithms.\n"
                    "Use --no-gui to select a single image and provide coordinates via dialog."
    )
    parser.add_argument("-a", "--algorithm", choices=["SIFT", "AKAZE", "ORB", "BRISK", "KAZE"],
                        default=None, help="Feature detection algorithm (default: auto-select)")
    parser.add_argument("-no-gui", action="store_true",
                        help="Run in headless mode; select coordinates via dialog.")
    args = parser.parse_args()

    algorithm = args.algorithm.upper() if args.algorithm else None
    main(algorithm, no_gui=args.no_gui)