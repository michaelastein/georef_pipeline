import os
import cv2
import json
import numpy as np
from tkinter import Tk, Canvas, Frame, Scrollbar, Label, LEFT, RIGHT, Y, NW
from tkinter.filedialog import askopenfilenames, askopenfilename
from PIL import Image, ImageTk
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------- GUI file selection -----------------
root = Tk()
root.withdraw()

# Select images
img_paths = askopenfilenames(title="Select images", filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
if len(img_paths) == 0:
    print("No images selected.")
    exit()

# Select JSON
json_file = askopenfilename(title="Select JSON file", filetypes=[("JSON files","*.json")])
if not json_file:
    print("No JSON file selected.")
    exit()

# ----------------- Load images -----------------
images = [cv2.imread(p) for p in img_paths]
orig_sizes = [(img.shape[1], img.shape[0]) for img in images]

# ----------------- Load JSON module points -----------------
with open(json_file, "r") as f:
    module_points = json.load(f)

# Convert to dict: module_id -> list of (image_idx, x, y)
tracks_dict = {}
for entry in module_points:
    mid = entry["module_id"]
    tracks_dict.setdefault(mid, []).append((entry["image_idx"], entry["x"], entry["y"]))

# ----------------- Feature detector -----------------
try:
    detector = cv2.SIFT_create()
    descriptor_type = 'float'
except:
    try:
        detector = cv2.AKAZE_create()
        descriptor_type = 'binary'
    except:
        detector = cv2.ORB_create(nfeatures=3000)
        descriptor_type = 'binary'

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

# ----------------- Compute keypoints & descriptors -----------------
kp_list, des_list = [], []
for img in images:
    gray = preprocess(img)
    kp, des = detector.detectAndCompute(gray, None)
    kp_list.append(kp)
    if descriptor_type=='float':
        des_list.append(des.astype(np.float32) if des is not None else None)
    else:
        des_list.append(des)

# ----------------- Matcher -----------------
if descriptor_type=='float':
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
else:
    matcher = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2), dict(checks=50))

# ----------------- Compute homographies -----------------
H_dict = {}
ratio_test = 0.7
min_inliers = 15
ransac_thresh = 4.0

def match_and_compute(i,j):
    des_i, des_j = des_list[i], des_list[j]
    kp_i, kp_j = kp_list[i], kp_list[j]
    if des_i is None or des_j is None: return None
    knn = matcher.knnMatch(des_i, des_j, k=2)
    good = [m for m,n in knn if m.distance < ratio_test*n.distance]
    if len(good) < min_inliers: return None
    src_pts = np.float32([kp_i[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp_j[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if H is None: return None
    return i,j,H

pairs = [(i,j) for i in range(len(images)) for j in range(i+1,len(images))]
with ThreadPoolExecutor() as ex:
    futures = {ex.submit(match_and_compute,i,j):(i,j) for i,j in pairs}
    for fut in as_completed(futures):
        res = fut.result()
        if res is not None:
            i,j,H = res
            H_dict[(i,j)] = H
            try:
                H_dict[(j,i)] = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                continue

# ----------------- Graph helpers -----------------
adj = {}
for a,b in H_dict.keys():
    adj.setdefault(a,set()).add(b)
    adj.setdefault(b,set()).add(a)

def shortest_path(u,v):
    if u==v: return [u]
    from collections import deque
    q = deque([u])
    parent = {u: None}
    while q:
        cur = q.popleft()
        for nb in adj.get(cur,()):
            if nb not in parent:
                parent[nb] = cur
                if nb==v:
                    path=[v]
                    p=v
                    while parent[p] is not None:
                        p=parent[p]
                        path.append(p)
                    path.reverse()
                    return path
                q.append(nb)
    return None

def propagate_point(pt,start_idx,target_idx):
    path = shortest_path(start_idx,target_idx)
    if path is None: return None
    cur_pt = np.array([pt[0],pt[1],1.0]).reshape(3,1)
    for k in range(len(path)-1):
        a,b = path[k], path[k+1]
        H = H_dict.get((a,b))
        if H is None: return None
        cur_pt = H @ cur_pt
        cur_pt = cur_pt/cur_pt[2,0]
    return cur_pt[:2,0]

# ----------------- Visualization -----------------
root = Tk()
root.title("Tracked Solar Modules with Lines")
canvas_side = Canvas(root,width=1400,height=800,bg="black")
scrollbar = Scrollbar(root,orient="vertical",command=canvas_side.yview)
canvas_side.configure(yscrollcommand=scrollbar.set)
canvas_side.pack(side=LEFT,fill="both",expand=1)
scrollbar.pack(side=RIGHT,fill=Y)
frame = Frame(canvas_side,bg="black")
canvas_side.create_window((0,0),window=frame,anchor=NW)

photo_refs=[]
img_positions=[]
markers=[]
cols=6
img_size=200
colors={mid: tuple([random.randint(0,255) for _ in range(3)]) for mid in tracks_dict.keys()}

for i,img in enumerate(images):
    h,w=img.shape[:2]
    scale=min(img_size/h,img_size/w)
    disp_w,disp_h=int(w*scale),int(h*scale)
    img_resized=cv2.resize(img,(disp_w,disp_h))
    img_rgb=cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
    im_pil=Image.fromarray(img_rgb)
    im_tk=ImageTk.PhotoImage(im_pil)
    photo_refs.append(im_tk)
    img_frame=Frame(frame,bd=1,relief="solid",bg="black")
    img_frame.grid(row=i//cols,column=i%cols,padx=5,pady=5)
    lbl=Label(img_frame,image=im_tk)
    lbl.pack()
    name_lbl=Label(img_frame,text=os.path.basename(img_paths[i]),fg="white",bg="black",font=("Arial",9))
    name_lbl.pack(fill="x")
    img_positions.append((lbl,scale,disp_w,disp_h))

# Place markers and draw lines
for mid, track in tracks_dict.items():
    points_by_img = {}
    for img_idx,x,y in track:
        points_by_img[img_idx] = np.array([x,y])
    # propagate to all connected images
    for img_idx in range(len(images)):
        if img_idx in points_by_img: continue
        for orig_idx, pt in points_by_img.items():
            propagated = propagate_point(pt, orig_idx, img_idx)
            if propagated is not None:
                points_by_img[img_idx] = propagated
                break
    # Draw markers & lines
    sorted_imgs = sorted(points_by_img.keys())
    prev_disp = None
    for img_idx in sorted_imgs:
        lbl,scale,_,_ = img_positions[img_idx]
        x_disp, y_disp = points_by_img[img_idx]*scale
        m=Label(lbl.master,bg="#%02x%02x%02x"%colors[mid])
        m.place(in_=lbl,x=int(x_disp),y=int(y_disp),anchor=NW,width=6,height=6)
        markers.append(m)
        # draw line from previous point
        if prev_disp is not None:
            lbl1,_,_,_ = img_positions[prev_disp[0]]
            x1,y1 = prev_disp[1:]
            lbl1.create_line(x1,y1,x_disp,y_disp,fill="#%02x%02x%02x"%colors[mid],width=2)
        prev_disp = (img_idx, x_disp, y_disp)

canvas_side.update_idletasks()
canvas_side.config(scrollregion=canvas_side.bbox("all"))
root.mainloop()
