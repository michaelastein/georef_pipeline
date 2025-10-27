import cv2
import numpy as np
from tkinter import Tk, Canvas, Frame, Scrollbar, Label, BOTH, NW, LEFT, RIGHT, Y, X, VERTICAL, HORIZONTAL, BOTTOM
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilenames
import time

# --- Load images ---
root = Tk()
root.withdraw()
img_paths = askopenfilenames(title="Select all images")
if not img_paths:
    exit()

images = [cv2.imread(p) for p in img_paths]
num_images = len(images)
print(f"Number of images: {num_images}")

# --- ORB + BFMatcher ---
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Compute homographies between neighbors (15 nearest) ---
positions = np.arange(len(images)).reshape(-1,1)  # Dummy positions for neighbor selection
neighbor_count = 15

homographies = {}  # (i,j) -> H
for i in range(len(images)):
    img_i = images[i]
    kp_i, des_i = orb.detectAndCompute(img_i, None)
    # choose neighbors
    start = max(0, i-neighbor_count//2)
    end = min(len(images), i+neighbor_count//2+1)
    for j in range(start, end):
        if i >= j:
            continue
        img_j = images[j]
        kp_j, des_j = orb.detectAndCompute(img_j, None)
        matches = bf.match(des_i, des_j)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 4:
            continue
        src_pts = np.float32([kp_i[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp_j[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            homographies[(i,j)] = H
            print(f"Stored homography {i}↔{j} (inliers: {np.sum(mask)})")

# --- Tkinter scrollable frame setup ---
root.deiconify()
root.title("Image Grid Viewer")

frame_main = Frame(root)
frame_main.pack(fill=BOTH, expand=1)

canvas_scroll = Canvas(frame_main)
scrollbar_v = Scrollbar(frame_main, orient=VERTICAL, command=canvas_scroll.yview)
scrollbar_h = Scrollbar(frame_main, orient=HORIZONTAL, command=canvas_scroll.xview)
canvas_scroll.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)

scrollbar_v.pack(side=RIGHT, fill=Y)
scrollbar_h.pack(side=BOTTOM, fill=X)
canvas_scroll.pack(side=LEFT, fill=BOTH, expand=1)

frame_inner = Frame(canvas_scroll)
canvas_scroll.create_window((0,0), window=frame_inner, anchor=NW)

# --- Display images in a grid (6 columns) ---
cols = 6
thumb_w, thumb_h = 200, 150
photo_refs = []
labels = []

for idx, img in enumerate(images):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((thumb_w, thumb_h))
    imtk = ImageTk.PhotoImage(img_pil)
    photo_refs.append(imtk)  # keep reference

    lbl = Label(frame_inner, image=imtk, bd=2, relief="solid")
    lbl.grid(row=idx//cols, column=idx%cols, padx=5, pady=5)
    lbl.image_idx = idx
    labels.append(lbl)

# --- Click event ---
marker_radius = 5
current_markers = []

def click_event(event, idx):
    global current_markers
    # remove previous markers
    for m in current_markers:
        m.destroy()
    current_markers = []

    x = event.x
    y = event.y
    print(f"Clicked image {idx} at ({x},{y})")

    lbl = labels[idx]
    # marker on clicked image
    mc = Canvas(lbl, width=lbl.winfo_width(), height=lbl.winfo_height(),
                bg=None, highlightthickness=0)
    mc.place(x=0, y=0)
    mc.create_oval(x-marker_radius, y-marker_radius, x+marker_radius, y+marker_radius,
                   outline="red", width=2)
    current_markers.append(mc)

    # propagate marker to neighbors using homographies
    for jdx in range(len(images)):
        if jdx == idx:
            continue
        H = None
        if (idx,jdx) in homographies:
            H = homographies[(idx,jdx)]
        elif (jdx,idx) in homographies:
            H = np.linalg.inv(homographies[(jdx,idx)])
        if H is None:
            continue
        pt = np.array([[x, y, 1]]).T
        pt_trans = H @ pt
        pt_trans /= pt_trans[2]
        xt, yt = pt_trans[0,0]*(thumb_w/labels[jdx].winfo_width()), pt_trans[1,0]*(thumb_h/labels[jdx].winfo_height())
        mc2 = Canvas(labels[jdx], width=labels[jdx].winfo_width(),
                     height=labels[jdx].winfo_height(), bg=None, highlightthickness=0)
        mc2.place(x=0, y=0)
        mc2.create_oval(xt-marker_radius, yt-marker_radius, xt+marker_radius, yt+marker_radius,
                        outline="red", width=2)
        current_markers.append(mc2)

for idx, lbl in enumerate(labels):
    lbl.bind("<Button-1>", lambda e, ix=lbl.image_idx: click_event(e, ix))

def update_scrollregion(event):
    canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))

frame_inner.bind("<Configure>", update_scrollregion)

root.mainloop()
