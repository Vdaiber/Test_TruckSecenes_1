#!/usr/bin/env python3
import os
import cv2
import numpy as np

from truckscenes import TruckScenes
from truckscenes.utils.geometry_utils import view_points, transform_matrix, BoxVisibility
from pyquaternion import Quaternion
from oft.utils.config import load_config
from oft.utils.visualization import get_camera_intrinsic, get_sensor_extrinsic

def main():
    # 1) Config + TS init
    cfg = load_config()
    ts  = TruckScenes(version=cfg["dataset"]["version"],
                      dataroot=cfg["dataset"]["dataroot"])

    # 2) Wähle eine Szene + den ersten Sample-Token
    scene0 = ts.scene[0]["token"]
    sample = ts.get("sample", ts.get("scene", scene0)["first_sample_token"])
    print("→ sample_token:", sample["token"])

    # 3) Welche Kanäle gibt es?
    print("→ sample['data'] keys:", list(sample["data"].keys()))

    # 4) Kamera-Channel aus Config
    cam_ch = cfg["visualization"]["camera_channel"]
    print("→ Verwende Kamera-Channel:", cam_ch)
    if cam_ch not in sample["data"]:
        raise ValueError(f"Channel {cam_ch} nicht gefunden!")

    # 5) SampleData + Bild laden
    sd_tk = sample["data"][cam_ch]
    sd    = ts.get("sample_data", sd_tk)
    img_fn= sd["filename"]
    if not os.path.isabs(img_fn):
        img_fn = os.path.join(ts.dataroot, img_fn)
    img   = cv2.imread(img_fn)
    print("→ Bild geladen:", img_fn, "Shape:", img.shape)

    # 6) Matrices
    calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ts.get("ego_pose",         sd["ego_pose_token"])
    K     = get_camera_intrinsic(calib)
    H     = get_sensor_extrinsic(ego, calib)
    print("→ Intrinsic K:\n", K)
    print("→ Extrinsic H (world→cam):\n", H)

    # 7) Roh-Annos und sample_data-get
    print("\n--- GT-Boxen via get_sample_data ---")
    # box_vis_level ANY damit wir nur sichtbare holen
    _, boxes, _ = ts.get_sample_data(sd_tk, box_vis_level=BoxVisibility.ANY)
    print("→ boxes returned by get_sample_data:", len(boxes))

    # 8) debug: alle Z-Werte der Eckpunkte ausrechnen
    zs_all = []
    for i,box in enumerate(boxes):
        # corners homogeneous
        c = box.corners()                     # (3,8)
        homo = np.vstack((c, np.ones((1,8),dtype=np.float32)))
        pts = (H @ homo)[:3,:]                # (3,8)
        zs_all.append(pts[2,:])
        print(f" Box {i:2d}: Z-Werte min/max = {pts[2,:].min():.2f}/{pts[2,:].max():.2f}")

    zs_all = np.concatenate(zs_all)
    print(f"→ insgesamt {zs_all.size} Eck-Z-Werte,  min={zs_all.min():.2f}, max={zs_all.max():.2f}")

    # 9) apply your z-threshold
    z_thr = cfg["render"].get("z_threshold", 0.0)
    keep  = zs_all > z_thr
    print(f"→ mit z_threshold={z_thr:.2f}m bleiben {keep.sum()} Eckpunkte über Threshold")
    # aber wir wollen Boxen filtern: mind. eine Ecke > threshold
    boxes_filtered = []
    for i,box in enumerate(boxes):
        c = box.corners(); homo = np.vstack((c, np.ones((1,8))))
        pts = (H @ homo)[:3,:]
        if np.any(pts[2,:] > z_thr):
            boxes_filtered.append(box)
    print(f"→ Boxen vor/after threshold: {len(boxes)} → {len(boxes_filtered)}\n")

    # 10) Zeichne zum Vergleich beide Versionen nebeneinander
    def draw(img, box_list, title):
        tmp = img.copy()
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for box in box_list:
            c = box.corners(); homo = np.vstack((c, np.ones((1,8))))
            pts_cam = (H @ homo)[:3,:]
            pts_img = view_points(pts_cam, K, normalize=True)
            xs, ys, zs = pts_img
            if not np.any(zs>z_thr): continue
            for i,j in edges:
                p1 = (int(xs[i]), int(ys[i])); p2 = (int(xs[j]), int(ys[j]))
                cv2.line(tmp, p1, p2, (0,255,0), 2)
        cv2.putText(tmp, title, (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        return tmp

    left  = draw(img, boxes,           "vor Filter")
    right = draw(img, boxes_filtered,  "nach Filter")
    canvas = np.hstack((left, right))
    cv2.imwrite("debug_compare.jpg", canvas)
    print("→ debug_compare.jpg geschrieben (links=vor, rechts=nach Filter)")

if __name__=="__main__":
    main()