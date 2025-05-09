#!/usr/bin/env python3
import os
import cv2
import numpy as np
from truckscenes import TruckScenes
from oft.utils.visualization import render_boxes, world_to_camera
from truckscenes.utils.geometry_utils import view_points  # nur view_points hier

def run_ground_truth(ts, cfg, scenes, num_frames, history_window):
    out_dir   = cfg["output"]["ground_truth_dir"]
    os.makedirs(out_dir, exist_ok=True)

    color     = tuple(cfg["render"]["colors"]["gt"])
    thickness = cfg["render"]["line_thickness"]
    gt_max    = cfg.get("gt_max_boxes", None)
    sensor_chan = cfg["visualization"]["camera_channel"]

    for scene_token in scenes:
        scene = ts.get("scene", scene_token)
        weather = scene.get("description","").split()[0] if scene.get("description") else ""
        sample_token = scene["first_sample_token"]
        count = 0

        while sample_token and (num_frames<1 or count < num_frames):
            sample = ts.get("sample", sample_token)
            anns = sample["anns"][:gt_max] if gt_max else sample["anns"]

            # 1) Wireframe + Labels
            boxes = [ts.get_box(ann_tk) for ann_tk in anns]
            img = render_boxes(
                ts, sample_token, boxes,
                color=color, thickness=thickness,
                sensor_channel=sensor_chan
            )

            # 2) Kategorie-Text
            for ann_tk, box in zip(anns, boxes):
                ann = ts.get("sample_annotation", ann_tk)
                cat_token = ann.get("category_token") or ann.get("category") or None
                cat = ts.get("category", cat_token)["name"] if cat_token else getattr(box, "name", "unknown")

                # corners in world
                corners_w = box.corners()
                # extrinsics
                sd = ts.get("sample_data", sample["data"][sensor_chan])
                calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
                ego   = ts.get("ego_pose", sd["ego_pose_token"])

                # world→camera
                corners_c = world_to_camera(corners_w, ego, calib)
                K = np.array(calib.get("camera_intrinsic", np.eye(3)), dtype=np.float32).reshape(3,3)
                pt2d = view_points(corners_c, K, normalize=True)[:2,0].astype(int)

                cv2.putText(img, cat, (int(pt2d[0]), int(pt2d[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # 3) Wetter-Tag
            if weather:
                h,w = img.shape[:2]
                (tw,th),_ = cv2.getTextSize(weather, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(img, weather, ((w-tw)//2, h-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Save & Log
            out_p = os.path.join(out_dir, f"{scene_token}_{sample_token}.jpg")
            cv2.imwrite(out_p, img)
            print(f"[GT] {scene_token}/{sample_token}: {len(anns)} Boxen → {out_p}")

            sample_token = sample.get("next")
            count += 1

if __name__=="__main__":
    raise RuntimeError("Nur via examples/visualize.py aufrufen.")