#!/usr/bin/env python3
import os, numpy as np
from truckscenes import TruckScenes
from oft.utils.visualization import render_boxes
from oft.fusion.nms_3d import nms_3d

def add_noise(box, trans_std, rot_std):
    import copy
    b = copy.deepcopy(box)
    b.center      = (np.array(b.center) + np.random.normal(scale=trans_std, size=3)).tolist()
    b.orientation = float(b.orientation + np.random.normal(scale=rot_std))
    return b

def run_fusion(ts, cfg, scenes, num_frames, history_window):
    out_dir       = cfg["output"]["fusion_dir"]
    os.makedirs(out_dir, exist_ok=True)
    sensor_types  = cfg["fusion"]["sensor_types"]
    iou_th        = cfg["fusion"]["iou_threshold"]
    noise_cfg     = cfg["fusion"].get("noise", {})
    color         = tuple(cfg["render"]["colors"]["fusion"])
    thickness     = cfg["render"]["line_thickness"]
    sensor_chan   = cfg["visualization"]["camera_channel"]

    for scene_token in scenes:
        scene        = ts.get("scene", scene_token)
        sample_token = scene["first_sample_token"]
        count = 0

        while sample_token and (num_frames<1 or count < num_frames):
            sample    = ts.get("sample", sample_token)
            all_boxes = []
            all_scores= []

            # Ground-Truth als „Sensor-Detections“ mit Rauschen
            for s in sensor_types:
                for ann_tk in sample["anns"]:
                    box_noisy = add_noise(
                        ts.get_box(ann_tk),
                        *noise_cfg.get(s, {}).get("translation_std", 0.0),
                        *noise_cfg.get(s, {}).get("rotation_std", 0.0)
                    )
                    all_boxes.append(box_noisy)
                    all_scores.append(1.0)

            import torch
            B = torch.tensor([b.to_lwhyaw() for b in all_boxes], dtype=torch.float32)
            S = torch.tensor(all_scores, dtype=torch.float32)
            keep = nms_3d(B, S, iou_threshold=iou_th).cpu().numpy().tolist()
            fused = [all_boxes[i] for i in keep]

            # Rendern (ohne box_style)
            img = render_boxes(
                ts,
                sample_token,
                fused,
                color=color,
                thickness=thickness,
                sensor_channel=sensor_chan
            )
            out_p = os.path.join(out_dir, f"fusion_{sample_token}.jpg")
            cv2.imwrite(out_p, img)
            print(f"[FU] {scene_token}/{sample_token}: {len(fused)} fused → {out_p}")

            sample_token = sample.get("next")
            count += 1