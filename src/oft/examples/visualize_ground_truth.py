#!/usr/bin/env python3
"""
examples/visualize_ground_truth.py

Ground-Truth-Render-Pipeline:
  - lädt aus der Config output/ground_truth_dir
  - selektiert Szenen & Samples
  - filtert nach Sichtbarkeit & z-Threshold
  - zeichnet Wireframe-Boxen mit render_sample_boxes()
  - speichert JPGs
"""

import os
from truckscenes import TruckScenes
from truckscenes.utils.geometry_utils import BoxVisibility
from oft.utils.visualization import render_sample_boxes
from oft.utils.config import load_config

def run_ground_truth(ts, cfg, scenes, num_frames, history_window):
    out_dir = cfg["output"]["ground_truth_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Parameter aus pipeline.yaml
    sensor_chan = cfg["visualization"]["camera_channel"]
    thickness   = cfg["render"]["line_thickness"]
    z_thresh    = cfg["render"]["z_threshold"]
    box_vis     = cfg["render"]["box_visibility"]
    visibility  = BoxVisibility[box_vis]                   # ANY | ALL | NONE
    gt_max      = cfg["dataset"].get("gt_max_boxes", None)  # None = alle

    for scene_token in scenes:
        sample_token = ts.get("scene", scene_token)["first_sample_token"]
        count = 0

        while sample_token and (num_frames < 1 or count < num_frames):
            sample = ts.get("sample", sample_token)

            # ggf. Beschränkung der GT-Annots
            anns = sample["anns"]
            if gt_max is not None:
                anns = anns[:gt_max]

            # 1) render_sample_boxes lädt Bild, filtert & zeichnet
            out_path = os.path.join(out_dir, f"{scene_token}_{sample_token}.jpg")
            img = render_sample_boxes(
                ts,
                sample_token=sample_token,
                sensor_channel=sensor_chan,
                visibility=visibility,
                z_threshold=z_thresh,
                thickness=thickness,
                out_path=out_path
            )

            print(f"[GT] {scene_token}/{sample_token}: {len(anns)} Boxen → {out_path}")

            # nächstes Sample
            sample_token = sample.get("next")
            count += 1

if __name__ == "__main__":
    raise RuntimeError("Bitte nur via examples/visualize.py aufrufen.")