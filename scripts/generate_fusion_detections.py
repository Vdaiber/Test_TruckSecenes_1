#!/usr/bin/env python3
import argparse
import yaml
import json
import os
import numpy as np

from oft.data.old_dataset import TruckScenesDataset
from oft.fusion.nms_3d import nms_bev_3d

def cast_py(val):
    """Rekursiv alle numpy-Typen in native Python-Typen umwandeln."""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, (list, tuple)):
        return [cast_py(v) for v in val]
    if isinstance(val, dict):
        return {k: cast_py(v) for k, v in val.items()}
    return val

def main():
    p = argparse.ArgumentParser(
        description="Stage 4: Apply BEV-NMS on noisy GT boxes"
    )
    p.add_argument(
        "-c","--pipeline",
        required=True,
        help="Path to config/pipeline.yaml"
    )
    args = p.parse_args()

    # 1) Config laden
    cfg = yaml.safe_load(open(args.pipeline, "r"))

    # 2) Wenn Fusion deaktiviert → abbrechen
    if not cfg["fusion"].get("enabled", False):
        print("Fusion ist in der Config deaktiviert → nichts zu tun.")
        return

    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    ocfg = cfg["output"]

    # 3) Dataset-Instanzen (clean vs. noisy)
    ds_clean = TruckScenesDataset(
        dataroot=dcfg["dataroot"],
        version=dcfg["version"].strip(),
        history_window=vcfg.get("history_window", 0),
        max_boxes=dcfg.get("gt_max_boxes") or 0,
        augment_noise_std=0.0
    )
    ds_noisy = TruckScenesDataset(
        dataroot=dcfg["dataroot"],
        version=dcfg["version"].strip(),
        history_window=vcfg.get("history_window", 0),
        max_boxes=dcfg.get("gt_max_boxes") or 0,
        augment_noise_std=dcfg.get("augment_noise_std", 0.0)
    )

    # 4) Szenen auswählen
    scenes = vcfg.get("scenes", [])
    if not scenes:
        scenes = sorted({
            ds_clean.ts.get("sample", tok)["scene_token"]
            for tok in ds_clean.samples
        })

    # 5) Pro Szene + Frame NMS anwenden
    dets = {}
    for scene in scenes:
        toks = [
            tok for tok in ds_clean.samples
            if ds_clean.ts.get("sample", tok)["scene_token"] == scene
        ]
        num = vcfg.get("num_frames", 0)
        if num > 0:
            toks = toks[:num]

        for tok in toks:
            idx = ds_clean.samples.index(tok)
            arr = ds_noisy[idx]["current"]  # np.ndarray (N,7)
            if arr.shape[0] == 0:
                dets[tok] = []
                continue

            # Dummy-Scores (alle 1.0)
            scores = np.ones(arr.shape[0], dtype=np.float32)

            # NMS
            keep = nms_bev_3d(
                boxes=arr,
                scores=scores,
                iou_threshold=cfg["fusion"].get("iou_threshold", 0.5)
            )

            out_boxes = []
            for i in keep:
                i_py = int(i)  # numpy → Python int
                out_boxes.append({
                    "index": i_py,
                    "box":       arr[i_py].tolist(),        # Python floats
                    "score":     float(scores[i_py])        # Python float
                })
            dets[tok] = out_boxes
            print(f"  Scene {scene}, Sample {tok}: {len(out_boxes)} kept")

    # 6) JSON schreiben
    os.makedirs(os.path.dirname(ocfg["dets_json"]), exist_ok=True)
    with open(ocfg["dets_json"], "w", encoding="utf-8") as f:
        json.dump(cast_py(dets), f, indent=2)
    print("✓ Fusion-Detections geschrieben nach", ocfg["dets_json"])


if __name__ == "__main__":
    main()