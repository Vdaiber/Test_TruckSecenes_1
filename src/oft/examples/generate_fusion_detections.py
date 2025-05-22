#!/usr/bin/env python3
"""
Apply BEV-NMS on noisy GT boxes and write fusion_detections.json,
inkl. Index jedes behalteten Box-Eintrags für spätere Zuordnung.
"""
import os
import json
import yaml
import argparse
import numpy as np

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(
        description="Apply BEV-NMS on noisy GT boxes and write fusion_detections.json"
    )
    parser.add_argument(
        "--pipeline", "-c",
        required=True,
        help="Path to pipeline.yaml"
    )
    args = parser.parse_args()

    # 1) Config laden
    cfg     = load_config(args.pipeline)
    dcfg    = cfg["dataset"]
    vcfg    = cfg["visualization"]
    ocfg    = cfg["output"]
    out_dir = ocfg["fusion_dir"]
    fn_json = ocfg["dets_json"]

    os.makedirs(out_dir, exist_ok=True)

    # 2) Dataset‐Instanzen
    from oft.data.dataset import TruckScenesDataset
    common = dict(
        dataroot       = dcfg["dataroot"],
        version        = dcfg["version"].strip(),
        history_window = vcfg.get("history_window", 0),
        max_boxes      = dcfg.get("gt_max_boxes") or 0
    )
    ds_clean = TruckScenesDataset(**common, augment_noise_std=0.0)
    ds_noisy = TruckScenesDataset(**common,
                                   augment_noise_std=dcfg.get("augment_noise_std", 0.0))

    # 3) NMS-Funktion
    from oft.fusion.nms_3d import nms_bev_3d

    # 4) Szenen auswählen
    scenes = vcfg.get("scenes", [])
    if not scenes:
        scenes = sorted({
            ds_clean.ts.get("sample", tok)["scene_token"]
            for tok in ds_clean.samples
        })

    dets: dict = {}

    # 5) Pro Szene und Frame NMS anwenden
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

            # Dummy-Scores
            scores = np.ones(arr.shape[0], dtype=np.float32)

            # NMS
            keep = nms_bev_3d(
                boxes=arr,
                scores=scores,
                iou_threshold=cfg["fusion"].get("iou_threshold", 0.5)
            )

            # Ausgabe als JSON-serialisierbare Dicts
            out_boxes = []
            for i in keep:
                # i kann numpy.int64 sein → cast zu Python-int
                py_i = int(i)
                # box7 wird per .tolist() native Python floats
                box7  = arr[py_i].tolist()
                score = float(scores[py_i])
                out_boxes.append({
                    "index": py_i,
                    "box":   box7,
                    "score": score
                })
            dets[tok] = out_boxes

    # 6) In JSON schreiben
    with open(fn_json, "w", encoding="utf-8") as f:
        json.dump(dets, f, indent=2)

    print(f"✓ Fusion-Detections written to {fn_json}")

if __name__ == "__main__":
    main()