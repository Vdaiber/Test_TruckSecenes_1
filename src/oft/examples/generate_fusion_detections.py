#!/usr/bin/env python3
import os
import json
import yaml
import argparse
import numpy as np

# pipeline.yaml laden
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(
        description="Apply BEV-NMS on noisy GT boxes and write fusion_detections.json")
    parser.add_argument("--pipeline", "-c", required=True,
                        help="Path to pipeline.yaml")
    args = parser.parse_args()

    cfg      = load_config(args.pipeline)
    dcfg     = cfg["dataset"]
    vcfg     = cfg["visualization"]
    out_dir  = cfg["output"]["fusion_results"]
    fn_json  = cfg["output"]["dets_json"]
    os.makedirs(out_dir, exist_ok=True)

    # common kwargs für unsere Dataset-Instanzen
    common = dict(
        dataroot       = dcfg["dataroot"],
        version        = dcfg["version"].strip(),
        history_window = vcfg.get("history_window", 0),
        max_boxes      = dcfg.get("gt_max_boxes") or 0
    )

    # GT-clean (wird hier nur zum Auslesen der sample-Tokens benutzt)
    from oft.data.dataset import TruckScenesDataset
    ds_clean = TruckScenesDataset(**common, augment_noise_std=0.0)
    # noisy → die „Detections“ (hier auf GT-Basis mit Noise)
    ds_noisy = TruckScenesDataset(**common,
                                   augment_noise_std=dcfg.get("augment_noise_std", 0.0))

    # NMS-Funktion importieren
    from oft.fusion.nms_3d import nms_bev_3d

    # Welche Szenen?
    # wenn vcfg["scenes"] leer ← alle Szenen
    scenes = vcfg.get("scenes", [])
    if not scenes:
        # alle scene_tokens aus ds_clean aufscannen
        scenes = sorted({ ds_clean.ts.get("sample", t)["scene_token"]
                          for t in ds_clean.samples })

    dets = {}  # wird sample_token → list of detections

    for scene in scenes:
        # alle sample_tokens dieser Szene in Reihenfolge
        toks = [tok for tok in ds_clean.samples
                if ds_clean.ts.get("sample", tok)["scene_token"] == scene]

        # begrenzen auf num_frames (0 = alle,  >0 = head-only)
        num = vcfg.get("num_frames", 0)
        if num > 0:
            toks = toks[:num]

        for tok in toks:
            idx = ds_clean.samples.index(tok)
            # noisy-Boxes (N×7) aus Dataset
            arr = ds_noisy[idx]["current"]   # np.ndarray (N,7)
            if arr.shape[0] == 0:
                dets[tok] = []
                continue

            # dummy-Scores: alle gleich 1.0 (später durch echte Scores ersetzen)
            scores = np.ones(arr.shape[0], dtype=np.float32)

            # NMS anwenden
            keep = nms_bev_3d(arr, scores, iou_threshold=cfg["fusion"].get("iou_threshold", 0.5))

            # als Liste von Dicts speichern
            out_boxes = []
            for i in keep:
                box7 = arr[i].tolist()
                out_boxes.append({
                    "box":      box7,
                    "score":    float(scores[i])
                })
            dets[tok] = out_boxes

    # in JSON schreiben
    with open(fn_json, "w") as f:
        json.dump(dets, f, indent=2)
    print(f"✓ fusion detections written to {fn_json}")

if __name__ == "__main__":
    main()