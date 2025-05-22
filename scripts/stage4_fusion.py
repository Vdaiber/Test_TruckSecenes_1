#!/usr/bin/env python3
import json
import numpy as np
from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset
from oft.fusion.nms_3d import nms_bev_3d

def main():
    cfg  = load_config()
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    ocfg = cfg["output"]
    fusion_cfg = cfg["fusion"]

    # Dataset clean vs noisy
    common = dict(
        dataroot       = dcfg["dataroot"],
        version        = dcfg["version"].strip(),
        history_window = vcfg.get("history_window", 0),
        max_boxes      = dcfg.get("gt_max_boxes") or 0
    )
    ds_clean = TruckScenesDataset(**common, augment_noise_std=0.0)
    ds_noisy = TruckScenesDataset(**common,
                                  augment_noise_std=dcfg.get("augment_noise_std",0.0))

    # Szenen-Auswahl
    scenes = vcfg.get("scenes", [])
    if not scenes:
        scenes = sorted({ ds_clean.ts.get("sample", tok)["scene_token"]
                          for tok in ds_clean.samples })

    entries = []
    num = vcfg.get("num_frames", 0)
    for scene in scenes:
        samples = [tok for tok in ds_clean.samples
                   if ds_clean.ts.get("sample", tok)["scene_token"] == scene]
        if num > 0:
            samples = samples[:num]
        for tok in samples:
            idx   = ds_clean.samples.index(tok)
            noisy = ds_noisy[idx]["current"]
            dets  = []
            if noisy is not None and noisy.shape[0] > 0:
                scores = np.ones(noisy.shape[0], dtype=float)
                keep   = nms_bev_3d(noisy, scores,
                                   iou_threshold=fusion_cfg.get("iou_threshold",0.5))
                for i in keep:
                    b = noisy[i]
                    dets.append({
                        "translation": [float(b[0]), float(b[1]), float(b[2])],
                        "wlh":         [float(b[3]), float(b[4]), float(b[5])],
                        "yaw":         float(b[6]),
                        "score":       float(scores[i])
                    })
            entries.append({
                "sample_token": tok,
                "dets": { vcfg["camera_channel"]: dets }
            })

    with open(ocfg["dets_json"], "w") as f:
        json.dump(entries, f, indent=2)
    print(f"✓ Stage 4: wrote {len(entries)} entries → {ocfg['dets_json']}")

if __name__=="__main__":
    main()