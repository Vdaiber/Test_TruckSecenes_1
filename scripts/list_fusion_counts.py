#!/usr/bin/env python3
"""
Listet Fusion-Detections pro Sample-Index für CAMERA_LEFT_FRONT.
Aufruf im Projekt-Root: python scripts/list_fusion_counts.py
"""
import json
from oft.utils.config import load_config
from oft.data.old_dataset import TruckScenesDataset

def main():
    # 1) Config & Dataset
    cfg = load_config()
    d = cfg["dataset"]
    v = cfg["visualization"]
    ds = TruckScenesDataset(
        dataroot           = d["dataroot"],
        version            = d["version"].strip(),
        history_window     = v.get("history_window", 0),
        max_boxes          = d.get("gt_max_boxes") or 0,
        augment_noise_std  = d.get("augment_noise_std", 0.0)
    )

    # 2) Load Fusion JSON
    fn = cfg["output"]["dets_json"]
    with open(fn, "r", encoding="utf-8") as f:
        entries = json.load(f)
    # Fallback falls Dict
    if isinstance(entries, dict):
        entries = [{"sample_token": k, "dets": {v["camera_channel"]: entries[k]}}
                   for k in entries]

    cam_ch = v["camera_channel"]

    # 3) Baue Mapping sample_token → count
    count_map = {
        e["sample_token"]: len(e["dets"].get(cam_ch, []))
        for e in entries
    }

    # 4) Drucke Index, Token, Count
    print(f"{'Idx':>3}  {'#Detections':>10}  Sample-Token")
    print("-"*50)
    for idx, tok in enumerate(ds.samples):
        cnt = count_map.get(tok, 0)
        print(f"{idx:3d}  {cnt:10d}  {tok}")

if __name__ == "__main__":
    main()