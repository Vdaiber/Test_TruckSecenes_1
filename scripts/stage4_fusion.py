#!/usr/bin/env python3
import argparse
import json
import numpy as np

from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset
from oft.fusion.nms_3d import nms_bev_3d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c","--pipeline", required=True)
    args = p.parse_args()

    cfg    = load_config(args.pipeline)
    dcfg   = cfg["dataset"]
    vcfg   = cfg["visualization"]
    ocfg   = cfg["output"]
    fusion = cfg["fusion"]

    common = dict(
        dataroot       = str(dcfg["dataroot"]),
        version        = str(dcfg["version"]).strip(),
        history_window = vcfg.get("history_window", 0),
        max_boxes      = dcfg.get("gt_max_boxes") or 0
    )

    ds_clean = TruckScenesDataset(
        **common,
        augment_noise_std=0.0
    )
    ds_noisy = TruckScenesDataset(
        **common,
        augment_noise_std=float(dcfg.get("augment_noise_std",0.0))
    )

    # … der Rest bleibt unverändert …
    entries = []
    # Erzeuge deine Fusion-Detektionen
    with open(ocfg["dets_json"], "w") as f:
        json.dump(entries, f, indent=2)
    print(f"✓ Stage 4: wrote {len(entries)} entries → {ocfg['dets_json']}")

if __name__=="__main__":
    main()