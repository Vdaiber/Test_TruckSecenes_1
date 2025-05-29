#!/usr/bin/env python3
import numpy as np
from oft.utils.config import load_config
from oft.data.old_dataset     import TruckScenesDataset

cfg  = load_config()
dcfg = cfg["dataset"]
std  = dcfg.get("augment_noise_std", 0.0)

if std > 0:
    # sauber vs. noisy
    ds_clean = TruckScenesDataset(
        dataroot           = dcfg["dataroot"],
        version            = dcfg["version"].strip(),
        history_window     = 0,
        max_boxes          = 0,
        augment_noise_std  = 0.0
    )
    ds_noisy = TruckScenesDataset(
        dataroot           = dcfg["dataroot"],
        version            = dcfg["version"].strip(),
        history_window     = 0,
        max_boxes          = 0,
        augment_noise_std  = std
    )
    c = ds_clean[0]["current"]
    n = ds_noisy[0]["current"]
    if c.size and n.size:
        assert not np.allclose(c, n), "⚠ noisy == clean – augentation hat nicht gegriffen!"
        print("✅ 03_noise_test: Noise-Augmentation OK")
    else:
        print("ℹ️ 03_noise_test: Keine Boxen im ersten Sample → übersprungen")
else:
    print("ℹ️ 03_noise_test: augment_noise_std=0 → übersprungen")