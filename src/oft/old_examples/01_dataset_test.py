#!/usr/bin/env python3
from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset

cfg = load_config()
dcfg = cfg["dataset"]
H    = cfg["visualization"]["history_window"]

ds = TruckScenesDataset(
    dataroot           = dcfg["dataroot"],
    version            = dcfg["version"].strip(),
    history_window     = H,
    max_boxes          = dcfg.get("gt_max_boxes") or 0,
    augment_noise_std  = 0.0
)
assert len(ds) > 0, "⚠ Dataset ist leer!"
sample = ds[0]
assert isinstance(sample["history"], list) and len(sample["history"]) == H, \
       f"⚠ History-Länge muss {H} sein, ist aber {len(sample['history'])}"
print(" 01_dataset_test: Dataset + History OK")