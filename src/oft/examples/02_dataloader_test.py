#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset
from oft.data.collate  import collate_fn

cfg  = load_config()
dcfg = cfg["dataset"]
B    = cfg["visualization"]["batch_size"]

ds = TruckScenesDataset(
    dataroot           = dcfg["dataroot"],
    version            = dcfg["version"].strip(),
    history_window     = cfg["visualization"]["history_window"],
    max_boxes          = dcfg.get("gt_max_boxes") or 0,
    augment_noise_std  = 0.0
)
loader = DataLoader(ds, batch_size=B, collate_fn=collate_fn)
batch = next(iter(loader))
assert batch["current"].ndim == 3 and batch["current"].shape[-1] == 7, \
       "⚠ current Tensor muss Form (B, M, 7) haben"
print(f"✅ 02_dataloader_test: OK, current.shape = {tuple(batch['current'].shape)}")