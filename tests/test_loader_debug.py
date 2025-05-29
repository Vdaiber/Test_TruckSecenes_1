#!/usr/bin/env python3
import numpy as np
from torch.utils.data import DataLoader
from oft.utils.config import load_config
from oft.data.old_dataset import TruckScenesDataset
from oft.data.collate  import collate_fn

def main():
    cfg = load_config()
    ds = TruckScenesDataset(
        dataroot = cfg["dataset"]["dataroot"],
        version  = cfg["dataset"]["version"],
        history_window  = cfg["visualization"]["history_window"],
        max_boxes       = cfg["dataset"].get("gt_max_boxes", None),
        augment_noise_std = cfg["dataset"].get("augment_noise_std", 0.0)
    )
    loader = DataLoader(
        ds,
        batch_size   = cfg["visualization"]["batch_size"],
        collate_fn   = collate_fn,
        shuffle      = False,
        num_workers  = 0
    )
    batch = next(iter(loader))
    print("=== Debug Loader ===")
    print("Current  boxes per sample:", batch["current"].shape)   # (B, M, 7)
    print("History  boxes per sample:", batch["history"].shape)   # (B, H, M, 7)
    print("Mask shape            :", batch["padding_mask"].shape)# (B, H+1, M)
    # Zeige die ersten Box-Zentren, damit du die Werte siehst
    print("Erste Box-Zentren sample[0]:")
    print(batch["current"][0,:5,:3].numpy())  # nur x,y,z der ersten 5 Boxen

if __name__ == "__main__":
    main()