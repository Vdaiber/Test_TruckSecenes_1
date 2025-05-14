from oft.data.dataset import TruckScenesDataset
from oft.utils.config     import load_config
import numpy as np

cfg = load_config()
# set some nonzero noise
cfg["dataset"]["augment_noise_std"] = 5.0

# fresh dataset instances
ds_clean = TruckScenesDataset(
    dataroot=cfg["dataset"]["dataroot"],
    version=cfg["dataset"]["version"],
    history_window=0,
    max_boxes=50,
    augment_noise_std=0.0
)
ds_noisy = TruckScenesDataset(
    dataroot=cfg["dataset"]["dataroot"],
    version=cfg["dataset"]["version"],
    history_window=0,
    max_boxes=50,
    augment_noise_std=cfg["dataset"]["augment_noise_std"]
)

# grab sample 0
item_clean = ds_clean[0]
item_noisy = ds_noisy[0]

# print first 5 centers
print("Clean centers (first 5):")
print(item_clean["current"][:5, :3])
print("Noisy centers (first 5):")
print(item_noisy["current"][:5, :3])