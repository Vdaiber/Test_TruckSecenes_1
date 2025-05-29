# test_loader.py
from torch.utils.data import DataLoader
from oft.utils.config import load_config
from oft.data.old_dataset import TruckScenesDataset
from oft.data.collate import collate_fn

cfg = load_config()
ds = TruckScenesDataset(
    dataroot=cfg["dataset"]["dataroot"],
    version=cfg["dataset"]["version"],
    history_window=cfg["visualization"]["history_window"],
    max_boxes=cfg.get("gt_max_boxes", 50),
    augment_noise_std=cfg["dataset"].get("augment_noise_std", 0.0)
)
loader = DataLoader(
    ds,
    batch_size=cfg["visualization"]["batch_size"],
    collate_fn=collate_fn,
    shuffle=False,
    num_workers=0
)

batch = next(iter(loader))
print("current.shape     ", batch["current"].shape)
print("history.shape     ", batch["history"].shape)
print("padding_mask.shape", batch["padding_mask"].shape)
print("scene tokens      ", batch["scene"])
print("scene_meta        ", batch["scene_meta"])