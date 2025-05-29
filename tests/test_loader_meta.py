# test_loader_meta.py
from torch.utils.data import DataLoader
from oft.utils.config import load_config
from oft.data.old_dataset import TruckScenesDataset

cfg = load_config()
ds = TruckScenesDataset(
    dataroot=cfg["dataset"]["dataroot"],
    version=cfg["dataset"]["version"],
    history_window=cfg["visualization"]["history_window"],
    max_boxes=cfg.get("gt_max_boxes", 50),
    augment_noise_std=cfg["dataset"].get("augment_noise_std", 0.0)
)
loader = DataLoader(ds, batch_size=1,  shuffle=False)  # leave collate_fn=None

batch = next(iter(loader))

print("scene_token:", batch["scene"][0])
print("scene_meta:", batch["scene_meta"])
# if you want the weather string:
print("weather:", batch["scene_meta"]["weather"][0])