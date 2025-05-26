from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset

cfg = load_config()
ds = TruckScenesDataset(
    dataroot=cfg["dataset"]["dataroot"],
    version=cfg["dataset"]["version"],
    history_window=cfg["visualization"]["history_window"],
    max_boxes=cfg.get("gt_max_boxes", 50),
    augment_noise_std=cfg["dataset"].get("augment_noise_std", 0.0)
)

item = ds[0]
print("scene_token:", item["scene"])
print("scene_meta: ", item["scene_meta"])