import numpy as np
from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset
from oft.tracking.hungarian import hungarian_match

def main():
    cfg  = load_config()
    dcfg = cfg["dataset"]
    H    = cfg["visualization"]["history_window"]
    ds   = TruckScenesDataset(
        dataroot=dcfg["dataroot"],
        version=dcfg["version"].strip(),
        history_window=H,
        max_boxes=dcfg.get("gt_max_boxes") or 0,
        augment_noise_std=dcfg.get("augment_noise_std",0.0)
    )
    # nimm erste Probe mit History
    sample = ds[0]
    # extrahiere prev-/curr-Mittelpunkte
    prev = np.vstack([h[:,:2] for h in sample["history"] if h is not None] + [sample["current"][:,:2]])
    curr = sample["current"][:,:2]
    matches = hungarian_match(prev, curr, prev_velocities=None, ego_transform=None, dt=1.0, max_distance=5.0)
    assert isinstance(matches, list), "⚠ Hungarian-Matching fehlgeschlagen"
    print("Matches:", matches)

if __name__=="__main__":
    main()