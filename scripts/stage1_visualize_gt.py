from oft.examples.visualize_ground_truth import run_ground_truth
from truckscenes import TruckScenes
from oft.utils.config import load_config

def main():
    cfg = load_config()
    ts  = TruckScenes(version=cfg["dataset"]["version"], dataroot=cfg["dataset"]["dataroot"])
    scenes = cfg["visualization"]["scenes"] or [s["token"] for s in ts.scene]
    run_ground_truth(ts, cfg, scenes, cfg["visualization"]["num_frames"], cfg["visualization"]["history_window"])
    # Test-Check:
    import os
    out = cfg["output"]["ground_truth_dir"]
    assert os.listdir(out), f"⚠ Keine GT-Bilder in {out}"

if __name__=="__main__":
    main()