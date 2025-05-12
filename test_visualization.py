#!/usr/bin/env python3
from truckscenes import TruckScenes
from oft.utils.config        import load_config
from oft.utils.visualization import render_sample_boxes
from truckscenes.utils.geometry_utils import BoxVisibility

def main():
    cfg = load_config()
    ts  = TruckScenes(
        version = cfg["dataset"]["version"],
        dataroot = cfg["dataset"]["dataroot"]
    )

    # wähle erste Szene & Sample
    scene0 = (cfg["visualization"]["scenes"] or [s["token"] for s in ts.scene])[0]
    first  = ts.get("scene", scene0)["first_sample_token"]

    rv = cfg["render"]
    img = render_sample_boxes(
        ts,
        sample_token   = first,
        sensor_channel = cfg["visualization"]["camera_channel"],
        visibility     = BoxVisibility[rv["box_visibility"]],
        z_threshold    = rv["z_threshold"],
        thickness      = rv["line_thickness"],
        out_path       = "test_output.jpg"
    )
    print(f"[Test] test_output.jpg gespeichert (Sample {first})")

if __name__=="__main__":
    main()