#!/usr/bin/env python3
"""
example/visualize.py

Master-Script zum Rendern von Ground-Truth und Fusion-Ergebnissen.
Liest config/pipeline.yaml, wählt Szenen aus und ruft intern
die DevKit-basierten Visualizer auf.
"""

import os
import sys

from truckscenes import TruckScenes
from oft.utils.config import load_config
from oft.utils.common_utils import list_scenes

# unsere Sub-Pipelines
from oft.old_examples.visualize_ground_truth import run_ground_truth
from oft.old_examples.visualize_fusion       import run_fusion


def main():
    # 1) Config laden
    cfg = load_config()

    # 2) TruckScenes-Objekt
    ts = TruckScenes(
        version=cfg["dataset"]["version"],
        dataroot=cfg["dataset"]["dataroot"]
    )

    # 3) Szenen-Selektion:
    raw = cfg["visualization"]["scenes"]
    if not raw:
        # leer = alle Tokens abfragen
        scenes = list_scenes(
            cfg["dataset"]["dataroot"],
            cfg["dataset"]["version"]
        )
    else:
        scenes = []
        for entry in raw:
            if isinstance(entry, int):
                scenes.append(ts.scene[entry]["token"])
            else:
                scenes.append(entry)

    num_frames     = cfg["visualization"]["num_frames"]
    history_window = cfg["visualization"]["history_window"]

    # 4) Ground-Truth rendern
    run_ground_truth(ts, cfg, scenes, num_frames, history_window)

    # 5) Fusion rendern (falls aktiviert)
    if cfg["fusion"]["enabled"]:
        run_fusion(ts, cfg, scenes, num_frames, history_window)


if __name__ == "__main__":
    main()