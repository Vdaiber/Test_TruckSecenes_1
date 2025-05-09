#!/usr/bin/env python3
"""
examples/visualize.py

Master-Script zum Rendern von Ground-Truth und Fusion-Ergebnissen.
Ruft intern die DevKit-basierten Visualizer auf.
"""

import os
import sys

from truckscenes import TruckScenes
from oft.utils.config import load_config
from oft.utils.common_utils import list_scenes

# Wir importieren unsere beiden Sub-Pipelines
from oft.examples.visualize_ground_truth import run_ground_truth
from oft.examples.visualize_fusion import run_fusion


def main():
    # 1) Konfiguration und Dataset initialisieren
    cfg = load_config()
    ts = TruckScenes(
        version=cfg["dataset"]["version"],
        dataroot=cfg["dataset"]["dataroot"]
    )

    # 2) Szenen festlegen (leer = alle)
    scenes = cfg["visualization"]["scenes"] or \
             list_scenes(cfg["dataset"]["dataroot"],
                         cfg["dataset"]["version"])

    num_frames     = cfg["visualization"]["num_frames"]
    history_window = cfg["visualization"]["history_window"]

    # 3) Ground-Truth rendern
    run_ground_truth(ts, cfg, scenes, num_frames, history_window)

    # 4) Fusion rendern (falls aktiviert)
    if cfg["fusion"]["enabled"]:
        run_fusion(ts, cfg, scenes, num_frames, history_window)


if __name__ == "__main__":
    main()
