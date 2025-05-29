#!/usr/bin/env python3
import os
import sys
import shutil
import yaml
import json
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# Projekt-Root ins PYTHONPATH
HERE     = os.path.abspath(__file__)
SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from oft.utils.config          import load_config
from oft.data.old_dataset          import TruckScenesDataset
from oft.data.collate          import collate_fn
from torch.utils.data          import DataLoader
from oft.old_examples.visualize_fusion_comparison import main as visualize_fusion_comparison

def test_dataset(cfg):
    dcfg = cfg["dataset"]
    ds = TruckScenesDataset(
        dataroot=dcfg["dataroot"],
        version=dcfg["version"].strip(),
        history_window=cfg["visualization"]["history_window"],
        max_boxes=dcfg.get("gt_max_boxes") or 0,
        augment_noise_std=0.0
    )
    assert len(ds) > 0, "Dataset ist leer"
    sample = ds[0]
    # Prüfe history-Länge
    H = cfg["visualization"]["history_window"]
    assert isinstance(sample["history"], list) and len(sample["history"]) == H, \
        f"History-Liste muss Länge {H} haben"
    print("✔ Dataset-Load + History: OK")

def test_dataloader(cfg):
    dcfg = cfg["dataset"]
    ds = TruckScenesDataset(
        dataroot=dcfg["dataroot"],
        version=dcfg["version"].strip(),
        history_window=cfg["visualization"]["history_window"],
        max_boxes=dcfg.get("gt_max_boxes") or 0,
        augment_noise_std=0.0
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["visualization"]["batch_size"],
        collate_fn=collate_fn
    )
    batch = next(iter(loader))
    # current Tensor sollte (B, M, 7) sein
    assert batch["current"].ndim == 3 and batch["current"].shape[-1] == 7
    print("✔ DataLoader + Collate: OK, current shape =", batch["current"].shape)

def test_noise(cfg):
    dcfg = cfg["dataset"]
    std = dcfg.get("augment_noise_std", 0)
    if std > 0:
        ds_clean = TruckScenesDataset(**dcfg, history_window=0, max_boxes=0, augment_noise_std=0.0)
        ds_noisy = TruckScenesDataset(**dcfg, history_window=0, max_boxes=0, augment_noise_std=std)
        cur_clean = ds_clean[0]["current"]
        cur_noisy = ds_noisy[0]["current"]
        if cur_clean.size and cur_noisy.size:
            assert not np.allclose(cur_clean, cur_noisy), "Noisy vs. clean sollte sich unterscheiden"
            print("✔ Noise-Applikation: OK")
        else:
            print("ℹ Keine Boxen im ersten Sample – überspringe Noise-Test")
    else:
        print("ℹ augment_noise_std=0 → überspringe Noise-Test")

def test_fusion_json(cfg):
    fn = cfg["output"]["dets_json"]
    assert os.path.isfile(fn), f"Fusion-JSON nicht gefunden: {fn}"
    with open(fn, "r") as f:
        data = json.load(f)
    assert isinstance(data, list), "Fusion-JSON muss eine Liste sein"
    print("✔ Fusion-JSON existiert und ist Liste mit", len(data), "Einträgen")

def test_hungarian_matching(cfg):
    # Wir nehmen das erste Sample und checken, dass Matching keine Fehler wirft
    dcfg  = cfg["dataset"]
    vcfg  = cfg["visualization"]
    ocfg  = cfg["output"]
    ds    = TruckScenesDataset(**dcfg, history_window=0, max_boxes=0, augment_noise_std=0.0)
    idx   = vcfg.get("sample_idx", 0)
    sample_token = ds.samples[idx]
    # GT-Mittelpunkte
    samp = ds.ts.get("sample", sample_token)
    gt_boxes = [ds.ts.get_box(a) for a in samp["anns"]]
    gt_centers = np.array([b.center for b in gt_boxes])
    # Fusion-Mittelpunkte
    fjs = json.load(open(ocfg["dets_json"], "r"))
    frow = next((e for e in fjs if e["sample_token"] == sample_token), None)
    assert frow is not None, "Kein Eintrag in Fusion-JSON für dieses Sample"
    dets = frow["dets"][vcfg["camera_channel"]]
    fus_centers = np.array([d["translation"] for d in dets])
    if gt_centers.size and fus_centers.size:
        cost = np.linalg.norm(gt_centers[:,None,:] - fus_centers[None,:,:], axis=-1)
        row,col = linear_sum_assignment(cost)
        print(f"✔ Hungarian Matching: {len(row)} Matches berechnet")
    else:
        print("ℹ Keine GT oder Fusion-Boxen zum Matching")

def test_visualization(cfg):
    # alten Ordner löschen, dann Visualisierung aufrufen und Ergebnis prüfen
    outdir = cfg["output"]["noise_comparison_dir"].replace("noise_comparison","fusion_comparison")
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    visualize_fusion_comparison()
    # sollte mindestens eine Datei im outdir erzeugt haben
    files = os.listdir(outdir)
    assert files, "Keine Visualisierung im Ordner " + outdir
    print("✔ Visualisierung erzeugt:", files)

if __name__ == "__main__":
    cfg = load_config()
    test_dataset(cfg)
    test_dataloader(cfg)
    test_noise(cfg)
    test_fusion_json(cfg)
    test_hungarian_matching(cfg)
    test_visualization(cfg)
    print("\n=== Alle Tests erfolgreich durchlaufen ===")