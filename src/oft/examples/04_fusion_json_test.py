#!/usr/bin/env python3
import os, json
from oft.utils.config import load_config

cfg = load_config()
fn  = cfg["output"]["dets_json"]
assert os.path.isfile(fn), f"⚠ Fusion-JSON nicht gefunden: {fn}"
data = json.load(open(fn, "r"))
assert isinstance(data, list), "⚠ Fusion-JSON muss Liste sein"
print(f" 04_fusion_json_test: OK, {len(data)} Einträge")