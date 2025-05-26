#!/usr/bin/env python3
import shutil, os
from oft.utils.config import load_config
from oft.examples.visualize_fusion_comparison import main as viz_fusion
from oft.examples.visualize_noise_comparison   import main as viz_noise

cfg   = load_config()
out_n = cfg["output"]["noise_comparison_dir"]
out_f = cfg["output"]["noise_comparison_dir"].replace("noise_comparison","fusion_comparison")

# alten Ordner löschen
for d in (out_n, out_f):
    if os.path.isdir(d):
        shutil.rmtree(d)

# Visualisierungen starten
viz_noise()
viz_fusion()

# Prüfung
assert os.listdir(out_n), f"⚠ Kein Output in {out_n}"
assert os.listdir(out_f), f"⚠ Kein Output in {out_f}"
print("✅ 06_visualization_test: beide Visualisierungen erzeugt")