from oft.examples.visualize_fusion_comparison import main as viz_fus
from oft.utils.config import load_config

def main():
    cfg = load_config()
    viz_fus()
    # Test-Check:
    import os
    out = cfg["output"]["noise_comparison_dir"].replace("noise_comparison","fusion_comparison")
    assert os.listdir(out), f"⚠ Keine Fusion-Vergleichs-Bilder in {out}"
    print("✓ Fusion-Vergleichsbilder erstellt")

if __name__=="__main__":
    main()