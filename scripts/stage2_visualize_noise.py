from oft.examples.visualize_noise_comparison import main as viz_noise
from oft.utils.config import load_config

def main():
    cfg = load_config()
    viz_noise()
    # Test-Check:
    import os
    out = cfg["output"]["noise_comparison_dir"]
    assert os.listdir(out), f"⚠ Keine Noise-Vergleichs-Bilder in {out}"

if __name__=="__main__":
    main()