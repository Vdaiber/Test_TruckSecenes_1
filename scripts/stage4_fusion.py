from oft.examples.generate_fusion_detections import main as gen_fusion
from oft.utils.config import load_config

parser = argparse.ArgumentParser(...)
parser.add_argument(
    "--pipeline", "-c",
    default="config/pipeline.yaml",    # ← Default setzen
    help="Path to pipeline.yaml"
)

def main():
    cfg = load_config()
    gen_fusion()
    # Test-Check:
    import os, json
    fn = cfg["output"]["dets_json"]
    assert os.path.isfile(fn), f"⚠ {fn} fehlt"
    data = json.load(open(fn))
    assert isinstance(data, dict), "⚠ Fusion-JSON muss Dict sein"
    print(f"✓ Fusion JSON mit {len(data)} Samples erzeugt")

if __name__=="__main__":
    main()