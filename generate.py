#!/usr/bin/env python3
import argparse
from oft.utils.common_utils import list_scenes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per‑scene sample token lists.")
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the TruckScenes dataset version folder (e.g. /data/v1.0-mini)"
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory where scene lists will be written"
    )
    args = parser.parse_args()

    list_scenes(dataroot=args.dataset_path, output_dir=args.output_dir)
    print(f"[generate_scenes] dataroot={args.dataset_path}, output={args.output_dir}")