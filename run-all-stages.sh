#!/usr/bin/env bash
set -e

python scripts/stage1_visualize_gt.py     -c config/pipeline.yaml
python scripts/stage2_visualize_noise.py  -c config/pipeline.yaml
python scripts/stage3_track_history.py    -c config/pipeline.yaml
python scripts/stage4_fusion.py           -c config/pipeline.yaml
python scripts/stage5_visualize_fusion.py -c config/pipeline.yaml