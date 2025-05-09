import numpy as np
import torch
from torch.utils.data import DataLoader
from oft.data.dataset import TruckScenesDataset
from oft.data.collate import collate_fn

# Dataset mit Augmentation
dataset = TruckScenesDataset(
    dataroot='/data',
    version='v1.0-mini',
    history_window=5,
    max_boxes=50,
    augment_noise_std=0.1
)
# DataLoader mit batch_size=2
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Erstes Batch (2 Samples) ziehen
batch = next(iter(loader))
cur = batch['current'].numpy()        # shape [2, N, 7]
# Vergleiche die ersten Boxzentren der beiden Samples
center1 = cur[0, 0, :3]
center2 = cur[1, 0, :3]

print('Center Sample 1:', center1)
print('Center Sample 2:', center2)
print('L2-Distanz:', np.linalg.norm(center1 - center2))