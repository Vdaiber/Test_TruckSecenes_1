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
loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

# Zwei aufeinanderfolgende Samples ziehen
batch1 = next(iter(loader))['current'][0].numpy()
batch2 = next(iter(loader))['current'][0].numpy()

print("Box[0] sample1:", batch1[0])
print("Box[0] sample2:", batch2[0])
# Vergleiche, ob sich die ersten Box-Koordinaten unterscheiden
print("Unterschied (L2) center:", 
      ((batch1[0,:3] - batch2[0,:3])**2).sum()**0.5)
