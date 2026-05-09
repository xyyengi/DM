#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断数据加载问题"""

import sys
import traceback

print("Step 1: Importing numpy...")
import numpy as np
print("  OK")

print("Step 2: Importing torch...")
import torch
print("  OK")

print("Step 3: Importing dataset_multivariate...")
try:
    from dataset_multivariate import get_dataloader_multivariate, MultiChannelWindScenarioDataset
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Step 4: Checking data files...")
import os
data_path = './input_4.27/'
files = ['train_pred.npy', 'train_res.npy', 'val_pred.npy', 'val_res.npy']
for f in files:
    path = os.path.join(data_path, f)
    if os.path.exists(path):
        data = np.load(path)
        print(f"  {f}: shape={data.shape}")
    else:
        print(f"  {f}: NOT FOUND!")

print("Step 5: Creating dataset (train)...")
try:
    print("  5.1: Initializing dataset...")
    dataset = MultiChannelWindScenarioDataset(
        data_path=data_path, 
        mode='train', 
        n_intervals=10
    )
    print(f"  OK, samples={len(dataset)}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Step 6: Creating dataloader...")
try:
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    print(f"  OK, batches={len(loader)}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Step 7: Testing batch iteration...")
try:
    batch = next(iter(loader))
    print(f"  OK, keys={list(batch.keys())}")
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: {v.shape}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nAll tests passed!")