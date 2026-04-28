import numpy as np
import os

data_dir = 'input_4.27'
files = ['pred.npy', 'test_pred.npy', 'test_res.npy', 'train_pred.npy', 'train_res.npy', 
         'val_pred.npy', 'val_res.npy', 'true.npy', 'metrics.npy']

print("=" * 60)
print("input_4.27 数据文件分析")
print("=" * 60)

for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        data = np.load(path)
        print(f"{f}: shape={data.shape}, dtype={data.dtype}")
        if len(data.shape) > 0:
            print(f"  sample values: {data[0] if len(data.shape)==1 else data[0,:5] if len(data.shape)==2 else data[0,0,:5]}")
    else:
        print(f"{f}: not found")