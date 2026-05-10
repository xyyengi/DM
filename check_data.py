import numpy as np
import os

data_path = 'c:/Users/小语言/DM_local/input_4.27'
files = [f for f in os.listdir(data_path) if f.endswith('.npy')]

print("=" * 50)
print("数据集规模分析")
print("=" * 50)

for f in sorted(files):
    arr = np.load(os.path.join(data_path, f))
    print(f"{f}: {arr.shape}")

# 计算训练样本数
train_pred = np.load(os.path.join(data_path, 'train_pred.npy'))
train_res = np.load(os.path.join(data_path, 'train_res.npy'))
val_pred = np.load(os.path.join(data_path, 'val_pred.npy'))
val_res = np.load(os.path.join(data_path, 'val_res.npy'))

print("\n" + "=" * 50)
print("样本统计:")
print(f"  训练集: {train_pred.shape[0]} 样本")
print(f"  验证集: {val_pred.shape[0]} 样本")
print(f"  总计: {train_pred.shape[0] + val_pred.shape[0]} 样本")
print("=" * 50)