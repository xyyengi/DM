#!/usr/bin/env python3
"""
清除条件矩阵缓存，因为条件区间定义已修复
"""
import os
import glob

def clear_cond_cache(data_path='./input_4.27/'):
    """清除条件矩阵缓存文件"""
    cache_files = [
        os.path.join(data_path, 'cond_matrix_train.npy'),
        os.path.join(data_path, 'cond_matrix_val.npy'),
        os.path.join(data_path, 'cond_matrix_test.npy'),
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            print(f"删除缓存文件: {cache_file}")
            os.remove(cache_file)
        else:
            print(f"缓存文件不存在: {cache_file}")
    
    print("\n条件矩阵缓存已清除，下次运行时会重新生成（使用修复后的功率值区间）")

if __name__ == '__main__':
    clear_cond_cache()
