# CSDI
This is the github repository for the NeurIPS 2021 paper "[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)".

## 重要说明

当前代码用于复现论文 `2023-Conditional_Diffusion_Model.pdf`（风电场景生成）。

**条件c构建方式的差异：**

| 论文要求 | 当前代码实现 |
|---------|-------------|
| c = [c_down, c_up] 基于预测误差分布的区间 | c = time_embed + feature_embed + cond_mask |
| c_up = min(1, f + K_h(f)) | 时间嵌入 + 特征嵌入 + 条件掩码 |
| c_down = max(0, f - K_h(f)) | CSDI NeurIPS 2021的填补任务实现 |

如需正确复现论文，需要修改 `main_model.py` 中的 `get_side_info` 方法，按照论文公式9构建条件c。


## Requirement

Please install the packages in requirements.txt

## Preparation
### Download the healthcare dataset 
```shell
python download.py physio
```
### Download the air quality dataset 
```shell
python download.py pm25
```

### Download the elecricity dataset 
Please put files in [GoogleDrive](https://drive.google.com/drive/folders/1krZQofLdeQrzunuKkLXy8L_kMzQrVFI_?usp=drive_link) to the "data" folder.

## Experiments 

### training and imputation for the healthcare dataset
```shell
python exe_physio.py --testmissingratio [missing ratio] --nsample [number of samples]
```

### imputation for the healthcare dataset with pretrained model
```shell
python exe_physio.py --modelfolder pretrained --testmissingratio [missing ratio] --nsample [number of samples]
```

### training and imputation for the healthcare dataset
```shell
python exe_pm25.py --nsample [number of samples]
```

### training and forecasting for the electricity dataset
```shell
python exe_forecasting.py --datatype electricity --nsample [number of samples]
```

### Visualize results
'visualize_examples.ipynb' is a notebook for visualizing results.

## Acknowledgements

A part of the codes is based on [BRITS](https://github.com/caow13/BRITS) and [DiffWave](https://github.com/lmnt-com/diffwave)

## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
