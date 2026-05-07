#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练主程序

功能:
1. 模型训练流程
2. 自动数据集划分
3. 完整的评估和可视化

作者: AI Assistant
创建时间: 2024
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.data_loader import PVDataLoader
from core.data.dataset import PVDataset
from core.models.spatial_lstm_model import SpatialLSTMModel
from core.data.spatial_cache_manager import SpatialCacheManager
from core.mlflow import MLflowTracker, register_and_promote_model
from core.utils.plot_utils import setup_chinese_fonts
from core.models.model_manager import ModelManager, get_default_model_tag
from core.models.model_config import MODEL_CONFIG


class EarlyStopping:
    """早停机制类"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best_weights: bool = True):
        """
        初始化早停机制
        
        Args:
            patience: 验证损失不改善的最大轮数
            min_delta: 认为是改善的最小损失减少量
            restore_best_weights: 是否在早停时恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_loss: 当前验证损失
            model: 当前模型
            
        Returns:
            是否应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            # 验证损失有改善
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            # 验证损失没有改善
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    
        return self.stop_training
    
    def reset(self):
        """重置早停状态"""
        self.best_loss = float('inf')
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False


class SpatialLSTMTrainer:
    """训练器 - 完整的训练流程管理"""

    def __init__(self, enable_mlflow: bool = True):
        """
        初始化训练器
        
        Args:
           enable_mlflow: 是否启用 MLflow 追踪
        """
        self.data_loader = PVDataLoader()
        # 使用默认的 SpatialLSTM 模型 tag（可通过环境变量覆盖）
        self.model_manager = ModelManager(tag=get_default_model_tag("spatial_lstm"))
        self.model_config = MODEL_CONFIG
        self.enable_mlflow = enable_mlflow

        # 配置中文字体
        setup_chinese_fonts()

        logger.info("训练器初始化完成")

    def train_model(self,
                    region: str = 'shaanxi',
                    data_start_date: datetime = datetime(2023, 1, 1),
                    data_end_date: datetime = datetime(2025, 8, 17),
                    device: str = 'cuda',
                    results_dir: str = 'results') -> Dict[str, Any]:
        """
        训练模型的完整流程
        
        Args:
            region: 地区名称
            data_start_date: 数据开始日期
            data_end_date: 数据结束日期
            device: 设备 ('cuda' 或 'cpu')
            results_dir: 结果保存目录
            enable_mlflow: 是否启用 MLflow 追踪
            
        Returns:
            训练结果字典
        """
        # 初始化 MLflow 追踪器
        # 初始化 MLflow 追踪器
        self.tracker = None
        run_context = None

        if self.enable_mlflow:
            self.tracker = MLflowTracker(model_type='spatial_lstm', region=region)
            run_context = self.tracker.start_run(
                run_name=f"spatial_lstm_{region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            run_context.__enter__()
            logger.info("🔗 MLflow 追踪已启动")
        
        try:
            logger.info("=" * 80)
            logger.info("开始训练模型")
            logger.info("=" * 80)
            logger.info(
                f"训练参数: batch_size={MODEL_CONFIG.batch_size}, num_epochs={MODEL_CONFIG.num_epochs}, learning_rate={MODEL_CONFIG.learning_rate}")

            # 创建结果目录
            os.makedirs(results_dir, exist_ok=True)
            
            # 检查设备可用性
            device = self._check_device(device)
            logger.info(f"使用设备: {device}")

            # 步骤1: 数据加载和准备
            logger.info("步骤1: 数据加载和准备...")
            train_df, val_df, test_df = self.data_loader.load_and_prepare_data(
                region, data_start_date, data_end_date
            )

            # 验证数据质量
            validation = self.data_loader.validate_pipeline_output(train_df, val_df, test_df)
            if not validation['is_valid']:
                raise ValueError(f"数据验证失败: {validation['issues']}")
            
            # 获取 locations_hash 供后续分析使用（从 data_loader 获取，因为聚合后的数据已无 location_id 列）
            locations_hash = self.data_loader.get_locations_hash()
            logger.info(f"获取 locations_hash: {locations_hash}")

            # 步骤2: 创建数据集（空间池化模式）
            logger.info("步骤2: 创建数据集...")
            train_dataset = PVDataset(
                train_df, MODEL_CONFIG.hist_len, MODEL_CONFIG.pred_len, MODEL_CONFIG.stride
            )
            val_dataset = PVDataset(
                val_df, MODEL_CONFIG.hist_len, MODEL_CONFIG.pred_len, MODEL_CONFIG.stride
            )
            test_dataset = PVDataset(
                test_df, MODEL_CONFIG.hist_len, MODEL_CONFIG.pred_len, MODEL_CONFIG.stride
            )

            logger.info(f"空间池化数据集创建完成:")
            logger.info(f"  训练集: {len(train_dataset)}个样本")
            logger.info(f"  验证集: {len(val_dataset)}个样本")
            logger.info(f"  测试集: {len(test_dataset)}个样本")

            # 步骤3: 创建数据加载器
            logger.info("步骤3: 创建数据加载器...")
            train_loader = DataLoader(train_dataset, batch_size=MODEL_CONFIG.batch_size, shuffle=True,
                                      num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=MODEL_CONFIG.batch_size, shuffle=False,
                                    num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=MODEL_CONFIG.batch_size, shuffle=False,
                                     num_workers=0)

            # 步骤4: 初始化空间池化模型
            logger.info("步骤4: 初始化空间池化模型...")
            
            # 获取空间信息
            spatial_info = train_dataset.get_spatial_info()
            
            # 排除cluster_id等非模型特征
            exclude_features = {'cluster_id'}
            model_feature_names = [f for f in spatial_info['feature_names'] if f not in exclude_features]
            actual_feature_dim = len(model_feature_names)
            
            logger.info(f"特征过滤: 原始{len(spatial_info['feature_names'])}个 -> 模型{actual_feature_dim}个")
            logger.info(f"排除的特征: {exclude_features}")
            
            model = SpatialLSTMModel(
                input_dim=actual_feature_dim,  # 使用过滤后的特征维度
                d_model=MODEL_CONFIG.spatial_pooling.d_model,
                lstm_layers=MODEL_CONFIG.lstm_layers,  # 使用配置文件中的层数
                num_clusters=spatial_info['num_clusters']
            ).to(device)

            logger.info(f"空间池化模型初始化完成: K={spatial_info['num_clusters']}, "
                           f"F={actual_feature_dim}, d_model={MODEL_CONFIG.spatial_pooling.d_model}")
            
            # 记录 MLflow 参数
            if self.tracker:
                self.tracker.log_params({
                    'batch_size': MODEL_CONFIG.batch_size,
                    'num_epochs': MODEL_CONFIG.num_epochs,
                    'learning_rate': MODEL_CONFIG.learning_rate,
                    'hist_len': MODEL_CONFIG.hist_len,
                    'pred_len': MODEL_CONFIG.pred_len,
                    'num_clusters': spatial_info['num_clusters'],
                    'feature_dim': actual_feature_dim,
                    'd_model': MODEL_CONFIG.spatial_pooling.d_model,
                    'lstm_layers': MODEL_CONFIG.lstm_layers,
                    'early_stopping_patience': MODEL_CONFIG.early_stopping_patience
                })
                self.tracker.set_tag('train_samples', str(len(train_dataset)))
                self.tracker.set_tag('val_samples', str(len(val_dataset)))
                self.tracker.set_tag('test_samples', str(len(test_dataset)))

            # 步骤5: 训练模型
            logger.info("步骤5: 开始模型训练...")
            logger.info(f"早停配置: patience={MODEL_CONFIG.early_stopping_patience}, "
                           f"min_delta={MODEL_CONFIG.early_stopping_min_delta}")
            training_history = self._train_model(
                model, train_loader, val_loader, MODEL_CONFIG.num_epochs, MODEL_CONFIG.learning_rate,
                device
            )
            
            # 记录训练结果
            if training_history.get('early_stopped', False):
                logger.info(f"✓ 训练通过早停机制完成，实际训练轮数: {training_history['stopped_epoch']}/{MODEL_CONFIG.num_epochs}")
            else:
                logger.info(f"✓ 训练完成全部 {MODEL_CONFIG.num_epochs} 轮")

            # 步骤6: 注意力分析
            logger.info("步骤6: 注意力模式分析...")
            attention_analysis = self._analyze_attention_patterns(
                model, val_loader, device, region, locations_hash
            )
            training_history['attention_analysis'] = attention_analysis

            # 步骤7: 全面评估
            logger.info("步骤7: 模型评估...")
            evaluation_results = self._comprehensive_model_evaluation(
                model, test_loader, test_dataset, device, results_dir
            )
            
            
            # 记录 MLflow 指标
            if self.tracker:
                # 确定要使用的 step:如果早停,使用stopped_epoch;否则使用总轮数
                final_step = training_history.get('stopped_epoch', MODEL_CONFIG.num_epochs)
                
                self.tracker.log_metrics({
                    'test_R2': evaluation_results['r2'],
                    'test_rmse': evaluation_results['rmse'],
                    'test_mae': evaluation_results['mae'],
                    'test_mape': evaluation_results['mape'],
                    'test_smape': evaluation_results['smape'],
                    'final_val_r2': training_history['val_r2'][-1] if training_history['val_r2'] else 0,
                    'final_train_r2': training_history['train_r2'][-1] if training_history['train_r2'] else 0,
                    'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else 0
                }, step=final_step)
                
            # 步骤8: 保存模型
            logger.info("步骤8: 保存模型...")

            # 使用训练时已经计算好的model_feature_names和actual_feature_dim
            # 生成完整的空间聚合特征列名
            spatial_feature_names = []
            for feature in model_feature_names:
                for cluster_id in range(spatial_info['num_clusters']):
                    spatial_feature_names.append(f"{feature}__c{cluster_id:02d}")
            
            # 准备模型数据 - 使用过滤后的特征信息
            scaler_params = {
                'num_clusters': spatial_info['num_clusters'],
                'feature_dim': actual_feature_dim,         # 使用训练时的实际特征维度
                'feature_names': model_feature_names,      # 使用训练时的实际特征名
                'weather_scaler_mean': spatial_info['weather_scaler'].mean_.tolist(),
                'weather_scaler_scale': spatial_info['weather_scaler'].scale_.tolist(),
                'pv_scaler_mean': spatial_info['pv_scaler'].mean_.tolist(),
                'pv_scaler_scale': spatial_info['pv_scaler'].scale_.tolist(),
                # 添加实际的scaler特征维度信息
                'scaler_feature_count': len(spatial_info['weather_scaler'].mean_)
            }
            
            model_data = {
                'model_type': 'spatial_lstm',  # 模型类型标识
                'model_state_dict': model.state_dict(),
                'spatial_info': spatial_info,  # 保留原始spatial_info用于训练时使用
                'model_config': {
                    'input_dim': actual_feature_dim,  # 使用训练时的实际特征维度
                    'd_model': MODEL_CONFIG.spatial_pooling.d_model,
                    'lstm_layers': MODEL_CONFIG.lstm_layers
                },
                'spatial_pooling_config': MODEL_CONFIG.spatial_pooling.__dict__,
                'training_config': MODEL_CONFIG.get_training_config(),
                'scaler_params': scaler_params,  # 使用可序列化的scaler参数
                'weather_features': spatial_feature_names,  # 保存完整的空间聚合特征列表
                'data_period': {
                    'start_date': data_start_date.strftime('%Y-%m-%d'),
                    'end_date': data_end_date.strftime('%Y-%m-%d')
                }
            }

            # 使用版本管理保存模型
            try:
                versioned_model_path = self.model_manager.save_model_with_version(
                    model_data=model_data,
                    region=region,
                    performance_metrics=evaluation_results
                )
                logger.info(f"版本化模型已保存到: {versioned_model_path}")
            except Exception as e:
                logger.warning(f"版本化保存失败: {e}")
            
            # 保存模型到 MLflow 并注册
            if self.tracker:
                try:
                    # 构造 Input Example 和 Signature
                    input_example = None
                    signature = None
                    try:
                        import pandas as pd
                        import numpy as np
                        from mlflow.models import infer_signature
                        
                        K = spatial_info['num_clusters']
                        F = actual_feature_dim
                        T_h = MODEL_CONFIG.hist_len
                        T_f = MODEL_CONFIG.pred_len
                        
                        # 构造 Dummy Input (Batch=1)
                        # 注意: Wrapper 期望 DataFrame 的列包含数组/列表
                        # 使用 float64 以避免 MLflow 在 JSON 序列化验证时的类型不匹配警告
                        dummy_hist_weather = np.random.randn(T_h, K, F)
                        dummy_fut_weather = np.random.randn(T_f, K, F)
                        dummy_hist_power = np.random.randn(T_h, 1)
                        
                        input_example_df = pd.DataFrame([{
                            'hist_weather': dummy_hist_weather,
                            'future_weather': dummy_fut_weather,
                            'hist_power': dummy_hist_power
                        }])
                        
                        # 推断 Output
                        # 为了推断签名，我们需要模拟 Wrapper 的输出 (numpy array)
                        # 这里直接用模型跑一次得到 output shape
                        model.eval()
                        with torch.no_grad():
                            dummy_out = model(
                                torch.from_numpy(dummy_hist_weather).float().unsqueeze(0).to(device),
                                torch.from_numpy(dummy_fut_weather).float().unsqueeze(0).to(device),
                                torch.from_numpy(dummy_hist_power).float().unsqueeze(0).to(device)
                            )
                            dummy_prediction = dummy_out.cpu().numpy().flatten()
                        
                        signature = infer_signature(input_example_df, dummy_prediction)
                        input_example = input_example_df
                        logger.info("已成功推断模型签名")
                        
                    except Exception as sig_err:
                        logger.warning(f"推断模型签名失败: {sig_err}")

                    # 使用统一接口注册和提升模型
                    version = register_and_promote_model(
                        model=model,
                        model_type='spatial_lstm',
                        region=region,
                        model_config=model_data['model_config'],
                        scaler_params=scaler_params,
                        evaluation_results={
                            'R2': evaluation_results['r2'],
                            'val_R2': max(training_history.get('val_r2', [0])),
                            'MAE': evaluation_results['mae'],
                            'RMSE': evaluation_results['rmse']
                        },
                        run_id=self.tracker.get_run_id(),
                        weather_features=spatial_info['feature_names'],
                        signature=signature,
                        input_example=input_example
                    )
                    if version:
                        logger.info(f"✅ 模型已注册到 MLflow Registry: pv-spatial_lstm-{region} v{version}")
                except Exception as e:
                    logger.warning(f"MLflow 模型注册失败: {e}")

            # 汇总结果
            final_results = {
                'training_completed': True,
                'model_path': versioned_model_path if 'versioned_model_path' in locals() else None,
                'results_dir': results_dir,
                'training_history': training_history,
                'evaluation_metrics': evaluation_results,
                'data_summary': {
                    'region': region,
                    'time_range': (data_start_date, data_end_date),
                    'train_samples': len(train_dataset),
                    'val_samples': len(val_dataset),
                    'test_samples': len(test_dataset),
                    'weather_features': spatial_info['feature_dim']
                }
            }

            logger.info("=" * 80)
            logger.info("模型训练完成!")
            logger.info(f"最终评估指标: R²={evaluation_results['r2']:.4f}, "
                             f"RMSE={evaluation_results['rmse']:.2f}, "
                             f"MAPE={evaluation_results['mape']:.2f}%, "
                             f"SMAPE={evaluation_results['smape']:.2f}%")
            logger.info("=" * 80)

            return final_results

        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            raise
        finally:
            # 关闭 MLflow Run
            if run_context:
                try:
                    run_context.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"关闭 MLflow Run 失败: {e}")

    def _check_device(self, device: str) -> str:
        """检查设备可用性"""
        if device == 'cuda' and torch.cuda.is_available():
            logger.info(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
            return 'cuda'
        else:
            if device == 'cuda':
                logger.warning("CUDA不可用，切换到CPU")
            return 'cpu'

    def _train_model(self,
                     model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     num_epochs: int,
                     learning_rate: float,
                     device: str) -> Dict[str, list]:
        """
        训练模型
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            device: 设备
            
        Returns:
            训练历史
        """
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        
        # 根据配置选择初始学习率
        if MODEL_CONFIG.use_lr_scheduler:
            initial_lr = MODEL_CONFIG.initial_lr
            logger.info(f"🔧 启用学习率调度，初始学习率: {initial_lr}")
        else:
            initial_lr = learning_rate
            logger.info(f"🔧 使用固定学习率: {initial_lr}")
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=initial_lr,
            weight_decay=MODEL_CONFIG.weight_decay  # 使用配置文件中的权重衰减
        )
        
        # 学习率调度器
        scheduler = None
        if MODEL_CONFIG.use_lr_scheduler:
            scheduler = self._create_lr_scheduler(optimizer)

        # 初始化早停机制
        early_stopping = EarlyStopping(
            patience=MODEL_CONFIG.early_stopping_patience,
            min_delta=MODEL_CONFIG.early_stopping_min_delta,
            restore_best_weights=MODEL_CONFIG.early_stopping_restore_best
        )

        # 记录训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_r2': [],
            'val_r2': [],
            'early_stopped': False,
            'stopped_epoch': 0
        }

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_losses = []
            train_predictions = []
            train_targets = []

            for batch in train_loader:
                optimizer.zero_grad()

                # 准备数据
                hist_weather = batch['hist_weather'].to(device)
                future_weather = batch['future_weather'].to(device)
                hist_power = batch['hist_power'].to(device)
                target = batch['target'].to(device)

                # 前向传播
                if MODEL_CONFIG.spatial_pooling.attention_entropy_weight > 0:
                    output, attention_loss = model(hist_weather, future_weather, hist_power, return_attention_loss=True)
                    # 总损失 = MSE损失 + 熵损失
                    loss = criterion(output, target) + MODEL_CONFIG.spatial_pooling.attention_entropy_weight * attention_loss
                else:
                    output = model(hist_weather, future_weather, hist_power)
                    loss = criterion(output, target)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 记录
                train_losses.append(loss.item())
                train_predictions.append(output.detach().cpu().numpy())
                train_targets.append(target.detach().cpu().numpy())

            # 验证阶段
            model.eval()
            val_losses = []
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    hist_weather = batch['hist_weather'].to(device)
                    future_weather = batch['future_weather'].to(device)
                    hist_power = batch['hist_power'].to(device)
                    target = batch['target'].to(device)

                    output = model(hist_weather, future_weather, hist_power)
                    loss = criterion(output, target)

                    val_losses.append(loss.item())
                    val_predictions.append(output.cpu().numpy())
                    val_targets.append(target.cpu().numpy())

            # 计算指标
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            # 计算RMSE和R²
            train_pred_flat = np.concatenate(train_predictions).flatten()
            train_true_flat = np.concatenate(train_targets).flatten()
            val_pred_flat = np.concatenate(val_predictions).flatten()
            val_true_flat = np.concatenate(val_targets).flatten()

            train_rmse = np.sqrt(mean_squared_error(train_true_flat, train_pred_flat))
            val_rmse = np.sqrt(mean_squared_error(val_true_flat, val_pred_flat))
            train_r2 = r2_score(train_true_flat, train_pred_flat)
            val_r2 = r2_score(val_true_flat, val_pred_flat)

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_rmse'].append(train_rmse)
            history['val_rmse'].append(val_rmse)
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)

            # MLflow Epoch 级指标追踪
            if self.tracker:
                current_lr = optimizer.param_groups[0]['lr']
                self.tracker.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'learning_rate': current_lr,
                }, step=epoch + 1)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # 早停检查
            if early_stopping(val_loss, model):
                history['early_stopped'] = True
                history['stopped_epoch'] = epoch + 1
                logger.info(f"早停触发！在第 {epoch + 1} 轮停止训练")
                logger.info(f"最佳验证损失: {early_stopping.best_loss:.6f}")
                break

            # 打印进度
            if (epoch + 1) % 2 == 0 or epoch == 0:
                patience_info = f"patience: {early_stopping.wait}/{early_stopping.patience}"
                logger.info(f"Epoch [{epoch + 1}/{num_epochs}] "
                                 f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                                 f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, {patience_info}")
            
            # 学习率调度
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                if (epoch + 1) % 2 == 0:
                    logger.info(f"📈 学习率调整至: {current_lr:.6f}")

        return history

    def _comprehensive_model_evaluation(self,
                                        model: nn.Module,
                                        test_loader: DataLoader,
                                        test_dataset: PVDataset,
                                        device: str,
                                        output_dir: str) -> Dict[str, Any]:
        """
        全面的模型评估
        
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            test_dataset: 测试数据集
            device: 设备
            output_dir: 输出目录
            
        Returns:
            评估结果
        """
        model.eval()
        predictions = []
        actuals = []

        # 获取预测结果
        with torch.no_grad():
            for batch in test_loader:
                hist_weather = batch['hist_weather'].to(device)
                future_weather = batch['future_weather'].to(device)
                hist_power = batch['hist_power'].to(device)
                target = batch['target'].to(device)

                output = model(hist_weather, future_weather, hist_power)

                predictions.append(output.cpu().numpy())
                actuals.append(target.cpu().numpy())

        # 合并结果
        predictions = np.concatenate(predictions).flatten()
        actuals = np.concatenate(actuals).flatten()

        # 逆转换到原始尺度
        predictions_original = test_dataset.inverse_transform_pv(predictions)
        actuals_original = test_dataset.inverse_transform_pv(actuals)

        # 计算评估指标
        metrics = self._calculate_evaluation_metrics(predictions_original, actuals_original)

        # 生成可视化
        self._generate_evaluation_plots(predictions_original, actuals_original, output_dir)

        # 保存评估报告
        self._save_evaluation_report(metrics, output_dir)

        return metrics

    def _calculate_evaluation_metrics(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        # MAPE计算 - 专门针对电力预测场景优化
        # 方案1: 排除零值的MAPE (适合电力预测，夜间发电为0是正常的)
        zero_threshold = 50  # MW，认为低于50MW为"零发电"
        non_zero_mask = np.abs(actuals) > zero_threshold

        if np.sum(non_zero_mask) > 0:
            # 只对非零发电时段计算MAPE
            mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
        else:
            mape = 0.0  # 如果全是零值，MAPE为0

        # SMAPE (对称MAPE) - 更稳定的百分比误差指标
        epsilon = 1e-6
        smape = np.mean(2.0 * np.abs(predictions - actuals) / (np.abs(predictions) + np.abs(actuals) + epsilon)) * 100

        # 电力预测特定指标
        # 1. 峰值预测准确率
        peak_threshold = 0.8  # 80%峰值
        actual_peak = np.max(actuals) * peak_threshold
        pred_peak = np.max(predictions) * peak_threshold

        actual_peak_mask = actuals >= actual_peak
        pred_peak_mask = predictions >= pred_peak
        peak_accuracy = np.mean(actual_peak_mask == pred_peak_mask) * 100

        # 2. 零发电期准确率 (夜间等)
        zero_mask_actual = actuals < zero_threshold
        zero_mask_pred = predictions < zero_threshold
        zero_accuracy = np.mean(zero_mask_actual == zero_mask_pred) * 100

        r2 = r2_score(actuals, predictions)

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,  # 排除零值的MAPE
            'smape': smape,  # 对称MAPE
            'peak_accuracy': peak_accuracy,  # 峰值预测准确率
            'zero_accuracy': zero_accuracy,  # 零发电期准确率
            'r2': r2
        }

    def _generate_evaluation_plots(self, predictions: np.ndarray, actuals: np.ndarray, output_dir: str):
        """生成评估图表"""
        # 1. 预测值vs实际值散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        plt.xlabel('实际值 (MW)')
        plt.ylabel('预测值 (MW)')
        plt.title('预测值 vs 实际值')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 时间序列对比图 (前1000个点)
        n_points = min(1000, len(predictions))
        plt.figure(figsize=(15, 6))
        plt.plot(actuals[:n_points], label='实际值', linewidth=1)
        plt.plot(predictions[:n_points], label='预测值', linewidth=1)
        plt.xlabel('时间步')
        plt.ylabel('功率 (MW)')
        plt.title('预测值与实际值对比 (前1000个时间步)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'time_series_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 误差分布直方图
        errors = predictions - actuals
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('预测误差 (MW)')
        plt.ylabel('频次')
        plt.title('预测误差分布')
        plt.axvline(x=0, color='red', linestyle='--', label=f'均值: {np.mean(errors):.2f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"评估图表已保存到: {output_dir}")

    def _save_evaluation_report(self, metrics: Dict[str, float], output_dir: str):
        """保存评估报告"""
        report_path = os.path.join(output_dir, 'evaluation_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("光伏发电功率预测模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("评估指标:\n")
            f.write(f"  R²相关系数:        {metrics['r2']:.4f}\n")
            f.write(f"  均方根误差(RMSE):  {metrics['rmse']:.2f} MW\n")
            f.write(f"  平均绝对误差(MAE): {metrics['mae']:.2f} MW\n")
            f.write(f"  平均绝对百分比误差(MAPE): {metrics['mape']:.2f}% (排除零发电时段)\n")
            f.write(f"  对称平均绝对百分比误差(SMAPE): {metrics['smape']:.2f}%\n")
            f.write(f"\n电力预测特定指标:\n")
            f.write(f"  峰值预测准确率: {metrics['peak_accuracy']:.2f}%\n")
            f.write(f"  零发电期准确率: {metrics['zero_accuracy']:.2f}%\n")

        logger.info(f"评估报告已保存到: {report_path}")
    
    def _analyze_attention_patterns(self, model, val_loader, device, region: str, locations_hash: str):
        """分析注意力模式并生成可解释图"""
        model.eval()
        attention_weights_collection = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 10:  # 分析10个batch
                    break
                
                hist_weather = batch['hist_weather'].to(device)
                future_weather = batch['future_weather'].to(device)
                hist_power = batch['hist_power'].to(device)
                
                weights = model.get_attention_weights(hist_weather, future_weather, hist_power)  # (B, T, K)
                attention_weights_collection.append(weights)
        
        if attention_weights_collection:
            all_weights = np.concatenate(attention_weights_collection, axis=0)  # (N, T, K)
            
            # 计算长期平均贡献
            avg_weights = np.mean(all_weights, axis=(0, 1))  # (K,)
            
            # 计算时间方差
            temporal_variance = np.var(all_weights, axis=1).mean(axis=0)  # (K,)
            
            # 保存注意力分析
            analysis = {
                'average_cluster_importance': avg_weights.tolist(),
                'temporal_attention_variance': temporal_variance.tolist(),
                'attention_entropy': -np.sum(avg_weights * np.log(avg_weights + 1e-8)),
                'max_cluster_weight': float(np.max(avg_weights)),
                'weight_concentration': float(np.sum(avg_weights > 0.1))  # 权重>0.1的分区数
            }
            
            # 保存注意力权重数据（完整数据用于诊断，平均数据用于热力图）
            self._save_attention_data(all_weights, avg_weights, region, locations_hash)
            
            return analysis
        
        return {}
    
    def _save_attention_data(self, all_weights, avg_weights, region: str, locations_hash: str):
        """统一保存注意力权重数据"""
        try:
            spatial_config = MODEL_CONFIG.spatial_processing
            timestamp = datetime.now().isoformat()
            
            # 1. 保存完整权重数据（用于诊断分析）
            full_weight_data = {
                'attention_weights': all_weights.tolist(),  # (B, T, K)
                'metadata': {
                    'num_clusters': spatial_config.num_clusters,
                    'cluster_method': spatial_config.cluster_method,
                    'num_batches': all_weights.shape[0],
                    'num_time_steps': all_weights.shape[1],
                    'model_type': 'spatial_pooling',
                    'timestamp': timestamp,
                    'region': region,
                    'locations_hash': locations_hash
                }
            }
            
            full_weight_path = "results/cluster_attention_weights.json"
            with open(full_weight_path, 'w') as f:
                json.dump(full_weight_data, f, indent=2)
            
            logger.info(f"完整注意力权重数据已保存到: {full_weight_path}")
            
            # 2. 保存平均权重数据（用于热力图）
            heatmap_weight_data = {
                'cluster_weights': avg_weights.tolist(),
                'timestamp': timestamp,
                'num_clusters': spatial_config.num_clusters,
                'cluster_method': spatial_config.cluster_method,
                'region': region,
                'locations_hash': locations_hash
            }
            
            # 通过 SpatialCacheManager 获取分区映射和中心点信息
            cache_manager = SpatialCacheManager(silent_mode=True)
            mapping_data = cache_manager.get_cluster_analysis_data_by_hash(region, locations_hash)
            
            if mapping_data:
                centroids = mapping_data.get('cluster_centroids')
                if centroids:
                    heatmap_weight_data['cluster_centroids'] = centroids
                    logger.info("已加载分区中心点信息")
                else:
                    logger.info("分区映射文件中无中心点信息")
            else:
                logger.info(f"未找到分区映射文件: region={region}, hash={locations_hash}")
            
            heatmap_weight_path = "results/cluster_heatmap_weights.json"
            with open(heatmap_weight_path, 'w') as f:
                json.dump(heatmap_weight_data, f, indent=2)
            
            logger.info(f"注意力热力图权重已保存到: {heatmap_weight_path}")
            
        except Exception as e:
            logger.error(f"保存注意力权重数据失败: {e}")
    
    def _create_lr_scheduler(self, optimizer):
        """
        创建学习率调度器
        
        Args:
            optimizer: 优化器
            
        Returns:
            学习率调度器
        """
        config = MODEL_CONFIG
        
        # 创建组合调度器: warmup + step decay
        def lr_lambda(epoch):
            if epoch < config.warmup_epochs:
                # Warmup阶段: 从初始学习率线性增长到目标学习率
                warmup_factor = (epoch + 1) / config.warmup_epochs
                return config.learning_rate / config.initial_lr * warmup_factor
            else:
                # 衰减阶段: 每decay_epochs轮衰减一次
                decay_epochs = epoch - config.warmup_epochs
                decay_times = decay_epochs // config.decay_epochs
                decay_factor = config.decay_factor ** decay_times
                final_lr = config.learning_rate / config.initial_lr * decay_factor
                # 确保不低于最小学习率
                min_lr_ratio = config.min_lr / config.initial_lr
                return max(final_lr, min_lr_ratio)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        logger.info(f"📈 学习率调度策略:")
        logger.info(f"   - 初始学习率: {config.initial_lr}")
        logger.info(f"   - 目标学习率: {config.learning_rate}")
        logger.info(f"   - Warmup轮数: {config.warmup_epochs}")
        logger.info(f"   - 衰减间隔: {config.decay_epochs}轮")
        logger.info(f"   - 衰减因子: {config.decay_factor}")
        logger.info(f"   - 最小学习率: {config.min_lr}")
        
        return scheduler
