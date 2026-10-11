"""Lightweight contracts for Shandong91 V3 low-rank semantics."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

from datasets.shandong91_faithful24 import Shandong91Faithful24Dataset, fit_faithful24_state_thresholds
from datasets.shandong91_low_rank import FixedPCAFactorTransform, factor_sha256, fit_train_only_pca
from src.models.shandong91_low_rank_diffusion import (
    Shandong91HeterogeneousRawBodyV3,
    Shandong91HeterogeneousRawBodyV31,
    Shandong91LowRankDiffusion,
)
from generate_shandong91_v3 import atomic_json, open_array


class Shandong91V3LowRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=yaml.safe_load(Path("configs/shandong91/raw_body_v3_low_rank_common_factor.yaml").read_text("utf-8"))
        cls.v31_config=yaml.safe_load(Path("configs/shandong91/raw_body_v3_1_factor_coordinate_loss.yaml").read_text("utf-8"))
        cls.factors=fit_train_only_pca(cls.config["data"]["data_path"])
        cls.thresholds=fit_faithful24_state_thresholds(cls.config["data"]["data_path"])
        cls.dataset=Shandong91Faithful24Dataset(cls.config["data"]["data_path"],"train",cls.thresholds)

    def test_train_only_factor_contract(self):
        self.assertEqual(self.factors["sha256"],factor_sha256(self.factors))
        self.assertEqual([self.factors["resources"][n]["factor_count"] for n in ("Wind","Solar","Load")],[5,1,1])
        self.assertEqual(self.factors["resources"]["Load"]["complete_train_rows"],7296)
        self.assertGreater(self.factors["resources"]["Load"]["explained_variance_ratio_cumulative"],.999999)

    def test_decomposition_reconstruction_and_load_constraint(self):
        sample=self.dataset[0]; residual=sample["residual"][None].float(); mask=sample["effective_mask"][None]
        transform=FixedPCAFactorTransform(self.factors); factor,local=transform.decompose(residual,mask)
        restored=transform.reconstruct(factor,local,mask)
        self.assertLess(float((restored-residual).abs()[mask].max()),1e-5)
        self.assertEqual(float(local[...,2].abs().max()),0.0)
        self.assertLess(float(transform.project_factor(local,mask).abs().max()),2e-5)

    def test_half_precision_projection_solves_in_fp32_with_gradient(self):
        sample=self.dataset[0]; mask=sample["effective_mask"][None]
        values=sample["residual"][None].half().requires_grad_(True)
        transform=FixedPCAFactorTransform(self.factors)
        factors=transform.project_factor(values,mask)
        self.assertEqual(factors.dtype,torch.float32)
        self.assertTrue(torch.isfinite(factors).all())
        factors.square().mean().backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(float(values.grad.float().norm()),0)

    def test_projection_disables_active_autocast_for_linear_solve(self):
        sample=self.dataset[0]; mask=sample["effective_mask"][None]
        values=sample["residual"][None].float().requires_grad_(True)
        transform=FixedPCAFactorTransform(self.factors)
        with torch.autocast(device_type="cpu",dtype=torch.bfloat16,enabled=True):
            factors=transform.project_factor(values,mask)
            loss=factors.square().mean()
        self.assertEqual(factors.dtype,torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_precomputed_projection_matches_direct_solve(self):
        sample=self.dataset[0]
        mask=sample["effective_mask"][None].expand(3,-1,-1,-1)
        values=torch.randn(mask.shape,dtype=torch.float32)
        transform=FixedPCAFactorTransform(self.factors)
        operator=transform.prepare_projection(mask)
        direct=transform.project_factor(values,mask)
        fast=transform.project_factor(values,mask,operator)
        self.assertTrue(torch.allclose(direct,fast,atol=2e-5,rtol=2e-5))
        direct_local=transform.project_local(values,mask)
        fast_local=transform.project_local(values,mask,operator)
        self.assertTrue(torch.allclose(direct_local,fast_local,atol=2e-5,rtol=2e-5))

    def test_sampling_solves_only_when_preparing_window_projection(self):
        sample=self.dataset[0]; batch={k:v[None].expand(2,*v.shape) for k,v in sample.items()}
        for key in ("actual","forecast","residual","time_mark","recent_error","node_state"):
            batch[key]=batch[key].float()
        model_config=copy.deepcopy(self.config["model"])
        model_config.update(base_channels=8,channel_multipliers=[1,2,2],group_norm_groups=4,state_channels=[4,8,8],dropout=0.0)
        model=Shandong91LowRankDiffusion(
            Shandong91HeterogeneousRawBodyV3(model_config,self.dataset.node_features,self.dataset.adjacency_with_self.float()),
            self.config["diffusion"],self.factors)
        original=torch.linalg.solve
        with mock.patch("torch.linalg.solve",wraps=original) as solve:
            model.sample(batch,batch["effective_mask"],method="ddim",inference_steps=3,
                         generator=torch.Generator().manual_seed(17))
        self.assertEqual(solve.call_count,3)

    def test_fast_sampler_matches_reference_per_step_solve(self):
        sample=self.dataset[0]; batch={k:v[None] for k,v in sample.items()}
        for key in ("actual","forecast","residual","time_mark","recent_error","node_state"):
            batch[key]=batch[key].float()
        model_config=copy.deepcopy(self.config["model"])
        model_config.update(base_channels=8,channel_multipliers=[1,2,2],group_norm_groups=4,state_channels=[4,8,8],dropout=0.0)
        model=Shandong91LowRankDiffusion(
            Shandong91HeterogeneousRawBodyV3(model_config,self.dataset.node_features,self.dataset.adjacency_with_self.float()),
            self.config["diffusion"],self.factors).eval()
        initial_factor=torch.randn(1,168,7)
        initial_local=torch.randn(1,168,91,3)
        reference=model.sample(
            batch,batch["effective_mask"],method="ddim",inference_steps=3,
            initial_factor_noise=initial_factor,initial_local_noise=initial_local,
            generator=torch.Generator().manual_seed(23),_precompute_projection=False)
        fast=model.sample(
            batch,batch["effective_mask"],method="ddim",inference_steps=3,
            initial_factor_noise=initial_factor,initial_local_noise=initial_local,
            generator=torch.Generator().manual_seed(23))
        delta=(reference-fast).abs()
        self.assertTrue(
            torch.allclose(reference,fast,atol=3e-4,rtol=3e-5),
            f"max_abs={float(delta.max())}, mean_abs={float(delta.mean())}",
        )

    def test_disk_backed_generation_array_reopens_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"scenarios.npy"; progress=Path(directory)/"progress.json"
            array=open_array(path,shape=(2,3),dtype=np.float32,resume=False)
            array[0]=[1,2,3]; array.flush()
            atomic_json({"completed_windows":1},progress)
            resumed=open_array(path,shape=(2,3),dtype=np.float32,resume=True)
            self.assertTrue(np.array_equal(resumed[0],[1,2,3]))
            self.assertEqual(__import__("json").loads(progress.read_text("utf-8"))["completed_windows"],1)
            with self.assertRaises(ValueError):
                open_array(path,shape=(3,2),dtype=np.float32,resume=True)
            resumed._mmap.close(); array._mmap.close()

    def test_forward_backward_checkpoint_and_sampler(self):
        sample=self.dataset[0]; batch={k:v[None] for k,v in sample.items()}
        for key in ("actual","forecast","residual","time_mark","recent_error","node_state"): batch[key]=batch[key].float()
        model_config=copy.deepcopy(self.config["model"])
        model_config.update(base_channels=8,channel_multipliers=[1,2,2],group_norm_groups=4,state_channels=[4,8,8],dropout=0.0)
        denoiser=Shandong91HeterogeneousRawBodyV3(model_config,self.dataset.node_features,self.dataset.adjacency_with_self.float())
        model=Shandong91LowRankDiffusion(denoiser,self.config["diffusion"],self.factors)
        prediction,error=model.prediction_and_error(batch,torch.tensor([137])); loss=model.masked_loss(error,batch["effective_mask"])
        loss.backward(); self.assertEqual(tuple(prediction.shape),(1,168,91,3))
        self.assertTrue(torch.isfinite(loss)); self.assertGreater(float(denoiser.encoder_blocks[0].conv1.weight.grad.norm()),0)
        clone=Shandong91LowRankDiffusion(
            Shandong91HeterogeneousRawBodyV3(model_config,self.dataset.node_features,self.dataset.adjacency_with_self.float()),
            self.config["diffusion"],self.factors)
        clone.load_state_dict(model.state_dict(),strict=True); clone.eval(); model.eval()
        factor_noise=torch.randn(1,168,7); local_noise=torch.randn(1,168,91,3)
        with torch.no_grad():
            left=model.sample(batch,batch["effective_mask"],method="ddim",inference_steps=2,
                              initial_factor_noise=factor_noise,initial_local_noise=local_noise)
            right=clone.sample(batch,batch["effective_mask"],method="ddim",inference_steps=2,
                               initial_factor_noise=factor_noise,initial_local_noise=local_noise)
        self.assertTrue(torch.allclose(left,right,rtol=0,atol=1e-6)); self.assertTrue(torch.isfinite(left).all())

    def test_v31_versions_training_semantics_without_architecture_expansion(self):
        self.assertNotEqual(
            Shandong91HeterogeneousRawBodyV3.architecture,
            Shandong91HeterogeneousRawBodyV31.architecture,
        )
        old=copy.deepcopy(self.config["model"]); new=copy.deepcopy(self.v31_config["model"])
        old["architecture"]=new["architecture"]
        self.assertEqual(old,new)
        self.assertEqual(
            self.v31_config["loss"]["formal_objective"],
            "factor_coordinate_plus_projected_local_mean",
        )
        self.assertEqual(self.v31_config["training"]["amp_initial_scale"],128.0)
        self.assertEqual(self.v31_config["training"]["amp_growth_interval"],1000000)

    def test_v31_coordinate_loss_directly_supervises_all_factor_resources(self):
        sample=self.dataset[0]; mask=sample["effective_mask"][None]
        config=copy.deepcopy(self.v31_config["model"])
        config.update(base_channels=8,channel_multipliers=[1,2,2],group_norm_groups=4,state_channels=[4,8,8],dropout=0.0)
        model=Shandong91LowRankDiffusion(
            Shandong91HeterogeneousRawBodyV31(config,self.dataset.node_features,self.dataset.adjacency_with_self.float()),
            self.v31_config["diffusion"],self.factors,
        )
        predicted_factor=torch.zeros(1,168,7,requires_grad=True)
        predicted_local=torch.zeros(1,168,91,3,requires_grad=True)
        latents={
            "predicted_factor_noise":predicted_factor,
            "factor_noise":torch.ones_like(predicted_factor),
            "predicted_local_noise":predicted_local,
            "local_noise":torch.ones_like(predicted_local),
        }
        loss,terms=model.coordinate_loss(latents,mask)
        self.assertTrue(torch.allclose(loss,terms["factor_coordinate"]+terms["local_projected"]))
        loss.backward()
        norms=[]
        for start,stop in ((0,5),(5,6),(6,7)):
            value=float(predicted_factor.grad[...,start:stop].norm())
            self.assertGreater(value,0); self.assertTrue(np.isfinite(value)); norms.append(value)
        self.assertLess(max(norms)/min(norms),3.0)
        self.assertEqual(float(predicted_local.grad[...,2].abs().max()),0.0)

    def test_v31_pipeline_is_fail_closed_and_complete(self):
        pipeline=Path("run_shandong91_raw_body_v3_1_pipeline.sh").read_text("utf-8")
        finalize=Path("run_shandong91_raw_body_v3_1_finalize.sh").read_text("utf-8")
        ordered=("cuda_amp_preflight","formal_training","generation_evaluation_plots_archive","complete")
        positions=[pipeline.index(f'PHASE="{name}"') for name in ordered]
        self.assertEqual(positions,sorted(positions))
        self.assertIn('r["cuda_amp_safety"]=="PASS"',pipeline)
        self.assertIn('factor_gradient_scale_not_pca_attenuated',pipeline)
        self.assertIn('amp_initial_scale"]==128.0',pipeline)
        self.assertIn('generate_shandong91_v3.py',finalize)
        self.assertIn('evaluate_shandong91_v3.py',finalize)
        self.assertIn('plot_shandong91_v3.py',finalize)
        self.assertIn('tar -czf "$TEMP_ARCHIVE"',finalize)
        self.assertIn('--resume-output',finalize)


if __name__=="__main__": unittest.main()
