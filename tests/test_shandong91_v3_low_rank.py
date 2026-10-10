"""Lightweight contracts for Shandong91 V3 low-rank semantics."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

import torch
import yaml

from datasets.shandong91_faithful24 import Shandong91Faithful24Dataset, fit_faithful24_state_thresholds
from datasets.shandong91_low_rank import FixedPCAFactorTransform, factor_sha256, fit_train_only_pca
from src.models.shandong91_low_rank_diffusion import Shandong91HeterogeneousRawBodyV3, Shandong91LowRankDiffusion


class Shandong91V3LowRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=yaml.safe_load(Path("configs/shandong91/raw_body_v3_low_rank_common_factor.yaml").read_text("utf-8"))
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


if __name__=="__main__": unittest.main()
