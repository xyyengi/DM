from __future__ import annotations

import copy
import os
from pathlib import Path
import unittest

import torch
import yaml

from datasets.shandong91_faithful24 import Shandong91Faithful24Dataset, fit_faithful24_state_thresholds
from datasets.shandong91_system_load import fit_system_load_contract, SystemLoadTransform
from src.models.shandong91_faithful24_diffusion import Shandong91Faithful24Diffusion, Shandong91HeterogeneousRawBodyV2
from src.models.shandong91_route_variants import Shandong91V2MultiscaleGraph, SystemLoadDiffusionAdapter


class RouteABTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ.get("SHANDONG91_DATA", "reliable_channel_training_v2_solar_nonnegative"))
        cfg_path = Path("configs/shandong91/raw_body_v2_faithful24_solar_nonnegative.yaml")
        cls.cfg = yaml.safe_load(cfg_path.read_text("utf-8"))
        thresholds = fit_faithful24_state_thresholds(cls.root)
        cls.ds = Shandong91Faithful24Dataset(cls.root, "train", thresholds)

    def test_route_a_contract_and_single_load_supervision(self):
        contract = fit_system_load_contract(self.root)
        self.assertEqual(len(contract["active_node_ids"]), 77)
        self.assertAlmostEqual(contract["coefficient_sum"], 1.0, places=7)
        self.assertLess(contract["node_reconstruction_max_abs_mw"], 1e-2)
        self.assertLess(contract["node_reconstruction_max_relative_to_system"], 1e-7)
        transform = SystemLoadTransform(contract, self.ds.scales)
        sample = self.ds[0]; residual = sample["residual"].unsqueeze(0)
        reconstructed = transform.expand(transform.collapse(residual))
        mask = sample["effective_mask"][..., 2]
        self.assertTrue(torch.allclose(reconstructed[0][mask], residual[0, ..., 2][mask], atol=1e-5, rtol=1e-5))

        model_cfg = copy.deepcopy(self.cfg["model"])
        denoiser = Shandong91HeterogeneousRawBodyV2(model_cfg, self.ds.node_features, self.ds.adjacency_with_self.float())
        base = Shandong91Faithful24Diffusion(denoiser, self.cfg["diffusion"])
        adapter = SystemLoadDiffusionAdapter(base, transform)
        batch = {k: v.unsqueeze(0) for k, v in sample.items()}
        node_noise = torch.randn_like(batch["residual"]); system_noise = torch.randn(1, 168)
        _, predicted_system, errors = adapter.prediction_and_error(batch, torch.tensor([10]), node_noise, system_noise)
        self.assertEqual(tuple(predicted_system.shape), (1, 168))
        loss = adapter.masked_loss(errors, batch["effective_mask"]); self.assertTrue(torch.isfinite(loss))
        changed = {name: value.clone() for name, value in errors.items()}
        ws_invalid = ~batch["effective_mask"][..., :2]
        changed["wind_solar"][ws_invalid] = 1e6
        load_complete = batch["effective_mask"][..., 2][..., transform.active].all(dim=-1)
        changed["load_system"][~load_complete] = 1e6
        self.assertTrue(torch.allclose(loss, adapter.masked_loss(changed, batch["effective_mask"])))
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in adapter.parameters()))

    def test_route_b_shape_gradient_update_and_reload(self):
        model_cfg = copy.deepcopy(self.cfg["model"])
        model = Shandong91V2MultiscaleGraph(model_cfg, self.ds.node_features, self.ds.adjacency_with_self.float())
        sample = self.ds[0]; batch = {k: v.unsqueeze(0) for k, v in sample.items()}
        args = (batch["residual"], torch.tensor([10]), batch["forecast"], batch["forecast_valid_mask"],
                batch["time_mark"].float(), batch["recent_error"], batch["recent_error_valid_mask"], batch["node_state"])
        out = model(*args); self.assertEqual(tuple(out.shape), (1,168,91,3)); self.assertTrue(torch.isfinite(out).all())
        loss = (out.square() * batch["effective_mask"]).sum() / batch["effective_mask"].sum()
        loss.backward()
        graph_grads = [p.grad for n,p in model.named_parameters() if "encoder_graphs" in n or "decoder_graphs" in n]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in graph_grads))
        before = model.encoder_graphs[0].gate.detach().clone() if hasattr(model.encoder_graphs[0], "gate") else next(model.encoder_graphs[0].parameters()).detach().clone()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4); opt.step()
        after = model.encoder_graphs[0].gate.detach() if hasattr(model.encoder_graphs[0], "gate") else next(model.encoder_graphs[0].parameters()).detach()
        self.assertTrue(torch.isfinite(after).all()); self.assertFalse(torch.equal(before, after))
        state = model.state_dict(); clone = Shandong91V2MultiscaleGraph(model_cfg, self.ds.node_features, self.ds.adjacency_with_self.float())
        clone.load_state_dict(state, strict=True); clone.eval(); model.eval()
        with torch.no_grad(): self.assertTrue(torch.allclose(model(*args), clone(*args), atol=1e-6, rtol=1e-6))


if __name__ == "__main__": unittest.main()
