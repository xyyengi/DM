import unittest
import torch
from src.models.station_joint_multiresidual_tail import HaarMultiresolution,JointMultiresolutionResidualTail,moving_average_same


class JointMultiresidualTailTests(unittest.TestCase):
    def assets(self):
        feat=torch.zeros(24,5);feat[:13,0]=1;feat[13:,1]=1
        return feat,torch.eye(24),torch.linspace(1,2,24)

    def test_haar_is_exact_and_orthogonal(self):
        x=torch.randn(3,24,168);p=HaarMultiresolution(3);low,high=p.split(x)
        self.assertTrue(torch.allclose(low+high,x,atol=1e-7,rtol=1e-6))
        self.assertLess(float((low*high).sum().abs()),1e-4)

    def test_forecast_is_exactly_decomposed_and_route_zero_is_identity(self):
        feat,graph,cap=self.assets();m=JointMultiresolutionResidualTail(32,feat,graph,cap)
        forecast=torch.rand(2,24,168);hidden=torch.rand(2,24,32,168)
        low,high=m.forecast_components(forecast)
        self.assertTrue(torch.allclose(low+high,forecast,atol=1e-7,rtol=1e-6))
        out=m(hidden,forecast,route=0.)
        self.assertEqual(tuple(out.correction.shape),(2,24,168));self.assertTrue(torch.equal(out.correction,torch.zeros_like(out.correction)))

    def test_daily_forecast_trend_is_distinct_and_same_length(self):
        forecast=torch.randn(2,24,168)
        trend=moving_average_same(forecast,24)
        self.assertEqual(trend.shape,forecast.shape)
        self.assertFalse(torch.equal(trend,HaarMultiresolution(3).low(forecast)))

    def test_invalid_route_is_rejected(self):
        feat,graph,cap=self.assets();m=JointMultiresolutionResidualTail(32,feat,graph,cap)
        with self.assertRaisesRegex(ValueError,r"\[0, 1\]"):
            m(torch.rand(1,24,32,168),torch.rand(1,24,168),route=1.1)

    def test_all_output_heads_receive_gradients_after_zero_head_update(self):
        feat,graph,cap=self.assets();m=JointMultiresolutionResidualTail(32,feat,graph,cap)
        opt=torch.optim.Adam(m.parameters(),lr=1e-3);hidden=torch.rand(2,24,32,168);forecast=torch.rand(2,24,168)
        for _ in range(2):
            opt.zero_grad();out=m(hidden,forecast,route=1.);loss=(out.correction-torch.randn_like(out.correction)).square().mean();loss.backward();opt.step()
        for name in ('local_fast','local_slow','system_fast','system_slow'):
            head=getattr(m,name);self.assertIsNotNone(head.weight.grad);self.assertGreater(float(head.weight.grad.norm()),0.)
        self.assertGreater(float(m.hidden[0].weight.grad.norm()),0.)

    def test_wind_and_solar_can_both_change_without_separate_routes(self):
        feat,graph,cap=self.assets();m=JointMultiresolutionResidualTail(32,feat,graph,cap)
        with torch.no_grad():
            for head in (m.local_fast,m.local_slow):head.bias.copy_(torch.tensor([.2,-.3]))
        out=m(torch.rand(1,24,32,168),torch.rand(1,24,168),route=1.).correction
        self.assertGreater(float(out[:,:13].detach().abs().mean()),0.);self.assertGreater(float(out[:,13:].detach().abs().mean()),0.)

    def test_full_diffusion_freezes_raw_and_backpropagates_joint_tail(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel
        feat,graph,cap=self.assets()
        cfg={"architecture":"station24_resunet","spatial_mode":"fixed_graph","station_count":24,
             "sequence_length":32,"base_channels":4,"num_layers":3,"channel_multipliers":[1,2,4],
             "group_norm_groups":4,"dropout":0.,"timestep_embedding_dim":8,"num_steps":2,
             "beta_start":1e-4,"beta_end":.02,"use_body_tail_experts":True,
             "tail_expert_channels":4,"tail_gate_channels":4,"tail_gate_prior_probability":.08,
             "tail_gate_loss_weight":.1,"use_joint_multiresidual_tail":True,
             "joint_multiresidual_channels":8,"joint_multiresidual_haar_levels":3,
             "joint_multiresidual_tail_fraction":.15,"joint_multiresidual_decomposition_loss_weight":.2,
             "joint_multiresidual_structure_loss_weight":.1,"jstd_mask_loss_weight":0.,"jstd_issue_loss_weight":0.,
             "jstd_outside_zero_loss_weight":.05}
        model=Station24DiffusionModel(cfg,feat,graph,cap);trainable=model.configure_body_tail_training()
        self.assertTrue(trainable);self.assertTrue(all('joint_multiresidual_tail' in n for n in trainable))
        length=32;b=2;batch={"residual_target":torch.randn(b,24,length),"residual":torch.randn(b,24,length),
          "residual_scale":torch.ones(b,24,length),"forecast":torch.rand(b,24,length),"calendar":torch.rand(b,8,length),
          "lead":torch.rand(b,2,length),"valid_mask":torch.ones(b,24,length),"recent_error":torch.randn(b,24,24),
          "recent_error_mask":torch.ones(b,24,1),"jstd_event_active":torch.ones(b),
          "jstd_event_time_support":torch.ones(b,length),"jstd_event_station_support":torch.ones(b,24,length),
          "jstd_sample_weight":torch.ones(b),"jstd_slow_target":torch.zeros(b,24,length),
          "jstd_fast_target":torch.zeros(b,24,length),"jstd_slow24_target":torch.zeros(b,24,length)}
        loss=model(batch,timestep=torch.tensor([0,1]),noise=torch.randn(b,24,length));self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(any(p.grad is not None for n,p in model.named_parameters() if n in trainable))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if n not in trainable))

    def test_frozen_body_can_remain_eval_while_joint_tail_trains(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel
        feat,graph,cap=self.assets()
        cfg={"architecture":"station24_resunet","spatial_mode":"fixed_graph","station_count":24,
             "sequence_length":32,"base_channels":4,"num_layers":3,"channel_multipliers":[1,2,4],
             "group_norm_groups":4,"dropout":.2,"timestep_embedding_dim":8,"num_steps":2,
             "beta_start":1e-4,"beta_end":.02,"use_body_tail_experts":True,
             "tail_expert_channels":4,"tail_gate_channels":4,"tail_gate_prior_probability":.08,
             "tail_gate_loss_weight":0.,"use_joint_multiresidual_tail":True,
             "joint_multiresidual_channels":8,"joint_multiresidual_haar_levels":3,
             "joint_multiresidual_tail_fraction":.15,"joint_multiresidual_decomposition_loss_weight":.2,
             "joint_multiresidual_structure_loss_weight":.1,"jstd_mask_loss_weight":0.,"jstd_issue_loss_weight":0.,
             "jstd_outside_zero_loss_weight":.05}
        model=Station24DiffusionModel(cfg,feat,graph,cap);model.configure_body_tail_training()
        model.eval();model.denoiser.joint_multiresidual_tail.train()
        self.assertFalse(model.denoiser.training)
        self.assertTrue(model.denoiser.joint_multiresidual_tail.training)


if __name__=='__main__':unittest.main()
