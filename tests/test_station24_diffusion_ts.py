import unittest
import torch
import yaml
from pathlib import Path
from src.models.station_diffusion_ts import StationDiffusionTS


class DiffusionTSJointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)
        config=yaml.safe_load(Path("configs/station24_diffusion_ts_joint_v1.yaml").read_text())
        self.model=StationDiffusionTS(config["model"]).eval()
        self.batch={"forecast":torch.rand(2,24,168),"actual":torch.rand(2,24,168),
            "valid_mask":torch.ones(2,24,168),"recent_error":torch.rand(2,24,24),
            "recent_error_mask":torch.ones(2,24,1)}

    def test_decomposition_recomposition_and_zero_mean_residual(self):
        for layer in [self.model.forecast_embedding,self.model.history_embedding]:
            torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        x=torch.randn(2,168,24);t=torch.tensor([70,220])
        # Extra history tokens alter attention normalization; compare decomposition
        # recomposition with the identical conditioned path rather than claiming
        # equality to an unconditional network with a different memory length.
        parts=self.model.components(x,t,self.batch)
        self.assertTrue(torch.allclose(sum(parts),self.model.predict(x,t,self.batch)))
        self.assertEqual(parts[0].shape,(2,168,24))
        self.assertLess(float(parts[2].detach().mean(1).abs().max()),1e-5)

    def test_both_sources_receive_gradients_and_cross_channel_conditioning(self):
        x=torch.randn(2,168,24);t=torch.tensor([70,220])
        forecast=self.batch["forecast"].requires_grad_()
        pred=self.model.predict(x,t,self.batch)
        # Wind output must depend on solar forecast and vice versa in the joint model.
        wind=torch.autograd.grad(pred[:,:,:13].square().mean(),forecast,retain_graph=True)[0]
        solar=torch.autograd.grad(pred[:,:,13:].square().mean(),forecast)[0]
        self.assertGreater(float(wind[:,13:].abs().sum()),0)
        self.assertGreater(float(solar[:,:13].abs().sum()),0)

    def test_masked_missing_target_does_not_receive_supervision(self):
        # Hold noisy input fixed while changing a masked target value.
        b=self.batch;b["actual"].requires_grad_();b["valid_mask"][0,0,4]=0
        x=torch.randn(2,168,24,requires_grad=True)
        original=self.model.predict
        self.model.predict=lambda *args,**kwargs:x
        loss,_=self.model(b,torch.tensor([70,220]),torch.randn_like(x))
        loss.backward()
        self.assertEqual(float(b["actual"].grad[0,0,4]),0.)
        self.model.predict=original

    def test_sampler_causality_joint_shape_and_diversity(self):
        noise=torch.randn(2,168,24)
        a=self.model.sample(self.batch,noise,steps=3)
        b=dict(self.batch,actual=torch.zeros_like(self.batch["actual"]))
        self.assertTrue(torch.equal(a,self.model.sample(b,noise,steps=3)))
        self.assertEqual(a.shape,(2,168,24))
        self.assertTrue(torch.isfinite(a).all())
        self.assertFalse(torch.equal(a[0],a[1]))

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA/AMP requires target server")
    def test_cuda_fft_168_backward(self):
        self.model.cuda()
        b={k:v.cuda() for k,v in self.batch.items()}
        with torch.amp.autocast("cuda"):
            loss,_=self.model(b)
        loss.backward()
        for name,p in self.model.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)


if __name__=="__main__":unittest.main()
