import unittest
import torch
import yaml
from pathlib import Path
from src.models.station_diffusion_ts_inherited import StationDiffusionTSInherited


class InheritedJointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(71)
        cfg=yaml.safe_load(Path("configs/station24_diffusion_ts_joint_inherited_v2.yaml").read_text())
        features=torch.randn(24,5);features[:,:2]=0;features[:13,0]=1;features[13:,1]=1
        self.assets={"geographic":torch.eye(24)+.1,"historical":torch.eye(24)+torch.ones(24,24)*.2,
                     "station_features":features,"capacities":torch.ones(24)}
        self.model=StationDiffusionTSInherited(cfg["model"],self.assets).eval()
        self.batch={"forecast":torch.rand(2,24,168),"actual":torch.rand(2,24,168),
            "valid_mask":torch.ones(2,24,168),"recent_error":torch.rand(2,24,24),
            "recent_error_mask":torch.ones(2,24,1),"calendar":torch.rand(2,8,168),
            "lead":torch.rand(2,2,168),"node_state":torch.rand(2,24,4,168)}

    def test_all_parameters_update_buffers_do_not(self):
        before={k:p.clone() for k,p in self.model.named_parameters()}
        buffers={k:p.clone() for k,p in self.model.named_buffers()}
        opt=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=0)
        for _ in range(3):
            opt.zero_grad();loss,_=self.model(self.batch,torch.tensor([50,250]));loss.backward();opt.step()
        for k,p in self.model.named_parameters():
            self.assertIsNotNone(p.grad,k);self.assertTrue(torch.isfinite(p.grad).all(),k)
            self.assertFalse(torch.equal(before[k],p),k)
        for k,p in self.model.named_buffers():self.assertTrue(torch.equal(buffers[k],p),k)

    def test_structure_loss_perfect_zero_and_each_gradient(self):
        clean=self.batch["actual"].transpose(1,2)*2-1
        valid=self.batch["valid_mask"].transpose(1,2)
        f=self.batch["forecast"].transpose(1,2)*2-1
        exact=self.model.structure_losses(clean,clean,valid,f)
        self.assertTrue(all(float(v.abs().max())==0 for v in exact.values()))
        pred=(clean+.15*torch.randn_like(clean)).requires_grad_()
        parts=self.model.structure_losses(pred,clean,valid,f)
        for key,value in parts.items():
            g=torch.autograd.grad(value.mean(),pred,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(g).all(),key);self.assertGreater(float(g.abs().sum()),0,key)
        # Pure long bias and local shift have distinct, nonzero structural errors.
        pred=clean.clone();pred[:,40:60]-=.4
        parts=self.model.structure_losses(pred,clean,valid,f)
        self.assertGreater(float(parts["slow"].mean()),0)
        self.assertGreater(float(parts["ramp"].mean()),0)

    def test_future_labels_and_event_routes_never_enter_sampling(self):
        noise=torch.randn(2,168,24)
        a=self.model.sample(self.batch,noise,3)
        other=dict(self.batch,actual=torch.zeros_like(self.batch["actual"]),
                   residual_target=torch.randn_like(self.batch["actual"]),tail_expert_route=torch.ones(2))
        self.assertTrue(torch.equal(a,self.model.sample(other,noise,3)))
        self.assertEqual(a.shape,(2,168,24))

    def test_masked_labels_receive_no_structural_supervision(self):
        clean=torch.rand(2,168,24,requires_grad=True);valid=torch.ones_like(clean);valid[:,17,0]=0
        pred=torch.rand_like(clean);f=torch.zeros_like(clean)
        parts=self.model.structure_losses(pred,clean,valid,f)
        sum(v.mean() for v in parts.values()).backward()
        self.assertEqual(float(clean.grad[:,17,0].abs().max()),0.)

    @unittest.skipUnless(torch.cuda.is_available(),"target CUDA/AMP NOT RUN on CPU")
    def test_cuda_amp_training(self):
        self.model.cuda().train();batch={k:v.cuda() for k,v in self.batch.items()}
        with torch.amp.autocast("cuda"):
            loss,_=self.model(batch,torch.tensor([50,250],device="cuda"))
        loss.backward()
        for k,p in self.model.named_parameters():
            self.assertIsNotNone(p.grad,k);self.assertTrue(torch.isfinite(p.grad).all(),k)


if __name__=="__main__":unittest.main()
