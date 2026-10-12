import unittest
import torch
from src.models.station_joint_multiresidual_tail import JointMultiresolutionResidualTail

class TailCapacitySearchTests(unittest.TestCase):
    def assets(self):
        feat=torch.zeros(24,5);feat[:13,0]=1;feat[13:,1]=1
        return feat,torch.eye(24),torch.ones(24)
    def build(self,c,d,h):
        return JointMultiresolutionResidualTail(32,*self.assets(),channels=c,groups=8,haar_levels=h,fusion_layers=d)
    def test_reference(self):
        old=self.build(24,2,3);implicit=JointMultiresolutionResidualTail(32,*self.assets())
        self.assertEqual(sum(p.numel() for p in old.parameters()),20588)
        self.assertEqual(list(old.state_dict()),list(implicit.state_dict()))
    def test_all_candidates(self):
        hidden=torch.randn(2,24,32,168);forecast=torch.randn(2,24,168)
        for c in (16,24,32,48):
          for d in (1,2,3):
           for h in (2,3):
            m=self.build(c,d,h);off=m(hidden,forecast,route=0).correction
            self.assertTrue(torch.equal(off,torch.zeros_like(off)))
            opt=torch.optim.AdamW(m.parameters(),lr=1e-3);before={n:p.detach().clone() for n,p in m.named_parameters()}
            target=torch.randn_like(off)
            for _ in range(3):
                opt.zero_grad();out=m(hidden,forecast,route=1);(out.correction-target).square().mean().backward();opt.step()
            self.assertEqual(tuple(out.correction.shape),(2,24,168))
            changed={n for n,p in m.named_parameters() if not torch.equal(before[n],p)}
            # Haar high-pass annihilates constant fast-head biases by design;
            # require every functional branch, not mathematically null biases.
            for branch in ('hidden','fast_condition','slow_condition','fast_fuse','slow_fuse',
                           'local_fast','local_slow','system','system_fast','system_slow','system_loading'):
                self.assertTrue(any(n.startswith(branch+'.') for n in changed),(c,d,h,branch))
            low,high=m.haar.split(forecast);torch.testing.assert_close(low+high,forecast,rtol=0,atol=1e-6)
    def test_invalid_grouping(self):
        with self.assertRaises(ValueError): JointMultiresolutionResidualTail(32,*self.assets(),channels=18,groups=8)

if __name__=='__main__':unittest.main()
