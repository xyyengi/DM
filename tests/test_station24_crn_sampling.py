import unittest

import torch
import torch.nn as nn

from src.models.station_conditioned_diffusion import (
    StationGaussianDiffusion,
    keyed_ancestral_noise,
)


class ZeroDenoiser(nn.Module):
    use_jstd_tail = False
    use_joint_multiresidual_tail = False
    lightweight_v2 = False

    def forward(self, noisy, timestep, forecast, calendar, lead, **kwargs):
        return torch.zeros_like(noisy)


class Station24CRNSamplingTests(unittest.TestCase):
    def setUp(self):
        self.diffusion = StationGaussianDiffusion(
            ZeroDenoiser(), num_steps=4, beta_start=1e-4, beta_end=0.02
        )
        self.forecast = torch.zeros(2, 3, 8)
        self.calendar = torch.zeros(2, 8, 1)
        self.lead = torch.zeros(2, 8, 1)
        self.issue_indices = torch.tensor([7, 11], dtype=torch.long)

    def test_keyed_noise_replays_and_ignores_global_rng(self):
        kwargs = dict(
            forecast=self.forecast,
            calendar=self.calendar,
            lead=self.lead,
            n_samples=5,
            ancestral_noise_seed=271828,
            ancestral_issue_indices=self.issue_indices,
            ancestral_member_block_start=400,
        )
        first = self.diffusion.sample(**kwargs)
        torch.manual_seed(999)
        _ = torch.randn(1000)
        second = self.diffusion.sample(**kwargs)
        self.assertTrue(torch.equal(first, second))

    def test_different_issue_stage_and_member_block_have_distinct_noise(self):
        common = dict(
            base_seed=314159,
            members=4,
            stations=3,
            length=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        reference = keyed_ancestral_noise(
            issue_indices=torch.tensor([2]), member_block_start=400, stage=4, **common
        )
        replay = keyed_ancestral_noise(
            issue_indices=torch.tensor([2]), member_block_start=400, stage=4, **common
        )
        changed_issue = keyed_ancestral_noise(
            issue_indices=torch.tensor([3]), member_block_start=400, stage=4, **common
        )
        changed_stage = keyed_ancestral_noise(
            issue_indices=torch.tensor([2]), member_block_start=400, stage=3, **common
        )
        changed_block = keyed_ancestral_noise(
            issue_indices=torch.tensor([2]), member_block_start=300, stage=4, **common
        )
        self.assertTrue(torch.equal(reference, replay))
        self.assertFalse(torch.equal(reference, changed_issue))
        self.assertFalse(torch.equal(reference, changed_stage))
        self.assertFalse(torch.equal(reference, changed_block))

    def test_keyed_noise_is_invariant_to_runtime_chunk_partition(self):
        common = dict(
            base_seed=314159,
            issue_indices=torch.tensor([2]),
            stage=217,
            stations=3,
            length=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        whole = keyed_ancestral_noise(
            member_block_start=397, members=11, **common
        )
        partitioned = torch.cat(
            [
                keyed_ancestral_noise(
                    member_block_start=397, members=2, **common
                ),
                keyed_ancestral_noise(
                    member_block_start=399, members=6, **common
                ),
                keyed_ancestral_noise(
                    member_block_start=405, members=3, **common
                ),
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(whole, partitioned))

    def test_keyed_noise_is_invariant_to_issue_batch_partition(self):
        common = dict(
            base_seed=271828,
            member_block_start=400,
            stage=99,
            members=4,
            stations=3,
            length=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        together = keyed_ancestral_noise(
            issue_indices=torch.tensor([7, 11]), **common
        ).reshape(2, 4, 3, 8)
        separate = torch.stack(
            [
                keyed_ancestral_noise(
                    issue_indices=torch.tensor([issue]), **common
                )
                for issue in (7, 11)
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(together, separate))

    def test_explicit_noise_provider_replays_keyed_full_path(self):
        common = dict(
            base_seed=161803,
            issue_indices=self.issue_indices,
            member_block_start=400,
            members=5,
            stations=3,
            length=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        initial = keyed_ancestral_noise(stage=4, **common)

        def provider(step):
            return keyed_ancestral_noise(stage=step, **common)

        keyed = self.diffusion.sample(
            self.forecast,
            self.calendar,
            self.lead,
            n_samples=5,
            ancestral_noise_seed=161803,
            ancestral_issue_indices=self.issue_indices,
            ancestral_member_block_start=400,
        )
        explicit = self.diffusion.sample(
            self.forecast,
            self.calendar,
            self.lead,
            n_samples=5,
            ancestral_initial_noise=initial,
            ancestral_noise_provider=provider,
        )
        self.assertTrue(torch.equal(keyed, explicit))

    def test_old_and_explicit_reverse_step_are_numerically_identical(self):
        noisy = torch.randn(6, 3, 8)
        timestep = torch.full((6,), 2, dtype=torch.long)
        forecast = torch.zeros_like(noisy)
        calendar = torch.zeros(6, 8, 1)
        lead = torch.zeros(6, 8, 1)
        torch.manual_seed(20261014)
        old = self.diffusion.denoise_step(
            noisy, timestep, forecast, calendar, lead
        )
        torch.manual_seed(20261014)
        injected = self.diffusion.denoise_step(
            noisy,
            timestep,
            forecast,
            calendar,
            lead,
            ancestral_random_noise=torch.randn_like(noisy),
        )
        self.assertTrue(torch.equal(old, injected))

    def test_default_sampling_keeps_historical_global_rng_semantics(self):
        torch.manual_seed(1234)
        first = self.diffusion.sample(
            self.forecast, self.calendar, self.lead, n_samples=3
        )
        torch.manual_seed(1234)
        second = self.diffusion.sample(
            self.forecast, self.calendar, self.lead, n_samples=3
        )
        self.assertTrue(torch.equal(first, second))

    def test_keyed_noise_is_standard_normal_with_reasonable_tolerance(self):
        values = keyed_ancestral_noise(
            base_seed=161803,
            issue_indices=torch.arange(8),
            member_block_start=0,
            stage=4,
            members=64,
            stations=6,
            length=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertLess(abs(float(values.mean())), 0.02)
        self.assertLess(abs(float(values.std()) - 1.0), 0.02)


if __name__ == "__main__":
    unittest.main()
