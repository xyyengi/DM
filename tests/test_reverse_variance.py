import unittest


try:
    import torch
except ImportError:  # pragma: no cover - local lightweight environments may omit torch
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ReverseVarianceTests(unittest.TestCase):
    def make_diffusion(self, variance_type):
        from diff_models_multivariate import GaussianDiffusionMultivariate

        return GaussianDiffusionMultivariate(
            model=torch.nn.Identity(),
            num_steps=20,
            beta_start=1e-4,
            beta_end=0.04,
            schedule="linear",
            reverse_variance_type=variance_type,
        )

    def test_beta_mode_preserves_legacy_variance(self):
        diffusion = self.make_diffusion("beta")

        self.assertTrue(torch.equal(diffusion.reverse_variance(7), diffusion.beta[7]))

    def test_posterior_matches_closed_form_and_is_smaller(self):
        diffusion = self.make_diffusion("posterior")
        t = 7
        expected = (
            diffusion.beta[t]
            * (1.0 - diffusion.alpha_hat[t - 1])
            / (1.0 - diffusion.alpha_hat[t])
        )

        self.assertTrue(torch.allclose(diffusion.reverse_variance(t), expected))
        self.assertLess(diffusion.reverse_variance(t).item(), diffusion.beta[t].item())

    def test_final_step_has_zero_posterior_variance(self):
        diffusion = self.make_diffusion("posterior")

        self.assertEqual(diffusion.reverse_variance(0).item(), 0.0)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reverse_variance_type"):
            self.make_diffusion("unknown")


if __name__ == "__main__":
    unittest.main()
