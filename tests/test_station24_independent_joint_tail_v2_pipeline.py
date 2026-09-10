import unittest
from pathlib import Path


class IndependentJointTailV2PipelineTests(unittest.TestCase):
    def test_pipeline_is_complete_and_uses_locked_member_budget(self):
        script = Path("run_station24_independent_joint_tail_v2_pipeline.sh").read_text(encoding="utf-8")
        self.assertIn("/root/miniconda3/envs/dm_env/bin/python", script)
        self.assertIn("audit_station24_independent_joint_tail_preflight.py", script)
        self.assertIn("--body-member-limit 400", script)
        self.assertIn("--tail-member-limit 100", script)
        self.assertIn("evaluate_station24_jstd_events.py", script)
        self.assertIn("evaluate_station24_diffusion_ts.py", script)
        self.assertIn("plot_station24_independent_tail_mixture.py", script)
        self.assertIn("summarize_station24_independent_tail_v2.py", script)
        self.assertIn("tar -czf", script)
        self.assertNotIn("sha256sum", script)
        self.assertNotIn("rm -", script)

    def test_launcher_records_log_status_and_root(self):
        script = Path("launch_station24_independent_joint_tail_v2.sh").read_text(encoding="utf-8")
        self.assertIn("nohup bash -c", script)
        self.assertIn("state=completed", script)
        self.assertIn("state=failed", script)
        self.assertIn("Monitor: tail -f", script)

    def test_resume_never_retrains_or_regenerates(self):
        script = Path("run_station24_independent_joint_tail_v2_resume.sh").read_text(encoding="utf-8")
        self.assertNotIn("train_station24.py", script)
        self.assertNotIn("generate_station24.py", script)
        self.assertIn("postprocess_resume_", script)
        self.assertIn("RESULT_SUMMARY.md", script)
        self.assertIn("tar -czf", script)


if __name__ == "__main__":
    unittest.main()
