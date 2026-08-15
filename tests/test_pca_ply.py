import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


def load_demo():
    path = ROOT / "demo" / "12_pca_ply.py"
    spec = importlib.util.spec_from_file_location("pca_ply_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PcaPlyDemoTests(unittest.TestCase):
    def test_cli_defaults_to_wan_training_seed(self):
        demo = load_demo()

        args = demo.parse_args(
            ["--ply", "cloud.ply", "--output", "cloud_utonia_pca.ply"]
        )

        self.assertEqual(args.seed, 42)
        self.assertEqual(args.scale, 1.0)
        self.assertEqual(args.brightness, 1.2)

    @patch("torch.cuda.is_available", return_value=False)
    def test_rejects_non_cuda_hosts_before_loading_ply(self, _available):
        demo = load_demo()

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            demo.main(["--ply", "cloud.ply", "--output", "cloud_utonia_pca.ply"])


if __name__ == "__main__":
    unittest.main()
