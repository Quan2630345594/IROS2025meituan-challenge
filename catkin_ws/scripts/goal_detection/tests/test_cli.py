import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class CliTests(unittest.TestCase):
    def test_npz_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root/'input.npz', image_rgb=np.zeros((32, 32, 3), np.uint8),
                     points_lidar=np.array([[1, 0, 2]], dtype=np.float32),
                     K=np.array([[10, 0, 16], [0, 10, 16], [0, 0, 1]]),
                     T_camera_lidar=np.eye(4), T_robot_lidar=np.eye(4))
            (root/'config.json').write_text(json.dumps({'grid': {
                'x_min': 0, 'x_max': 4, 'y_min': -2, 'y_max': 2, 'resolution': 1}}))
            run = subprocess.run([sys.executable, '-m', 'bev.cli', '--input', str(root/'input.npz'),
                '--output', str(root/'out.npz'), '--config', str(root/'config.json')],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            with np.load(root/'out.npz') as result:
                self.assertEqual(result['pillars'].shape, (4, 4, 4))
                np.testing.assert_array_equal(result['point_indices'], [0])
                np.testing.assert_allclose(result['uv'], [[21, 16]])


if __name__ == '__main__':
    unittest.main()
