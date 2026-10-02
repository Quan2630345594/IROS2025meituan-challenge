import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class ModelTests(unittest.TestCase):
    def test_missing_camera_and_gradient(self):
        from bev.model import FusionModel
        model = FusionModel(classes=14)
        image = torch.rand(1, 3, 64, 64, requires_grad=True)
        pillars = torch.rand(1, 4, 16, 16)
        output = model(image, pillars, np.empty((0, 2)), [], [])
        self.assertEqual(tuple(output['heatmap'].shape), (1, 14, 16, 16))
        self.assertFalse(output['camera_mask'].any())
        output = model(image, pillars, [[12, 12], [16, 16]], [1, 1], [2, 2])
        self.assertEqual(int(output['camera_mask'].sum()), 1)
        output['fused'].sum().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
