import unittest
import numpy as np
from bev.geometry import Grid, transform, pillar_statistics, correspondence


class GeometryTests(unittest.TestCase):
    def test_boundaries_nan_and_empty(self):
        grid = Grid(0, 2, -1, 1, 1)
        points = np.array([[0, -1, 1], [0.1, -0.9, 3], [2, 0, 5], [1, 1, 5], [np.nan, 0, 1]])
        feature = pillar_statistics(points, grid)
        self.assertEqual(feature.shape, (4, 2, 2))
        np.testing.assert_allclose(feature[:, 0, 0], [np.log(3), 2, 3, 1])
        self.assertEqual(np.count_nonzero(feature[0]), 1)
        self.assertFalse(pillar_statistics(np.empty((0, 3)), grid).any())

    def test_transform_direction(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [1, 2, 3]
        np.testing.assert_allclose(transform([[2, 3, 4]], matrix), [[3, 5, 7]])

    def test_z_buffer_and_behind_camera(self):
        grid = Grid(0, 10, -5, 5, 1)
        K = np.array([[10, 0, 20], [0, 10, 20], [0, 0, 1]])
        points = np.array([[1, 0, 2], [2, 0, 4], [1, 0, -2], [9, 0, 1]])
        ids, uv, rows, cols = correspondence(points, K, np.eye(4), np.eye(4), (40, 40), grid)
        np.testing.assert_array_equal(ids, [0])
        np.testing.assert_allclose(uv, [[25, 20]])
        np.testing.assert_array_equal(rows, [5])
        np.testing.assert_array_equal(cols, [1])

    def test_empty_projection(self):
        result = correspondence(np.empty((0, 3)), np.eye(3), np.eye(4), np.eye(4), (40, 40), Grid())
        self.assertEqual(result[1].shape, (0, 2))

    def test_invalid_grid(self):
        with self.assertRaises(ValueError):
            _ = Grid(resolution=0).shape
        with self.assertRaises(ValueError):
            _ = Grid(resolution=0.3).shape


if __name__ == '__main__':
    unittest.main()
