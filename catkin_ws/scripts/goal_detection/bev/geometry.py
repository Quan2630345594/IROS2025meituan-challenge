"""Metric pillarization and optical-camera correspondence (NumPy reference)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    x_min: float = 0.0
    x_max: float = 50.0
    y_min: float = -25.0
    y_max: float = 25.0
    resolution: float = 0.1

    @property
    def shape(self):
        extents = np.array([self.y_max-self.y_min, self.x_max-self.x_min])
        if self.resolution <= 0 or np.any(extents <= 0):
            raise ValueError('Invalid grid extents/resolution')
        cells = extents / self.resolution
        if not np.allclose(cells, np.round(cells)):
            raise ValueError('Grid extents must be divisible by resolution')
        return tuple(np.round(cells).astype(int))


def transform(points, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite 4x4 transform')
    return np.asarray(points)[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]


def indices(points_robot, grid):
    h, w = grid.shape
    p = np.asarray(points_robot)[:, :3]
    valid = np.isfinite(p).all(axis=1)
    valid &= (p[:, 0] >= grid.x_min) & (p[:, 0] < grid.x_max)
    valid &= (p[:, 1] >= grid.y_min) & (p[:, 1] < grid.y_max)
    q = p[valid]
    cols = np.floor((q[:, 0]-grid.x_min)/grid.resolution).astype(int)
    rows = np.floor((q[:, 1]-grid.y_min)/grid.resolution).astype(int)
    return valid, np.clip(rows, 0, h-1), np.clip(cols, 0, w-1)


def pillar_statistics(points_robot, grid):
    """[log(1+count), mean_z, max_z, min_z]; zero means unobserved.

    This is a deterministic geometric baseline, not a trained PointPillars PFN.
    """
    h, w = grid.shape
    valid, rows, cols = indices(points_robot, grid)
    z = np.asarray(points_robot)[valid, 2]
    count = np.zeros((h, w), np.float32)
    total = np.zeros_like(count)
    maximum = np.full_like(count, -np.inf)
    minimum = np.full_like(count, np.inf)
    np.add.at(count, (rows, cols), 1)
    np.add.at(total, (rows, cols), z)
    np.maximum.at(maximum, (rows, cols), z)
    np.minimum.at(minimum, (rows, cols), z)
    observed = count > 0
    maximum[~observed] = minimum[~observed] = 0
    return np.stack([np.log1p(count), total/np.maximum(count, 1), maximum, minimum])


def correspondence(points_lidar, K, T_camera_lidar, T_robot_lidar, image_hw, grid):
    """Return original point indices, rectified-image pixels, BEV rows/cols.

    Camera must be optical (x right, y down, z forward). Input image is
    rectified and K belongs to that image. Nearest pixel z-buffer rejects
    farther points projecting onto the same pixel, not all surface occlusion.
    """
    height, width = image_hw
    K = np.asarray(K)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(height, width) <= 0:
        raise ValueError('Invalid intrinsics/image size')
    camera = transform(points_lidar, T_camera_lidar)
    robot = transform(points_lidar, T_robot_lidar)
    valid_grid, _, _ = indices(robot, grid)
    valid = valid_grid & np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    ids = np.flatnonzero(valid)
    projected = camera[ids] @ K.T
    uv = projected[:, :2]/projected[:, 2:3]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    ids, uv = ids[inside], uv[inside]
    pixel = np.floor(uv).astype(int)
    order = np.argsort(camera[ids, 2], kind='stable')
    _, first = np.unique(pixel[order, 1]*width+pixel[order, 0], return_index=True)
    keep = order[first]
    ids, uv = ids[keep], uv[keep]
    _, rows, cols = indices(robot[ids], grid)
    return ids, uv.astype(np.float32), rows, cols
