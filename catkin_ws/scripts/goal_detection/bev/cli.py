"""Offline preprocessing and raw network forward; see README for NPZ format."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import Grid, transform, pillar_statistics, correspondence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'config/bev.json')
    parser.add_argument('--forward', action='store_true', help='Untrained/raw network shape validation only')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    grid = Grid(**config['grid'])
    with np.load(args.input, allow_pickle=False) as data:
        image = data['image_rgb']
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('image_rgb must be rectified HxWx3 RGB uint8')
        points = data['points_lidar']
        robot = transform(points, data['T_robot_lidar'])
        pillars = pillar_statistics(robot, grid)
        ids, uv, rows, cols = correspondence(points, data['K'], data['T_camera_lidar'],
                                           data['T_robot_lidar'], image.shape[:2], grid)
    result = dict(pillars=pillars, point_indices=ids, uv=uv, rows=rows, cols=cols)
    if args.forward:
        import torch
        from .model import FusionModel
        torch.manual_seed(0)
        model = FusionModel(len(config['classes']), len(config['semantic_classes']), config['channels']).eval()
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float()/255
        with torch.no_grad():
            outputs = model(tensor, torch.from_numpy(pillars).unsqueeze(0), uv, rows, cols)
        result.update({name: value.cpu().numpy() for name, value in outputs.items()})
        print('UNTRAINED outputs: shape validation only, no usable detections')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(json.dumps({name: list(value.shape) for name, value in result.items()}, indent=2))


if __name__ == '__main__':
    main()
