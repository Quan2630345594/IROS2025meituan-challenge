"""Trainable compact reference: FPN + sparse lifting + gated fusion + heads.

No pretrained weights, target assignment, loss implementation or box decoding
are included. Raw outputs MUST NOT be connected to navigation.
"""
import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1),
                         nn.GroupNorm(8, cout), nn.SiLU())


class ImageFPN(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.stem = nn.Sequential(block(3, 32, 2), block(32, 64, 2))
        self.down3 = block(64, 128, 2)
        self.down4 = block(128, 256, 2)
        self.lateral = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in (64, 128, 256)])

    def forward(self, image):
        c2 = self.stem(image)
        c3 = self.down3(c2)
        c4 = self.down4(c3)
        p4 = self.lateral[2](c4)
        p3 = self.lateral[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        return p2, p3, p4


def lift(feature, uv, rows, cols, image_hw, bev_hw):
    """B=1 reference, differentiable feature sampling and mean scatter.

    uv is measured in original rectified pixels; no letterbox is used here.
    The feature pixel convention is u_feature=u_image*W_feature/W_image.
    """
    if feature.shape[0] != 1:
        raise ValueError('Reference lifting supports batch size 1')
    h, w = bev_hw
    ih, iw = image_hw
    fh, fw = feature.shape[-2:]
    uv = torch.as_tensor(uv, device=feature.device, dtype=feature.dtype)
    row = torch.as_tensor(rows, device=feature.device, dtype=torch.long)
    col = torch.as_tensor(cols, device=feature.device, dtype=torch.long)
    if uv.shape != (len(row), 2) or len(col) != len(row):
        raise ValueError('Correspondence lengths mismatch')
    pixels = uv * uv.new_tensor([fw/iw, fh/ih])
    grid = (pixels + 0.5) / uv.new_tensor([fw, fh]) * 2 - 1
    samples = F.grid_sample(feature, grid.reshape(1, 1, -1, 2), align_corners=False,
                            padding_mode='border')[0, :, 0] if len(row) else feature.new_zeros((feature.shape[1], 0))
    flat = row*w+col
    summed = feature.new_zeros((feature.shape[1], h*w)).index_add(1, flat, samples)
    count = feature.new_zeros(h*w).index_add(0, flat, feature.new_ones(len(row)))
    return (summed/count.clamp_min(1)).reshape(1, -1, h, w), (count > 0).reshape(1, 1, h, w)


class FusionModel(nn.Module):
    def __init__(self, classes=14, semantic_classes=6, channels=64):
        super().__init__()
        self.image = ImageFPN(channels)
        self.lidar = nn.Sequential(block(4, channels), block(channels, channels))
        self.gate = nn.Conv2d(channels*2+1, channels, 1)
        self.refine = block(channels, channels)
        self.object_heatmap = nn.Conv2d(channels, classes, 1)
        self.object_heatmap.bias.data.fill_(-2.19)
        # offset_xy, center_z, log(length,width,height), sin(yaw), cos(yaw)
        self.box = nn.Conv2d(channels, 8, 1)
        self.semantic = nn.Conv2d(channels, semantic_classes, 1)
        self.occupancy = nn.Conv2d(channels, 3, 1)
        self.image_logits = nn.Conv2d(channels, classes, 1)

    def forward(self, image, pillars, uv, rows, cols):
        p2, p3, p4 = self.image(image)
        camera, mask = lift(p2, uv, rows, cols, image.shape[-2:], pillars.shape[-2:])
        lidar = self.lidar(pillars)
        gate = torch.sigmoid(self.gate(torch.cat([lidar, camera, mask.to(lidar.dtype)], 1)))
        fused = self.refine(lidar + gate * camera * mask)
        return {'heatmap': self.object_heatmap(fused), 'box': self.box(fused),
                'semantic': self.semantic(fused), 'occupancy': self.occupancy(fused),
                'image_logits': self.image_logits(p2), 'camera_mask': mask,
                'fused': fused}
