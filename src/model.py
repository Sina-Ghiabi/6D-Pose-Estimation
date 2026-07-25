"""
EnhancedRCVPose: dual-branch (RGB + depth) 6D pose estimation model.

Architecture: ResNet50 backbones (one per modality) -> FPN -> attention -> fusion,
then two heads: pose (3-D translation + 6-D continuous rotation, Zhou et al. 2019)
and outside9 (per-pixel radius maps for 9 keypoints, upsampled to input resolution).

The 6-D rotation representation avoids the discontinuities and double-cover
ambiguity (q vs -q) that make quaternions harder to regress directly; ground-truth
poses are still stored/consumed as [t, quaternion] everywhere outside this module
(dataset, ICP, metrics) -- only the network's raw output and its loss decode to/from
the 6-D form, via rot6d_to_matrix / quat_to_matrix below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AttentionModule(nn.Module):
    """Self-attention over spatial positions of a feature map."""

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x


class FeaturePyramidNetwork(nn.Module):
    """Top-down FPN over a list of backbone feature maps."""

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        laterals = [conv(feature) for feature, conv in zip(features, self.lateral_convs)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:]
            )
        return [conv(lateral) for lateral, conv in zip(laterals, self.fpn_convs)]


class EnhancedRCVPose(nn.Module):
    """Main model: RGB + depth -> pose (9,: 3 trans + 6-D rotation) and outside9 radius maps (9, H, W)."""

    def __init__(self, fpn_out_channels: int = 256, pose_hidden: int = 128):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        self.rgb_layer1 = nn.Sequential(*list(resnet.children())[:5])   # 256 ch
        self.rgb_layer2 = list(resnet.children())[5]                    # 512 ch
        self.rgb_layer3 = list(resnet.children())[6]                    # 1024 ch
        self.rgb_layer4 = list(resnet.children())[7]                    # 2048 ch

        resnet_depth = models.resnet50(pretrained=True)
        resnet_depth.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet_depth.conv1.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))
        self.depth_layer1 = nn.Sequential(*list(resnet_depth.children())[:5])
        self.depth_layer2 = list(resnet_depth.children())[5]
        self.depth_layer3 = list(resnet_depth.children())[6]
        self.depth_layer4 = list(resnet_depth.children())[7]

        self.rgb_fpn = FeaturePyramidNetwork([512, 1024, 2048], out_channels=fpn_out_channels)
        self.depth_fpn = FeaturePyramidNetwork([512, 1024, 2048], out_channels=fpn_out_channels)
        self.rgb_attention = AttentionModule(fpn_out_channels)
        self.depth_attention = AttentionModule(fpn_out_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_out_channels * 2, fpn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pose_head = nn.Sequential(
            nn.Linear(fpn_out_channels, pose_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(pose_hidden, 9),  # 3 translation + 6 continuous rotation params
        )

        self.outside9_head = nn.Sequential(
            nn.Conv2d(fpn_out_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 9, kernel_size=1),
        )

    def freeze_backbone(self):
        """Freeze everything except pose_head/outside9_head (warm-up stage)."""
        for name, param in self.named_parameters():
            param.requires_grad = any(k in name for k in ('pose_head', 'outside9_head'))

    def unfreeze_backbone(self):
        """Unfreeze all parameters (main training stage)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, rgb, depth):
        x1 = self.rgb_layer1(rgb)
        x2 = self.rgb_layer2(x1)
        x3 = self.rgb_layer3(x2)
        x4 = self.rgb_layer4(x3)
        rgb_fpn_features = self.rgb_fpn([x2, x3, x4])

        d1 = self.depth_layer1(depth)
        d2 = self.depth_layer2(d1)
        d3 = self.depth_layer3(d2)
        d4 = self.depth_layer4(d3)
        depth_fpn_features = self.depth_fpn([d2, d3, d4])

        rgb_attended = self.rgb_attention(rgb_fpn_features[0])
        depth_attended = self.depth_attention(depth_fpn_features[0])
        combined = torch.cat([rgb_attended, depth_attended], dim=1)
        fused = self.fusion(combined)

        pooled = self.global_pool(fused)
        pose = self.pose_head(pooled.view(pooled.size(0), -1))

        outside9 = self.outside9_head(fused)
        target_size = (rgb.shape[2], rgb.shape[3])
        outside9 = F.interpolate(outside9, size=target_size, mode='bilinear', align_corners=False)

        return pose, outside9


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 2019 continuity-friendly 6-D -> 3x3 rotation matrix. x: (B, 6)."""
    a1, a2 = x[:, 0:3], x[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3), columns b1/b2/b3


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """q: (B, 4) as (x, y, z, w) -> (B, 3, 3) rotation matrix."""
    q = F.normalize(q, dim=1)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]
    R = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x ** 2 + y ** 2),
    ], dim=1).view(B, 3, 3)
    return R


class WeightedPoseLoss(nn.Module):
    """Weighted translation MSE + geodesic rotation loss (6-D repr) + radius-map MSE."""

    def __init__(self, w_trans: float = 1.0, w_rot: float = 10.0, w_pts: float = 1.0):
        super().__init__()
        self.w_trans = w_trans
        self.w_rot = w_rot
        self.w_pts = w_pts

    def forward(self, pred_pose, target_pose, pred_outside9, target_outside9):
        trans_loss = F.mse_loss(pred_pose[:, :3], target_pose[:, :3])

        R_pred = rot6d_to_matrix(pred_pose[:, 3:9])
        R_gt   = quat_to_matrix(target_pose[:, 3:7])
        # Geodesic angle of the relative rotation R_pred^T @ R_gt: trace = 1 + 2*cos(theta).
        # Same angular quantity as the old 2*arccos(|q_pred . q_gt|) quaternion formula.
        trace = torch.diagonal(torch.bmm(R_pred.transpose(1, 2), R_gt), dim1=-2, dim2=-1).sum(-1)
        cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
        rot_loss = torch.acos(cos_theta).mean()

        pts_loss = F.mse_loss(pred_outside9, target_outside9)

        total = self.w_trans * trans_loss + self.w_rot * rot_loss + self.w_pts * pts_loss
        return total, trans_loss, rot_loss, pts_loss
