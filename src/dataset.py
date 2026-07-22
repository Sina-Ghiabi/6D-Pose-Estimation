"""
PoseDataset: loads rgb/depth/pose per frame for one LINEMOD class and computes the
9 per-keypoint radius maps on-the-fly (from depth + mask + pose + Outside9.npy
keypoints) instead of reading precomputed `Out_pt*_dm/*.npy` files. This keeps
preprocessed disk usage to the raw sensor data only -- radius maps never touch disk.
"""

import os
import struct

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from PIL import Image
from numba import jit, prange
from scipy.spatial.transform import Rotation as SciRotation

# Standard LINEMOD camera intrinsics (same values used throughout this project's
# preprocessing and visualization notebooks).
CAMERA_K = np.array([
    [572.4114, 0.0, 325.2611],
    [0.0, 573.57043, 242.04899],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
_FX, _FY = CAMERA_K[0, 0], CAMERA_K[1, 1]
_CX, _CY = CAMERA_K[0, 2], CAMERA_K[1, 2]


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_radius_map(depth, kp_cam, fx, fy, cx, cy):
    """
    Per-pixel 3D distance to a single camera-space keypoint.
    depth  : (H, W) float64, depth in mm
    kp_cam : (3,)   float64, keypoint in camera space (mm)
    returns: (H, W) float64, distances in mm (0 for background)
    """
    H, W = depth.shape
    result = np.zeros((H, W), dtype=np.float64)
    for v in prange(H):
        for u in range(W):
            z = depth[v, u]
            if z == 0.0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            dx = x - kp_cam[0]
            dy = y - kp_cam[1]
            dz = z - kp_cam[2]
            result[v, u] = (dx * dx + dy * dy + dz * dz) ** 0.5
    return result


def read_depth_dpt(path: str) -> np.ndarray:
    """Read the binary `.dpt` format: [uint32 H][uint32 W][uint16 x H*W, mm]."""
    with open(path, 'rb') as f:
        h, w = struct.unpack('II', f.read(8))
        data = np.frombuffer(f.read(h * w * 2), dtype=np.uint16)
    return data.reshape(h, w).astype(np.float32)


def pose_matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    """(3,4) or (4,4) [R|t] (t in metres) -> (7,) [tx,ty,tz,qx,qy,qz,qw]."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        R_mat, t = matrix[:3, :3], matrix[:3, 3]
    else:
        R_mat, t = matrix[:, :3], matrix[:, 3]
    quat = SciRotation.from_matrix(R_mat).as_quat()  # (x, y, z, w)
    return np.concatenate([t, quat]).astype(np.float32)


def safe_collate(batch):
    """Drop samples that failed to load (returned None from __getitem__)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


class AdvancedAugmentation:
    """Colour jitter + small random rotation applied to rgb/depth/mask together."""

    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
        )

    def __call__(self, rgb, depth, mask):
        rgb = self.color_jitter(rgb)

        angle = np.random.uniform(-10, 10)
        rgb = transforms.functional.rotate(rgb, angle, expand=False)
        depth = transforms.functional.rotate(depth, angle, expand=False)
        mask = transforms.functional.rotate(mask, angle, expand=False)

        depth_np = np.array(depth).astype(np.float32)
        noise = np.random.normal(scale=0.01, size=depth_np.shape).astype(np.float32)
        depth = Image.fromarray(depth_np + noise, mode='F')

        return rgb, depth, mask


class PoseDataset(Dataset):
    """
    One LINEMOD class, one split ('train'/'val'/'test'). Each sample:
      - rgb          : (3, H, W) float, ImageNet-normalised
      - depth        : (1, H, W) float, metres
      - pose         : (7,) float, [tx,ty,tz,qx,qy,qz,qw]
      - radius_maps  : (num_radius_pts, H, W) float, metres -- computed on-the-fly
    """

    MEAN_RGB = [0.485, 0.456, 0.406]
    STD_RGB = [0.229, 0.224, 0.225]

    def __init__(self, obj_dir: str, split: str = 'train',
                 num_radius_pts: int = 9, augment: bool = False):
        self.obj_dir = obj_dir
        self.rgb_dir = os.path.join(obj_dir, 'rgb')
        self.depth_dir = os.path.join(obj_dir, 'depth')
        self.mask_dir = os.path.join(obj_dir, 'mask')
        self.pose_dir = os.path.join(obj_dir, 'pose')
        self.num_radius_pts = num_radius_pts

        split_path = os.path.join(obj_dir, 'Split', f'{split}.txt')
        if not os.path.isdir(obj_dir) or not os.path.isfile(split_path):
            raise FileNotFoundError(f'Split file not found: {split_path}')
        with open(split_path) as f:
            self.ids = [line.strip() for line in f if line.strip()]
        if not self.ids:
            raise FileNotFoundError(f'Split file is empty: {split_path}')

        kp_path = os.path.join(obj_dir, 'Outside9.npy')
        if not os.path.isfile(kp_path):
            raise FileNotFoundError(f'Outside9.npy not found: {kp_path}')
        self.keypoints = np.load(kp_path)[:num_radius_pts]  # (K, 3) mm

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN_RGB, std=self.STD_RGB),
        ])
        self.augmentation = AdvancedAugmentation() if augment else None

    def __len__(self):
        return len(self.ids)

    def _compute_radius_maps(self, depth_mm, mask, pose_matrix):
        depth64 = depth_mm.astype(np.float64).copy()
        depth64[mask == 0] = 0.0

        RT = np.asarray(pose_matrix, dtype=np.float64)
        if RT.shape == (4, 4):
            RT = RT[:3, :]
        RT_mm = RT.copy()
        RT_mm[:, 3] = RT[:, 3] * 1000.0  # metres -> mm

        maps = np.zeros((self.num_radius_pts,) + depth_mm.shape, dtype=np.float32)
        for i, kp in enumerate(self.keypoints):
            kp_cam = RT_mm[:, :3] @ kp + RT_mm[:, 3]
            rmap = compute_radius_map(depth64, kp_cam, _FX, _FY, _CX, _CY)
            maps[i] = (rmap / 1000.0).astype(np.float32)  # mm -> m
        return maps

    def __getitem__(self, idx):
        base = self.ids[idx]
        try:
            rgb_bgr = cv2.imread(os.path.join(self.rgb_dir, f'{base}.png'))
            rgb_img = Image.fromarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))

            depth_mm = read_depth_dpt(os.path.join(self.depth_dir, f'{base}.dpt'))
            mask = cv2.imread(os.path.join(self.mask_dir, f'{base}.png'), cv2.IMREAD_GRAYSCALE)

            pose_matrix = np.load(os.path.join(self.pose_dir, f'pose{base}.npy'))
            pose_vec = pose_matrix_to_vector(pose_matrix)

            # Radius maps are computed from the *original* (un-augmented) depth/mask/pose.
            radius_maps = self._compute_radius_maps(depth_mm, mask, pose_matrix)

            depth_img = Image.fromarray((depth_mm / 1000.0).astype(np.float32), mode='F')
            mask_img = Image.fromarray(mask, mode='L')

            if self.augmentation is not None:
                rgb_img, depth_img, mask_img = self.augmentation(rgb_img, depth_img, mask_img)

            rgb_t = self.rgb_transform(rgb_img)
            depth_t = torch.from_numpy(np.array(depth_img, dtype=np.float32)).unsqueeze(0)

            return {
                'rgb': rgb_t.float(),
                'depth': depth_t.float(),
                'pose': torch.from_numpy(pose_vec).float(),
                'radius_maps': torch.from_numpy(radius_maps).float(),
            }
        except Exception:
            return None
