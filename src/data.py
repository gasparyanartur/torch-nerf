import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import pathlib as pl
from torch.utils.data import Dataset, DataLoader
import pathlib as pl
import torch
from torch import nn
from torchvision import io
from torchvision import transforms
import json

digit_pattern = re.compile(r"\d+")


def load_transforms(
    tfs_path: pl.Path,
) -> tuple[float, list[float], list[torch.FloatTensor]]:
    with open(tfs_path, "r") as f:
        transforms = json.load(f)

    cam_angle_x = float(transforms["camera_angle_x"])
    rotations = []
    transform_matrixes = []

    for frame in transforms["frames"]:
        # Assume ordered
        rotation = float(frame["rotation"])
        transform_matrix = torch.FloatTensor(frame["transform_matrix"])

        rotations.append(rotation)
        transform_matrixes.append(transform_matrix)

    return cam_angle_x, rotations, transform_matrixes


def extract_digit_from_path_name(path: pl.Path) -> int:
    match = digit_pattern.search(path.name)

    if not match:
        return None

    return int(match.group(0))


def load_img_paths(imgs_path: pl.Path):
    paths = imgs_path.iterdir()  # Ordered lexagraphically
    paths = sorted(paths, key=extract_digit_from_path_name)  # Ordered numerically
    return paths


def load_frame(
    imgs_path: pl.Path,
    rotations: list[float],
    tf_matrixes: list[torch.FloatTensor],
    idx: int,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    img_path = imgs_path[idx]
    rotation = rotations[idx]
    tf_matrix = tf_matrixes[idx]
    img = io.read_image(str(img_path), mode=io.ImageReadMode.RGB_ALPHA)

    return img, rotation, tf_matrix


def get_rays(
    H: int, W: int, focal: float, c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Ported from https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L123

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="xy",
    )
    ds = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], dim=-1
    )
    rays_d = ds @ c2w[:3, :3].T
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)

    return rays_d, rays_o


class FrameDataset(Dataset):
    def __init__(
        self,
        data_path: pl.Path,
        data_mode: str,  # 'train', 'val', 'test'
        ex_idx: int = 5,
    ) -> None:
        super().__init__()

        self.imgs_path = data_path / data_mode
        self.tfs_path = data_path / f"transforms_{data_mode}.json"

        self.cam_angle_x, self.rotations, self.transform_matrixes = load_transforms(
            self.tfs_path
        )
        self.img_paths = load_img_paths(self.imgs_path)

        self.ex_img, *_ = self[ex_idx]
        self.C, self.H, self.W = self.ex_img.shape

        self.focal = 0.5 * self.W / np.tan(0.5 * self.cam_angle_x)

    @property
    def shape(self) -> tuple[float, float]:
        return self.H, self.W

    def __len__(self) -> int:
        return len(self.transform_matrixes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, torch.Tensor]:
        img, rot, c2w = load_frame(
            self.img_paths, self.rotations, self.transform_matrixes, idx
        )
        return img, rot, c2w


class RayDataset(Dataset):
    def __init__(self, frame_dataset: FrameDataset) -> None:
        super().__init__()

        self.frame_dataset = frame_dataset

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C_r, _, c2w = self.frame_dataset[idx]
        C_r = C_r / 255.0

        focal = self.frame_dataset.focal
        H, W = self.frame_dataset.shape

        r_o, r_d = get_rays(H, W, focal, c2w)
        return r_o, r_d, C_r
