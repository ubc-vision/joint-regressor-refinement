import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms

from warp import perturbation_helper, sampling_helper

from args import args

import constants

from SPIN.models import hmr, SMPL
import SPIN.config as config

import numpy as np

import create_smpl_gt

import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, AlphaCompositor, PointsRenderer

from utils import utils

from torch.nn import functional as F


class Renderer(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self):
        super(Renderer, self).__init__()

        self.sorted_indices = np.load("data/body_model/sorted_indices.npy")
        self.sorted_indices = torch.tensor(
            self.sorted_indices, dtype=torch.long).to(args.device)[:, 0]

    def forward(self, batch, smpl, raster_settings, subset=True, colored=True):

        # import time

        # start_time = time.time()

        focal_length = torch.stack(
            [batch['intrinsics'][:, 0, 0]/224, batch['intrinsics'][:, 1, 1]/224], dim=1).to(args.device)
        principal_point = torch.stack(
            [batch['intrinsics'][:, 0, 2]/-112+1, batch['intrinsics'][:, 1, 2]/-112+1], dim=1)
        # focal_length = torch.ones(
        #     batch["image"].shape[0], 2).to(args.device)*5000/224
        # principal_point = torch.zeros(batch["image"].shape[0], 2).to(args.device)

        pose = rot6d_to_rotmat(
            batch['pose'].reshape(-1, 6)).reshape(-1, 23, 3, 3)
        orient = rot6d_to_rotmat(
            batch['orient'].reshape(-1, 6)).reshape(-1, 1, 3, 3)

        point_cloud = smpl(betas=batch['betas'], body_pose=pose,
                           global_orient=orient, pose2rot=False).vertices

        if(subset):
            idx = self.sorted_indices[-700:]
            point_cloud = point_cloud[:, idx]

        cameras = PerspectiveCameras(device=args.device, T=batch['cam'],
                                     focal_length=focal_length, principal_point=principal_point)

        image_size = torch.tensor([224, 224]).unsqueeze(
            0).expand(batch['intrinsics'].shape[0], 2).to(args.device)

        feat = torch.ones(
            point_cloud.shape[0], point_cloud.shape[1], 4).to(args.device)

        point_cloud[:, :, 1] *= -1
        point_cloud[:, :, 0] *= -1
        point_cloud *= 2

        # pred_joints[:, :, 1] *= -1
        # pred_joints[:, :, 0] *= -1
        # pred_joints *= 2

        # pred_joints_2d = cameras.transform_points_screen(
        #     pred_joints, image_size)

        this_point_cloud = Pointclouds(points=point_cloud, features=feat)

        rasterizer = PointsRasterizer(
            cameras=cameras, raster_settings=raster_settings)

        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )

        rendered_image = renderer(this_point_cloud)

        return rendered_image.permute(0, 3, 1, 2)


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)
