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


class Renderer(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self, subset, image_size=256):
        super(Renderer, self).__init__()

        self.subset = subset

        self.image_size = image_size

        if(subset):
            self.raster_settings = PointsRasterizationSettings(
                image_size=self.image_size,
                radius=0.04,
                points_per_pixel=10
            )
            self.sorted_indices = np.load("data/body_model/sorted_indices.npy")
            self.sorted_indices = torch.tensor(
                self.sorted_indices, dtype=torch.long)[:, 0].to(args.device)
        else:
            self.raster_settings = PointsRasterizationSettings(
                image_size=self.image_size,
                radius=0.02,
                points_per_pixel=10
            )

        self.smpl = SMPL(
            '{}'.format("SPIN/data/smpl"),
            batch_size=1,
        ).to(args.device)

    def forward(self, batch, point_cloud=None):

        # import time

        # start_time = time.time()

        # focal_length = torch.stack(
        #     [batch['intrinsics'][:, 0, 0]/self.image_size, batch['intrinsics'][:, 1, 1]/self.image_size], dim=1).to(args.device)
        # principal_point = torch.stack(
        #     [batch['intrinsics'][:, 0, 2]/-112+1, batch['intrinsics'][:, 1, 2]/-112+1], dim=1)
        focal_length = torch.ones(
            batch["image"].shape[0], 2).to(args.device)*5000/self.image_size
        principal_point = torch.zeros(
            batch["image"].shape[0], 2).to(args.device)

        if(point_cloud is None):

            pose = utils.rot6d_to_rotmat(
                batch['pose'].reshape(-1, 6)).reshape(-1, 23, 3, 3)
            orient = utils.rot6d_to_rotmat(
                batch['orient'].reshape(-1, 6)).reshape(-1, 1, 3, 3)

            point_cloud = self.smpl(betas=batch['betas'], body_pose=pose,
                                    global_orient=orient, pose2rot=False).vertices

        point_cloud[:, :, 1] *= -1
        point_cloud[:, :, 0] *= -1
        point_cloud *= 2

        if(self.subset):
            idx = self.sorted_indices[-700:]
            point_cloud = point_cloud[:, idx]

        cameras = PerspectiveCameras(device=args.device, T=batch['cam'],
                                     focal_length=focal_length, principal_point=principal_point)

        image_size = torch.tensor([self.image_size, self.image_size]).unsqueeze(
            0).expand(batch['image'].shape[0], 2).to(args.device)

        feat = torch.ones(
            point_cloud.shape[0], point_cloud.shape[1], 4).to(args.device)

        # pred_verts_2d = cameras.transform_points_screen(
        #     point_cloud, image_size)

        # furthest_point = torch.max(pred_verts_2d[..., 2], dim=1).values
        # closest_point = torch.min(pred_verts_2d[..., 2], dim=1).values

        # furthest_point = furthest_point.unsqueeze(
        #     -1).expand(point_cloud.shape[:-1])
        # closest_point = closest_point.unsqueeze(
        #     -1).expand(point_cloud.shape[:-1])

        # dist = (pred_verts_2d[..., 2]-closest_point) / \
        #     (furthest_point-closest_point)

        # feat[..., 0] = 1-dist
        # feat[..., 1] = dist

        this_point_cloud = Pointclouds(points=point_cloud, features=feat)

        rasterizer = PointsRasterizer(
            cameras=cameras, raster_settings=self.raster_settings)

        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )

        rendered_image = renderer(this_point_cloud)

        return rendered_image.permute(0, 3, 1, 2)


def return_2d_joints(batch, smpl, J_regressor=None, mask=None):

    # start_time = time.time()

    # focal_length = torch.stack(
    #     [batch['intrinsics'][:, 0, 0]/224, batch['intrinsics'][:, 1, 1]/224], dim=1).to(args.device)
    # principal_point = torch.stack(
    #     [batch['intrinsics'][:, 0, 2]/-112+1, batch['intrinsics'][:, 1, 2]/-112+1], dim=1)
    focal_length = torch.ones(
        batch["image"].shape[0], 2).to(args.device)*5000/224
    principal_point = torch.zeros(batch["image"].shape[0], 2).to(args.device)

    pose = utils.rot6d_to_rotmat(
        batch['pose'].reshape(-1, 6)).reshape(-1, 23, 3, 3)
    orient = utils.rot6d_to_rotmat(
        batch['orient'].reshape(-1, 6)).reshape(-1, 1, 3, 3)

    if(J_regressor is not None):
        point_cloud = utils.find_joints(
            smpl, batch['betas'], orient, pose, J_regressor, mask=mask)
    else:

        point_cloud = smpl(betas=batch['betas'], body_pose=pose,
                           global_orient=orient, pose2rot=False).vertices

    point_cloud[:, :, 1] *= -1
    point_cloud[:, :, 0] *= -1
    point_cloud *= 2

    cameras = PerspectiveCameras(device=args.device, T=batch['cam'],
                                 focal_length=focal_length, principal_point=principal_point)

    image_size = torch.tensor([224, 224]).unsqueeze(
        0).expand(batch['intrinsics'].shape[0], 2).to(args.device)

    feat = torch.ones(
        point_cloud.shape[0], point_cloud.shape[1], 4).to(args.device)

    pred_verts_2d = cameras.transform_points_screen(
        point_cloud, image_size)

    return pred_verts_2d
