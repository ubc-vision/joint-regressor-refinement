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

from renderer import Renderer


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Pose_Refiner(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self):
        super(Pose_Refiner, self).__init__()

        self.num_inputs = 512
        self.num_joints = 24*6

        self.resnet = getattr(models, "resnet18")(
            pretrained=True, norm_layer=FrozenBatchNorm2d)
        # self.resnet = getattr(models, "resnet50")(pretrained=True)

        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-1])

        self.linears = nn.Sequential(
            nn.Linear(self.num_inputs, 512),
            nn.ReLU(),
            Residual_Block(512),
            Residual_Block(512),
        )

        # num_features = 512+144+10+3
        num_features = 512

        self.final_position_layer = nn.Linear(num_features, 144)
        self.position_layer = nn.Sequential(
            Residual_Block(num_features),
            Residual_Block(num_features),
            self.final_position_layer
        )
        self.final_shape_layer = nn.Linear(num_features, 10)
        self.shape_layer = nn.Sequential(
            Residual_Block(num_features),
            Residual_Block(num_features),
            self.final_shape_layer
        )
        self.final_cam_layer = nn.Linear(num_features, 3)
        self.cam_layer = nn.Sequential(
            Residual_Block(num_features),
            Residual_Block(num_features),
            self.final_cam_layer
        )

        # nn.init.xavier_uniform_(self.final_position_layer.weight, gain=0.01)
        # nn.init.xavier_uniform_(self.final_shape_layer.weight, gain=0.01)
        # nn.init.xavier_uniform_(self.final_cam_layer.weight, gain=0.01)

        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.img_renderer = Renderer(subset=False)

    def forward(self, batch):

        # rendered_image = self.img_renderer(batch)

        # final_image = .5*rendered_image[:, :3] + .5*batch['image']

        # final_image = self.normalize(final_image)

        final_image = torch.zeros(
            (batch["image"].shape[0], 3, 224, 224)).to(args.device)

        output = self.resnet(final_image)

        output = output.reshape(-1, self.num_inputs)

        output = self.linears(output)

        # output = torch.cat(
        #     [output, batch["orient"].reshape(-1, 1*6), batch["pose"].reshape(-1, 23*6), batch["betas"], batch["cam"]], dim=1)

        est_betas = self.shape_layer(output)/100
        # print("est_betas")
        # print(est_betas[0])
        est_betas += batch["betas"]

        
        est_cam = self.cam_layer(output)/100
        # print("est_cam") 
        # print(est_cam[0])
        est_cam += batch["cam"]

        est_pose = self.position_layer(output)/100
        est_pose = est_pose.reshape(-1, 24, 6)
        # print("est_pose") 
        # print(est_pose[0, 0])
        est_pose += torch.cat([batch["orient"], batch["pose"]], dim=1)

        return est_pose, est_betas, est_cam


class Residual_Block(nn.Module):
    def __init__(self, num_inputs):
        super(Residual_Block, self).__init__()

        self.linears = nn.Sequential(
            nn.Linear(num_inputs, num_inputs),
            nn.BatchNorm1d(num_inputs),
            nn.ReLU(),
            nn.Linear(num_inputs, num_inputs),
            nn.BatchNorm1d(num_inputs),
        )

    def forward(self, x):

        residual = x
        x = self.linears(x)
        x += residual
        x = nn.ReLU()(x)
        return x
