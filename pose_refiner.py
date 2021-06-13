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


class Pose_Refiner(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self):
        super(Pose_Refiner, self).__init__()

        self.num_inputs = 512
        self.num_joints = 24*6

        self.resnet = getattr(models, "resnet18")(pretrained=True)
        # self.resnet = getattr(models, "resnet50")(pretrained=True)

        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-1])

        self.linears = nn.Sequential(
            nn.Linear(self.num_inputs, 512),
            nn.ReLU(),
            Residual_Block(512),
            Residual_Block(512),
        )

        num_features = 512+144+10+3
        # num_features = 512

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

        rendered_image = self.img_renderer(batch)

        final_image = .5*rendered_image[:, :3] + .5*batch['image']

        final_image = self.normalize(final_image)

        # final_image = torch.zeros(
        #     (batch["image"].shape[0], 3, 224, 224)).to(args.device)

        output = self.resnet(final_image)

        output = output.reshape(-1, self.num_inputs)

        output = self.linears(output)

        output = torch.cat(
            [output, batch["orient"].reshape(-1, 1*6), batch["pose"].reshape(-1, 23*6), batch["betas"], batch["cam"]], dim=1)

        est_betas = self.shape_layer(output)/100 + batch["betas"]
        est_cam = self.cam_layer(output)/100 + batch["cam"]

        est_pose = self.position_layer(output)/100
        est_pose = est_pose.reshape(-1, 24, 6) + \
            torch.cat([batch["orient"], batch["pose"]], dim=1)

        return est_pose, est_betas, est_cam


class Residual_Block(nn.Module):
    def __init__(self, num_inputs):
        super(Residual_Block, self).__init__()

        self.linears = nn.Sequential(
            nn.Linear(num_inputs, num_inputs),
            nn.ReLU(),
            nn.Linear(num_inputs, num_inputs),
        )

    def forward(self, x):

        residual = x
        x = self.linears(x)
        x += residual
        x = nn.ReLU()(x)
        return x


class Pose_Refiner_Translation(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self):
        super(Pose_Refiner_Translation, self).__init__()

        self.num_inputs = 512
        self.num_joints = 17

        self.resnet = getattr(models, "resnet18")(pretrained=True)
        # self.resnet = getattr(models, "resnet50")(pretrained=True)

        self.conv1 = nn.Conv2d(6, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.resnet = nn.Sequential(
            *list(self.resnet.children())[3:-1])

        self.linear_1 = nn.Linear(self.num_inputs, 512)

        self.position_layer = nn.Linear(512, self.num_joints*3)

        nn.init.xavier_uniform_(self.position_layer.weight, gain=0.01)

        self.normalize = transforms.Normalize(
            (0.475, 0.475, 0.475, 0.485, 0.456, 0.406), (0.225, 0.225, 0.225, 0.229, 0.224, 0.225))

        self.img_renderer = Renderer(subset=False)

    def forward(self, batch):

        rendered_image = self.img_renderer(
            batch, point_cloud=batch["pred_vertices"])

        final_image = torch.cat([rendered_image[:, :3], batch['image']], dim=1)

        final_image = self.normalize(final_image)

        output = self.conv1(final_image)
        output = self.bn1(output)

        output = nn.ReLU(inplace=True)(output)

        output = self.resnet(output)

        output = output.reshape(-1, self.num_inputs)

        output = self.linear_1(output)

        output = nn.ReLU(inplace=True)(output)

        # concatenate original positions here?
        # output = torch.cat(
        #     [output, batch["orient"].reshape(-1, 1*6), batch["pose"].reshape(-1, 23*6), batch["betas"], batch["cam"]], dim=1)

        est_position = self.position_layer(
            output).reshape(-1, self.num_joints, 3)*.1 + batch["pred_j3d"]

        return est_position
