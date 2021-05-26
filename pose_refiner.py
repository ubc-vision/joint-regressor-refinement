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

        self.conv1 = nn.Conv2d(6, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.resnet = nn.Sequential(
            *list(self.resnet.children())[3:-1])

        self.linear_1 = nn.Linear(self.num_inputs, 512)

        self.position_layer = nn.Linear(512+144+10+3, 144)
        self.shape_layer = nn.Linear(512+144+10+3, 10)
        self.cam_layer = nn.Linear(512+144+10+3, 3)

        nn.init.xavier_uniform_(self.position_layer.weight, gain=0.01)
        nn.init.xavier_uniform_(self.shape_layer.weight, gain=0.01)
        nn.init.xavier_uniform_(self.cam_layer.weight, gain=0.01)

        self.normalize = transforms.Normalize(
            (0.475, 0.475, 0.475, 0.485, 0.456, 0.406), (0.225, 0.225, 0.225, 0.229, 0.224, 0.225))

        self.img_renderer = Renderer(subset=False)

    def forward(self, batch):

        rendered_image = self.img_renderer(batch)

        # drawing = (rendered_image[:, :3]+batch['image']*.5)

        # blt = utils.torch_img_to_np_img(drawing)

        # import matplotlib.pyplot as plt
        # from matplotlib.patches import Circle
        # for i in range(blt.shape[0]):

        #     plt.imshow(utils.torch_img_to_np_img(drawing)[i])

        #     ax = plt.gca()

        #     plt.savefig(
        #         f"output/refining_image_{i:03d}_2.png", dpi=300)
        #     plt.close()

        # exit()

        final_image = torch.cat([rendered_image[:, :3], batch['image']], dim=1)

        final_image = self.normalize(final_image)

        output = self.conv1(final_image)
        output = self.bn1(output)

        output = nn.ReLU(inplace=True)(output)

        output = self.resnet(output)

        output = output.reshape(-1, self.num_inputs)

        output = self.linear_1(output)

        output = nn.ReLU(inplace=True)(output)

        output = torch.cat(
            [output, batch["orient"].reshape(-1, 1*6), batch["pose"].reshape(-1, 23*6), batch["betas"], batch["cam"]], dim=1)

        est_pose = self.position_layer(output).reshape(-1, 24, 6) + \
            torch.cat([batch["orient"], batch["pose"]], dim=1)
        est_betas = self.shape_layer(output) + batch["betas"]
        est_cam = self.cam_layer(output) + batch["cam"]

        return est_pose, est_betas, est_cam
