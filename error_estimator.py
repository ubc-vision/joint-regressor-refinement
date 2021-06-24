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
from mesh_renderer import Mesh_Renderer


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


class Error_Estimator(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self):
        super(Error_Estimator, self).__init__()

        self.num_features = 512
        self.num_joints = 17

        self.resnet = getattr(models, "resnet18")(
            pretrained=True, norm_layer=FrozenBatchNorm2d)
        # self.resnet = getattr(models, "resnet50")(pretrained=True)

        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-1])

        self.linears = nn.Sequential(
            Residual_Block(self.num_features),
            Residual_Block(self.num_features),
            nn.Linear(self.num_features, 17)
        )

        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.img_renderer = Mesh_Renderer()

    def forward(self, batch):

        rendered_image = self.img_renderer(batch)

        final_image = .5 * \
            rendered_image[:, 3:].expand(
                batch['image'].shape) + .5*batch['image']

        # import matplotlib.pyplot as plt
        # plt.imshow(utils.torch_img_to_np_img(final_image)[0])
        # itern = batch["iteration"]
        # plt.savefig(f"output/img_{itern}.png")
        # plt.close()

        final_image = self.normalize(final_image)

        # final_image = torch.zeros(
        #     (batch["image"].shape[0], 3, 224, 224)).to(args.device)

        output = self.resnet(final_image)

        output = output.reshape(-1, self.num_features)

        output = self.linears(output)

        return output


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
