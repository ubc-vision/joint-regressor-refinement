import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms

from warp import perturbation_helper, sampling_helper

from args import args

import constants

from SPIN.models import hmr, SMPL
import SPIN.config as config
from SPIN.utils.geometry import rot6d_to_rotmat

import numpy as np

import create_smpl_gt

import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, AlphaCompositor, PointsRenderer

from utils import utils

import sys  # nopep8
sys.path.append('/scratch/iamerich/stylegan2-pytorch')  # nopep8

from style_gan_v2 import Discriminator
from mesh_renderer import Mesh_Renderer

from SPIN.models import hmr, SMPL


class Img_Disc(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self):
        super(Img_Disc, self).__init__()
        self.img_renderer = Mesh_Renderer(image_size=256)

        self.discriminator = Discriminator(
            256, channel_multiplier=2
        ).to(args.device)

        self.smpl = SMPL(
            '{}'.format("SPIN/data/smpl"),
            batch_size=1,
        ).to(args.device)

        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406, .45), (0.229, 0.224, 0.225, 0.225))

    def forward(self, batch, pred_pose):

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 24, 3, 3)

        pred_vertices = self.smpl(global_orient=pred_rotmat[:, :1], body_pose=pred_rotmat[:, 1:],
                                  betas=batch["betas"], pose2rot=False).vertices

        batch["pred_vertices"] = pred_vertices

        rendered_image = self.img_renderer(batch)

        image = torch.cat([batch['image'], rendered_image[:, 3:]], dim=1)

        # image = (rendered_image[:, 3:].expand(
        #     batch['image'].shape)*.5+batch['image']*.5)

        # import matplotlib.pyplot as plt
        # from time import sleep

        # print("saving image")
        # plt.imshow(utils.torch_img_to_np_img(image[0]))
        # plt.savefig("output/img.png")
        # sleep(2)

        # print this image

        image = self.normalize(image)

        disc_out = self.discriminator(image)

        return disc_out
        # take the batch and render it to an image
        # take the image and discriminate it
