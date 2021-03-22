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


class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x


class Render_Model(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self):
        super(Render_Model, self).__init__()

        self.num_inputs = 512
        self.num_joints = len(constants.GT_2_J17)
        self.num_units_per_joint = 10

        self.resnet = getattr(models, "resnet18")(norm_layer=MyGroupNorm)
        # self.resnet = nn.Sequential(
        #     *list(self.resnet.children())[:3], *list(self.resnet.children())[4:-2])
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-1])

        self.linear_operations = nn.Sequential(
            nn.Linear(self.num_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_joints*self.num_units_per_joint),
        ).to(args.device)

        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # self.spin_model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
        # checkpoint = torch.load(
        #     "SPIN/data/model_checkpoint.pt", map_location=args.device)
        # self.spin_model.load_state_dict(checkpoint['model'], strict=False)
        # self.spin_model.eval()

        self.smpl = SMPL(
            '{}'.format("SPIN/data/smpl"),
            batch_size=1,
        ).to(args.device)

        # self.J_regressor = torch.from_numpy(
        #     np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

        self.raster_settings = PointsRasterizationSettings(
            image_size=224,
            radius=0.01,
            points_per_pixel=10
        )

    def forward(self, batch):

        focal_length = torch.stack(
            [batch['intrinsics'][:, 0, 0]/224, batch['intrinsics'][:, 1, 1]/224], dim=1).to(args.device)
        principal_point = torch.stack(
            [batch['intrinsics'][:, 0, 2]/-112+1, batch['intrinsics'][:, 1, 2]/-112+1], dim=1)
        # principal_point = torch.zeros(focal_length.shape).to(args.device)

        point_cloud = self.smpl(betas=batch['pred_betas'], body_pose=batch['pose'],
                                global_orient=batch['orient'], pose2rot=False).vertices

        cameras = PerspectiveCameras(device=args.device, T=batch['estimated_translation'],
                                     focal_length=focal_length, principal_point=principal_point)

        image_size = torch.tensor([224, 224]).unsqueeze(
            0).expand(batch['intrinsics'].shape[0], 2).to(args.device)

        feat = torch.ones(
            point_cloud.shape[0], point_cloud.shape[1], 4).to(args.device)

        point_cloud[:, :, 1] *= -1
        point_cloud[:, :, 0] *= -1
        point_cloud *= 2

        pred_joints_2d = cameras.transform_points_screen(
            point_cloud, image_size)

        this_point_cloud = Pointclouds(points=point_cloud, features=feat)

        rasterizer = PointsRasterizer(
            cameras=cameras, raster_settings=self.raster_settings)

        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )

        rendered_image = renderer(this_point_cloud)

        final_image = (rendered_image.permute(0, 3, 1, 2)
                       [:, :3]*.5+batch['image']*.5)

        final_image = self.normalize(final_image)

        # blt = utils.torch_img_to_np_img(final_image)

        # import matplotlib.pyplot as plt
        # for i in range(final_image.shape[0]):

        #     im = plt.imshow(blt[i])
        #     plt.savefig(f"output/image_{i:03d}.png", dpi=300)
        #     plt.close()
        # exit()

        output = self.resnet(final_image)

        output = output.reshape(-1, self.num_inputs)

        linear_output = self.linear_operations(output)

        error_estimates = []

        for i in range(self.num_joints):
            error_estimates.append(torch.norm(
                linear_output[:, i*self.num_units_per_joint: (i+1)*self.num_units_per_joint], dim=-1).to(args.device))

        error_estimates = torch.stack(
            error_estimates, dim=-1).to(args.device)

        return error_estimates
