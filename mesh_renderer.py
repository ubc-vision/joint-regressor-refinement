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

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes


# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

from utils import utils

from torch.nn import functional as F


class Mesh_Renderer(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self, image_size=256):
        super(Mesh_Renderer, self).__init__()

        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        self.image_size = image_size

        # self.cameras = FoVPerspectiveCameras(device=args.device)

        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        verts, faces_idx, _ = load_obj("data/body_model/smpl_uv.obj")
        self.faces = faces_idx.verts_idx

        # self.indeces = torch.range(0, verts.shape[0]-1, 9, dtype=torch.long)
        # indices = np.arange(0, 6889, 9)

        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        self.textures = TexturesVertex(
            verts_features=verts_rgb.to(args.device))

        # meshes = Meshes(
        #     verts=[verts],
        #     faces=[self.faces],
        #     textures=self.textures
        # )

        # print(meshes.verts_normals_list())
        # print(meshes.verts_normals_list()[0].shape)

        # np.save("output/mesh.npy", verts.numpy())
        # np.save("output/normals.npy", meshes.verts_normals_list()[0].numpy())
        # exit()

        # self.faces = torch.stack([self.faces]*args.batch_size).to(args.device)

    def forward(self, batch):

        batch_size = batch["image"].shape[0]

        # focal_length = torch.stack(
        #     [batch['intrinsics'][:, 0, 0]/self.image_size, batch['intrinsics'][:, 1, 1]/self.image_size], dim=1).to(args.device)
        # principal_point = torch.stack(
        #     [batch['intrinsics'][:, 0, 2]/-112+1, batch['intrinsics'][:, 1, 2]/-112+1], dim=1)
        focal_length = torch.ones(
            batch["image"].shape[0], 2).to(args.device)*5000/self.image_size
        principal_point = torch.zeros(
            batch["image"].shape[0], 2).to(args.device)

        cameras = PerspectiveCameras(device=args.device, T=batch['cam'],
                                     focal_length=focal_length, principal_point=principal_point)

        # image_size = torch.tensor([self.image_size, self.image_size]).unsqueeze(
        #     0).expand(batch['intrinsics'].shape[0], 2).to(args.device)

        # pred_verts_2d = cameras.transform_points_screen(
        #     batch["pred_vertices"], image_size)

        # print("pred_verts_2d")
        # print(torch.max(pred_verts_2d))
        # print(torch.min(pred_verts_2d))

        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=self.blend_params)
        )

        # batch["pred_vertices"] *= 2
        # batch["pred_vertices"][..., :2] *= -1`

        meshes = Meshes(
            verts=batch["pred_vertices"],
            faces=torch.stack([self.faces]*batch_size).to(args.device),
            textures=self.textures
        )

        silhouete = silhouette_renderer(
            meshes_world=meshes)

        return silhouete.permute(0, 3, 1, 2)


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
