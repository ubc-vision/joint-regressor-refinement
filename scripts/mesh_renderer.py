import torch
from torch import nn

from scripts.args import args

import numpy as np

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes


# rendering components
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, TexturesVertex,
)


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

    def forward(self, batch, smpl_verts):

        batch_size = batch["image"].shape[0]

        focal_length = torch.ones(
            batch["image"].shape[0], 2).to(args.device)*5000/self.image_size
        principal_point = torch.zeros(
            batch["image"].shape[0], 2).to(args.device)

        cameras = PerspectiveCameras(device=args.device, T=batch['cam'],
                                     focal_length=focal_length, principal_point=principal_point)

        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=self.blend_params)
        )

        meshes = Meshes(
            verts=smpl_verts,
            faces=torch.stack([self.faces]*batch_size).to(args.device),
            textures=self.textures
        )

        silhouete = silhouette_renderer(
            meshes_world=meshes)

        return silhouete.permute(0, 3, 1, 2)
