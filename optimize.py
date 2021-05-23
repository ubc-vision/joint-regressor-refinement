from args import args
import wandb
import torch
from torch import nn, optim

from tqdm import tqdm

# from render_model import Render_Model
# from pose_estimator import Pose_Estimator
# from pose_refiner import Pose_Refiner
from discriminator import Discriminator
from renderer import Renderer
from mesh_renderer import Mesh_Renderer
# from pose_refiner_transformer import Pose_Refiner_Transformer

# from train import find_joints, move_pelvis

from pytorch3d.renderer import PerspectiveCameras

from data import load_data, data_set
# from visualizer import draw_gradients

import torchvision
from torchvision import transforms

# from pose_refiner import render_batch

from SPIN.models import hmr, SMPL
import SPIN.config as config

import numpy as np

import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, AlphaCompositor, PointsRenderer

from eval_utils import batch_compute_similarity_transform_torch

from SPIN.utils.geometry import rot6d_to_rotmat

from utils import utils

import os
import imageio

from torch.nn import functional as F


def find_joints(smpl, shape, orient, pose, J_regressor):

    J_regressor_batch = nn.ReLU()(J_regressor)
    J_regressor_batch = J_regressor_batch / torch.sum(J_regressor_batch, dim=1).unsqueeze(
        1).expand(J_regressor_batch.shape)

    pred_vertices = smpl(global_orient=orient, body_pose=pose,
                         betas=shape, pose2rot=False).vertices
    J_regressor_batch = J_regressor_batch[None, :].expand(
        pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
    pred_joints = torch.matmul(J_regressor_batch, pred_vertices)

    return pred_joints


def move_pelvis(j3ds):
    # move the hip location of gt to estimated
    pelvis = j3ds[:, [0], :].clone()

    j3ds_clone = j3ds.clone()

    j3ds_clone -= pelvis

    return j3ds_clone


def optimize_pose_refiner():

    spin_model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
    checkpoint = torch.load(
        "SPIN/data/model_checkpoint.pt", map_location=args.device)
    spin_model.load_state_dict(checkpoint['model'], strict=False)
    spin_model.eval()

    smpl = SMPL(
        '{}'.format("SPIN/data/smpl"),
        batch_size=1,
    ).to(args.device)

    maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True).to(args.device)
    maskrcnn.eval()

    J_regressor = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

    J_regressor_retrained = J_regressor.clone()
    J_regressor_retrained.requires_grad = True

    raster_settings = PointsRasterizationSettings(
        image_size=224,
        radius=0.04,
        points_per_pixel=10
    )
    raster_settings_img = PointsRasterizationSettings(
        image_size=224,
        radius=0.005,
        points_per_pixel=10
    )

    img_renderer = Renderer()
    # img_renderer = Mesh_Renderer()
    # img_renderer = nn.DataParallel(img_renderer)

    pose_discriminator = Discriminator().to(args.device)
    # checkpoint = torch.load(
    #     "models/pose_discriminator_epoch_0.pt", map_location=args.device)
    # pose_discriminator.load_state_dict(checkpoint)
    # pose_discriminator.eval()
    pose_discriminator.train()

    disc_optimizer = optim.Adam(
        pose_discriminator.parameters(), lr=args.disc_learning_rate)

    J_Regressor_optimizer = optim.Adam(
        [J_regressor_retrained], lr=1e-6)

    loss_function = nn.MSELoss()

    data = data_set("train")
    val_data = data_set("validation")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=True, drop_last=False)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for epoch in range(args.train_epochs):

        iterator = iter(loader)
        val_iterator = iter(val_loader)

        for iteration in tqdm(range(len(loader))):

            try:
                batch = next(iterator)
            except:
                import time
                time.sleep(1)
                print("problem loading batch")
                continue

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations"):
                    batch[item] = batch[item].to(args.device).float()

            batch['gt_j3d'] = move_pelvis(batch['gt_j3d'])

            spin_image = normalize(batch['image'])

            with torch.no_grad():
                spin_pred_pose, spin_pred_betas, pred_camera = spin_model(
                    spin_image)

            # pred_cam_t = torch.stack([-2*pred_camera[:, 1],
            #                           -2*pred_camera[:, 2],
            #                           2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
            # batch["cam"] = pred_cam_t
            batch["cam"] = batch["gt_translation"]

            pred_rotmat = rot6d_to_rotmat(spin_pred_pose).view(-1, 24, 3, 3)

            batch["pose"] = spin_pred_pose[:, 1:].clone()
            batch["orient"] = spin_pred_pose[:, 0].unsqueeze(1).clone()
            batch["betas"] = spin_pred_betas.clone()

            optimize_parameters = [batch["pose"],
                                   batch["orient"], batch["betas"], batch["cam"]]

            for item in optimize_parameters:
                item.requires_grad = True

            this_batch_optimizer = optim.Adam(
                optimize_parameters, lr=1e-2)

            import time

            for i in range(100):

                this_time = time.time()

                pred_rotmat_orient = rot6d_to_rotmat(
                    batch['orient'].reshape(-1, 6)).view(-1, 1, 3, 3)

                pred_rotmat_pose = rot6d_to_rotmat(
                    batch['pose'].reshape(-1, 6)).view(-1, 23, 3, 3)

                pred_joints = find_joints(
                    smpl, batch["betas"], pred_rotmat_orient, pred_rotmat_pose, J_regressor)

                joint_loss = loss_function(move_pelvis(
                    pred_joints), batch['gt_j3d']/1000)

                rendered_silhouette = img_renderer(
                    batch, smpl, raster_settings)
                # rendered_silhouette = img_renderer(
                #     batch, smpl)

                rendered_silhouette = rendered_silhouette[:, 3].unsqueeze(1)

                silhouette_loss = loss_function(
                    rendered_silhouette[batch["valid"]], batch["mask_rcnn"][batch["valid"]])

                pred_disc = pose_discriminator(
                    torch.cat([batch['orient'], batch['pose']], dim=1))

                discriminated_loss = loss_function(pred_disc, torch.ones(
                    pred_disc.shape).to(args.device))

                opt_loss = silhouette_loss*100+joint_loss*10000+discriminated_loss*10

                this_batch_optimizer.zero_grad()
                opt_loss.backward()
                this_batch_optimizer.step()

            pred_gt = pose_discriminator(spin_pred_pose)
            pred_disc = pose_discriminator(
                torch.cat([batch['orient'], batch['pose']], dim=1).detach())
            discriminator_loss = loss_function(pred_disc, torch.zeros(
                pred_disc.shape).to(args.device))+loss_function(pred_gt, torch.ones(
                    pred_disc.shape).to(args.device))
            disc_optimizer.zero_grad()
            discriminator_loss.backward()
            disc_optimizer.step()

            # get the joints from the joint regressor retrained
            # get error to gt
            # update j_regressor
            # relu and take norm
            pred_joints = find_joints(
                smpl, batch["betas"].detach(), pred_rotmat_orient.detach(), pred_rotmat_pose.detach(), J_regressor_retrained)
            j_regressor_error = loss_function(move_pelvis(
                pred_joints), batch['gt_j3d']/1000)
            J_Regressor_optimizer.zero_grad()
            j_regressor_error.backward()
            J_Regressor_optimizer.step()

            if(args.wandb_log):
                wandb.log(
                    {
                        "silhouette_loss": silhouette_loss.item(),
                        "joint_loss": joint_loss.item(),
                        "discriminated_loss": discriminated_loss.item(),
                        "discriminator_loss": discriminator_loss.item(),
                        "j_regressor_error": j_regressor_error.item(),
                    })

            # rendered_img = img_renderer(
            #     batch, smpl, raster_settings_img, subset=False, colored=False)

            # drawing = (rendered_img[:, 3].unsqueeze(1).expand(
            #     batch['image'].shape)+batch['image']*.5)

            # blt = utils.torch_img_to_np_img(drawing)

            # import matplotlib.pyplot as plt
            # from matplotlib.patches import Circle
            # for i in range(blt.shape[0]):

            #     plt.imshow(utils.torch_img_to_np_img(drawing)[i])

            #     ax = plt.gca()

            #     plt.savefig(
            #         f"output/refining_image_{i:03d}_2.png", dpi=300)
            #     plt.close()

            #     plt.imshow(utils.torch_img_to_np_img(rendered_silhouette)[i])

            #     ax = plt.gca()

            #     plt.savefig(
            #         f"output/refining_image_{i:03d}_1.png", dpi=300)
            #     plt.close()

            #     plt.imshow(utils.torch_img_to_np_img(batch["mask_rcnn"])[i])

            #     ax = plt.gca()

            #     plt.savefig(
            #         f"output/refining_image_{i:03d}_0.png", dpi=300)
            #     plt.close()

            # exit()

            if(iteration % 100 == 0):

                print("saving model and regressor")

                torch.save(pose_discriminator.state_dict(),
                           f"models/pose_discriminator_epoch_{epoch}.pt")
                np.save("models/retrained_J_Regressor.npy",
                        J_regressor_retrained.cpu().detach().numpy())

        # TODO reimplement
        # torch.save(pose_refiner.state_dict(),
        #            f"models/pose_refiner_epoch_{epoch}.pt")


# def render_batch(batch, smpl, raster_settings, colored=True):
#     focal_length = torch.stack(
#         [batch['intrinsics'][:, 0, 0]/224, batch['intrinsics'][:, 1, 1]/224], dim=1).to(args.device)
#     principal_point = torch.stack(
#         [batch['intrinsics'][:, 0, 2]/-112+1, batch['intrinsics'][:, 1, 2]/-112+1], dim=1)
#     # focal_length = torch.ones(
#     #     batch["image"].shape[0], 2).to(args.device)*5000/224
#     # principal_point = torch.zeros(batch["image"].shape[0], 2).to(args.device)

#     pose = rot6d_to_rotmat(batch['pose'].reshape(-1, 6)).reshape(-1, 23, 3, 3)
#     orient = rot6d_to_rotmat(
#         batch['orient'].reshape(-1, 6)).reshape(-1, 1, 3, 3)

#     point_cloud = smpl(betas=batch['betas'], body_pose=pose,
#                        global_orient=orient, pose2rot=False).vertices

#     cameras = PerspectiveCameras(device=args.device, T=batch['cam'],
#                                  focal_length=focal_length, principal_point=principal_point)

#     image_size = torch.tensor([224, 224]).unsqueeze(
#         0).expand(batch['intrinsics'].shape[0], 2).to(args.device)

#     feat = torch.ones(
#         point_cloud.shape[0], point_cloud.shape[1], 4).to(args.device)

#     point_cloud[:, :, 1] *= -1
#     point_cloud[:, :, 0] *= -1
#     point_cloud *= 2

#     pred_joints_2d = cameras.transform_points_screen(
#         point_cloud, image_size)

#     this_point_cloud = Pointclouds(points=point_cloud, features=feat)

#     rasterizer = PointsRasterizer(
#         cameras=cameras, raster_settings=raster_settings)

#     renderer = PointsRenderer(
#         rasterizer=rasterizer,
#         compositor=AlphaCompositor()
#     )
#     # renderer = nn.DataParallel(renderer)

#     rendered_image = renderer(this_point_cloud)

#     print("rendered_image.shape")
#     print(rendered_image.shape)

#     # final_image = (rendered_image.permute(0, 3, 1, 2)
#     #                [:, :3]*.5+batch['image']*.5)

#     # final_image = normalize(final_image)

#     return rendered_image.permute(0, 3, 1, 2)
