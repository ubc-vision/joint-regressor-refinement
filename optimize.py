from args import args
import wandb
import torch
from torch import nn, optim

from tqdm import tqdm

# from render_model import Render_Model
# from pose_estimator import Pose_Estimator
# from pose_refiner import Pose_Refiner
from discriminator import Discriminator
from renderer import Renderer, return_2d_joints
from mesh_renderer import Mesh_Renderer
# from pose_refiner_transformer import Pose_Refiner_Transformer

from pytorch3d.renderer import PerspectiveCameras

from data import data_set


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

from SPIN.utils.geometry import rot6d_to_rotmat

from utils import utils

import os
import imageio

from torch.nn import functional as F

import copy


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

    # J_regressor_retrained = J_regressor.clone()
    J_regressor_retrained = torch.load(
        "models/best_pose_refiner/retrained_J_Regressor.pt", map_location=args.device)
    J_regressor_retrained.requires_grad = True

    # raster_settings = PointsRasterizationSettings(
    #     image_size=224,
    #     radius=0.04,
    #     points_per_pixel=10
    # )
    # raster_settings_img = PointsRasterizationSettings(
    #     image_size=224,
    #     radius=0.005,
    #     points_per_pixel=10
    # )

    silhouette_renderer = Renderer(subset=True)
    img_renderer = Renderer(subset=False)
    # silhouette_renderer = Mesh_Renderer()
    # silhouette_renderer = nn.DataParallel(silhouette_renderer)

    pose_discriminator = Discriminator().to(args.device)
    checkpoint = torch.load(
        "models/best_pose_refiner/opt_disc.pt", map_location=args.device)
    pose_discriminator.load_state_dict(checkpoint)
    # pose_discriminator.eval()
    pose_discriminator.train()

    disc_optimizer = optim.Adam(
        pose_discriminator.parameters(), lr=args.opt_disc_learning_rate)
    checkpoint = torch.load(
        "models/best_pose_refiner/opt_disc_optim.pt", map_location=args.device)
    disc_optimizer.load_state_dict(checkpoint)

    J_Regressor_optimizer = optim.Adam(
        [J_regressor_retrained], lr=args.j_reg_lr)

    loss_function = nn.MSELoss()

    j_reg_mask = utils.find_j_reg_mask(J_regressor)

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

            batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

            spin_image = normalize(batch['image'])

            with torch.no_grad():
                spin_pred_pose, spin_pred_betas, pred_camera = spin_model(
                    spin_image)

            # pred_cam_t = torch.stack([-2*pred_camera[:, 1],
            #                           -2*pred_camera[:, 2],
            #                           2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
            # batch["cam"] = pred_cam_t

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

            # utils.render_batch(img_renderer, batch, "og")

            for i in range(100):

                pred_rotmat_orient = rot6d_to_rotmat(
                    batch['orient'].reshape(-1, 6)).view(-1, 1, 3, 3)

                pred_rotmat_pose = rot6d_to_rotmat(
                    batch['pose'].reshape(-1, 6)).view(-1, 23, 3, 3)

                pred_joints = utils.find_joints(
                    smpl, batch["betas"], pred_rotmat_orient, pred_rotmat_pose, J_regressor)

                joint_loss = loss_function(utils.move_pelvis(
                    pred_joints), batch['gt_j3d']/1000)

                rendered_silhouette = silhouette_renderer(batch)

                rendered_silhouette = rendered_silhouette[:, 3].unsqueeze(1)

                silhouette_loss = loss_function(
                    rendered_silhouette[batch["valid"]], batch["mask_rcnn"][batch["valid"]])

                pred_disc = pose_discriminator(
                    torch.cat([batch['orient'], batch['pose']], dim=1))

                discriminated_loss = loss_function(pred_disc, torch.ones(
                    pred_disc.shape).to(args.device))

                opt_loss = silhouette_loss*100+joint_loss*10000+discriminated_loss*100

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
            pred_joints = utils.find_joints(
                smpl, batch["betas"].detach(), pred_rotmat_orient.detach(), pred_rotmat_pose.detach(), J_regressor_retrained, mask=j_reg_mask)
            j_regressor_error = loss_function(utils.move_pelvis(
                pred_joints), batch['gt_j3d']/1000)
            J_Regressor_optimizer.zero_grad()
            j_regressor_error.backward()
            J_Regressor_optimizer.step()

            mpjpe_new_opt, pampjpe_new_opt = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            pred_joints = utils.find_joints(
                smpl, batch["betas"].detach(), pred_rotmat_orient.detach(), pred_rotmat_pose.detach(), J_regressor)

            mpjpe_old_opt, pampjpe_old_opt = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            if(args.wandb_log):
                wandb.log(
                    {
                        "silhouette_loss": silhouette_loss.item(),
                        "joint_loss": joint_loss.item(),
                        "discriminated_loss": discriminated_loss.item(),
                        "discriminator_loss": discriminator_loss.item(),
                        "j_regressor_error": j_regressor_error.item(),
                        "mpjpe": mpjpe_old_opt.item(),
                        "pampjpe": pampjpe_old_opt.item(),
                        "mpjpe difference": mpjpe_new_opt.item()-mpjpe_old_opt.item(),
                        "pampjpe difference": pampjpe_new_opt.item()-pampjpe_old_opt.item(),
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

            if(args.wandb_log and (iteration+1) % 100 == 0):

                print("saving model and regressor")

                torch.save(pose_discriminator.state_dict(),
                           f"models/pose_discriminator_epoch_{epoch}.pt")
                torch.save(disc_optimizer.state_dict(),
                           f"models/disc_optimizer_epoch_{epoch}.pt")

                torch.save(J_regressor_retrained,
                           "models/retrained_J_Regressor.pt")


def optimize_network():

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

    J_regressor_retrained = torch.load(
        "models/best_pose_refiner/retrained_J_Regressor.pt", map_location=args.device)
    J_regressor_retrained.requires_grad = True

    silhouette_renderer = Renderer(subset=True)
    img_renderer = Renderer(subset=False)
    # silhouette_renderer = Mesh_Renderer()
    # silhouette_renderer = nn.DataParallel(silhouette_renderer)

    J_Regressor_optimizer = optim.Adam(
        [J_regressor_retrained], lr=args.j_reg_lr)

    loss_function = nn.MSELoss()

    j_reg_mask = utils.find_j_reg_mask(J_regressor)

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

            batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

            spin_image = normalize(batch['image'])

            this_spin_model = copy.deepcopy(spin_model)
            this_spin_model.train()

            this_batch_optimizer = optim.Adam(
                this_spin_model.parameters(), lr=1e-5)

            for i in range(100):

                spin_pred_pose, spin_pred_betas, pred_camera = this_spin_model(
                    spin_image)

                pred_rotmat = rot6d_to_rotmat(
                    spin_pred_pose).view(-1, 24, 3, 3)

                batch["pose"] = spin_pred_pose[:, 1:].clone()
                batch["orient"] = spin_pred_pose[:, 0].unsqueeze(1).clone()
                batch["betas"] = spin_pred_betas.clone()

                # if(i == 0):
                #     utils.render_batch(img_renderer, batch, "og")

                pred_rotmat_orient = rot6d_to_rotmat(
                    batch['orient'].reshape(-1, 6)).view(-1, 1, 3, 3)

                pred_rotmat_pose = rot6d_to_rotmat(
                    batch['pose'].reshape(-1, 6)).view(-1, 23, 3, 3)

                pred_joints = utils.find_joints(
                    smpl, batch["betas"], pred_rotmat_orient, pred_rotmat_pose, J_regressor)

                joint_loss = loss_function(utils.move_pelvis(
                    pred_joints), batch['gt_j3d']/1000)

                # rendered_silhouette = silhouette_renderer(batch)

                # rendered_silhouette = rendered_silhouette[:, 3].unsqueeze(1)

                # silhouette_loss = loss_function(
                #     rendered_silhouette[batch["valid"]], batch["mask_rcnn"][batch["valid"]])

                opt_loss = joint_loss*10000

                # print(i)
                # print(
                #     f"silhouette_loss.item()*100: {silhouette_loss.item()*100}")
                # print(f"joint_loss.item()*10000: {joint_loss.item()*10000}")

                this_batch_optimizer.zero_grad()
                opt_loss.backward()
                this_batch_optimizer.step()

            # utils.render_batch(img_renderer, batch, "optimized")
            # exit()

            # get the joints from the joint regressor retrained
            # get error to gt
            # update j_regressor
            # relu and take norm
            pred_joints = utils.find_joints(
                smpl, batch["betas"].detach(), pred_rotmat_orient.detach(), pred_rotmat_pose.detach(), J_regressor_retrained, mask=j_reg_mask)
            j_regressor_error = loss_function(utils.move_pelvis(
                pred_joints), batch['gt_j3d']/1000)
            J_Regressor_optimizer.zero_grad()
            j_regressor_error.backward()
            J_Regressor_optimizer.step()

            mpjpe_new_opt, pampjpe_new_opt = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            pred_joints = utils.find_joints(
                smpl, batch["betas"].detach(), pred_rotmat_orient.detach(), pred_rotmat_pose.detach(), J_regressor)

            mpjpe_old_opt, pampjpe_old_opt = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            if(args.wandb_log):
                wandb.log(
                    {
                        # "silhouette_loss": silhouette_loss.item(),
                        "joint_loss": joint_loss.item(),
                        # "discriminated_loss": discriminated_loss.item(),
                        # "discriminator_loss": discriminator_loss.item(),
                        "j_regressor_error": j_regressor_error.item(),
                        "mpjpe": mpjpe_old_opt.item(),
                        "pampjpe": pampjpe_old_opt.item(),
                        "mpjpe difference": mpjpe_new_opt.item()-mpjpe_old_opt.item(),
                        "pampjpe difference": pampjpe_new_opt.item()-pampjpe_old_opt.item(),
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

            if(args.wandb_log and (iteration+1) % 1000 == 0):

                print("saving model and regressor")

                # torch.save(pose_discriminator.state_dict(),
                #            f"models/pose_discriminator_epoch_{epoch}.pt")
                # torch.save(disc_optimizer.state_dict(),
                #            f"models/disc_optimizer_epoch_{epoch}.pt")

                torch.save(J_regressor_retrained,
                           "models/retrained_J_Regressor.pt")
