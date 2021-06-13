from args import args
import wandb
import torch
from torch import nn, optim

from tqdm import tqdm


from pose_refiner import Pose_Refiner, Pose_Refiner_Translation
from renderer import Renderer, return_2d_joints
from discriminator import Discriminator


from data import load_data, data_set

from torchvision import transforms

# from create_smpl_gt import quaternion_to_rotation_matrix, find_translation_and_pose, find_direction_to_gt, batch_rodrigues, optimize_pose, optimize_translation, quaternion_multiply, rotation_matrix_to_quaternion

from SPIN.models import hmr, SMPL
import SPIN.config as config

import numpy as np

import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, AlphaCompositor, PointsRenderer

from utils import utils

from eval_utils import batch_compute_similarity_transform_torch

from SPIN.utils.geometry import rot6d_to_rotmat

import constants

import time


def train_pose_refiner_model():

    spin_model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
    checkpoint = torch.load(
        "SPIN/data/model_checkpoint.pt", map_location=args.device)
    spin_model.load_state_dict(checkpoint['model'], strict=False)
    spin_model.eval()

    smpl = SMPL(
        '{}'.format("SPIN/data/smpl"),
        batch_size=1,
    ).to(args.device)

    J_regressor = torch.load(
        'models/best_pose_refiner/retrained_J_Regressor.pt').to(args.device)
    J_regressor_initial = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

    j_reg_mask = utils.find_j_reg_mask(J_regressor_initial)

    num_networks = 1

    silhouette_renderer = Renderer(subset=True)
    img_renderer = Renderer(subset=False)

    pose_refiners = [Pose_Refiner().to(args.device)
                     for _ in range(num_networks)]
    # if(args.wandb_log):
    #     wandb.watch(pose_refiners[0], log_freq=10)
    # checkpoint = torch.load(
    #     "models/pose_refiner_epoch_6.pt", map_location=args.device)
    # pose_refiner.load_state_dict(checkpoint)
    for pose_refiner in pose_refiners:
        pose_refiner.train()

        for param in pose_refiner.resnet.parameters():
            param.requires_grad = False
    # pose_refiner.eval()
    # print("model load worked succesfully")
    # exit()
    pose_discriminator = Discriminator().to(args.device)
    pose_discriminator.train()

    pose_optimizers = [optim.Adam(
        pose_refiners[i].parameters(), lr=args.learning_rate) for i in range(num_networks)]

    disc_optimizer = optim.Adam(
        pose_discriminator.parameters(), lr=args.disc_learning_rate)

    loss_function = nn.MSELoss()

    data = data_set("train")
    val_data = data_set("validation")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=True, drop_last=True)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for epoch in range(args.train_epochs):

        iterator = iter(loader)
        val_iterator = iter(val_loader)

        for iteration in tqdm(range(len(loader))):

            try:
                batch = next(iterator)
            except:
                print("problem loading batch")
                time.sleep(1)
                continue

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations"):
                    batch[item] = batch[item].to(args.device).float()

            # train generator

            batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

            spin_image = normalize(batch['image'])

            with torch.no_grad():
                spin_pred_pose, pred_betas, pred_camera = spin_model(
                    spin_image)

            pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                      -2*pred_camera[:, 2],
                                      2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
            batch["cam"] = pred_cam_t

            pred_rotmat = rot6d_to_rotmat(spin_pred_pose).view(-1, 24, 3, 3)

            batch["orient"] = spin_pred_pose[:, :1]
            batch["pose"] = spin_pred_pose[:, 1:]
            batch["betas"] = pred_betas

            # utils.render_batch(img_renderer, batch, "initial")

            pred_joints = utils.find_joints(
                smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor, mask=j_reg_mask)

            mpjpe_before_refinement, pampjpe_before_refinement = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            for i in range(num_networks):

                est_pose, est_betas, est_cam = pose_refiners[i](batch)

                batch["orient"] = est_pose[:, :1]
                batch["pose"] = est_pose[:, 1:]
                batch["betas"] = est_betas
                batch["cam"] = est_cam

                # utils.render_batch(img_renderer, batch, "refined")
                # exit()

                pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

                pred_joints = utils.find_joints(
                    smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor, mask=j_reg_mask)

                pred_joints = utils.move_pelvis(pred_joints)

                joint_loss = loss_function(
                    pred_joints, batch['gt_j3d']/1000)

                pred_disc = pose_discriminator(est_pose)

                discriminated_loss = loss_function(pred_disc, torch.ones(
                    pred_disc.shape).to(args.device))

                # rendered_silhouette = silhouette_renderer(batch)

                # rendered_silhouette = rendered_silhouette[:, 3].unsqueeze(1)

                # silhouette_loss = loss_function(
                #     rendered_silhouette[batch["valid"]], batch["mask_rcnn"][batch["valid"]])

                joints_2d = return_2d_joints(
                    batch, smpl, J_regressor=J_regressor, mask=j_reg_mask)

                loss_2d = loss_function(joints_2d[..., :2], batch["gt_j2d"])

                loss = joint_loss+discriminated_loss/1000+loss_2d/100000

                pose_optimizers[i].zero_grad()
                loss.backward()
                pose_optimizers[i].step()

                for item in ["orient", "pose", "betas", "cam"]:
                    batch[item] = batch[item].detach()

            mpjpe_after, pampjpe_after = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            # train discriminator

            pred_gt = pose_discriminator(spin_pred_pose)
            pred_disc = pose_discriminator(est_pose.detach())

            discriminator_loss = loss_function(pred_disc, torch.zeros(
                pred_disc.shape).to(args.device))+loss_function(pred_gt, torch.ones(
                    pred_disc.shape).to(args.device))

            disc_optimizer.zero_grad()
            discriminator_loss.backward()
            disc_optimizer.step()

            if(args.wandb_log):

                wandb.log(
                    {
                        "joint_loss": joint_loss.item(),
                        "discriminated_loss": discriminated_loss,
                        "loss_2d": loss_2d.item(),
                        # "silhouette_loss": silhouette_loss,
                        "discriminator_loss": discriminator_loss,
                        "mpjpe_before_refinement": mpjpe_before_refinement.item(),
                        "pampjpe_before_refinement": pampjpe_before_refinement.item(),
                        "mpjpe_after": mpjpe_after.item(),
                        "pampjpe_after": pampjpe_after.item(),
                        "mpjpe_difference": mpjpe_after.item()-mpjpe_before_refinement.item(),
                        "pampjpe_difference": pampjpe_after.item()-pampjpe_before_refinement.item(), })

            if(args.wandb_log and iteration % 100 == 0):

                with torch.no_grad():

                    for pose_refiner in pose_refiners:
                        pose_refiner.eval()

                    batch = next(val_iterator)

                    for item in batch:
                        if(item != "valid" and item != "path" and item != "pixel_annotations"):
                            batch[item] = batch[item].to(args.device).float()

                    spin_image = normalize(batch['image'])

                    spin_pred_pose, pred_betas, pred_camera = spin_model(
                        spin_image)

                    pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                              -2*pred_camera[:, 2],
                                              2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
                    batch["cam"] = pred_cam_t

                    pred_rotmat = rot6d_to_rotmat(
                        spin_pred_pose).view(-1, 24, 3, 3)

                    batch["pose"] = spin_pred_pose[:, 1:]
                    batch["orient"] = spin_pred_pose[:, 0].unsqueeze(1)
                    batch["betas"] = pred_betas

                    pred_joints = utils.find_joints(
                        smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor, mask=j_reg_mask)

                    mpjpe_before_refinement, pampjpe_before_refinement = utils.evaluate(
                        pred_joints, batch['gt_j3d'])

                    for i in range(num_networks):

                        est_pose, est_betas, est_cam = pose_refiners[i](batch)

                        batch["orient"] = est_pose[:, :1]
                        batch["pose"] = est_pose[:, 1:]
                        batch["betas"] = est_betas
                        batch["cam"] = est_cam

                        # utils.render_batch(img_renderer, batch, "refined")
                        # exit()

                    pred_rotmat = rot6d_to_rotmat(
                        est_pose).view(-1, 24, 3, 3)

                    pred_joints = utils.find_joints(
                        smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor, mask=j_reg_mask)

                    pred_joints = utils.move_pelvis(pred_joints)

                    mpjpe_after, pampjpe_after = utils.evaluate(
                        pred_joints, batch['gt_j3d'])

                    for pose_refiner in pose_refiners:
                        pose_refiner.train()

                wandb.log(
                    {
                        "validation mpjpe_before_refinement": mpjpe_before_refinement.item(),
                        "validation pampjpe_before_refinement": pampjpe_before_refinement.item(),
                        "validation mpjpe_after": mpjpe_after.item(),
                        "validation pampjpe_after": pampjpe_after.item(),
                        "validation mpjpe_difference": mpjpe_after.item()-mpjpe_before_refinement.item(),
                        "validation pampjpe_difference": pampjpe_after.item()-pampjpe_before_refinement.item(), })

        print(f"epoch: {epoch}")

        for i in range(num_networks):
            torch.save(pose_refiners[i].state_dict(),
                       f"models/pose_refiner_{i}_epoch_{epoch}.pt")


def train_pose_refiner_translation_model():

    import os
    os.chdir("/scratch/iamerich/MeshTransformer/")  # noqa

    import transformer

    metro, smpl, mesh = transformer.load_transformer()

    os.chdir("/scratch/iamerich/human-body-pose/")  # noqa

    metro.eval()

    loss_function = nn.MSELoss()

    data = data_set("train")
    val_data = data_set("validation")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    pose_refiner = Pose_Refiner_Translation().to(args.device)

    for param in pose_refiner.resnet.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        pose_refiner.parameters(), lr=args.translation_lr)

    for epoch in range(args.train_epochs):

        iterator = iter(loader)
        val_iterator = iter(val_loader)

        for iteration in tqdm(range(len(loader))):

            try:
                batch = next(iterator)
            except:
                print("problem loading batch")
                time.sleep(1)
                continue

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations"):
                    batch[item] = batch[item].to(args.device).float()

            # train generator

            batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

            spin_image = normalize(batch['image'])

            with torch.no_grad():
                pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = metro(
                    spin_image, smpl, mesh)

            pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

            pred_3d_joints_from_smpl = utils.move_pelvis(
                pred_3d_joints_from_smpl)

            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,
                                                                constants.J17_2_METRO]

            batch["pred_j3d"] = pred_3d_joints_from_smpl
            batch["pred_vertices"] = pred_vertices

            pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                      -2*pred_camera[:, 2],
                                      2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
            batch["cam"] = pred_cam_t

            mpjpe_before, pampjpe_before = utils.evaluate(
                batch["pred_j3d"], batch['gt_j3d'])

            batch["pred_j3d"] = pose_refiner(batch)

            mpjpe_after, pampjpe_after = utils.evaluate(
                batch["pred_j3d"], batch['gt_j3d'])

            loss = loss_function(batch["pred_j3d"], batch["gt_j3d"]/1000)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(args.wandb_log):

                wandb.log(
                    {
                        "loss": loss.item(),
                        "mpjpe_before": mpjpe_before.item(),
                        "pampjpe_before": pampjpe_before.item(),
                        "mpjpe_after": mpjpe_after.item(),
                        "pampjpe_after": pampjpe_after.item(),
                        "mpjpe_difference": mpjpe_after.item()-mpjpe_before.item(),
                        "pampjpe_difference": pampjpe_after.item()-pampjpe_before.item(), })

            if(args.wandb_log and iteration % 100 == 0):

                pose_refiner.eval()

                batch = next(val_iterator)

                for item in batch:
                    if(item != "valid" and item != "path" and item != "pixel_annotations"):
                        batch[item] = batch[item].to(args.device).float()

                # train generator

                batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

                spin_image = normalize(batch['image'])

                with torch.no_grad():
                    pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = metro(
                        spin_image, smpl, mesh)

                pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

                pred_3d_joints_from_smpl = utils.move_pelvis(
                    pred_3d_joints_from_smpl)

                pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,
                                                                    constants.J17_2_METRO]

                batch["pred_j3d"] = pred_3d_joints_from_smpl
                batch["pred_vertices"] = pred_vertices

                pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                          -2*pred_camera[:, 2],
                                          2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
                batch["cam"] = pred_cam_t

                mpjpe_before, pampjpe_before = utils.evaluate(
                    batch["pred_j3d"], batch['gt_j3d'])

                batch["pred_j3d"] = pose_refiner(batch)

                mpjpe_after, pampjpe_after = utils.evaluate(
                    batch["pred_j3d"], batch['gt_j3d'])

                pose_refiner.train()

                wandb.log(
                    {
                        "val_mpjpe_before": mpjpe_before.item(),
                        "val_pampjpe_before": pampjpe_before.item(),
                        "val_mpjpe_after": mpjpe_after.item(),
                        "val_pampjpe_after": pampjpe_after.item(),
                        "val_mpjpe_difference": mpjpe_after.item()-mpjpe_before.item(),
                        "val_pampjpe_difference": pampjpe_after.item()-pampjpe_before.item(), })

        print(f"epoch {epoch}")
        torch.save(pose_refiner.state_dict(),
                   f"models/transformer_refiner_epoch_{epoch}.pt")


def train_joint_regressor():

    spin_model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
    checkpoint = torch.load(
        "SPIN/data/model_checkpoint.pt", map_location=args.device)
    spin_model.load_state_dict(checkpoint['model'], strict=False)
    spin_model.eval()

    smpl = SMPL(
        '{}'.format("SPIN/data/smpl"),
        batch_size=1,
    ).to(args.device)

    J_regressor = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

    J_regressor.requires_grad = True

    j_reg_mask = utils.find_j_reg_mask(J_regressor)

    pose_refiner = Pose_Refiner().to(args.device)
    checkpoint = torch.load(
        "models/pose_refiner_epoch_6.pt", map_location=args.device)
    pose_refiner.load_state_dict(checkpoint)
    pose_refiner.eval()

    loss_function = nn.MSELoss()

    img_renderer = Renderer(subset=False)

    data = data_set("train")
    val_data = data_set("validation")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    optimizer = optim.Adam([J_regressor], lr=args.j_reg_lr)

    for epoch in range(5):

        iterator = iter(loader)
        val_iterator = iter(val_loader)

        for iteration in tqdm(range(len(loader))):

            try:
                batch = next(iterator)
            except:
                continue

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations"):
                    batch[item] = batch[item].to(args.device).float()

            # train generator

            batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

            spin_image = normalize(batch['image'])

            with torch.no_grad():
                spin_pred_pose, pred_betas, pred_camera = spin_model(
                    spin_image)

            pred_rotmat = rot6d_to_rotmat(spin_pred_pose).view(-1, 24, 3, 3)

            batch["orient"] = spin_pred_pose[:, :1]
            batch["pose"] = spin_pred_pose[:, 1:]
            batch["betas"] = pred_betas

            pred_joints = utils.find_joints(
                smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor, mask=j_reg_mask)

            est_pose, est_betas, est_cam = pose_refiner(batch)

            batch["orient"] = est_pose[:, :1]
            batch["pose"] = est_pose[:, 1:]
            batch["betas"] = est_betas
            batch["cam"] = est_cam

            pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

            pred_joints = utils.find_joints(
                smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor, mask=j_reg_mask)

            pred_joints = utils.move_pelvis(pred_joints)

            loss = loss_function(pred_joints, batch["gt_j3d"]/1000)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(args.wandb_log):
                wandb.log({"loss": loss.item()})

            if(args.wandb_log and iteration % 100 == 0):

                with torch.no_grad():
                    batch = next(val_iterator)

                    for item in batch:

                        if(item != "valid" and item != "path" and item != "pixel_annotations"):
                            batch[item] = batch[item].to(args.device).float()

                    batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

                    spin_image = normalize(batch['image'])

                    spin_pred_pose, pred_betas, pred_camera = spin_model(
                        spin_image)

                    pred_rotmat = rot6d_to_rotmat(
                        spin_pred_pose).view(-1, 24, 3, 3)

                    batch["orient"] = spin_pred_pose[:, :1]
                    batch["pose"] = spin_pred_pose[:, 1:]
                    batch["betas"] = pred_betas

                    pred_joints = utils.find_joints(
                        smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor)

                    est_pose, est_betas, est_cam = pose_refiner(batch)

                    batch["orient"] = est_pose[:, :1]
                    batch["pose"] = est_pose[:, 1:]
                    batch["betas"] = est_betas
                    batch["cam"] = est_cam

                    pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

                    pred_joints = utils.find_joints(
                        smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

                    pred_joints = utils.move_pelvis(pred_joints)

                    loss = loss_function(pred_joints, batch["gt_j3d"]/1000)

                    wandb.log({"val_loss": loss.item()})

        torch.save(J_regressor,
                   f"models/j_regressor_epoch_{epoch}.pt")
