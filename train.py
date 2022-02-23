from args import args
import wandb
import torch
from torch import nn, optim

from tqdm import tqdm


from pose_refiner import Pose_Refiner
from transformer_refiner import Transformer_Refiner
from renderer import Renderer, return_2d_joints
from mesh_renderer import Mesh_Renderer
from discriminator import Discriminator
from img_disc import Img_Disc


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

    def run_epoch(batch, val=False):
        batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

        spin_image = normalize(batch['spin_image'])

        with torch.no_grad():
            spin_pred_pose, pred_betas, pred_camera = spin_model(
                spin_image)

        pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                  -2*pred_camera[:, 2],
                                  2*5000/(256 * pred_camera[:, 0] + 1e-9)], dim=-1)
        batch["cam"] = pred_cam_t

        initial_pose = spin_pred_pose.clone()

        # if(not val):
        #     initial_pose += torch.randn(
        #         spin_pred_pose.shape).to(args.device)*.05

        pred_rotmat = rot6d_to_rotmat(initial_pose).view(-1, 24, 3, 3)

        batch["orient"] = initial_pose[:, :1]
        batch["pose"] = initial_pose[:, 1:]
        batch["betas"] = pred_betas

        orient_offset_batch = orient_offset.unsqueeze(
            0).expand(batch["orient"].shape)
        pose_offset_batch = pose_offset.unsqueeze(
            0).expand(batch["pose"].shape)
        shape_offset_batch = shape_offset.unsqueeze(
            0).expand(batch["betas"].shape)
        cam_offset_batch = cam_offset.unsqueeze(
            0).expand(batch["cam"].shape)

        pred_vertices = smpl(global_orient=pred_rotmat[:, :1], body_pose=pred_rotmat[:, 1:],
                             betas=batch["betas"], pose2rot=False).vertices

        batch["pred_vertices"] = pred_vertices

        # batch["pred_vertices"] *= 2
        # batch["pred_vertices"][..., :2] *= -1

        # joints_2d = return_2d_joints(
        #     batch, smpl, J_regressor=J_regressor)
        # utils.render_batch(img_renderer, batch,
        #                    "smpl", [joints_2d])
        # utils.render_batch(img_renderer, batch,
        #                    "gt", [batch["gt_j2d"]])
        # exit()

        # utils.render_batch(img_renderer, batch, "initial")

        pred_joints = utils.find_joints(
            smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor)

        mpjpe_before, pampjpe_before = utils.evaluate(
            pred_joints, batch['gt_j3d'])

        est_pose, est_betas, est_cam = pose_refiner(batch)
        batch["orient"] = est_pose[:, :1]
        batch["pose"] = est_pose[:, 1:]
        batch["betas"] = est_betas
        batch["cam"] = est_cam

        # batch["orient"] = batch["orient"] + orient_offset_batch
        # batch["pose"] = batch["pose"] + pose_offset_batch
        # batch["betas"] = batch["betas"] + shape_offset_batch
        # batch["cam"] = batch["cam"] + cam_offset_batch
        # est_pose = torch.cat([batch["orient"], batch["pose"]], dim=1)

        pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

        pred_vertices = smpl(global_orient=pred_rotmat[:, :1], body_pose=pred_rotmat[:, 1:],
                             betas=batch["betas"], pose2rot=False).vertices

        batch["pred_vertices"] = pred_vertices

        # utils.render_batch(img_renderer, batch, "refined")
        # exit()

        # pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

        pred_joints = utils.find_joints(
            smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

        mpjpe_after, pampjpe_after = utils.evaluate(
            pred_joints, batch['gt_j3d'])

        if(val):
            return mpjpe_before, mpjpe_after, pampjpe_before, pampjpe_after

        return pred_joints, est_pose, spin_pred_pose, mpjpe_before, mpjpe_after, pampjpe_before, pampjpe_after

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
    # J_regressor = torch.load(
    #     'models/retrained_J_Regressor.pt').to(args.device)
    # j_reg_mask = utils.find_j_reg_mask(J_regressor)
    # J_regressor = J_regressor*j_reg_mask
    # J_regressor = nn.ReLU()(J_regressor)
    # J_regressor = J_regressor / torch.sum(J_regressor, dim=1).unsqueeze(
    #     1).expand(J_regressor.shape)

    silhouette_renderer = Renderer(subset=True)
    img_renderer = Mesh_Renderer()

    pose_refiner = Pose_Refiner().to(args.device)
    pose_refiner.train()

    for param in pose_refiner.resnet.parameters():
        param.requires_grad = False

    # pose_discriminator = Img_Disc().to(args.device)
    pose_discriminator = Discriminator().to(args.device)
    pose_discriminator.train()

    # orient_offset = torch.zeros((1, 6)).to(args.device)
    # pose_offset = torch.zeros((23, 6)).to(args.device)
    # shape_offset = torch.zeros((10)).to(args.device)
    # cam_offset = torch.zeros((3)).to(args.device)

    # orient_offset.requires_grad = True
    # pose_offset.requires_grad = True
    # shape_offset.requires_grad = True
    # cam_offset.requires_grad = True

    orient_offset = torch.torch.load(
        "models/orient_offset.pt").float().to(args.device)
    pose_offset = torch.torch.load(
        "models/pose_offset.pt").float().to(args.device)
    shape_offset = torch.torch.load(
        "models/shape_offset.pt").float().to(args.device)
    cam_offset = torch.torch.load(
        "models/cam_offset.pt").float().to(args.device)

    pose_optimizer = optim.Adam(
        pose_refiner.parameters(), lr=args.learning_rate)

    disc_optimizer = optim.Adam(
        pose_discriminator.parameters(), lr=args.disc_learning_rate)

    mse_loss = nn.MSELoss()

    data = data_set("train")
    val_data = data_set("validation")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True, drop_last=True)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for epoch in range(args.train_epochs):

        iterator = iter(loader)
        val_iterator = iter(val_loader)

        for iteration in tqdm(range(len(loader))):
            # for iteration in tqdm(range(601)):

            try:
                batch = next(iterator)
            except:
                print("problem loading batch")
                time.sleep(1)
                continue

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations" and item != "inc_gt"):
                    batch[item] = batch[item].to(args.device).float()

            pred_joints, est_pose, spin_pred_pose, mpjpe_before, mpjpe_after, pampjpe_before, pampjpe_after = run_epoch(
                batch)

            pred_joints = utils.move_pelvis(pred_joints)

            joint_loss = mse_loss(
                pred_joints[batch["inc_gt"]], batch['gt_j3d'][batch["inc_gt"]]/1000)

            pred_disc = pose_discriminator(est_pose)

            # disc_weight = (1-batch["inc_gt"].to(torch.float32))*4+1

            # disc_weight = disc_weight.view(-1, 1,
            #                                1).expand(pred_disc.shape).to(args.device)

            # print(disc_weight)

            # discriminated_loss = mse_loss(disc_weight*pred_disc, disc_weight*torch.ones(
            #     pred_disc.shape).to(args.device))
            discriminated_loss = mse_loss(pred_disc, torch.ones(
                pred_disc.shape).to(args.device))

            # rendered_silhouette = silhouette_renderer(batch)

            # rendered_silhouette = rendered_silhouette[:, 3].unsqueeze(1)

            # silhouette_loss = mse_loss(
            #     rendered_silhouette[batch["valid"]], batch["mask_rcnn"][batch["valid"]])

            joints_2d = return_2d_joints(
                batch, smpl, J_regressor=J_regressor)

            loss_2d = mse_loss(
                joints_2d[..., :2][batch["inc_gt"]], batch["gt_j2d"][batch["inc_gt"]])

            loss = joint_loss+discriminated_loss/1000+loss_2d/100000
            # loss = discriminated_loss

            pose_optimizer.zero_grad()
            loss.backward()
            pose_optimizer.step()

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations"):
                    batch[item] = batch[item].detach()

            # mpjpe_after, pampjpe_after = utils.evaluate(
            #     pred_joints, batch['gt_j3d'])

            # train discriminator

            pred_gt = pose_discriminator(spin_pred_pose)
            pred_disc = pose_discriminator(est_pose.detach())

            discriminator_loss = mse_loss(pred_disc, torch.zeros(
                pred_disc.shape).to(args.device))+mse_loss(pred_gt, torch.ones(
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
                        "mpjpe_before": mpjpe_before.item(),
                        "pampjpe_before": pampjpe_before.item(),
                        "mpjpe_after": mpjpe_after.item(),
                        "pampjpe_after": pampjpe_after.item(),
                        "mpjpe_difference": mpjpe_after.item()-mpjpe_before.item(),
                        "pampjpe_difference": pampjpe_after.item()-pampjpe_before.item(), })

            if(args.wandb_log and iteration % 100 == 0):

                with torch.no_grad():

                    pose_refiner.eval()

                    batch = next(val_iterator)

                    for item in batch:
                        if(item != "valid" and item != "path" and item != "pixel_annotations"):
                            batch[item] = batch[item].to(args.device).float()

                    mpjpe_before, mpjpe_after, pampjpe_before, pampjpe_after = run_epoch(
                        batch, val=True)

                    pose_refiner.train()

                wandb.log(
                    {
                        "validation mpjpe_before": mpjpe_before.item(),
                        "validation pampjpe_before": pampjpe_before.item(),
                        "validation mpjpe_after": mpjpe_after.item(),
                        "validation pampjpe_after": pampjpe_after.item(),
                        "validation mpjpe_difference": mpjpe_after.item()-mpjpe_before.item(),
                        "validation pampjpe_difference": pampjpe_after.item()-pampjpe_before.item(), })

        print(f"saving model for epoch: {epoch}")

        torch.save(pose_refiner.state_dict(),
                   f"models/pose_refiner_epoch_{epoch}.pt")
        torch.save(pose_optimizer.state_dict(),
                   f"models/pose_optimizers_epoch_{epoch}.pt")

        # torch.save(orient_offset, f"models/orient_offset.pt")
        # torch.save(pose_offset, f"models/pose_offset.pt")
        # torch.save(shape_offset, f"models/shape_offset.pt")
        # torch.save(cam_offset, f"models/cam_offset.pt")


def train_pose_refiner_offset():

    def run_epoch(batch, val=False):
        batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

        spin_image = normalize(batch['spin_image'])

        with torch.no_grad():
            spin_pred_pose, pred_betas, pred_camera = spin_model(
                spin_image)

        pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                  -2*pred_camera[:, 2],
                                  2*5000/(256 * pred_camera[:, 0] + 1e-9)], dim=-1)
        batch["cam"] = pred_cam_t

        initial_pose = spin_pred_pose.clone()

        # if(not val):
        #     initial_pose += torch.randn(
        #         spin_pred_pose.shape).to(args.device)*.05

        pred_rotmat = rot6d_to_rotmat(initial_pose).view(-1, 24, 3, 3)

        batch["orient"] = initial_pose[:, :1]
        batch["pose"] = initial_pose[:, 1:]
        batch["betas"] = pred_betas

        orient_offset_batch = orient_offset.unsqueeze(
            0).expand(batch["orient"].shape)
        pose_offset_batch = pose_offset.unsqueeze(
            0).expand(batch["pose"].shape)
        shape_offset_batch = shape_offset.unsqueeze(
            0).expand(batch["betas"].shape)
        cam_offset_batch = cam_offset.unsqueeze(
            0).expand(batch["cam"].shape)

        # pred_vertices = smpl(global_orient=pred_rotmat[:, :1], body_pose=pred_rotmat[:, 1:],
        #                      betas=batch["betas"], pose2rot=False).vertices

        # batch["pred_vertices"] = pred_vertices

        # utils.render_batch(img_renderer, batch, "initial")

        pred_joints = utils.find_joints(
            smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor)

        mpjpe_before, pampjpe_before = utils.evaluate(
            pred_joints, batch['gt_j3d'])

        # est_pose, est_betas, est_cam = pose_refiner(batch)

        batch["orient"] = batch["orient"] + orient_offset_batch
        batch["pose"] = batch["pose"] + pose_offset_batch
        batch["betas"] = batch["betas"] + shape_offset_batch
        batch["cam"] = batch["cam"] + cam_offset_batch
        est_pose = torch.cat([batch["orient"], batch["pose"]], dim=1)

        pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

        # pred_vertices = smpl(global_orient=pred_rotmat[:, :1], body_pose=pred_rotmat[:, 1:],
        #                      betas=batch["betas"], pose2rot=False).vertices

        # batch["pred_vertices"] = pred_vertices

        # utils.render_batch(img_renderer, batch, "refined")
        # exit()

        # pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

        pred_joints = utils.find_joints(
            smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

        mpjpe_after, pampjpe_after = utils.evaluate(
            pred_joints, batch['gt_j3d'])

        if(val):
            return mpjpe_before, mpjpe_after, pampjpe_before, pampjpe_after

        return pred_joints, est_pose, spin_pred_pose, mpjpe_before, mpjpe_after, pampjpe_before, pampjpe_after

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
        'models/retrained_J_Regressor.pt').to(args.device)

    with torch.no_grad():
        j_reg_mask = utils.find_j_reg_mask(J_regressor)
        J_regressor = J_regressor*j_reg_mask

        J_regressor = nn.ReLU()(J_regressor)
        J_regressor = J_regressor / torch.sum(J_regressor, dim=1).unsqueeze(
            1).expand(J_regressor.shape)

    silhouette_renderer = Renderer(subset=True)
    img_renderer = Mesh_Renderer()

    pose_refiner = Pose_Refiner().to(args.device)
    pose_refiner.train()

    for param in pose_refiner.resnet.parameters():
        param.requires_grad = False

    # pose_discriminator = Img_Disc().to(args.device)
    pose_discriminator = Discriminator().to(args.device)
    pose_discriminator.train()

    orient_offset = torch.zeros((1, 6)).to(args.device)
    pose_offset = torch.zeros((23, 6)).to(args.device)
    shape_offset = torch.zeros((10)).to(args.device)
    cam_offset = torch.zeros((3)).to(args.device)

    orient_offset.requires_grad = True
    pose_offset.requires_grad = True
    shape_offset.requires_grad = True
    cam_offset.requires_grad = True

    # orient_offset = torch.torch.load(
    #     "models/orient_offset.pt").float().to(args.device)
    # pose_offset = torch.torch.load(
    #     "models/pose_offset.pt").float().to(args.device)
    # shape_offset = torch.torch.load(
    #     "models/shape_offset.pt").float().to(args.device)
    # cam_offset = torch.torch.load(
    #     "models/cam_offset.pt").float().to(args.device)

    pose_optimizer = optim.Adam(
        [orient_offset, pose_offset, shape_offset, cam_offset], lr=1e-3)

    disc_optimizer = optim.Adam(
        pose_discriminator.parameters(), lr=args.disc_learning_rate)

    mse_loss = nn.MSELoss()

    data = data_set("train")
    val_data = data_set("validation")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True, drop_last=True)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for epoch in range(args.train_epochs):

        iterator = iter(loader)
        val_iterator = iter(val_loader)

        # for iteration in tqdm(range(len(loader))):
        for iteration in tqdm(range(601)):

            try:
                batch = next(iterator)
            except:
                print("problem loading batch")
                time.sleep(1)
                continue

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations" and item != "inc_gt"):
                    batch[item] = batch[item].to(args.device).float()

            pred_joints, est_pose, spin_pred_pose, mpjpe_before, mpjpe_after, pampjpe_before, pampjpe_after = run_epoch(
                batch)

            pred_joints = utils.move_pelvis(pred_joints)

            joint_loss = mse_loss(
                pred_joints[batch["inc_gt"]], batch['gt_j3d'][batch["inc_gt"]]/1000)

            pred_disc = pose_discriminator(est_pose)

            # disc_weight = (1-batch["inc_gt"].to(torch.float32))*4+1

            # disc_weight = disc_weight.view(-1, 1,
            #                                1).expand(pred_disc.shape).to(args.device)

            # print(disc_weight)

            # discriminated_loss = mse_loss(disc_weight*pred_disc, disc_weight*torch.ones(
            #     pred_disc.shape).to(args.device))
            discriminated_loss = mse_loss(pred_disc, torch.ones(
                pred_disc.shape).to(args.device))

            # rendered_silhouette = silhouette_renderer(batch)

            # rendered_silhouette = rendered_silhouette[:, 3].unsqueeze(1)

            # silhouette_loss = mse_loss(
            #     rendered_silhouette[batch["valid"]], batch["mask_rcnn"][batch["valid"]])

            joints_2d = return_2d_joints(
                batch, smpl, J_regressor=J_regressor)

            loss_2d = mse_loss(
                joints_2d[..., :2][batch["inc_gt"]], batch["gt_j2d"][batch["inc_gt"]])

            loss = joint_loss+discriminated_loss/1000+loss_2d/100000
            # loss = discriminated_loss

            pose_optimizer.zero_grad()
            loss.backward()
            pose_optimizer.step()

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations"):
                    batch[item] = batch[item].detach()

            # mpjpe_after, pampjpe_after = utils.evaluate(
            #     pred_joints, batch['gt_j3d'])

            # train discriminator

            pred_gt = pose_discriminator(spin_pred_pose)
            pred_disc = pose_discriminator(est_pose.detach())

            discriminator_loss = mse_loss(pred_disc, torch.zeros(
                pred_disc.shape).to(args.device))+mse_loss(pred_gt, torch.ones(
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
                        "mpjpe_before": mpjpe_before.item(),
                        "pampjpe_before": pampjpe_before.item(),
                        "mpjpe_after": mpjpe_after.item(),
                        "pampjpe_after": pampjpe_after.item(),
                        "mpjpe_difference": mpjpe_after.item()-mpjpe_before.item(),
                        "pampjpe_difference": pampjpe_after.item()-pampjpe_before.item(), })

            if(args.wandb_log and iteration % 100 == 0):

                with torch.no_grad():

                    # pose_refiner.eval()

                    batch = next(val_iterator)

                    for item in batch:
                        if(item != "valid" and item != "path" and item != "pixel_annotations"):
                            batch[item] = batch[item].to(args.device).float()

                    mpjpe_before, mpjpe_after, pampjpe_before, pampjpe_after = run_epoch(
                        batch, val=True)

                    # pose_refiner.train()

                wandb.log(
                    {
                        "validation mpjpe_before": mpjpe_before.item(),
                        "validation pampjpe_before": pampjpe_before.item(),
                        "validation mpjpe_after": mpjpe_after.item(),
                        "validation pampjpe_after": pampjpe_after.item(),
                        "validation mpjpe_difference": mpjpe_after.item()-mpjpe_before.item(),
                        "validation pampjpe_difference": pampjpe_after.item()-pampjpe_before.item(), })

        print(f"saving model for epoch: {epoch}")

        torch.save(orient_offset, f"models/orient_offset.pt")
        torch.save(pose_offset, f"models/pose_offset.pt")
        torch.save(shape_offset, f"models/shape_offset.pt")
        torch.save(cam_offset, f"models/cam_offset.pt")


# def train_transformer_refiner():

#     def run_epoch(epoch, train=True):
#         batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

#         spin_image = normalize(batch['image'])

#         pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, final_layer, image_feats, num_joints = metro(
#             spin_image, smpl, mesh)

#         if(train):

#             noisy_feats = final_layer[:, num_joints:] + torch.randn(
#                 final_layer[:, num_joints:].shape).to(args.device)*2.0

#             batch["final_layer"] = noisy_feats

#             pred_joints, pred_vertices = find_joints_from_updated_head(
#                 final_cls_head, noisy_feats, image_feats, num_joints, metro.upsampling, metro.upsampling2, smpl)

#         else:

#             batch["final_layer"] = final_layer[:, num_joints:]

#             pred_joints = smpl.get_h36m_joints(pred_vertices)

#             pred_joints = utils.move_pelvis(
#                 pred_joints)

#             pred_joints = pred_joints[:, constants.J17_2_METRO]

#         batch["pred_j3d"] = pred_joints

#         pred_vertices *= 2
#         pred_vertices[..., :2] *= -1

#         batch["pred_vertices"] = pred_vertices

#         # print("batch[final_layer].shape")
#         # print(batch["final_layer"].shape)

#         pred_cam_t = torch.stack([-2*pred_camera[:, 1],
#                                   -2*pred_camera[:, 2],
#                                   2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
#         batch["cam"] = pred_cam_t.detach()

#         mpjpe_before, pampjpe_before = utils.evaluate(
#             batch["pred_j3d"], batch['gt_j3d'])

#         refined_feats = transformer_refiner(batch)

#         pred_joints, _ = find_joints_from_updated_head(
#             final_cls_head, refined_feats, image_feats, num_joints, metro.upsampling, metro.upsampling2, smpl)

#         mpjpe_after, pampjpe_after = utils.evaluate(
#             pred_joints, batch['gt_j3d'])

#         return pred_joints, mpjpe_before, pampjpe_before, mpjpe_after, pampjpe_after

#     import copy

#     import os
#     os.chdir("/scratch/iamerich/MeshTransformer/")  # noqa

#     from transformer import load_transformer, find_joints_from_updated_head

#     metro, smpl, mesh, final_cls_head = load_transformer()

#     metro.to(args.device)
#     final_cls_head.to(args.device)

#     os.chdir("/scratch/iamerich/human-body-pose/")  # noqa

#     metro.eval()

#     transformer_refiner = Transformer_Refiner().to(args.device)

#     transformer_refiner.train()
#     for param in transformer_refiner.resnet.parameters():
#         param.requires_grad = False
#     optimizer = optim.Adam(
#         transformer_refiner.parameters(), lr=args.learning_rate)

#     loss_function = nn.MSELoss()

#     data = data_set("train")
#     val_data = data_set("validation")

#     loader = torch.utils.data.DataLoader(
#         data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
#     val_loader = torch.utils.data.DataLoader(
#         val_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)

#     normalize = transforms.Normalize(
#         (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

#     for epoch in range(args.train_epochs):

#         iterator = iter(loader)
#         val_iterator = iter(val_loader)

#         batch = next(iterator)

#         for item in batch:
#             if(item != "valid" and item != "path" and item != "pixel_annotations"):
#                 batch[item] = batch[item].to(args.device).float()

#         for iteration in tqdm(range(len(loader))):

#             try:
#                 batch = next(iterator)
#             except:
#                 print("problem loading batch")
#                 time.sleep(1)
#                 continue

#             for item in batch:
#                 if(item != "valid" and item != "path" and item != "pixel_annotations"):
#                     batch[item] = batch[item].to(args.device).float()

#             pred_joints, mpjpe_before, pampjpe_before, mpjpe_after, pampjpe_after = run_epoch(
#                 batch)

#             joint_loss = loss_function(
#                 pred_joints, batch['gt_j3d']/1000)

#             optimizer.zero_grad()
#             joint_loss.backward()
#             optimizer.step()

#             if(args.wandb_log):

#                 wandb.log(
#                     {
#                         # "loss": loss.item(),
#                         "mpjpe_before": mpjpe_before.item(),
#                         "pampjpe_before": pampjpe_before.item(),
#                         "mpjpe_after": mpjpe_after.item(),
#                         "pampjpe_after": pampjpe_after.item(),
#                         "mpjpe_difference": mpjpe_after.item()-mpjpe_before.item(),
#                         "pampjpe_difference": pampjpe_after.item()-pampjpe_before.item(), })

#             if(args.wandb_log and iteration % 100 == 0):

#                 with torch.no_grad():

#                     batch = next(val_iterator)

#                     for item in batch:
#                         if(item != "valid" and item != "path" and item != "pixel_annotations"):
#                             batch[item] = batch[item].to(args.device).float()

#                     pred_joints, mpjpe_before, pampjpe_before, mpjpe_after, pampjpe_after = run_epoch(
#                         batch, train=False)

#                 wandb.log(
#                     {
#                         # "loss": loss.item(),
#                         "validation_mpjpe_before": mpjpe_before.item(),
#                         "validation_pampjpe_before": pampjpe_before.item(),
#                         "validation_mpjpe_after": mpjpe_after.item(),
#                         "validation_pampjpe_after": pampjpe_after.item(),
#                         "validation_mpjpe_difference": mpjpe_after.item()-mpjpe_before.item(),
#                         "validation_pampjpe_difference": pampjpe_after.item()-pampjpe_before.item(), })

#         print(f"epoch {epoch}")
#         torch.save(model.state_dict(),
#                    f"models/transformer_refiner_epoch_{epoch}.pt")


# def train_error_estimator_parametric():

#     from error_estimator import Error_Estimator
#     import copy

#     spin = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
#     checkpoint = torch.load(
#         "SPIN/data/model_checkpoint.pt", map_location=args.device)
#     spin.load_state_dict(checkpoint['model'], strict=False)
#     spin.eval()

#     smpl = SMPL(
#         '{}'.format("SPIN/data/smpl"),
#         batch_size=1,
#     ).to(args.device)

#     J_regressor = torch.from_numpy(
#         np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

#     J_regressor.requires_grad = True

#     j_reg_mask = utils.find_j_reg_mask(J_regressor)

#     spin.eval()

#     model = Error_Estimator().to(args.device)
#     model.train()
#     for param in model.resnet.parameters():
#         param.requires_grad = False
#     optimizer = optim.Adam(
#         model.parameters(), lr=args.learning_rate)

#     loss_function = nn.MSELoss()

#     data = data_set("train")
#     val_data = data_set("validation")

#     loader = torch.utils.data.DataLoader(
#         data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
#     val_loader = torch.utils.data.DataLoader(
#         val_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)

#     num_optim_iters = 10

#     normalize = transforms.Normalize(
#         (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

#     for epoch in range(args.train_epochs):

#         iterator = iter(loader)
#         val_iterator = iter(val_loader)

#         this_batch = next(iterator)
#         # batch = next(iterator)

#         for iteration in tqdm(range(len(loader))):

#             batch = copy.deepcopy(this_batch)

#             for item in batch:
#                 if(item != "valid" and item != "path" and item != "pixel_annotations"):
#                     batch[item] = batch[item].to(args.device).float()

#             # try:
#             #     batch = next(iterator)
#             # except:
#             #     print("problem loading batch")
#             #     time.sleep(1)
#             #     continue

#             # for item in batch:
#             #     if(item != "valid" and item != "path" and item != "pixel_annotations"):
#             #         batch[item] = batch[item].to(args.device).float()

#             batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

#             spin_image = normalize(batch['image'])

#             with torch.no_grad():
#                 spin_pred_pose, pred_betas, pred_camera = spin(
#                     spin_image)

#             spin_pred_pose.requires_grad = True

#             this_batch_optim = optim.Adam(
#                 [spin_pred_pose], lr=args.opt_lr)

#             all_pred_vertices = []
#             all_gt_errors = []

#             model.eval()

#             for i in range(num_optim_iters):

#                 batch["iteration"] = i

#                 pred_rotmat = rot6d_to_rotmat(
#                     spin_pred_pose).view(-1, 24, 3, 3)

#                 batch["orient"] = spin_pred_pose[:, :1]
#                 batch["pose"] = spin_pred_pose[:, 1:]
#                 batch["betas"] = pred_betas

#                 pred_joints, pred_verts = utils.find_joints(
#                     smpl,
#                     batch["betas"],
#                     pred_rotmat[:, 0].unsqueeze(1),
#                     pred_rotmat[:, 1:],
#                     J_regressor,
#                     mask=j_reg_mask,
#                     return_verts=True)

#                 pred_verts[:, :, 1] *= -1
#                 pred_verts[:, :, 0] *= -1
#                 pred_verts *= 2

#                 batch["pred_vertices"] = pred_verts

#                 pred_cam_t = torch.stack([-2*pred_camera[:, 1],
#                                           -2*pred_camera[:, 2],
#                                           2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
#                 batch["cam"] = pred_cam_t.detach()

#                 all_pred_vertices.append(pred_verts.detach())

#                 if(i == 0):
#                     mpjpe_before, pampjpe_before = utils.evaluate(
#                         pred_joints, batch['gt_j3d'])

#                 estimated_error = model(batch)

#                 gt_errors = torch.sqrt(((pred_joints - batch["gt_j3d"]/1000) **
#                                         2).sum(dim=-1))
#                 all_gt_errors.append(gt_errors.detach())

#                 estimated_error = torch.mean(estimated_error)

#                 this_batch_optim.zero_grad()
#                 estimated_error.backward()
#                 this_batch_optim.step()

#             model.train()

#             mpjpe_after, pampjpe_after = utils.evaluate(
#                 pred_joints, batch['gt_j3d'])

#             # print("mpjpe_after")
#             # print(mpjpe_after)
#             # print("pampjpe_after")
#             # print(pampjpe_after)

#             all_pred_vertices = torch.cat(all_pred_vertices, dim=0)
#             all_gt_errors = torch.cat(all_gt_errors, dim=0)

#             error_batch = {
#                 "image": torch.cat([batch["image"]]*num_optim_iters, dim=0),
#                 "cam": torch.cat([batch["cam"]]*num_optim_iters, dim=0),
#                 "pred_vertices": all_pred_vertices,
#                 "iteration": 99,
#             }

#             estimated_error = model(error_batch)

#             loss = loss_function(estimated_error, all_gt_errors)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if(args.wandb_log):

#                 wandb.log(
#                     {
#                         "loss": loss.item(),
#                         "mpjpe_before": mpjpe_before.item(),
#                         "pampjpe_before": pampjpe_before.item(),
#                         "mpjpe_after": mpjpe_after.item(),
#                         "pampjpe_after": pampjpe_after.item(),
#                         "mpjpe_difference": mpjpe_after.item()-mpjpe_before.item(),
#                         "pampjpe_difference": pampjpe_after.item()-pampjpe_before.item(), })

#         exit()
#         print(f"epoch {epoch}")
#         torch.save(model.state_dict(),
#                    f"models/transformer_refiner_epoch_{epoch}.pt")


##############################################################################################################################

    # spin_model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
    # checkpoint = torch.load(
    #     "SPIN/data/model_checkpoint.pt", map_location=args.device)
    # spin_model.load_state_dict(checkpoint['model'], strict=False)
    # spin_model.eval()

    # smpl = SMPL(
    #     '{}'.format("SPIN/data/smpl"),
    #     batch_size=1,
    # ).to(args.device)

    # J_regressor = torch.from_numpy(
    #     np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

    # J_regressor.requires_grad = True

    # j_reg_mask = utils.find_j_reg_mask(J_regressor)

    # pose_refiner = Pose_Refiner().to(args.device)
    # checkpoint = torch.load(
    #     "models/pose_refiner_epoch_6.pt", map_location=args.device)
    # pose_refiner.load_state_dict(checkpoint)
    # pose_refiner.eval()

    # loss_function = nn.MSELoss()

    # img_renderer = Renderer(subset=False)

    # data = data_set("train")
    # val_data = data_set("validation")

    # loader = torch.utils.data.DataLoader(
    #     data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(
    #     val_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)

    # normalize = transforms.Normalize(
    #     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # optimizer = optim.Adam([J_regressor], lr=args.j_reg_lr)

    # for epoch in range(5):

    #     iterator = iter(loader)
    #     val_iterator = iter(val_loader)

    #     for iteration in tqdm(range(len(loader))):

    #         try:
    #             batch = next(iterator)
    #         except:
    #             continue

    #         for item in batch:
    #             if(item != "valid" and item != "path" and item != "pixel_annotations"):
    #                 batch[item] = batch[item].to(args.device).float()

    #         # train generator

    #         batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

    #         spin_image = normalize(batch['image'])

    #         with torch.no_grad():
    #             spin_pred_pose, pred_betas, pred_camera = spin_model(
    #                 spin_image)

    #         pred_rotmat = rot6d_to_rotmat(spin_pred_pose).view(-1, 24, 3, 3)

    #         batch["orient"] = spin_pred_pose[:, :1]
    #         batch["pose"] = spin_pred_pose[:, 1:]
    #         batch["betas"] = pred_betas

    #         pred_joints = utils.find_joints(
    #             smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor, mask=j_reg_mask)

    #         est_pose, est_betas, est_cam = pose_refiner(batch)

    #         batch["orient"] = est_pose[:, :1]
    #         batch["pose"] = est_pose[:, 1:]
    #         batch["betas"] = est_betas
    #         batch["cam"] = est_cam

    #         pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

    #         pred_joints = utils.find_joints(
    #             smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor, mask=j_reg_mask)

    #         pred_joints = utils.move_pelvis(pred_joints)

    #         loss = loss_function(pred_joints, batch["gt_j3d"]/1000)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         if(args.wandb_log):
    #             wandb.log({"loss": loss.item()})

    #         if(args.wandb_log and iteration % 100 == 0):

    #             with torch.no_grad():
    #                 batch = next(val_iterator)

    #                 for item in batch:

    #                     if(item != "valid" and item != "path" and item != "pixel_annotations"):
    #                         batch[item] = batch[item].to(args.device).float()

    #                 batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

    #                 spin_image = normalize(batch['image'])

    #                 spin_pred_pose, pred_betas, pred_camera = spin_model(
    #                     spin_image)

    #                 pred_rotmat = rot6d_to_rotmat(
    #                     spin_pred_pose).view(-1, 24, 3, 3)

    #                 batch["orient"] = spin_pred_pose[:, :1]
    #                 batch["pose"] = spin_pred_pose[:, 1:]
    #                 batch["betas"] = pred_betas

    #                 pred_joints = utils.find_joints(
    #                     smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor)

    #                 est_pose, est_betas, est_cam = pose_refiner(batch)

    #                 batch["orient"] = est_pose[:, :1]
    #                 batch["pose"] = est_pose[:, 1:]
    #                 batch["betas"] = est_betas
    #                 batch["cam"] = est_cam

    #                 pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

    #                 pred_joints = utils.find_joints(
    #                     smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

    #                 pred_joints = utils.move_pelvis(pred_joints)

    #                 loss = loss_function(pred_joints, batch["gt_j3d"]/1000)

    #                 wandb.log({"val_loss": loss.item()})

    #     torch.save(J_regressor,
    #                f"models/j_regressor_epoch_{epoch}.pt")
