from args import args
import wandb
import torch
from torch import nn, optim

from tqdm import tqdm


from pose_refiner import Pose_Refiner
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

from eval_utils import batch_compute_similarity_transform_torch

from SPIN.utils.geometry import rot6d_to_rotmat


def evaluate(pred_j3ds, target_j3ds):

    with torch.no_grad():

        pred_j3ds = pred_j3ds.clone().detach()
        target_j3ds = target_j3ds.clone().detach()
        target_j3ds /= 1000

        # print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        pred_pelvis = pred_j3ds[:, [0], :].clone()
        target_pelvis = target_j3ds[:, [0], :].clone()

        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis

        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds - target_j3ds) **
                             2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(
            pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(
            ((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        m2mm = 1000

        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm

        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
        }

        # log_str = ' '.join(
        #     [f'{k.upper()}: {v:.4f},'for k, v in eval_dict.items()])
        # print(log_str)

    return mpjpe, pa_mpjpe


def find_joints(smpl, shape, orient, pose, J_regressor):

    pred_vertices = smpl(global_orient=orient, body_pose=pose,
                         betas=shape, pose2rot=False).vertices
    J_regressor_batch = J_regressor[None, :].expand(
        pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
    pred_joints = torch.matmul(J_regressor_batch, pred_vertices)

    return pred_joints


def move_pelvis(j3ds):
    # move the hip location of gt to estimated
    pelvis = j3ds[:, [0], :].clone()

    j3ds -= pelvis

    return j3ds


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

    J_regressor = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

    pose_refiner = Pose_Refiner().to(args.device)
    checkpoint = torch.load(
        "models/best_pose_refiner/pose_refiner_epoch_0.pt", map_location=args.device)
    spin_model.load_state_dict(checkpoint['model'], strict=False)
    pose_refiner.train()
    print("model load worked succesfully")
    exit()
    pose_discriminator = Discriminator().to(args.device)
    pose_discriminator.train()

    for param in pose_refiner.resnet.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        pose_refiner.parameters(), lr=args.learning_rate)
    disc_optimizer = optim.Adam(
        pose_discriminator.parameters(), lr=args.learning_rate)

    loss_function = nn.MSELoss()

    data = data_set("train")
    val_data = data_set("validation")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.training_batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.training_batch_size, num_workers=1, pin_memory=True, shuffle=True, drop_last=True)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for epoch in range(args.train_epochs):

        total_loss = 0

        iterator = iter(loader)
        val_iterator = iter(val_loader)

        for iteration in tqdm(range(len(loader))):

            batch = next(iterator)

            for item in batch:
                batch[item] = batch[item].to(args.device).float()

            # train generator

            batch['gt_j3d'] = move_pelvis(batch['gt_j3d'])

            spin_image = normalize(batch['image'])

            with torch.no_grad():
                pred_pose, pred_betas, pred_camera = spin_model(
                    spin_image)

            pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                      -2*pred_camera[:, 2],
                                      2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
            batch["gt_translation"] = pred_cam_t

            pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 24, 3, 3)

            batch["pose"] = pred_pose[:, 1:]
            batch["orient"] = pred_pose[:, 0].unsqueeze(1)
            batch["betas"] = pred_betas

            pred_joints = find_joints(
                smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor)

            mpjpe_before_refinement, pampjpe_before_refinement = evaluate(
                pred_joints, batch['gt_j3d'])

            pred_rotmat, pred_rot6d = pose_refiner(batch)

            batch["orient"] = pred_rot6d[:, :1]
            batch["pose"] = pred_rot6d[:, 1:]

            pred_joints = find_joints(
                smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

            pred_joints = move_pelvis(pred_joints)

            mpjpe_after, pampjpe_after = evaluate(
                pred_joints, batch['gt_j3d'])

            joint_loss = loss_function(pred_joints, batch['gt_j3d']/1000)

            # add a loss so the estimates dont stray too far from original
            pred_disc = pose_discriminator(pred_rot6d)

            discriminated_loss = loss_function(pred_disc, torch.ones(
                pred_disc.shape).to(args.device))

            loss = joint_loss+discriminated_loss/1000

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train discriminator

            pred_gt = pose_discriminator(pred_pose)
            pred_disc = pose_discriminator(pred_rot6d.detach())

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
                        "discriminator_loss": discriminator_loss,
                        "mpjpe_before_refinement": mpjpe_before_refinement.item(),
                        "pampjpe_before_refinement": pampjpe_before_refinement.item(),
                        "mpjpe_after": mpjpe_after.item(),
                        "pampjpe_after": pampjpe_after.item(),
                        "mpjpe_difference": mpjpe_after.item()-mpjpe_before_refinement.item(),
                        "pampjpe_difference": pampjpe_after.item()-pampjpe_before_refinement.item(), })

            if(args.wandb_log and iteration % 100 == 0):

                with torch.no_grad():

                    pose_refiner.eval()

                    batch = next(val_iterator)

                    for item in batch:
                        batch[item] = batch[item].to(args.device).float()

                    spin_image = normalize(batch['image'])

                    pred_pose, pred_betas, pred_camera = spin_model(
                        spin_image)

                    pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                              -2*pred_camera[:, 2],
                                              2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
                    batch["gt_translation"] = pred_cam_t

                    pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 24, 3, 3)

                    batch["pose"] = pred_pose[:, 1:]
                    batch["orient"] = pred_pose[:, 0].unsqueeze(1)
                    batch["betas"] = pred_betas

                    pred_joints = find_joints(
                        smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor)

                    pred_joints = move_pelvis(pred_joints)

                    mpjpe_before_refinement, pampjpe_before_refinement = evaluate(
                        pred_joints, batch['gt_j3d'])

                    pred_rotmat, pred_rot6d = pose_refiner(batch)

                    batch["pose"] = pred_rot6d[:, 1:]
                    batch["orient"] = pred_rot6d[:, 0].unsqueeze(1)

                    pred_joints = find_joints(
                        smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

                    mpjpe_after, pampjpe_after = evaluate(
                        pred_joints, batch['gt_j3d'])

                    pose_refiner.train()

                wandb.log(
                    {
                        "validation mpjpe_before_refinement": mpjpe_before_refinement.item(),
                        "validation pampjpe_before_refinement": pampjpe_before_refinement.item(),
                        "validation mpjpe_after": mpjpe_after.item(),
                        "validation pampjpe_after": pampjpe_after.item(),
                        "validation mpjpe_difference": mpjpe_after.item()-mpjpe_before_refinement.item(),
                        "validation pampjpe_difference": pampjpe_after.item()-pampjpe_before_refinement.item(), })

        print(f"epoch: {epoch}, loss: {total_loss}")

        torch.save(pose_refiner.state_dict(),
                   f"models/pose_refiner_epoch_{epoch}.pt")
