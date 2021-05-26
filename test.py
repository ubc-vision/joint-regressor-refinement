import torch
from torch import nn, optim

import os

from utils import utils

from tqdm import tqdm

from args import args

from data import load_data, data_set

import numpy as np

from smpl import SMPL, SMPL_MODEL_DIR

from torchvision import transforms

from warp import perturbation_helper, sampling_helper

# from create_smpl_gt import find_crop, rotation_matrix_to_quaternion, quaternion_to_rotation_matrix, find_joints

import copy

from pose_refiner import Pose_Refiner
from renderer import Renderer, return_2d_joints


# SPIN
import sys
# sys.path.append('/scratch/iamerich/SPIN')
from SPIN.models import hmr, SMPL
import SPIN.config as config
from SPIN.utils.geometry import rot6d_to_rotmat


def test_pose_refiner_model():

    spin_model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
    checkpoint = torch.load(
        "SPIN/data/model_checkpoint.pt", map_location=args.device)
    spin_model.load_state_dict(checkpoint['model'], strict=False)
    spin_model.eval()

    smpl = SMPL(
        '{}'.format("SPIN/data/smpl"),
        batch_size=1,
    ).to(args.device)

    # J_regressor = torch.from_numpy(
    #     np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)
    J_regressor = torch.load(
        'models/best_pose_refiner/j_regressor.pt')[0].to(args.device)

    # new_J_regressor = torch.load(
    #     'models/j_regressor_epoch_0.pt')[0].to(args.device)

    pose_refiner = Pose_Refiner().to(args.device)
    checkpoint = torch.load(
        "models/pose_refiner_epoch_0.pt", map_location=args.device)
    pose_refiner.load_state_dict(checkpoint)
    pose_refiner.eval()

    for param in pose_refiner.resnet.parameters():
        param.requires_grad = False

    loss_function = nn.MSELoss()

    img_renderer = Renderer(subset=False)

    data = data_set("validation")
    # data = data_set("train")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    mpjpe_before = []
    pampjpe_before = []
    mpjpe_after = []
    pampjpe_after = []
    mpjpe_after_j_reg = []
    pampjpe_after_j_reg = []

    total_loss = 0

    iterator = iter(loader)

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

        # pred_cam_t = torch.stack([-2*pred_camera[:, 1],
        #                           -2*pred_camera[:, 2],
        #                           2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
        # batch["cam"] = pred_cam_t

        pred_rotmat = rot6d_to_rotmat(spin_pred_pose).view(-1, 24, 3, 3)

        batch["orient"] = spin_pred_pose[:, :1]
        batch["pose"] = spin_pred_pose[:, 1:]
        batch["betas"] = pred_betas

        # utils.render_batch(img_renderer, batch, "initial")

        pred_joints = utils.find_joints(
            smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor)

        mpjpe_before_refinement, pampjpe_before_refinement = utils.evaluate(
            pred_joints, batch['gt_j3d'])

        mpjpe_before.append(torch.tensor(mpjpe_before_refinement))
        pampjpe_before.append(torch.tensor(pampjpe_before_refinement))

        est_pose, est_betas, est_cam = pose_refiner(batch)

        batch["orient"] = est_pose[:, :1]
        batch["pose"] = est_pose[:, 1:]
        batch["betas"] = est_betas
        batch["cam"] = est_cam

        pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

        pred_joints = utils.find_joints(
            smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

        # joints_2d = return_2d_joints(batch, smpl, J_regressor=J_regressor)

        # utils.render_batch(img_renderer, batch, "reg_0", joints_2d)

        pred_joints = utils.move_pelvis(pred_joints)

        mpjpe_after_refinement, pampjpe_after_refinement = utils.evaluate(
            pred_joints, batch['gt_j3d'])

        mpjpe_after.append(torch.tensor(mpjpe_after_refinement))
        pampjpe_after.append(torch.tensor(pampjpe_after_refinement))

        # verify new joint regressor
        # pred_joints = utils.find_joints(
        #     smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], new_J_regressor)

        # joints_2d = return_2d_joints(batch, smpl, J_regressor=new_J_regressor)

        # utils.render_batch(img_renderer, batch, "reg_1", joints_2d)
        # exit()

        pred_joints = utils.move_pelvis(pred_joints)

        mpjpe_after_refinement_j_reg, pampjpe_after_refinement_j_reg = utils.evaluate(
            pred_joints, batch['gt_j3d'])

        mpjpe_after_j_reg.append(torch.tensor(mpjpe_after_refinement_j_reg))
        pampjpe_after_j_reg.append(
            torch.tensor(pampjpe_after_refinement_j_reg))

    print("mpjpe_before")
    print(torch.mean(torch.stack(mpjpe_before)))
    print("pampjpe_before")
    print(torch.mean(torch.stack(pampjpe_before)))
    print("mpjpe_after")
    print(torch.mean(torch.stack(mpjpe_after)))
    print("pampjpe_after")
    print(torch.mean(torch.stack(pampjpe_after)))
    print("mpjpe_after_j_reg")
    print(torch.mean(torch.stack(mpjpe_after_j_reg)))
    print("pampjpe_after_j_reg")
    print(torch.mean(torch.stack(pampjpe_after_j_reg)))
