import time
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

import constants


# SPIN
# import sys
# sys.path.append('/scratch/iamerich/SPIN')
from SPIN.models import hmr, SMPL
import SPIN.config as config
from SPIN.utils.geometry import rot6d_to_rotmat

import sys  # noqa
sys.path.append('/scratch/iamerich/VIBE')  # noqa

from lib.utils.demo_utils import download_ckpt
from lib.models.vibe import VIBE_Demo

# import sys  # noqa
# sys.path.append('/scratch/iamerich/MEVA')  # noqa

# import os

# from meva.lib.meva_model import MEVA_demo
# from meva.utils.video_config import update_cfg


def test_pose_refiner_model():

    model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
    checkpoint = torch.load(
        "SPIN/data/model_checkpoint.pt", map_location=args.device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    smpl = SMPL(
        '{}'.format("SPIN/data/smpl"),
        batch_size=1,
    ).to(args.device)

    initial_J_regressor = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)
    J_regressor = torch.load(
        'models/best_pose_refiner/retrained_J_Regressor.pt').to(args.device)
    # J_regressor = torch.load(
    #     'models/pose_refiner_epoch_0.pt').to(args.device)
    # J_regressor = torch.load(
    #     'models/retrained_J_Regressor.pt').to(args.device)
    # J_regressor = torch.from_numpy(
    #     np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

    # new_J_regressor = torch.load(
    #     'models/j_regressor_epoch_0.pt')[0].to(args.device)

    num_networks = 5

    pose_refiners = [Pose_Refiner().to(args.device)
                     for _ in range(num_networks)]
    for i in range(num_networks):
        checkpoint = torch.load(
            f"models/pose_refiner_{i}_epoch_0.pt", map_location=args.device)
        pose_refiners[i].load_state_dict(checkpoint)
        pose_refiners[i].eval()

        for param in pose_refiners[i].resnet.parameters():
            param.requires_grad = False

    loss_function = nn.MSELoss()

    img_renderer = Renderer(subset=False)

    data = data_set("validation")
    # data = data_set("train")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    mpjpe_before = []
    pampjpe_before = []

    mpjpe_after = [[] for i in range(num_networks)]
    pampjpe_after = [[] for i in range(num_networks)]

    iterator = iter(loader)

    for iteration in tqdm(range(len(loader))):

        with torch.no_grad():

            batch = next(iterator)

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations"):
                    batch[item] = batch[item].to(args.device).float()

            # train generator

            batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

            spin_image = normalize(batch['image'])

            spin_pred_pose, pred_betas, pred_camera = model(
                spin_image)

            pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                      -2*pred_camera[:, 2],
                                      2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
            batch["cam"] = pred_cam_t

            pred_rotmat = rot6d_to_rotmat(spin_pred_pose).view(-1, 24, 3, 3)

            batch["orient"] = spin_pred_pose[:, :1]
            batch["pose"] = spin_pred_pose[:, 1:]
            batch["betas"] = pred_betas

            # initial_joints_2d = return_2d_joints(
            #     batch, smpl, J_regressor=initial_J_regressor)

            # joints_2d = return_2d_joints(batch, smpl, J_regressor=J_regressor)
            # utils.render_batch(img_renderer, batch, "initial",
            #                    [joints_2d, initial_joints_2d, batch["gt_j2d"]])

            pred_joints = utils.find_joints(
                smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor)

            mpjpe_before_refinement, pampjpe_before_refinement = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            mpjpe_before.append(torch.tensor(mpjpe_before_refinement))
            pampjpe_before.append(torch.tensor(pampjpe_before_refinement))

            for i in range(num_networks):

                # utils.render_batch(img_renderer, batch, f"refine_{i}")

                est_pose, est_betas, est_cam = pose_refiners[i](batch)

                batch["orient"] = est_pose[:, :1]
                batch["pose"] = est_pose[:, 1:]
                batch["betas"] = est_betas
                batch["cam"] = est_cam

                pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

                pred_joints = utils.find_joints(
                    smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

                mpjpe_after_refinement, pampjpe_after_refinement = utils.evaluate(
                    pred_joints, batch['gt_j3d'])

                mpjpe_after[i].append(torch.tensor(mpjpe_after_refinement))
                pampjpe_after[i].append(torch.tensor(pampjpe_after_refinement))

    print("MPJPE")
    print(
        f"{torch.mean(torch.stack(mpjpe_before)):.4f}")
    print("PAMPJPE")
    print(
        f"{torch.mean(torch.stack(pampjpe_before)):.4f}")

    for i in range(num_networks):
        print()
        print(f"after {i}")
        print("MPJPE")
        print(
            f"{torch.mean(torch.stack(mpjpe_after[i])):.4f}")
        print("PAMPJPE")
        print(
            f"{torch.mean(torch.stack(pampjpe_after[i])):.4f}")


def test_pose_refiner_model_VIBE():

    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(args.device)
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # os.chdir("/scratch/iamerich/MEVA/")  # noqa

    # pretrained_file = f"/scratch/iamerich/MEVA/results/meva/train_meva_2/model_best.pth.tar"

    # config_file = "/scratch/iamerich/MEVA/meva/cfg/train_meva_2.yml"
    # cfg = update_cfg(config_file)
    # model = MEVA_demo(
    #     n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
    #     batch_size=cfg.TRAIN.BATCH_SIZE,
    #     seqlen=cfg.DATASET.SEQLEN,
    #     hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
    #     add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
    #     bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
    #     use_residual=cfg.MODEL.TGRU.RESIDUAL,
    #     cfg=cfg.VAE_CFG,
    # ).to(args.device)
    # ckpt = torch.load(pretrained_file)
    # # print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    # ckpt = ckpt['gen_state_dict']
    # model.load_state_dict(ckpt)
    # model.eval()

    # os.chdir("/scratch/iamerich/human-body-pose/")  # noqa

    initial_J_regressor = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)
    J_regressor = torch.load(
        'models/retrained_J_Regressor.pt').to(args.device)

    J_regressor = nn.ReLU()(J_regressor)
    J_regressor = J_regressor / torch.sum(J_regressor, dim=1).unsqueeze(
        1).expand(J_regressor.shape)

    # new_J_regressor = torch.load(
    #     'models/j_regressor_epoch_0.pt')[0].to(args.device)

    pose_refiner = Pose_Refiner().to(args.device)
    checkpoint = torch.load(
        "models/pose_refiner_epoch_2.pt", map_location=args.device)
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

    iterator = iter(loader)

    # os.chdir("/scratch/iamerich/MEVA/")  # noqa

    for iteration in tqdm(range(len(loader))):

        batch = next(iterator)

        for item in batch:
            if(item != "valid" and item != "path" and item != "pixel_annotations"):
                batch[item] = batch[item].to(args.device).float()

        # train generator

        batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

        spin_image = normalize(batch['image'])

        # print("J_regressor.shape")
        # print(J_regressor.shape)

        with torch.no_grad():
            output = model(
                spin_image, J_regressor=J_regressor)[0]

        output["kp_3d"] = output["kp_3d"][:, 0]

        # print(output["kp_3d"].shape)
        # print(batch["gt_j3d"].shape)

        output["kp_3d"] = utils.move_pelvis(output["kp_3d"])

        mpjpe_before_refinement, pampjpe_before_refinement = utils.evaluate(
            output["kp_3d"], batch['gt_j3d'])

        mpjpe_before.append(torch.tensor(mpjpe_before_refinement))
        pampjpe_before.append(torch.tensor(pampjpe_before_refinement))

        # est_pose, est_betas, est_cam = pose_refiner(batch)

        # batch["orient"] = est_pose[:, :1]
        # batch["pose"] = est_pose[:, 1:]
        # batch["betas"] = est_betas
        # batch["cam"] = est_cam

        # pred_rotmat = rot6d_to_rotmat(est_pose).view(-1, 24, 3, 3)

        # pred_joints = utils.find_joints(
        #     smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

        # # initial_joints_2d = return_2d_joints(
        # #     batch, smpl, J_regressor=initial_J_regressor)

        # # joints_2d = return_2d_joints(batch, smpl, J_regressor=J_regressor)

        # # utils.render_batch(img_renderer, batch, "reg_0",
        # #                    [joints_2d, initial_joints_2d, batch["gt_j2d"]])
        # # exit()

        # pred_joints = utils.move_pelvis(pred_joints)

        # mpjpe_after_refinement, pampjpe_after_refinement = utils.evaluate(
        #     pred_joints, batch['gt_j3d'])

        # mpjpe_after.append(torch.tensor(mpjpe_after_refinement))
        # pampjpe_after.append(torch.tensor(pampjpe_after_refinement))

        # # verify new joint regressor
        # # pred_joints = utils.find_joints(
        # #     smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], new_J_regressor)

        # # joints_2d = return_2d_joints(batch, smpl, J_regressor=new_J_regressor)

        # # utils.render_batch(img_renderer, batch, "reg_1", joints_2d)
        # # exit()

        # # pred_joints = utils.move_pelvis(pred_joints)

        # # mpjpe_after_refinement_j_reg, pampjpe_after_refinement_j_reg = utils.evaluate(
        # #     pred_joints, batch['gt_j3d'])

        # # mpjpe_after_j_reg.append(torch.tensor(mpjpe_after_refinement_j_reg))
        # # pampjpe_after_j_reg.append(
        #     torch.tensor(pampjpe_after_refinement_j_reg))
    print("MPJPE")
    print(
        f"{torch.mean(torch.stack(mpjpe_before)):.4f}")
    print("PAMPJPE")
    print(
        f"{torch.mean(torch.stack(pampjpe_before)): .4f}")


def test_pose_refiner_translation_model():

    import os
    os.chdir("/scratch/iamerich/MeshTransformer/")  # noqa

    import transformer

    metro, smpl, mesh = transformer.load_transformer()

    os.chdir("/scratch/iamerich/human-body-pose/")  # noqa

    metro.eval()

    loss_function = nn.MSELoss()

    data = data_set("validation")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    img_renderer = Renderer(subset=True)

    mpj = []
    pampj = []

    iterator = iter(loader)

    for iteration in tqdm(range(len(loader))):

        batch = next(iterator)

        for item in batch:
            if(item != "valid" and item != "path" and item != "pixel_annotations"):
                batch[item] = batch[item].to(args.device).float()

        # train generator

        batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

        spin_image = normalize(batch['image'])

        with torch.no_grad():
            # pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, _, _ = metro(
            #     spin_image, smpl, mesh)
            pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = metro(
                spin_image, smpl, mesh)

        pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                  -2*pred_camera[:, 2],
                                  2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
        batch["cam"] = pred_cam_t

        # utils.render_batch(img_renderer, batch, "transformer",
        #                    vertices=pred_vertices)

        pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

        pred_3d_joints_from_smpl = utils.move_pelvis(
            pred_3d_joints_from_smpl)

        pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,
                                                            constants.J17_2_METRO]

        # mpjpe, pampjpe = utils.evaluate(
        #     pred_joints, batch['gt_j3d'])
        mpjpe, pampjpe = utils.evaluate(
            pred_3d_joints_from_smpl, batch['gt_j3d'])

        mpj.append(torch.tensor(mpjpe))
        pampj.append(torch.tensor(pampjpe))

        print("MPJPE")
        print(
            f"{torch.mean(torch.stack(mpj)):.4f}")
        print("PAMPJPE")
        print(
            f"{torch.mean(torch.stack(pampj)):.4f}")

    # print(mpjpe)
    # print(pampjpe)
