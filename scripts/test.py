import time
import torch
from torch import nn, optim

import os

from scripts import utils

from tqdm import tqdm

from scripts.args import args

from scripts.data import data_set

import numpy as np

from scripts.smpl import SMPL, SMPL_MODEL_DIR

from torchvision import transforms


# SPIN
# import sys
# sys.path.append('/scratch/iamerich/SPIN')
from SPIN.models import hmr, SMPL
import SPIN.config as config
from SPIN.utils.geometry import rot6d_to_rotmat

import sys  # noqa
sys.path.append('/scratch/iamerich/MEVA')  # noqa


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

    J_regressor = torch.load(
        'models/retrained_J_Regressor.pt').to(args.device)
    J_regressor_initial = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

    J_regressor_og = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)
    j_reg_mask = utils.find_j_reg_mask(J_regressor_og)

    loss_function = nn.MSELoss()

    # img_renderer = Mesh_Renderer()

    data = data_set("validation")
    # data = data_set("train")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    mpjpe_before = []
    pampjpe_before = []

    mpjpe_after = []
    pampjpe_after = []

    iterator = iter(loader)

    for iteration in tqdm(range(len(loader))):

        with torch.no_grad():

            batch = next(iterator)
            # except:
            #     print("problem loading batch")
            #     import time
            #     time.sleep(1)
            #     continue

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations"):
                    batch[item] = batch[item].to(args.device).float()

            # train generator

            batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

            spin_image = normalize(batch['spin_image'])

            spin_pred_pose, pred_betas, pred_camera = model(
                spin_image)

            pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                      -2*pred_camera[:, 2],
                                      2*5000/(256 * pred_camera[:, 0] + 1e-9)], dim=-1)
            batch["cam"] = pred_cam_t

            pred_rotmat = rot6d_to_rotmat(spin_pred_pose).view(-1, 24, 3, 3)

            pred_joints = utils.find_joints(
                smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor_initial, mask=j_reg_mask)

            mpjpe_before_refinement, pampjpe_before_refinement = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            mpjpe_before.append(torch.tensor(mpjpe_before_refinement))
            pampjpe_before.append(torch.tensor(pampjpe_before_refinement))

            pred_joints = utils.find_joints(
                smpl, batch["betas"], pred_rotmat[:, 0].unsqueeze(1), pred_rotmat[:, 1:], J_regressor, mask=j_reg_mask)

            mpjpe_after_refinement, pampjpe_after_refinement = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            mpjpe_after.append(torch.tensor(mpjpe_after_refinement))
            pampjpe_after.append(torch.tensor(pampjpe_after_refinement))

    print("MPJPE")
    print(
        f"{torch.mean(torch.stack(mpjpe_before)):.4f}")
    print("PAMPJPE")
    print(
        f"{torch.mean(torch.stack(pampjpe_before)):.4f}")
    print()
    print(f"after")
    print("MPJPE")
    print(
        f"{torch.mean(torch.stack(mpjpe_after)):.4f}")
    print("PAMPJPE")
    print(
        f"{torch.mean(torch.stack(pampjpe_after)):.4f}")


def test_pose_refiner_model_VIBE_MEVA(vibe=True):

    import os

    if(vibe):
        print("testing on VIBE")
        import sys  # noqa
        sys.path.append('/scratch/iamerich/VIBE')  # noqa

        from lib.utils.demo_utils import download_ckpt
        from lib.models.vibe import VIBE_Demo

        model = VIBE_Demo(
            seqlen=16,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
        ).to(args.device)
        pretrained_file = download_ckpt(use_3dpw=False)
        ckpt = torch.load(pretrained_file)
        print(
            f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
        ckpt = ckpt['gen_state_dict']
        model.load_state_dict(ckpt, strict=False)
        model.eval()
    else:

        print("testing on MEVA")

        from meva.lib.meva_model import MEVA_demo
        from meva.utils.video_config import update_cfg

        os.chdir("/scratch/iamerich/MEVA/")  # noqa

        pretrained_file = f"/scratch/iamerich/MEVA/results/meva/train_meva_2/model_best.pth.tar"

        config_file = "/scratch/iamerich/MEVA/meva/cfg/train_meva_2.yml"
        cfg = update_cfg(config_file)
        model = MEVA_demo(
            n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            seqlen=cfg.DATASET.SEQLEN,
            hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
            add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
            bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
            use_residual=cfg.MODEL.TGRU.RESIDUAL,
            cfg=cfg.VAE_CFG,
        ).to(args.device)
        ckpt = torch.load(pretrained_file)
        # print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
        ckpt = ckpt['gen_state_dict']
        model.load_state_dict(ckpt)
        model.eval()
        os.chdir("/scratch/iamerich/human-body-pose/")  # noqa

    J_regressor = torch.load(
        'models/retrained_J_Regressor.pt').to(args.device)
    J_regressor_initial = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

    J_regressor_og = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)
    j_reg_mask = utils.find_j_reg_mask(J_regressor_og)

    J_regressor = nn.ReLU()(J_regressor)
    J_regressor = J_regressor / torch.sum(J_regressor, dim=1).unsqueeze(
        1).expand(J_regressor.shape)

    J_regressor_initial = nn.ReLU()(J_regressor_initial)
    J_regressor_initial = J_regressor_initial / torch.sum(J_regressor_initial, dim=1).unsqueeze(
        1).expand(J_regressor_initial.shape)

    # for param in pose_refiner.resnet.parameters():
    #     param.requires_grad = False

    loss_function = nn.MSELoss()

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

    os.chdir("/scratch/iamerich/MEVA/")  # noqa

    for iteration in tqdm(range(len(loader))):

        batch = next(iterator)

        for item in batch:
            if(item != "valid" and item != "path" and item != "pixel_annotations"):
                batch[item] = batch[item].to(args.device).float()

        # train generator

        batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

        spin_image = normalize(batch['spin_image'])

        # print("J_regressor.shape")
        # print(J_regressor.shape)

        with torch.no_grad():
            output = model(
                spin_image, J_regressor=J_regressor_initial)[0]

        output["kp_3d"] = output["kp_3d"][:, 0]

        # print(output["kp_3d"].shape)
        # print(batch["gt_j3d"].shape)

        output["kp_3d"] = utils.move_pelvis(output["kp_3d"])

        mpjpe_before_refinement, pampjpe_before_refinement = utils.evaluate(
            output["kp_3d"], batch['gt_j3d'])

        mpjpe_before.append(torch.tensor(mpjpe_before_refinement))
        pampjpe_before.append(torch.tensor(pampjpe_before_refinement))

        with torch.no_grad():
            output = model(
                spin_image, J_regressor=J_regressor)[0]

        output["kp_3d"] = output["kp_3d"][:, 0]

        # print(output["kp_3d"].shape)
        # print(batch["gt_j3d"].shape)

        output["kp_3d"] = utils.move_pelvis(output["kp_3d"])

        mpjpe_after_refinement, pampjpe_after_refinement = utils.evaluate(
            output["kp_3d"], batch['gt_j3d'])

        mpjpe_after.append(torch.tensor(mpjpe_after_refinement))
        pampjpe_after.append(torch.tensor(pampjpe_after_refinement))

    print("MPJPE")
    print(
        f"{torch.mean(torch.stack(mpjpe_before)):.4f}")
    print("PAMPJPE")
    print(
        f"{torch.mean(torch.stack(pampjpe_before)):.4f}")
    print()
    print(f"after")
    print("MPJPE")
    print(
        f"{torch.mean(torch.stack(mpjpe_after)):.4f}")
    print("PAMPJPE")
    print(
        f"{torch.mean(torch.stack(pampjpe_after)):.4f}")


# def test_pose_refiner_translation_model():

#     import os
#     os.chdir("/scratch/iamerich/MeshTransformer/")  # noqa

#     import transformer

#     metro, smpl, mesh = transformer.load_transformer()

#     os.chdir("/scratch/iamerich/human-body-pose/")  # noqa

#     metro.eval()

#     loss_function = nn.MSELoss()

#     data = data_set("validation")

#     loader = torch.utils.data.DataLoader(
#         data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)

#     normalize = transforms.Normalize(
#         (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

#     img_renderer = Renderer(subset=True)

#     mpj = []
#     pampj = []

#     iterator = iter(loader)

#     for iteration in tqdm(range(len(loader))):

#         batch = next(iterator)

#         for item in batch:
#             if(item != "valid" and item != "path" and item != "pixel_annotations"):
#                 batch[item] = batch[item].to(args.device).float()

#         # train generator

#         batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

#         spin_image = normalize(batch['image'])

#         with torch.no_grad():
#             # pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, _, _ = metro(
#             #     spin_image, smpl, mesh)
#             pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = metro(
#                 spin_image, smpl, mesh)

#         pred_cam_t = torch.stack([-2*pred_camera[:, 1],
#                                   -2*pred_camera[:, 2],
#                                   2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
#         batch["cam"] = pred_cam_t

#         # utils.render_batch(img_renderer, batch, "transformer",
#         #                    vertices=pred_vertices)

#         pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

#         pred_3d_joints_from_smpl = utils.move_pelvis(
#             pred_3d_joints_from_smpl)

#         pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,
#                                                             constants.J17_2_METRO]

#         # mpjpe, pampjpe = utils.evaluate(
#         #     pred_joints, batch['gt_j3d'])
#         mpjpe, pampjpe = utils.evaluate(
#             pred_3d_joints_from_smpl, batch['gt_j3d'])

#         mpj.append(torch.tensor(mpjpe))
#         pampj.append(torch.tensor(pampjpe))

#         print("MPJPE")
#         print(
#             f"{torch.mean(torch.stack(mpj)):.4f}")
#         print("PAMPJPE")
#         print(
#             f"{torch.mean(torch.stack(pampj)):.4f}")

#     # print(mpjpe)
#     # print(pampjpe)
