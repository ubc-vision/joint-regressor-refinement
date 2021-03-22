from args import args
import wandb
import torch
from torch import nn, optim

from tqdm import tqdm

from render_model import Render_Model

from data import load_data, data_set, load_precompted
# from visualizer import draw_gradients

from torchvision import transforms

from create_smpl_gt import find_crop

from SPIN.models import hmr, SMPL
import SPIN.config as config

import numpy as np

import create_smpl_gt

import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, AlphaCompositor, PointsRenderer


def find_joints(smpl, shape, orient, pose, J_regressor):

    pred_vertices = smpl(global_orient=orient, body_pose=pose,
                         betas=shape, pose2rot=False).vertices
    J_regressor_batch = J_regressor[None, :].expand(
        pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
    pred_joints = torch.matmul(J_regressor_batch, pred_vertices)

    return pred_joints


def move_gt_pelvis(gt_j3ds, j3ds):
    # move the hip location of gt to estimated
    pred_pelvis = (j3ds[:, [2], :] + j3ds[:, [3], :]) / 2.0
    gt_pelvis = (gt_j3ds[:, [2], :] + gt_j3ds[:, [3], :]) / 2.0

    gt_j3ds -= gt_pelvis
    gt_j3ds += pred_pelvis

    return gt_j3ds


def find_pose_add_noise(normalize, spin_model, smpl, J_regressor, batch):

    batch_size = batch['image'].shape[0]

    initial_image = batch['image'].clone()

    image, min_x, min_y, scale, intrinsics = find_crop(
        batch['image'], batch['gt_j2d'], batch['intrinsics'])

    image = normalize(image)

    cropped_gt_2d = batch['gt_j2d'].clone()
    cropped_gt_2d[..., 0] -= min_x.view(-1, 1).expand(cropped_gt_2d.shape[:-1])
    cropped_gt_2d[..., 1] -= min_y.view(-1, 1).expand(cropped_gt_2d.shape[:-1])
    cropped_gt_2d *= 224 / \
        (scale.view(-1, 1, 1).expand(cropped_gt_2d.shape)*1000)

    orient, pose, pred_betas, pose_initial, orient_initial, estimated_translation = create_smpl_gt.find_translation_and_pose(
        image,
        batch['gt_j3d'],
        cropped_gt_2d,
        intrinsics,
        spin_model,
        smpl,
        J_regressor
    )

    focal_length = torch.stack(
        [intrinsics[:, 0, 0]/224, intrinsics[:, 1, 1]/224], dim=1).to(args.device)
    principal_point = torch.stack(
        [intrinsics[:, 0, 2]/-112+1, intrinsics[:, 1, 2]/-112+1], dim=1).to(args.device)

    with torch.no_grad():

        cameras = PerspectiveCameras(device=args.device, T=estimated_translation,
                                     focal_length=focal_length, principal_point=principal_point)

        j3d_with_noise = find_joints(smpl, pred_betas,
                                     orient, pose, J_regressor)

    j3d_with_noise_flipped = j3d_with_noise.clone()
    j3d_with_noise_flipped[:, :, 1] *= -1
    j3d_with_noise_flipped[:, :, 0] *= -1
    j3d_with_noise_flipped *= 2

    image_size = torch.tensor([224, 224]).unsqueeze(
        0).expand(batch_size, 2).to(args.device)

    j2d_with_noise = cameras.transform_points_screen(
        j3d_with_noise_flipped, image_size)

    this_gt_j3d = batch['gt_j3d']/1000

    this_gt_j3d = move_gt_pelvis(this_gt_j3d, j3d_with_noise)

    mpjpe_2d = (cropped_gt_2d - j2d_with_noise[..., :2]).norm(dim=-1)
    mpjpe_3d = (this_gt_j3d - j3d_with_noise).norm(dim=-1)

    output_dict = {
        "image": image,
        "initial_image": initial_image,
        "gt_j3d": batch['gt_j3d'],
        "gt_j2d": cropped_gt_2d,
        "intrinsics": intrinsics,
        "joints3d": j3d_with_noise,
        "joints2d": j2d_with_noise,
        "mpjpe_2d": mpjpe_2d,
        "mpjpe_3d": mpjpe_3d,
        "orient": orient,
        "pose": pose,
        "pred_betas": pred_betas,
        "estimated_translation": estimated_translation,
        "pose_initial": pose_initial,
        "orient_initial": orient_initial,
    }

    return output_dict


def create_buffer():

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

    data_dict = load_data("train")

    this_data_set = data_set(data_dict)

    loader = torch.utils.data.DataLoader(
        this_data_set, batch_size=256, num_workers=3, shuffle=False, drop_last=False)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for i in range(2, 10):
        iterator = iter(loader)

        j3d_with_noise = []
        j2d_with_noise = []
        mpjpe_2d = []
        mpjpe_3d = []
        orient = []
        pose = []
        pred_betas = []
        estimated_translation = []

        for iteration in tqdm(range(len(loader))):

            batch = next(iterator)

            for item in batch:
                batch[item] = batch[item].to(args.device).float()

            batch_augmented = find_pose_add_noise(
                normalize, spin_model, smpl, J_regressor, batch)

            j3d_with_noise.append(batch_augmented['joints3d'].detach())
            j2d_with_noise.append(batch_augmented['joints2d'].detach())
            mpjpe_2d.append(batch_augmented['mpjpe_2d'].detach())
            mpjpe_3d.append(batch_augmented['mpjpe_3d'].detach())
            orient.append(batch_augmented['orient'].detach())
            pose.append(batch_augmented['pose'].detach())
            pred_betas.append(batch_augmented['pred_betas'].detach())
            estimated_translation.append(
                batch_augmented['estimated_translation'].detach())

            del batch_augmented

        # save the output from this

        j3d_with_noise = torch.cat(j3d_with_noise, dim=0)
        torch.save(j3d_with_noise,
                   f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/j3d_with_noise.pt")
        j2d_with_noise = torch.cat(j2d_with_noise, dim=0)
        torch.save(j2d_with_noise,
                   f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/j2d_with_noise.pt")
        mpjpe_2d = torch.cat(mpjpe_2d, dim=0)
        torch.save(
            mpjpe_2d, f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/mpjpe_2d.pt")
        mpjpe_3d = torch.cat(mpjpe_3d, dim=0)
        torch.save(
            mpjpe_3d, f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/mpjpe_3d.pt")
        orient = torch.cat(orient, dim=0)
        torch.save(
            orient, f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/orient.pt")
        pose = torch.cat(pose, dim=0)
        torch.save(
            pose, f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/pose.pt")
        pred_betas = torch.cat(pred_betas, dim=0)
        torch.save(
            pred_betas, f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/pred_betas.pt")
        estimated_translation = torch.cat(estimated_translation, dim=0)
        torch.save(estimated_translation,
                   f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/estimated_translation.pt")


def train_render_model():

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

    model = Render_Model().to(args.device)
    # model.load_state_dict(torch.load(
    #     f"models/linearized_model_6_epoch8.pt", map_location=args.device))
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    data_dict = load_data("train")

    val_data_dict = load_data("validation")

    precomputed_dict_val = load_precompted()
    val_data_set = data_set(val_data_dict, precomputed_dict_val)

    val_loader = torch.utils.data.DataLoader(
        val_data_set, batch_size=args.training_batch_size, num_workers=1, shuffle=True, drop_last=True)

    for epoch in range(args.train_epochs):

        precomputed_dict = load_precompted(epoch % 10)
        # precomputed_dict = load_precompted()

        this_data_set = data_set(data_dict, precomputed_dict)

        loss_function = nn.MSELoss()

        loader = torch.utils.data.DataLoader(
            this_data_set, batch_size=args.training_batch_size, num_workers=4, shuffle=True, drop_last=True)

        total_loss = 0

        iterator = iter(loader)

        val_iterator = iter(val_loader)

        for iteration in tqdm(range(len(loader))):

            batch = next(iterator)

            image_crop, _, _, _, intrinsics = find_crop(batch['image'], batch['gt_j2d'],
                                                        intrinsics=batch['intrinsics'])

            batch['intrinsics'] = intrinsics
            batch['image'] = image_crop

            for item in batch:
                batch[item] = batch[item].to(args.device).float()

            estimated_loss = model(batch)

            estimated_loss = estimated_loss/17

            loss = loss_function(
                estimated_loss, batch['mpjpe_3d'])

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            if(args.wandb_log):

                wandb.log(
                    {"loss": loss.item()})

            optimizer.step()

            del batch

            if(iteration % 100 == 0):
                # if(False):

                # with torch.no_grad():

                model.eval()
                val_batch = next(val_iterator)

                image_crop, _, _, _, intrinsics = find_crop(val_batch['image'], val_batch['gt_j2d'],
                                                            intrinsics=val_batch['intrinsics'])

                val_batch['intrinsics'] = intrinsics
                val_batch['image'] = image_crop

                for item in val_batch:
                    val_batch[item] = val_batch[item].to(args.device)

                with torch.no_grad():
                    estimated_loss = model.forward(val_batch)

                estimated_loss = estimated_loss/17

                val_loss = loss_function(
                    estimated_loss, val_batch['mpjpe_3d'])

                if(args.wandb_log):
                    wandb.log(
                        {"validation loss": val_loss.item()}, commit=False)

                del val_batch

                model.train()

        print(f"epoch: {epoch}, loss: {total_loss}")

        # draw_gradients(model, "train", "train")
        # draw_gradients(model, "validation", "validation")

        torch.save(model.state_dict(),
                   f"models/render_model_epoch{epoch}.pt")

        # scheduler.step()

    return model

# def train_render_model():

#     spin_model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
#     checkpoint = torch.load(
#         "SPIN/data/model_checkpoint.pt", map_location=args.device)
#     spin_model.load_state_dict(checkpoint['model'], strict=False)
#     spin_model.eval()

#     smpl = SMPL(
#         '{}'.format("SPIN/data/smpl"),
#         batch_size=1,
#     ).to(args.device)

#     J_regressor = torch.from_numpy(
#         np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

#     model = Render_Model().to(args.device)
#     # model.load_state_dict(torch.load(
#     #     f"models/linearized_model_6_epoch8.pt", map_location=args.device))
#     model.train()

#     optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

#     data_dict = load_data("train")

#     normalize = transforms.Normalize(
#         (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

#     val_data_dict = load_data("validation")

#     this_data_set = data_set(data_dict)

#     val_data_set = data_set(val_data_dict)

#     loss_function = nn.MSELoss()

#     loader = torch.utils.data.DataLoader(
#         this_data_set, batch_size=args.training_batch_size, num_workers=2, shuffle=True, drop_last=True)

#     val_loader = torch.utils.data.DataLoader(
#         val_data_set, batch_size=args.training_batch_size, num_workers=2, shuffle=True)

#     for epoch in range(args.train_epochs):

#         total_loss = 0

#         iterator = iter(loader)

#         val_iterator = iter(val_loader)

#         # for iteration in tqdm(range(len(loader))):
#         for iteration in tqdm(range(len(loader))):

#             batch = next(iterator)

#             for item in batch:
#                 batch[item] = batch[item].to(args.device).float()

#             batch_augmented = find_pose_add_noise(
#                 normalize, spin_model, smpl, J_regressor, batch)

#             estimated_loss = model(batch_augmented)

#             loss = loss_function(
#                 estimated_loss, batch_augmented['mpjpe_3d'])

#             total_loss += loss.item()

#             optimizer.zero_grad()
#             loss.backward()

#             if(args.wandb_log):

#                 wandb.log(
#                     {"loss": loss.item()})

#             optimizer.step()

#             if(iteration % 100 == 0):
#                 # if(False):

#                 # with torch.no_grad():

#                 model.eval()
#                 val_batch = next(val_iterator)

#                 for item in val_batch:
#                     val_batch[item] = val_batch[item].to(args.device)

#                 val_augmented_batch = find_pose_add_noise(
#                     normalize, spin_model, smpl, J_regressor, val_batch)

#                 with torch.no_grad():
#                     estimated_loss = model.forward(val_augmented_batch)

#                 val_loss = loss_function(
#                     estimated_loss, val_augmented_batch['mpjpe_3d'])

#                 if(args.wandb_log):
#                     wandb.log(
#                         {"validation loss": val_loss.item()}, commit=False)

#                 model.train()

#         print(f"epoch: {epoch}, loss: {total_loss}")

#         # draw_gradients(model, "train", "train")
#         # draw_gradients(model, "validation", "validation")

#         torch.save(model.state_dict(),
#                    f"models/render_model_epoch{epoch}.pt")

#         # scheduler.step()

#     return model
