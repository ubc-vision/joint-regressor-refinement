from args import args
import wandb
import torch
from torch import nn, optim

from tqdm import tqdm

from render_model import Render_Model
from pose_estimator import Pose_Estimator
from pose_refiner import Pose_Refiner
from pose_discriminator import Pose_Discriminator

from data import load_data, data_set, data_set_refine, load_precompted, load_ground_truth, find_crop
# from visualizer import draw_gradients

from torchvision import transforms

from create_smpl_gt import quaternion_to_rotation_matrix, find_translation_and_pose, find_direction_to_gt, batch_rodrigues, optimize_pose, optimize_translation, quaternion_multiply, rotation_matrix_to_quaternion

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

    orient, pose, pred_betas, pose_initial, orient_initial, estimated_translation = find_translation_and_pose(
        image,
        batch['gt_j3d'],
        cropped_gt_2d,
        intrinsics,
        spin_model,
        smpl,
        J_regressor
    )

    # focal_length = torch.stack(
    #     [intrinsics[:, 0, 0]/224, intrinsics[:, 1, 1]/224], dim=1).to(args.device)
    # principal_point = torch.stack(
    #     [intrinsics[:, 0, 2]/-112+1, intrinsics[:, 1, 2]/-112+1], dim=1).to(args.device)

    # with torch.no_grad():

    #     cameras = PerspectiveCameras(device=args.device, T=estimated_translation,
    #                                  focal_length=focal_length, principal_point=principal_point)

    #     j3d_with_noise = find_joints(smpl, pred_betas,
    #                                  orient, pose, J_regressor)

    # j3d_with_noise_flipped = j3d_with_noise.clone()
    # j3d_with_noise_flipped[:, :, 1] *= -1
    # j3d_with_noise_flipped[:, :, 0] *= -1
    # j3d_with_noise_flipped *= 2

    # image_size = torch.tensor([224, 224]).unsqueeze(
    #     0).expand(batch_size, 2).to(args.device)

    # j2d_with_noise = cameras.transform_points_screen(
    #     j3d_with_noise_flipped, image_size)

    # this_gt_j3d = batch['gt_j3d']/1000

    # this_gt_j3d = move_pelvis(this_gt_j3d, j3d_with_noise)

    # mpjpe_2d = (cropped_gt_2d - j2d_with_noise[..., :2]).norm(dim=-1)
    # mpjpe_3d = (this_gt_j3d - j3d_with_noise).norm(dim=-1)

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

    # for i in range(2, 10):
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

    orient = torch.cat(orient, dim=0)
    torch.save(
        orient, f"/scratch/iamerich/human36m/processed/saved_output_train/ground_truth_pose/orient.pt")
    pose = torch.cat(pose, dim=0)
    torch.save(
        pose, f"/scratch/iamerich/human36m/processed/saved_output_train/ground_truth_pose/pose.pt")
    estimated_translation = torch.cat(estimated_translation, dim=0)
    torch.save(estimated_translation,
               f"/scratch/iamerich/human36m/processed/saved_output_train/ground_truth_pose/estimated_translation.pt")
    pred_betas = torch.cat(pred_betas, dim=0)
    torch.save(
        pred_betas, f"/scratch/iamerich/human36m/processed/saved_output_train/ground_truth_pose/betas.pt")

    # save the output from this

    # j3d_with_noise = torch.cat(j3d_with_noise, dim=0)
    # torch.save(j3d_with_noise,
    #            f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/j3d_with_noise.pt")
    # j2d_with_noise = torch.cat(j2d_with_noise, dim=0)
    # torch.save(j2d_with_noise,
    #            f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/j2d_with_noise.pt")
    # mpjpe_2d = torch.cat(mpjpe_2d, dim=0)
    # torch.save(
    #     mpjpe_2d, f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/mpjpe_2d.pt")
    # mpjpe_3d = torch.cat(mpjpe_3d, dim=0)
    # torch.save(
    #     mpjpe_3d, f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/mpjpe_3d.pt")
    # orient = torch.cat(orient, dim=0)
    # torch.save(
    #     orient, f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/orient.pt")
    # pose = torch.cat(pose, dim=0)
    # torch.save(
    #     pose, f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/pose.pt")
    # pred_betas = torch.cat(pred_betas, dim=0)
    # torch.save(
    #     pred_betas, f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/pred_betas.pt")
    # estimated_translation = torch.cat(estimated_translation, dim=0)
    # torch.save(estimated_translation,
    #            f"/scratch/iamerich/human36m/processed/saved_output_train/{i}/estimated_translation.pt")


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

    pose_refiner_1 = Pose_Refiner().to(args.device)
    pose_refiner_1.train()
    pose_discriminator_1 = Pose_Discriminator().to(args.device)
    pose_discriminator_1.train()
    pose_refiner_2 = Pose_Refiner().to(args.device)
    pose_refiner_2.train()
    pose_discriminator_2 = Pose_Discriminator().to(args.device)
    pose_discriminator_2.train()

    for param in pose_refiner_1.resnet.parameters():
        param.requires_grad = False
    for param in pose_refiner_2.resnet.parameters():
        param.requires_grad = False

    optimizer_1 = optim.Adam(
        pose_refiner_1.parameters(), lr=args.learning_rate)
    disc_optimizer_1 = optim.Adam(
        pose_discriminator_1.parameters(), lr=args.learning_rate)
    optimizer_2 = optim.Adam(
        pose_refiner_2.parameters(), lr=args.learning_rate)
    disc_optimizer_2 = optim.Adam(
        pose_discriminator_2.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()

    # data_dict = load_data("train")
    # val_data_dict = load_data("validation")
    # exit()

    precomputed_dict = load_ground_truth("train")
    val_precomputed_dict = load_ground_truth("validation")
    data = data_set_refine(precomputed_dict)
    val_data = data_set_refine(val_precomputed_dict)

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

            pred_rotmat, pred_rot6d = pose_refiner_1(batch)

            batch["orient"] = pred_rot6d[:, :1]
            batch["pose"] = pred_rot6d[:, 1:]

            pred_joints = find_joints(
                smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

            pred_joints = move_pelvis(pred_joints)

            mpjpe_after_1, pampjpe_after_1 = evaluate(
                pred_joints, batch['gt_j3d'])

            joint_loss_1 = loss_function(pred_joints, batch['gt_j3d']/1000)

            # add a loss so the estimates dont stray too far from original
            pred_disc = pose_discriminator_1(pred_rot6d)

            discriminated_loss_1 = loss_function(pred_disc, torch.ones(
                pred_disc.shape).to(args.device))

            loss = joint_loss_1+discriminated_loss_1/1000

            optimizer_1.zero_grad()
            loss.backward()
            optimizer_1.step()

            # train discriminator

            pred_gt = pose_discriminator_1(pred_pose)
            pred_disc = pose_discriminator_1(pred_rot6d.detach())

            discriminator_loss_1 = loss_function(pred_disc, torch.zeros(
                pred_disc.shape).to(args.device))+loss_function(pred_gt, torch.ones(
                    pred_disc.shape).to(args.device))

            disc_optimizer_1.zero_grad()
            discriminator_loss_1.backward()
            disc_optimizer_1.step()

            for item in batch:
                batch[item] = batch[item].detach()

            # train generator

            pred_rotmat, pred_rot6d = pose_refiner_2(batch)

            batch["orient"] = pred_rot6d[:, :1]
            batch["pose"] = pred_rot6d[:, 1:]

            pred_joints = find_joints(
                smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

            pred_joints = move_pelvis(pred_joints)

            mpjpe_after_2, pampjpe_after_2 = evaluate(
                pred_joints, batch['gt_j3d'])

            joint_loss_2 = loss_function(pred_joints, batch['gt_j3d']/1000)

            # add a loss so the estimates dont stray too far from original
            pred_disc = pose_discriminator_2(pred_rot6d)

            discriminated_loss_2 = loss_function(pred_disc, torch.ones(
                pred_disc.shape).to(args.device))

            loss = joint_loss_2+discriminated_loss_2/1000

            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()

            # train discriminator

            pred_gt = pose_discriminator_2(pred_pose)
            pred_disc = pose_discriminator_2(pred_rot6d.detach())

            discriminator_loss_2 = loss_function(pred_disc, torch.zeros(
                pred_disc.shape).to(args.device))+loss_function(pred_gt, torch.ones(
                    pred_disc.shape).to(args.device))

            disc_optimizer_2.zero_grad()
            discriminator_loss_2.backward()
            disc_optimizer_2.step()

            if(args.wandb_log):

                wandb.log(
                    {
                        "joint_loss_1": joint_loss_1.item(),
                        "joint_loss_2": joint_loss_2.item(),
                        "discriminated_loss_1": discriminated_loss_1,
                        "discriminated_loss_2": discriminated_loss_2,
                        "discriminator_loss_1": discriminator_loss_1,
                        "discriminator_loss_2": discriminator_loss_2,
                        "mpjpe_before_refinement": mpjpe_before_refinement.item(),
                        "pampjpe_before_refinement": pampjpe_before_refinement.item(),
                        "mpjpe_after_1": mpjpe_after_1.item(),
                        "pampjpe_after_1": pampjpe_after_1.item(),
                        "mpjpe_after_2": mpjpe_after_2.item(),
                        "pampjpe_after_2": pampjpe_after_2.item(),
                        "mpjpe_difference_1": mpjpe_after_1.item()-mpjpe_before_refinement.item(),
                        "pampjpe_difference_1": pampjpe_after_1.item()-pampjpe_before_refinement.item(),
                        "mpjpe_difference_2": mpjpe_after_2.item()-mpjpe_after_1.item(),
                        "pampjpe_difference_2": pampjpe_after_2.item()-pampjpe_after_1.item(), })

            if(args.wandb_log and iteration % 100 == 0):

                with torch.no_grad():

                    pose_refiner_1.eval()
                    pose_refiner_2.eval()

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

                    pred_rotmat, pred_rot6d = pose_refiner_1(batch)

                    batch["pose"] = pred_rot6d[:, 1:]
                    batch["orient"] = pred_rot6d[:, 0].unsqueeze(1)

                    pred_joints = find_joints(
                        smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

                    mpjpe_after_1, pampjpe_after_1 = evaluate(
                        pred_joints, batch['gt_j3d'])

                    pred_rotmat, pred_rot6d = pose_refiner_2(batch)

                    batch["orient"] = pred_rot6d[:, :1]
                    batch["pose"] = pred_rot6d[:, 1:]

                    pred_joints = find_joints(
                        smpl, batch["betas"], pred_rotmat[:, :1], pred_rotmat[:, 1:], J_regressor)

                    pred_joints = move_pelvis(pred_joints)

                    mpjpe_after_2, pampjpe_after_2 = evaluate(
                        pred_joints, batch['gt_j3d'])

                    pose_refiner_1.train()
                    pose_refiner_2.train()

                wandb.log(
                    {
                        "validation mpjpe_before_refinement": mpjpe_before_refinement.item(),
                        "validation pampjpe_before_refinement": pampjpe_before_refinement.item(),
                        "validation mpjpe_after_1": mpjpe_after_1.item(),
                        "validation pampjpe_after_1": pampjpe_after_1.item(),
                        "validation mpjpe_after_2": mpjpe_after_2.item(),
                        "validation pampjpe_after_2": pampjpe_after_2.item(),
                        "validation mpjpe_difference_1": mpjpe_after_1.item()-mpjpe_before_refinement.item(),
                        "validation pampjpe_difference_1": pampjpe_after_1.item()-pampjpe_before_refinement.item(),
                        "validation mpjpe_difference_2": mpjpe_after_2.item()-mpjpe_after_1.item(),
                        "validation pampjpe_difference_2": pampjpe_after_2.item()-pampjpe_after_1.item(), })

        print(f"epoch: {epoch}, loss: {total_loss}")

        torch.save(pose_refiner_1.state_dict(),
                   f"models/pose_refiner_1_epoch_{epoch}.pt")
        torch.save(pose_refiner_2.state_dict(),
                   f"models/pose_refiner_2_epoch_{epoch}.pt")
