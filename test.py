import torch
from torch import nn, optim

import os

from eval_utils import batch_compute_similarity_transform_torch

from utils import utils

from tqdm import tqdm

from args import args

from data import load_data, data_set

import numpy as np

from smpl import SMPL, SMPL_MODEL_DIR

from torchvision import transforms

from warp import perturbation_helper, sampling_helper

from create_smpl_gt import find_crop, rotation_matrix_to_quaternion, quaternion_to_rotation_matrix, find_joints

import copy

from train import find_pose_add_noise


# SPIN
import sys
# sys.path.append('/scratch/iamerich/SPIN')
from SPIN.models import hmr, SMPL
import SPIN.config as config
from SPIN.utils.geometry import rot6d_to_rotmat


def evaluate(pred_j3ds, target_j3ds):

    with torch.no_grad():

        pred_j3ds = pred_j3ds.clone().detach()
        target_j3ds = target_j3ds.clone().detach()
        target_j3ds /= 1000

        # print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
        target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

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

        log_str = ' '.join(
            [f'{k.upper()}: {v:.4f},'for k, v in eval_dict.items()])
        print(log_str)

    return mpjpe, pa_mpjpe


# # find crop and pass in 224x224 image
# def find_crop(image, joints_2d):

#     batch_size = joints_2d.shape[0]
#     min_x = torch.min(joints_2d[..., 0], dim=1)[0]
#     max_x = torch.max(joints_2d[..., 0], dim=1)[0]
#     min_y = torch.min(joints_2d[..., 1], dim=1)[0]
#     max_y = torch.max(joints_2d[..., 1], dim=1)[0]

#     min_x = (min_x-500)/500
#     max_x = (max_x-500)/500
#     min_y = (min_y-500)/500
#     max_y = (max_y-500)/500

#     average_x = (min_x+max_x)/2
#     average_y = (min_y+max_y)/2

#     scale_x = (max_x-min_x)*1.2
#     scale_y = (max_y-min_y)*1.2

#     scale = torch.where(scale_x > scale_y, scale_x, scale_y)

#     # print(scale[:3])
#     # print(average_x[:3])

#     scale /= 2

#     min_x = (average_x-scale)*500+500
#     min_y = (average_y-scale)*500+500

#     zeros = torch.zeros(batch_size).to(args.device)
#     ones = torch.ones(batch_size).to(args.device)

#     bilinear_sampler = sampling_helper.DifferentiableImageSampler(
#         'bilinear', 'zeros')

#     vec = torch.stack([zeros, scale, scale, average_x /
#                        scale, average_y/scale], dim=1)

#     transformation_mat = perturbation_helper.vec2mat_for_similarity(vec)

#     image = bilinear_sampler.warp_image(
#         image, transformation_mat, out_shape=[224, 224]).contiguous()

#     return image, min_x, min_y, scale


def convert_back_to_original_dimensions(image, pred_joints, pred_camera, min_x, min_y, image_scale):

    camera_translation = torch.stack(
        [pred_camera[:, 1], pred_camera[:, 2], 2*5000/(112 * pred_camera[:, 0] + 1e-9)], dim=-1)

    camera_translation = camera_translation.unsqueeze(
        1).expand(pred_joints.shape[0], pred_joints.shape[1], 3)

    camera_scale = pred_camera[:,
                               0].unsqueeze(-1).unsqueeze(-1).expand(pred_joints.shape)

    image_scale = image_scale.unsqueeze(
        -1).unsqueeze(-1).expand(pred_joints.shape)

    pred_joints += camera_translation

    pred_joints *= camera_scale
    pred_joints *= 112
    pred_joints += 112

    pred_joints = pred_joints*1000/224*image_scale

    pred_joints[:, :, 0] += min_x.unsqueeze(-1).expand(pred_joints.shape[:-1])
    pred_joints[:, :, 1] += min_y.unsqueeze(-1).expand(pred_joints.shape[:-1])

    # return pred_joints, camera_scale*112*1000/224*image_scale
    return pred_joints, camera_scale*112*1000/224*image_scale


# use ground truth crop on image
def spin_estimate_vertices(spin_model, batch, smpl, J_regressor):

    # get the crop for the image
    image, _, _, _, _ = find_crop(
        batch['image'], batch['gt_j2d'], batch['intrinsics'])

    pred_rotmat, pred_betas, pred_camera = spin_model(image)

    # get back into crop space
    # then back into original image space

    pred_joints = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(
        1), pose2rot=False).vertices
    J_regressor_batch = J_regressor[None, :].expand(
        pred_joints.shape[0], -1, -1).to(args.device)
    pred_joints = torch.matmul(J_regressor_batch, pred_joints)

    return pred_joints


def test_render_model(model):
    model.eval()

    J_regressor = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

    smpl = SMPL(
        '{}'.format("SPIN/data/smpl"),
        batch_size=args.optimization_batch_size,
        # joint_type="cocoplus"
    ).to(args.device)

    spin_model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
    checkpoint = torch.load(
        "SPIN/data/model_checkpoint.pt", map_location=args.device)
    spin_model.load_state_dict(checkpoint['model'], strict=False)
    spin_model.eval()

    data_dict = load_data("validation")

    mse_loss = nn.MSELoss()

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    this_data_set = data_set(data_dict)

    loader = torch.utils.data.DataLoader(
        this_data_set, batch_size=args.optimization_batch_size, num_workers=0, shuffle=True)
    iterator = iter(loader)

    before_mpjpe = []
    before_pa_mpjpe = []
    after_mpjpe = []
    after_pa_mpjpe = []

    for iteration in tqdm(range(len(iterator))):

        batch = next(iterator)

        for item in batch:
            batch[item] = batch[item].to(args.device).float()

        batch_augmented = find_pose_add_noise(
            normalize, spin_model, smpl, J_regressor, batch)

        pred_pose_quat = batch_augmented['pose_initial']
        pred_orient_quat = batch_augmented['orient_initial']

        optimizer = optim.SGD(
            [pred_pose_quat, pred_orient_quat], lr=args.optimization_rate)

        for i in range(args.opt_steps):

            optimizer.zero_grad()

            pred_joints = find_joints(smpl, batch_augmented['pred_betas'], pred_orient_quat,
                                      pred_pose_quat, J_regressor)

            if(i == 0):
                print("before")
                mpjpe, pa_mpjpe = evaluate(pred_joints, batch['gt_j3d'])

                before_mpjpe.append(mpjpe)
                before_pa_mpjpe.append(pa_mpjpe)

            batch_augmented['orient'] = quaternion_to_rotation_matrix(
                pred_orient_quat.reshape(-1, 4)).reshape(-1, 1, 3, 3)
            batch_augmented['pose'] = quaternion_to_rotation_matrix(
                pred_pose_quat.reshape(-1, 4)).reshape(-1, 23, 3, 3)

            estimated_loss = model.forward(batch_augmented)

            estimated_loss = torch.mean(estimated_loss)

            print(f"estimated loss {estimated_loss.item()}")

            estimated_loss.backward()

            optimizer.step()
            print(f"{i}")

            # if(i%10==0):
            # batch['estimated_j3d'] = optimized_joints
            mpjpe, pampjpe = evaluate(pred_joints, batch['gt_j3d'])

            # mpjpe_errors[int(i/10)]+= mpjpe
            # pampjpe_errors[int(i/10)]+= pampjpe

            # print("best")
            # evaluate(best_poses, batch['gt_j3d'])
            # print(f"loss {estimated_loss.item()}, iteration {i}")
        mpjpe, pa_mpjpe = evaluate(pred_joints, batch['gt_j3d'])

        after_mpjpe.append(mpjpe)
        after_pa_mpjpe.append(pa_mpjpe)

        print(f"before_mpjpe: {torch.mean(torch.tensor(before_mpjpe))}")
        print(f"before_pa_mpjpe: {torch.mean(torch.tensor(before_pa_mpjpe))}")
        print(f"after_mpjpe: {torch.mean(torch.tensor(after_mpjpe))}")
        print(f"after_pa_mpjpe: {torch.mean(torch.tensor(after_pa_mpjpe))}")

        # print("after")
        # evaluate(batch['estimated_j3d'], batch['gt_j3d'])

        # estimated_j3d[batch['indices']] = batch['estimated_j3d'].cpu().detach()

        # print("initial j3d")
        # print(evaluate(initial_j3d, data_dict['gt_j3d']))

        # print("mpjpe")
        # print(np.array(mpjpe_errors)/(iteration+1))
        # print("pa mpjpe")
        # print(np.array(pampjpe_errors)/(iteration+1))

        # wandb_viz(batch, "optimized", estimated_loss_per_pose[0].item(), smpl)

        # exit()

        # return the error

    print("initial j3d")
    print(evaluate(initial_j3d, data_dict['gt_j3d']))

    print("mpjpe")
    print(mpjpe_errors/len(iterator))
    print("pa mpjpe")
    print(pampjpe_errors/len(iterator))

    # print("estimated j3d")
    # evaluate(estimated_j3d, data_dict['gt_j3d'])

    # print(f"epoch: {epoch}, estimated_loss_total: {estimated_loss_total}, pose_differences_total: {pose_differences_total}")

    return model


def wandb_viz(batch, name, estimated_loss, smpl):

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import wandb

    joints2d = projection(batch['estimated_j3d'], batch['cam'])
    joints2d_gt = projection(batch['gt_j3d'], batch['gt_cam'])

    dims_before = batch['dims_before']

    mpjpe, pa_mpjpe = evaluate(
        batch['estimated_j3d'][0][None], batch['gt_j3d'][0][None])

    label = f"{name}\nmpjpe: {mpjpe}\npa_mpjpe: {pa_mpjpe}\nestimated_loss: {estimated_loss}"

    if(dims_before[0, 0] == 1920):
        offset = [420, 0]
    else:
        offset = [0, 420]

    des_bboxes = batch['bboxes'].unsqueeze(1).expand(-1, joints2d.shape[1], -1)

    joints2d[:, :, 0] *= des_bboxes[:, :, 2]/2*1.1
    joints2d[:, :, 0] += des_bboxes[:, :, 0]
    joints2d[:, :, 1] *= des_bboxes[:, :, 3]/2*1.1
    joints2d[:, :, 1] += des_bboxes[:, :, 1]

    joints2d_gt[:, :, 0] *= des_bboxes[:, :, 2]/2*1.1
    joints2d_gt[:, :, 0] += des_bboxes[:, :, 0]
    joints2d_gt[:, :, 1] *= des_bboxes[:, :, 3]/2*1.1
    joints2d_gt[:, :, 1] += des_bboxes[:, :, 1]

    # draw gradients
    plt.imshow(utils.torch_img_to_np_img(batch['image'])[0])
    ax = plt.gca()

    for i in range(joints2d_gt.shape[1]):

        circ = Circle((joints2d_gt[0, i, 0]+offset[0],
                       joints2d_gt[0, i, 1]+offset[1]), 10, color='b')

        ax.add_patch(circ)

    for i in range(joints2d.shape[1]):

        circ = Circle((joints2d[0, i, 0]+offset[0],
                       joints2d[0, i, 1]+offset[1]), 10, color='r')

        ax.add_patch(circ)

    estimated_vertices = find_vertices(
        batch['estimated_pose'], batch['estimated_shape'], smpl).cpu().detach().numpy()
    gt_vertices = find_vertices(
        batch['gt_pose'], batch['gt_shape'], smpl).cpu().detach().numpy()

    points = np.ones((estimated_vertices.shape[1]*2, 4))

    points[:estimated_vertices.shape[1], :3] = estimated_vertices[0]
    points[estimated_vertices.shape[1]:, :3] = gt_vertices[0]
    points[estimated_vertices.shape[1]:, 3:] += 10

    wandb.log({f"3d pose": wandb.Object3D(
        {
            "type": "lidar/beta",
            "points": points,
        }
    )}, commit=False)
    # wandb.log({f"{name}_3d_pose_gt": wandb.Object3D(batch['gt_j3d'][0].cpu().detach().numpy())}, commit=False)
    wandb.log({f"overlayed pose on image": wandb.Image(plt, caption=label)})
    plt.close()
