import torch
from torch import nn, optim

import sys

import os

from eval_utils import batch_compute_similarity_transform_torch

from utils import utils

from tqdm import tqdm

from args import args

from data import load_data, data_set, find_joints, find_vertices, projection

import numpy as np

from smpl import SMPL, SMPL_MODEL_DIR



def evaluate(pred_j3ds, target_j3ds):

    pred_j3ds = pred_j3ds.clone().detach()
    target_j3ds = target_j3ds.clone().detach()
    # print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
    pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
    target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0


    pred_j3ds -= pred_pelvis
    target_j3ds -= target_pelvis

    # Absolute error (MPJPE)
    errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
    errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    m2mm = 1000

    mpjpe = np.mean(errors) * m2mm
    pa_mpjpe = np.mean(errors_pa) * m2mm

    eval_dict = {
    'mpjpe': mpjpe,
    'pa-mpjpe': pa_mpjpe,
    }

    log_str = ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
    # print(log_str)

    return mpjpe, pa_mpjpe



def test_crop_model(model):
    model.eval()

    J_regressor = torch.from_numpy(np.load('data/vibe_data/J_regressor_h36m.npy')).float().to(args.device)
            
    smpl = SMPL(
        '{}'.format(SMPL_MODEL_DIR),
        batch_size=64,
        create_transl=False
    ).to(args.device)

    data_dict = load_data("validation")

    initial_j3d = data_dict['estimated_j3d'].clone()
    estimated_j3d = data_dict['estimated_j3d'].clone()

    mse_loss = nn.MSELoss()

    estimated_loss_total = 0
    pose_differences_total = 0

    # data_dict['estimated_j3d'] = estimated_j3d

    # this_data_set = data_set(data_dict, training=False)
    this_data_set = data_set(data_dict, training=False)

    loader = torch.utils.data.DataLoader(this_data_set, batch_size = args.optimization_batch_size, num_workers=0, shuffle=False)
    iterator = iter(loader)

    evaluate(estimated_j3d, data_dict['gt_j3d'])

    mpjpe_errors = [0]*11
    pampjpe_errors = [0]*11

    for iteration in tqdm(range(len(iterator))):

        batch = next(iterator)

        # save the 
        

        for item in batch:
            batch[item] = batch[item].to(args.device) 

        estimated_vertices = find_vertices(batch['estimated_pose'], batch['estimated_shape'], smpl).cpu().detach().numpy()

        print("estimated_vertices[0].shape")
        print(estimated_vertices[0].shape)
        print("batch['cam'][0].shape")
        print(batch['cam'][0].shape)
        print("batch['image'][0].shape")
        print(batch['image'][0].shape)
        np.save("cam.npy", batch['cam'][0].cpu().numpy())
        np.save("estimated_vertices.npy", estimated_vertices[0])
        exit()


        optimized_pose = torch.cat([batch['estimated_pose'], batch['estimated_shape']], dim=1)
        # print("optimized_pose.shape")
        # print(optimized_pose.shape)

        initial_joints = find_joints(optimized_pose[:, :72], optimized_pose[:, 72:], smpl, J_regressor)
        

        optimized_pose.requires_grad = True

        optimizer = optim.SGD([optimized_pose], lr=args.optimization_rate)

        min_losses = torch.ones(optimized_pose.shape[0]).to(args.device)
        best_poses = initial_joints.clone()

        # print(best_poses.shape)

        for i in range(args.opt_steps):

            optimizer.zero_grad()

            optimized_joints = find_joints(optimized_pose[:, :72], optimized_pose[:, 72:], smpl, J_regressor)

            batch['estimated_pose'] = optimized_pose[:, :72]
            batch['estimated_shape'] = optimized_pose[:, 72:]

            batch['estimated_j3d'] = optimized_joints

            estimated_loss = model.forward(batch)

            estimated_loss_per_pose = torch.mean(estimated_loss, dim=-1)

            # print(estimated_loss.shape)

            if(i==0):
                wandb_viz(batch, "initial", estimated_loss_per_pose[0].item(), smpl)

            # individul_losses = torch.mean(estimated_loss, dim=-1)

            # best_poses[individul_losses<min_losses] = optimized_joints[individul_losses<min_losses]

            # min_losses = torch.where(individul_losses<min_losses, individul_losses, min_losses)

            estimated_loss = torch.mean(estimated_loss)
            

            pose_differences = mse_loss(initial_joints, optimized_joints)

            # estimated_loss_total += estimated_loss.item()
            # pose_differences_total += pose_differences.item()

            loss = pose_differences*1e-1 + estimated_loss

            loss.backward()
            
            optimizer.step()

            if(i%10==0):
            # batch['estimated_j3d'] = optimized_joints
                mpjpe, pampjpe = evaluate(batch['estimated_j3d'], batch['gt_j3d'])

                mpjpe_errors[int(i/10)]+= mpjpe
                pampjpe_errors[int(i/10)]+= pampjpe


            # print("best")
            # evaluate(best_poses, batch['gt_j3d'])
            # print(f"loss {estimated_loss.item()}, iteration {i}")

        # print("after")
        # evaluate(batch['estimated_j3d'], batch['gt_j3d'])

        estimated_j3d[batch['indices']] = batch['estimated_j3d'].cpu().detach()

        print("initial j3d")
        print(evaluate(initial_j3d, data_dict['gt_j3d']))

        print("mpjpe")
        print(np.array(mpjpe_errors)/(iteration+1))
        print("pa mpjpe")
        print(np.array(pampjpe_errors)/(iteration+1))

        wandb_viz(batch, "optimized", estimated_loss_per_pose[0].item(), smpl)

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


    mpjpe, pa_mpjpe = evaluate(batch['estimated_j3d'][0][None], batch['gt_j3d'][0][None])

    label = f"{name}\nmpjpe: {mpjpe}\npa_mpjpe: {pa_mpjpe}\nestimated_loss: {estimated_loss}"

    if(dims_before[0, 0]==1920):
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

        circ = Circle((joints2d_gt[0, i, 0]+offset[0],joints2d_gt[0, i, 1]+offset[1]),10, color = 'b')

        ax.add_patch(circ)

    for i in range(joints2d.shape[1]):

        circ = Circle((joints2d[0, i, 0]+offset[0],joints2d[0, i, 1]+offset[1]),10, color = 'r')

        ax.add_patch(circ)


    estimated_vertices = find_vertices(batch['estimated_pose'], batch['estimated_shape'], smpl).cpu().detach().numpy()
    gt_vertices = find_vertices(batch['gt_pose'], batch['gt_shape'], smpl).cpu().detach().numpy()

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
