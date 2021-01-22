import torch
from torch import nn, optim

import sys

import os

from eval_utils import batch_compute_similarity_transform_torch

from utils import utils

from tqdm import tqdm

from args import args

from data import load_data, data_set, find_joints

import numpy as np

from smpl import SMPL, SMPL_MODEL_DIR



def evaluate(pred_j3ds, target_j3ds):

    pred_j3ds = pred_j3ds.clone().detach()
    target_j3ds = target_j3ds.clone().detach()
    print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
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
    print(log_str)

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

    for epoch in range(args.train_epochs):

        estimated_loss_total = 0
        pose_differences_total = 0

        data_dict['estimated_j3d'] = estimated_j3d

        # this_data_set = data_set(data_dict, training=False)
        this_data_set = data_set(data_dict, training=False)

        loader = torch.utils.data.DataLoader(this_data_set, batch_size = args.optimization_batch_size, num_workers=0, shuffle=True)
        iterator = iter(loader)

        for iteration in tqdm(range(len(loader))):

            batch = next(iterator)

            for item in batch:
                batch[item] = batch[item].to(args.device) 


            optimized_pose = torch.cat([batch['estimated_pose'], batch['estimated_shape']], dim=1)
            print("optimized_pose.shape")
            print(optimized_pose.shape)

            initial_joints = find_joints(optimized_pose[:, :72], optimized_pose[:, 72:], smpl, J_regressor)
            

            optimized_pose.requires_grad = True

            optimizer = optim.SGD([optimized_pose], lr=args.optimization_rate)

            min_losses = torch.ones(optimized_pose.shape[0]).to(args.device)
            best_poses = initial_joints.clone()

            print(best_poses.shape)

            for i in range(10000):

                optimizer.zero_grad()

                optimized_joints = find_joints(optimized_pose[:, :72], optimized_pose[:, 72:], smpl, J_regressor)

                batch['estimated_j3d'] = optimized_joints

                estimated_loss = model.forward(batch)

                individul_losses = torch.mean(estimated_loss, dim=-1)

                best_poses[individul_losses<min_losses] = optimized_joints[individul_losses<min_losses]

                min_losses = torch.where(individul_losses<min_losses, individul_losses, min_losses)


                estimated_loss = torch.mean(estimated_loss)
                

                pose_differences = mse_loss(initial_joints, optimized_joints)

                estimated_loss_total += estimated_loss.item()
                pose_differences_total += pose_differences.item()

                loss = pose_differences*1e-1 + estimated_loss

                loss.backward()
                
                optimizer.step()

                if(i%10==0):

                    batch['estimated_j3d'] = optimized_joints
                    evaluate(batch['estimated_j3d'], batch['gt_j3d'])
                    print("best")
                    evaluate(best_poses, batch['gt_j3d'])
                    print(f"loss {estimated_loss.item()}, iteration {i}")

            estimated_j3d[batch['indices']] = batch['estimated_j3d'].cpu().detach()

            # return the error

        evaluate(estimated_j3d, data_dict['gt_j3d'])

        print(f"epoch: {epoch}, estimated_loss_total: {estimated_loss_total}, pose_differences_total: {pose_differences_total}")
    
    return model




