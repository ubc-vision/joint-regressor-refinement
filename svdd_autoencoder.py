# "borowed" from https://github.com/pytorch/examples/blob/master/vae/main.py


from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import sys
sys.path.append('../VIBE')
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.utils.eval_utils import batch_compute_similarity_transform_torch

sys.path.append('/home/willow/Documents/smplpytorch')
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from display_utils import display_model


# from spacepy import pycdf
import numpy as np
import glob
import pickle
import random
from time import sleep

import math

# TODO this is going to need a changing width
def load_VIBE(dataset, data_type, step=1, window_size=11, add_ends=False):

    poses = sorted(glob.glob("3DPW_data/{}/*_{}_theta.npy".format(dataset, data_type)))

    posarinos = []

    for pose in poses:

        smpl = np.load(pose)

        if(add_ends):

            for x in range(int(-1*window_size/2), smpl.shape[0], step):
        
                indece = np.clip(np.arange(window_size)+x, 0, smpl.shape[0]-1)

                posarinos.append(smpl[indece][np.newaxis])

        else:

            for x in range(0, smpl.shape[0]-window_size, step):

                posarinos.append(smpl[x:x+window_size][np.newaxis])

    posarinos = np.concatenate(posarinos, axis=0)

    print("posarinos.shape")
    print(posarinos.shape)
    
    return posarinos


class VAE(nn.Module):
    def __init__(self, num_inputs):
        super(VAE, self).__init__()

        self.num_inputs = num_inputs

        self.fc1 = nn.Linear(self.num_inputs, int(self.num_inputs/2))
        self.fc21 = nn.Linear(int(self.num_inputs/2), int(self.num_inputs/4))
        self.fc22 = nn.Linear(int(self.num_inputs/2), int(self.num_inputs/4))
        self.fc3 = nn.Linear(int(self.num_inputs/4), int(self.num_inputs/2))
        self.fc4 = nn.Linear(int(self.num_inputs/2), self.num_inputs)

    def encode(self, x):
        x = x.reshape(-1, self.num_inputs)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        out = self.fc4(h3)
        out = out.view(-1, self.shape[-2], self.shape[-1])
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function_vae(recon_data, data, mu, logvar, kl_weight = 1e-3):

    data = data.view(-1, 69)
    recon_data = recon_data.view(-1, 69)

    pose_loss = (data-recon_data)**2
    pose_loss = torch.sum(pose_loss)

    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KLD = 0

    return KLD*kl_weight + pose_loss


def train_vae(model, data, optimizer, initialize = False):

    data = data[:, :, 6:75]

    print("data.shape")
    print(data.shape)

    if(initialize == False):
        for param in model.parameters():
            param.requires_grad = True

    model.shape = data.shape

    this_batch = data.to('cuda')

    for epoch in range(args.train_epochs):

        model.train()

        sleep(.1)

        train_loss = 0
    
        optimizer.zero_grad()
        recon_data, mu, logvar = model.forward(this_batch)
        loss = loss_function_vae(recon_data, this_batch, mu, logvar, kl_weight=args.kl_weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if(epoch %100==0):
            print("epoch {}".format(epoch))
            print("loss: {}".format(train_loss))

    torch.save(mu, "output/mu.pt")
    
    torch.save(model.state_dict(), "models/vae.pt")

class SVDD_AE(nn.Module):
    def __init__(self, num_inputs):
        super(SVDD_AE, self).__init__()

        # initialize R at 0
        # use line search after the first couple epochs
        # self.R = torch.tensor(0, device=self.device)
        self.R = torch.tensor(0)
        self.warm_up_epochs = 10
        self.nu = args.nu

        self.num_inputs = num_inputs

        self.fc1 = nn.Linear(self.num_inputs, int(self.num_inputs/2))
        self.fc2 = nn.Linear(int(self.num_inputs/2), int(self.num_inputs/4))
        self.fc3 = nn.Linear(int(self.num_inputs/4), int(self.num_inputs/2))
        self.fc4 = nn.Linear(int(self.num_inputs/2), self.num_inputs)

    def encode(self, x):
        x = x.reshape(-1, self.num_inputs)
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        out = self.fc4(h3)
        out = out.view(-1, self.shape[-2], self.shape[-1])
        return out

    def forward(self, x):
        z = self.encode(x)

        return self.decode(z), z

    def init_center_c(self, data, eps=0.1):
        self.eval()

        with torch.no_grad():
            """Initialize hypersphere center c as the mean from an initial forward pass on the data."""

            print("data.shape")
            print(data.shape)

            z = self.encode(data)

            self.c = torch.mean(z, dim=0)

            # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
            self.c[(abs(self.c) < eps) & (self.c < 0)] = -eps
            self.c[(abs(self.c) < eps) & (self.c > 0)] = eps

            # np.save("models/centre_temp", self.c.cpu().detach().numpy())

            print("self.c")
            print(self.c)


# Reconstruction + Deep One-Class Classification
def loss_function_svdd_ae(recon_data, z, data, c, R, nu, epoch):

    data = data.view(-1, 69)
    recon_data = recon_data.view(-1, 69)

    # _, original_joints = smpl_layer(data)
    # _, reconstructed_joints = smpl_layer(recon_data)

    # pose_loss = (original_joints-reconstructed_joints)**2
    pose_loss = (data-recon_data)**2
    pose_loss = torch.sum(pose_loss)

    latent_dist = torch.norm((z - c), dim=1)
    scores = latent_dist - float(R)
    one_class_loss = float(R) + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

    return one_class_loss + pose_loss*args.tr_pose_weight, one_class_loss, pose_loss



def train_svdd_ae(model, data, val_target, val_pred, optimizer, initialize = False):

    data = data[:, :, 6:75]
    val_target = val_target[:, :, 6:75]
    val_pred = val_pred[:, :, 6:75]

    if(initialize):
        # initialize c
        model.init_center_c(data)
        model.shape = data.shape
    else:
        for param in model.parameters():
            param.requires_grad = True

    print("data.shape")
    print(data.shape)

    this_batch = data.to('cuda')

    for epoch in range(args.train_epochs):

        model.train()

        sleep(.1)

        train_loss = 0
        train_one_class_loss = 0
        train_pose_loss = 0
    
        
        optimizer.zero_grad()
        recon_data, z = model.forward(this_batch)
        loss, one_class_loss, pose_loss = loss_function_svdd_ae(recon_data, z, this_batch, model.c, model.R, model.nu, epoch)
        loss.backward()
        train_loss += loss.item()
        train_one_class_loss += one_class_loss.item()
        train_pose_loss += pose_loss.item()
        optimizer.step()

        if(epoch %100==0):
            print("epoch {}".format(epoch))
            print("loss: {}".format(train_loss))
            print("one class loss {}, pose loss {}".format(train_one_class_loss, train_pose_loss))
            print("R: {}".format(model.R))
            print("training anomoly score")
            check_anomoly_score(data, model)
            print("val anomoly score")
            check_anomoly_score(val_target, model)
            print("vibe anomoly score")
            check_anomoly_score(val_pred, model)

        if(epoch > model.warm_up_epochs or initialize == False):
            model.R = torch.tensor(get_new_radius(model, data, model.c, model.nu))

        if(epoch %100 == 0):
            # np.save("models/R_temp", model.R)
            # torch.save(model.state_dict(), "models/model_temp.pt")
            print("R value is {}".format(model.R))


        

# borrowed from https://github.com/gulvarol/smplpytorch
def th_posemap_axisang(pose_vectors):
    '''
    Converts axis-angle to rotmat
    pose_vectors (Tensor (batch_size x 72)): pose parameters in axis-angle representation
    '''
    rot_nb = int(pose_vectors.shape[1] / 3)
    rot_mats = []
    for joint_idx in range(rot_nb):
        axis_ang = pose_vectors[:, joint_idx * 3:(joint_idx + 1) * 3]
        rot_mat = batch_rodrigues(axis_ang)
        rot_mats.append(rot_mat)

    rot_mats = torch.cat(rot_mats, 1)
    return rot_mats

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def return_joints(smpl_input, smpl, J_regressor):

    smpl_input = smpl_input.clone()

    smpl_input[:, :6] = 0

    th_pose_rotmat = th_posemap_axisang(smpl_input[:, 3:75]).view(-1, 24, 3, 3)
    
    pred_vertices = smpl(global_orient=th_pose_rotmat[:, 0].unsqueeze(1), body_pose=th_pose_rotmat[:, 1:], betas=smpl_input[:,75:], pose2rot=False).vertices
    J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
    pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
    pred_joints = pred_joints[:, H36M_TO_J14, :]
    pred_pelvis = (pred_joints[:,[2],:] + pred_joints[:,[3],:]) / 2.0

    pred_joints -= pred_pelvis

    return pred_joints

def return_joints_gt(smpl_input, smpl, J_regressor):

    th_pose_rotmat = th_posemap_axisang(smpl_input[:, 3:75]).view(-1, 24, 3, 3)
    
    pred_vertices = smpl(global_orient=th_pose_rotmat[:, 0].unsqueeze(1), body_pose=th_pose_rotmat[:, 1:], betas=smpl_input[:,75:], pose2rot=False).vertices
    J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
    pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
    pred_joints = pred_joints[:, H36M_TO_J14, :]
    pred_pelvis = (pred_joints[:,[2],:] + pred_joints[:,[3],:]) / 2.0

    pred_joints -= pred_pelvis

    return pred_joints

def optimize_latent_vector(pred, target, model, window_size):

    J_regressor = torch.from_numpy(np.load('../VIBE/data/vibe_data/J_regressor_h36m.npy')).float().to('cuda')

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    if(args.is_svdd_ae == 1):
        R = model.R
        c = model.c

    smpl = SMPL(
        '../VIBE/{}'.format(SMPL_MODEL_DIR),
        batch_size=64,
        create_transl=False
    ).to('cuda')
    # smpl_layer = SMPL_Layer(
    #     center_idx=0,
    #     gender='neutral',
    #     model_root='models/smpl_model').to('cuda')

    print("pred.shape")
    print(pred.shape)

    batch_size = 150

    this_data = []
    gt_data = []
    gt_gt_data = []
    initial_gt_data = []

    # initial_to_gt_diff = 0

    with torch.no_grad():
        for x in range(0, pred.shape[0], batch_size):

            size = min(pred.shape[0]-x, batch_size)

            # zeros = torch.zeros(window_size*(size), 3).to(device)

            smpl_input = target[x:x+size].reshape(-1, target[x:x+size].shape[-1])

            gt_joints = return_joints(smpl_input, smpl, J_regressor)
            gt_joints = gt_joints.reshape(-1, window_size, gt_joints.shape[-2], gt_joints.shape[-1])
            gt_data.append(gt_joints)


            smpl_input = pred[x:x+size].reshape(-1, pred[x:x+size].shape[-1])

            initial_joints = return_joints(smpl_input, smpl, J_regressor)
            initial_joints = initial_joints.reshape(-1, window_size, gt_joints.shape[-2], gt_joints.shape[-1])
            this_data.append(initial_joints)


            smpl_input = target[x:x+size].reshape(-1, target[x:x+size].shape[-1])

            gt_gt_joints = return_joints_gt(smpl_input, smpl, J_regressor)
            gt_gt_joints = gt_gt_joints.reshape(-1, window_size, gt_gt_joints.shape[-2], gt_gt_joints.shape[-1])
            gt_gt_data.append(gt_gt_joints)

            smpl_input = pred[x:x+size].reshape(-1, pred[x:x+size].shape[-1])

            initial_gt_joints = return_joints_gt(smpl_input, smpl, J_regressor)
            initial_gt_joints = initial_gt_joints.reshape(-1, window_size, initial_gt_joints.shape[-2], initial_gt_joints.shape[-1])
            initial_gt_data.append(initial_gt_joints)



    this_data = torch.cat(this_data, 0)
    gt_data = torch.cat(gt_data, 0)
    gt_gt_data = torch.cat(gt_gt_data, 0)
    initial_gt_data = torch.cat(initial_gt_data, 0)
    optimized_data = pred[:, :, 6:75].clone()

    if(args.is_svdd_ae == 1):
        recon_data, z = model.forward(optimized_data)
        optimized_data.requires_grad = True
        optimizer = optim.Adam([optimized_data], lr=args.svdd_ae_optimizer_lr)
        dist_from_cent = torch.norm(z-c, dim = 1)-R
    else:
        recon_data, z, logvar = model.forward(optimized_data)
        optimized_data.requires_grad = True
        optimizer = optim.Adam([optimized_data], lr=args.vae_optimizer_lr)
        dist_from_cent = 0

    optimizer.zero_grad()

    gt_loss = []
    current_loss = []
    pose_losses = []
    svdd_losses = []

    lowest_to_gt_diff_pa = None
    last_epoch_updated = 0

    for epoch in range(args.opt_epochs):

        permutation = torch.randperm(pred.size()[0])

        pose_loss = 0
        svdd_loss = 0
        current_to_gt_diff = 0
        current_to_gt_diff_pa = 0
        initial_to_gt_diff = 0
        initial_to_gt_diff_pa = 0



        for batch in range(0, pred.shape[0], batch_size):

            size = min(pred.shape[0]-batch, batch_size)

            indices = permutation[batch:batch+size]

            this_batch = optimized_data[indices].to('cuda')

            if(args.is_svdd_ae == 1):
                recon_data, z = model.forward(this_batch)
            else:
                recon_data, z, logvar = model.forward(this_batch)

            

            recon_data = recon_data.reshape(-1, recon_data.shape[-1])

            orientation = pred[indices][:, :, :6].reshape(-1, 6)
            this_batch = this_batch.reshape(-1, 69)
            betas = pred[indices][:, :, 75:].reshape(-1, 10)


            this_batch = torch.cat((orientation, this_batch, betas), dim=-1)
            recon_data = torch.cat((orientation, recon_data, betas), dim=-1)

            #going to need to run this for both the optimized and recon data
            optimized_joints = return_joints(this_batch, smpl, J_regressor)
            recon_joints = return_joints(recon_data, smpl, J_regressor)

            optimized_joints = optimized_joints.reshape(size, window_size, optimized_joints.shape[-2], optimized_joints.shape[-1])
            recon_joints = recon_joints.reshape(size, window_size, recon_joints.shape[-2], recon_joints.shape[-1])

            if(args.is_svdd_ae == 1):
                dist_from_cent_loss = torch.norm(z-c, dim = 1)-R

                zeros = torch.zeros(dist_from_cent_loss.shape).to('cuda')
                dist_from_cent_loss = torch.where(dist_from_cent_loss > 0, dist_from_cent_loss, zeros)

                dist_from_cent_loss = torch.sum(dist_from_cent_loss)
            else:
                dist_from_cent_loss = torch.zeros(1).to('cuda')

            initial_joints = this_data[indices].to('cuda')
            initial_joints = initial_joints.reshape(size, window_size, initial_joints.shape[-2], initial_joints.shape[-1])

            initial_pose_differences = torch.sum(torch.sqrt(((optimized_joints - initial_joints) ** 2).sum(dim=-1)).view(size, -1).mean(dim=-1))
            pose_differences = torch.sum(torch.sqrt(((optimized_joints - recon_joints) ** 2).sum(dim=-1)).view(size, -1).mean(dim=-1))

            gt_joints = gt_data[indices].to('cuda')
            gt_joints = gt_joints.reshape(size, window_size, gt_joints.shape[-2], gt_joints.shape[-1])

            current_to_gt_diff += torch.sum(torch.sqrt(((optimized_joints - gt_joints) ** 2).sum(dim=-1)).view(size, -1).mean(dim=-1)).item()

            initial_to_gt_diff += torch.sum(torch.sqrt(((initial_joints - gt_joints) ** 2).sum(dim=-1)).view(size, -1).mean(dim=-1)).item()

            pose_loss += pose_differences.item()
            svdd_loss += dist_from_cent_loss.item() 

            loss = pose_differences*args.opt_pose_weight + initial_pose_differences*args.opt_initial_pose_weight + dist_from_cent_loss
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        current_to_gt_diff *= 1000/pred.shape[0]
        # current_to_initial_diff *= 1000/pred.shape[0]
        initial_to_gt_diff *= 1000/pred.shape[0]
        # current_to_gt_diff_pa *= 1000/pred.shape[0]
        # initial_to_gt_diff_pa *= 1000/pred.shape[0]
        

        current_loss.append(current_to_gt_diff)
        gt_loss.append(initial_to_gt_diff)
        pose_losses.append(pose_loss)
        svdd_losses.append(svdd_loss)


        print("epoch")
        print(epoch)
        print("pose_loss")
        print(pose_loss)
        print("svdd_loss")
        print(svdd_loss)
        # print("current_to_initial_diff")
        # print(current_to_initial_diff)
        print("current_to_gt_diff")
        print(current_to_gt_diff)
        print("initial_to_gt_diff")
        print(initial_to_gt_diff)
        # print("current_to_gt_diff_pa")
        # print(current_to_gt_diff_pa)
        # print("initial_to_gt_diff_pa")
        # print(initial_to_gt_diff_pa)
        print()

        

        if(epoch%10==0):
        # if (False):

            this_pickle = {}

            current_to_gt_diff = 0

            initial_to_gt_diff = 0
            current_to_gt_distances = []
            initial_to_gt_distances = []

            gt_saved_joints = []
            these_saved_joints = []
            initial_saved_joints = []


            with torch.no_grad():
                for x in range(0, pred.shape[0], batch_size):

                    size = min(pred.shape[0]-x, batch_size)

                    orientation = pred[x:x+size, int(window_size/2), :6]
                    betas = pred[x:x+size, int(window_size/2), 75:]
                    this_batch = optimized_data[x:x+size, int(window_size/2)].to('cuda')

                    print(orientation.shape)
                    print(betas.shape)
                    print(this_batch.shape)

                    smpl_input = torch.cat((orientation, this_batch, betas), dim=-1)
                    these_joints = return_joints_gt(smpl_input, smpl, J_regressor)
                    these_joints = these_joints.reshape(size, 14, 3)

                    gt_saved_joints.append(gt_gt_data[x:x+size, int(window_size/2)])
                    initial_saved_joints.append(initial_gt_data[x:x+size, int(window_size/2)])
                    these_saved_joints.append(these_joints)

                    current_to_gt_diff += torch.sum(torch.sqrt(((these_joints - gt_gt_data[x:x+size, int(window_size/2)]) ** 2).sum(dim=-1)).view(size, -1).mean(dim=-1)).item()

                    print(these_joints.shape)
                    print(these_joints[0])
                    print(gt_gt_data[x:x+size, int(window_size/2)].shape)
                    print(gt_gt_data[0, int(window_size/2)])

                    S1_hat = batch_compute_similarity_transform_torch(these_joints.view(size, -1, 3), gt_gt_data[x:x+size, int(window_size/2)].view(size, -1, 3))
                    S1_hat = S1_hat.reshape(size, -1, 3)
                    # distances 
                    distances = torch.sqrt(((S1_hat - gt_gt_data[x:x+size, int(window_size/2)]) ** 2).sum(dim=-1)).view(size, -1).mean(dim=-1)
                    current_to_gt_diff_pa += torch.sum(distances).item()
                    current_to_gt_distances.append(distances)

                    initial_to_gt_diff += torch.sum(torch.sqrt(((initial_gt_data[x:x+size, int(window_size/2)] - gt_gt_data[x:x+size, int(window_size/2)]) ** 2).sum(dim=-1)).view(size, -1).mean(dim=-1)).item()

                    S1_hat = batch_compute_similarity_transform_torch(initial_gt_data[x:x+size, int(window_size/2)].view(size, -1, 3), gt_gt_data[x:x+size, int(window_size/2)].view(size, -1, 3))
                    S1_hat = S1_hat.reshape(size, -1, 3)
                    distances = torch.sqrt(((S1_hat - gt_gt_data[x:x+size, int(window_size/2)]) ** 2).sum(dim=-1)).view(size, -1).mean(dim=-1)
                    initial_to_gt_diff_pa += torch.sum(distances).item()
                    initial_to_gt_distances.append(distances)


            gt_saved_joints = torch.cat(gt_saved_joints) 
            these_saved_joints = torch.cat(these_saved_joints) 
            initial_saved_joints = torch.cat(initial_saved_joints)

            print(gt_saved_joints.shape)
            print(these_saved_joints.shape)

            this_pickle['gt_saved_joints'] = gt_saved_joints
            this_pickle['these_saved_joints'] = these_saved_joints
            this_pickle['initial_gt_data'] = initial_saved_joints


            current_to_gt_distances = torch.cat(current_to_gt_distances)
            initial_to_gt_distances = torch.cat(initial_to_gt_distances)

            print("torch.mean(current_to_gt_distances)")
            print(torch.mean(current_to_gt_distances))

            this_pickle['initial_dist_from_cent'] = dist_from_cent

            this_pickle['current_to_gt_distances'] = current_to_gt_distances
            this_pickle['initial_to_gt_distances'] = initial_to_gt_distances

            current_to_gt_diff *= 1000/pred.shape[0]
            initial_to_gt_diff *= 1000/pred.shape[0]
            current_to_gt_diff_pa *= 1000/pred.shape[0]
            initial_to_gt_diff_pa *= 1000/pred.shape[0]

            print("current_to_gt_diff")
            print(current_to_gt_diff)
            this_pickle['current_to_gt_distances'] = current_to_gt_distances
            print("initial_to_gt_distances")
            print(initial_to_gt_distances)
            this_pickle['initial_to_gt_distances'] = initial_to_gt_distances
            print("initial_to_gt_diff")
            print(initial_to_gt_diff)
            this_pickle['initial_to_gt_diff'] = initial_to_gt_diff
            print("current_to_gt_diff_pa")
            print(current_to_gt_diff_pa)
            this_pickle['current_to_gt_diff_pa'] = current_to_gt_diff_pa
            if(lowest_to_gt_diff_pa is None):
                lowest_to_gt_diff_pa = current_to_gt_diff_pa
            elif(current_to_gt_diff_pa < lowest_to_gt_diff_pa):
                lowest_to_gt_diff_pa = current_to_gt_diff_pa
                last_epoch_updated = epoch
            print("lowest_to_gt_diff_pa")
            print(lowest_to_gt_diff_pa)
            print("initial_to_gt_diff_pa")
            print(initial_to_gt_diff_pa)
            this_pickle['initial_to_gt_diff_pa'] = initial_to_gt_diff_pa

            pickle.dump(this_pickle, open(f'out_folder/{args.name}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            print()

            # if(epoch - last_epoch_updated > 40):
            #     return
                


def check_anomoly_score(data, model, print_stuff = False):

    R = model.R
    c = model.c

    
    recon_data, z = model.forward(data)

    latent_dist = torch.norm((z - c), dim=1) - R

    ones = torch.ones(latent_dist.shape).to('cuda')
    zeros = torch.zeros(latent_dist.shape).to('cuda')

    print("percent outside of threshold")
    print(torch.mean(torch.where(latent_dist > 0, ones, zeros)).data)


def get_new_radius(model, data, c, nu):

    recon_data, z = model.forward(data)

    latent_dist = torch.norm((z - c), dim=1)

    return get_radius(latent_dist, nu)


# taken from https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py
def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)


def draw_data(batches, name, scores=None):
    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root='models/smpl_model').to('cuda')
    

    print("totoal lenggth {}".format(batches[0].shape[0]))
    for i in range(batches[0].shape[0]):
        # Forward from the SMPL layer

        verts = []
        joints = []
        for batch in batches:

            zeros = torch.zeros(batch[i].shape[0], 6).to('cuda')

            smpl_input = torch.cat((zeros, batch[i]), dim=-1)

            these_verts, these_joints = smpl_layer(smpl_input)
            verts.append(these_verts)
            joints.append(these_joints)


        for j in range(batch.shape[1]):

            if(len(batches) > 1):
                # Draw output vertices and joints
                fig, ax = display_model(
                    {'verts': verts[0].cpu().detach(),
                    'joints': joints[0].cpu().detach()},
                    model_faces=smpl_layer.th_faces,
                    with_joints=True,
                    kintree_table=smpl_layer.kintree_table,
                    batch_idx=j,
                    # savepath=f'images/{name}_{i:04d}_{j:02d}.png',
                    show=False)

                fig, ax = display_model(
                    {'verts': verts[1].cpu().detach(),
                    'joints': joints[1].cpu().detach()},
                    model_faces=smpl_layer.th_faces,
                    with_joints=True,
                    fig=fig,
                    ax=ax,
                    kintree_table=smpl_layer.kintree_table,
                    batch_idx=j,
                    savepath='images/{}_{:04d}_{:02d}.png'.format(name, i, j),
                    title="score: {}".format(scores[i]),
                    show=False)
            else:
                display_model(
                    {'verts': verts[0].cpu().detach(),
                    'joints': joints[0].cpu().detach()},
                    model_faces=smpl_layer.th_faces,
                    with_joints=True,
                    kintree_table=smpl_layer.kintree_table,
                    batch_idx=j,
                    savepath='images/{}_{:04d}_{:02d}.png'.format(name, i, j),
                    title="score: {}".format(scores[i]),
                    show=False)

            plt.close()
    


def load_VAE(name, shape):
    vae_model = SVDD_AE(shape).to(device)
    vae_model.load_state_dict(torch.load(name))
    vae_model.eval()

    return vae_model

def load_data(step = 1, window_size = 11, add_ends = False):


    train_target = load_VIBE("train", "target", step=step, window_size = window_size, add_ends = add_ends)
    train_target = torch.Tensor(train_target).to('cuda')
    train_pred = load_VIBE("train", "pred", step=step, window_size = window_size, add_ends = add_ends)
    train_pred = torch.Tensor(train_pred).to('cuda')
    test_target = load_VIBE("test", "target", step=step, window_size = window_size, add_ends = add_ends)
    test_target = torch.Tensor(test_target).to('cuda')
    test_pred = load_VIBE("test", "pred", step=step, window_size = window_size, add_ends = add_ends)
    test_pred = torch.Tensor(test_pred).to('cuda')
    val_target = load_VIBE("validation", "target", step=step, window_size = window_size, add_ends = add_ends)
    val_target = torch.Tensor(val_target).to('cuda')
    val_pred = load_VIBE("validation", "pred", step=step, window_size = window_size, add_ends = add_ends)
    val_pred = torch.Tensor(val_pred).to('cuda')

    return train_target, train_pred, test_target, test_pred, val_target, val_pred
    


def train_model(model=None):
    train_target, train_pred, test_target, test_pred, val_target, val_pred = load_data(window_size=args.window_size)

    train_target = torch.cat([train_target, test_target, val_target], dim=0)


    num_inputs = train_target.shape[-2]*(train_target.shape[-1]-16)

    initialize = False

    
    # model = load_VAE("models/model.pt", train_target.shape)
    # model.R = torch.from_numpy(np.load("models/R.npy")).to(device)
    # model.c = torch.from_numpy(np.load("models/centre.npy")).to(device)

    
    if(args.is_svdd_ae==1):
        if model is None:
            model = SVDD_AE(num_inputs).to('cuda')
            initialize = True
        optimizer = optim.Adam(model.parameters(), lr=args.svdd_ae_lr)
        train_svdd_ae(model, train_target, test_target, test_pred, optimizer, initialize=initialize)
    else:
        if model is None:
            model = VAE(num_inputs).to('cuda')
            initialize = True
        optimizer = optim.Adam(model.parameters(), lr=args.vae_lr)
        train_vae(model, train_target, optimizer, initialize=initialize)

    return model

def test_model(model):

    train_target, train_pred, test_target, test_pred, val_target, val_pred = load_data(window_size = args.window_size, add_ends=True)

    # num_inputs = train_target.shape[-2]*(train_target.shape[-1]-16)
    # model = load_VAE("models/model_temp.pt", num_inputs)
    # model.R = torch.from_numpy(np.load("models/R_temp.npy")).to('cuda')
    # model.c = torch.from_numpy(np.load("models/centre_temp.npy")).to('cuda')

    optimize_latent_vector(test_pred, test_target, model, window_size=args.window_size)

    # check_anomoly_score(test_pred, model, print_stuff = True)


if __name__ == "__main__":

    # to get the time aspect, load all in an array 
    # get the ground truth to be the new frame. 
    # play around with a different number of frames and such. 
    # 

    parser = argparse.ArgumentParser()

    device = torch.device("cuda")

    parser.add_argument('--name', default="out_pickle",
                        help='name of the pickle file')
    parser.add_argument('--is_svdd_ae', type=int, default=1,
                        help='0 for vae, 1 for svdd_ae')
    parser.add_argument('--opt_pose_weight', type=float, default=20,
                        help='weight of pose loss on optimizing')
    parser.add_argument('--opt_initial_pose_weight', type=float, default=20,
                        help='weight of pose loss on optimizing')
    parser.add_argument('--tr_pose_weight', type=float, default=1e-5,
                        help='weight of pose loss on training')
    parser.add_argument('--train_epochs', type=int, default=1000,
                        help='number of epochs to train between optimization steps')
    parser.add_argument('--opt_epochs', type=int, default=101,
                        help='number of epochs to train between optimization steps')
    parser.add_argument('--vae_lr', type=float, default=1e-4,
                        help='learning rate for the autoencoder')
    parser.add_argument('--svdd_ae_lr', type=float, default=1e-4,
                        help='learning rate for the autoencoder')
    parser.add_argument('--svdd_ae_optimizer_lr', type=float, default=1e-4,
                        help='learning rate for the optimizer')
    parser.add_argument('--vae_optimizer_lr', type=float, default=1e-4,
                        help='learning rate for the optimizer')
    parser.add_argument('--nu', type=float, default=1e-1,
                        help='nu')
    parser.add_argument('--kl_weight', type=float, default=1e-2,
                        help='kl weight for the vae')
    parser.add_argument('--window_size', type=int, default=11,
                        help='number of subsequent frames')
    args = parser.parse_args()

    print("args")
    print(args)


    model = train_model()
    test_model(model)

    # for i in range(9):
    #     model = train_model(model)
    #     test_model(model)

