# "borowed" from https://github.com/pytorch/examples/blob/master/vae/main.py


from __future__ import print_function
import argparse
import torch
from torch import nn, optim
import torchvision.models as models
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Arrow

import sys

import os

import wandb


# from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from eval_utils import batch_compute_similarity_transform_torch

# from spacepy import pycdf
import numpy as np
import joblib
import glob
import pickle
import random
from time import sleep

import imageio
from utils.backbone import FrozenBatchNorm2d
from utils import utils
from warp import perturbation_helper, sampling_helper 

import math

from tqdm import tqdm


class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=num_channels,
                                 eps=1e-5, affine=True)
    
    def forward(self, x):
        x = self.norm(x)
        return x



class MPJPE_Model(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self):
        super(MPJPE_Model, self).__init__()

        self.num_inputs = 512*4*4
        self.num_joints = 14

        # os.environ['TORCH_HOME'] = 'models\\resnet_pretrained'

        self.resnet = getattr(models, "resnet18")(norm_layer=MyGroupNorm)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:3], *list(self.resnet.children())[4:-2])

        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        self.linearized_sampler = sampling_helper.DifferentiableImageSampler('linearized', 'zeros')
        self.crop_scalar = args.crop_scalar
        self.crop_size = [64, 64]
        self.linear_operations = []

        for i in range(14):

            linear_operation = nn.Sequential(
                nn.Linear(self.num_inputs, 512),
                # nn.GroupNorm(32, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            ).to(device)
            
            self.linear_operations.append(linear_operation)

        self.linear_operations = nn.ModuleList(self.linear_operations)


        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.data_transforms = transforms.Compose([
                self.normalize,
                transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.3, hue=0)
                ])
        

        # for param in self.linear_operations.parameters():
        #     param.requires_grad = True


    def forward(self, input_dict):  

        image = input_dict['image']
        dims_before = input_dict['dims_before']
        joints3d = input_dict['estimated_j3d']
        cam = input_dict['cam']
        # gt_j2d = input_dict['gt_j2d']
        bboxes = input_dict['bboxes']

        training = input_dict["training"][0]

        self.input_images = []

        # for every joint, get the crop centred at the reprojected 2d location

        joints2d = projection(joints3d, cam)

        bboxes = bboxes.unsqueeze(1).expand(-1, joints2d.shape[1], -1)

        joints2d[:, :, 0] *= bboxes[:, :, 2]/2*1.1
        joints2d[:, :, 0] += bboxes[:, :, 0]
        joints2d[:, :, 1] *= bboxes[:, :, 3]/2*1.1
        joints2d[:, :, 1] += bboxes[:, :, 1]

        zeros = torch.zeros(image.shape[0]).to(device)
        ones = torch.ones(image.shape[0]).to(device)

        # print("image.shape")
        # print(image.shape)

        error_estimates = []

        for i in range(14):
            
            dx = joints2d[:, i, 0]/(dims_before[:, 1]/2)-1
            x_mult = torch.where(dims_before[:, 0]==1920, 1080/1920*ones, ones)
            dx = dx*x_mult

            dy = joints2d[:, i, 1]/(dims_before[:, 0]/2)-1
            y_mult = torch.where(dims_before[:, 0]==1080, 1080/1920*ones, ones)
            dy = dy*y_mult

            vec = torch.stack([zeros, ones/self.crop_scalar, ones/self.crop_scalar, self.crop_scalar*dx, self.crop_scalar*dy], dim=1)

            transformation_mat = perturbation_helper.vec2mat_for_similarity(vec)

            linearized_transformed_image = self.linearized_sampler.warp_image(image, transformation_mat, out_shape=self.crop_size).contiguous()

            if(training):
                linearized_transformed_image = self.data_transforms(linearized_transformed_image)
            else:
                linearized_transformed_image = self.normalize(linearized_transformed_image)

            output = self.resnet(linearized_transformed_image)
            
            output = output.reshape(-1, self.num_inputs)

            error_estimate = self.linear_operations[i](output)

            error_estimate = torch.norm(error_estimate, dim=-1)/150

            error_estimates.append(error_estimate)

        error_estimates = torch.stack(error_estimates, dim=-1)

        return error_estimates

        # if(i == 6):
        #     blt = utils.torch_img_to_np_img(image)

        #     blt_crop = utils.torch_img_to_np_img(linearized_transformed_image)


        #     for j in range(image.shape[0]):

        #         if(dims_before[j, 0]==1920):
        #             offset = [420, 0]
        #         else:
        #             offset = [0, 420]
                
        #         plt.imshow(blt[j])
        #         ax = plt.gca()
        #         for k in range(joints2d.shape[1]):
        #             circ = Circle((gt_j2d[j, k, 0]+offset[0],gt_j2d[j, k, 1]+offset[1]),10, color = 'b')
        #             ax.add_patch(circ)
        #         for k in range(joints2d.shape[1]):
        #             circ = Circle((joints2d[j, k, 0]+offset[0],joints2d[j, k, 1]+offset[1]),10, color = 'r')
        #             ax.add_patch(circ)
        #         plt.savefig(f"{j:03d}_figure")
        #         plt.close()
                
        #         plt.imshow(blt_crop[j])
        #         plt.savefig(f"{j:03d}_crop")
        #         plt.close()
        #     exit()



def load_image(images, index, training):

    image = imageio.imread(f"{images[index]}")/255.0
    image = utils.np_img_to_torch_img(image).float()


    # plt.imshow(  image.permute(1, 2, 0)  )
    # plt.savefig(f"before_transform_{index}")

    # for i in range(10):
    #     after_transform = data_transform(image)
    #     plt.imshow(  after_transform.permute(1, 2, 0)  )
    #     plt.savefig(f"after_transform_{index}_{i}")
    #     plt.close()
    # exit()
    dims_before = torch.Tensor([image.shape[1], image.shape[2]])

    if(dims_before[0]==1080):
        top_bottom_pad = torch.nn.ZeroPad2d((0, 0, 420, 420))
        image = top_bottom_pad(image)
    else:
        sides_pad = torch.nn.ZeroPad2d((420, 420, 0, 0))
        image = sides_pad(image)

    assert image.shape[1] == 1920 and image.shape[2] == 1920

    return image, dims_before


class data_set(Dataset):
    def __init__(self, input_dict, training=True):
        self.images = input_dict['images']
        self.estimated_j3d = input_dict['estimated_j3d']
        self.gt_j3d = input_dict['gt_j3d']
        self.gt_j2d = input_dict['gt_j2d']
        self.gt_cam = input_dict['gt_cam']
        self.pred_cam = input_dict['pred_cam']
        self.bboxes = input_dict['bboxes']
        self.mpjpe = input_dict['mpjpe']
        self.training = training

    
    def __getitem__(self, index):

        # print(f"self.images[index] {self.images[index]}")

        image, dims_before = load_image(self.images, index, self.training)

        # only add the noide to gt if training
        if(self.training):

            # vector pointing to ground truth from estimated
            ground_truth_vector = self.gt_j3d[index]-self.estimated_j3d[index]

            # chose a point along the path between estimated and ground truth
            estimated_j3d = torch.rand(1)*ground_truth_vector+self.estimated_j3d[index]

            # add noise to the point
            estimated_j3d = estimated_j3d + (torch.rand(self.gt_j3d[index].shape)*.1)-.05

            cam = self.gt_cam[index]

            # project with can and get 2d error
            projected_2d_estimated_joints = projection(estimated_j3d[None], cam[None])
            projected_2d_gt_joints = projection(self.gt_j3d[index][None], cam[None])
            
            mpjpe = torch.sqrt(((projected_2d_estimated_joints - projected_2d_gt_joints) ** 2).sum(dim=-1))[0]

            gt_gradient = 2*(estimated_j3d-self.gt_j3d[index])

            
        else:

            estimated_j3d = self.estimated_j3d[index]

            cam = self.pred_cam[index]

            mpjpe = self.mpjpe[index]

            gt_gradient = 2*(estimated_j3d-self.gt_j3d[index])



        output_dict = {'indices': index, 'image': image, 'dims_before': dims_before, 'estimated_j3d': estimated_j3d, 'gt_j3d':self.gt_j3d[index], 'gt_j2d': self.gt_j2d[index], 'gt_gradient': gt_gradient, 'cam': cam, 'bboxes': self.bboxes[index], 'mpjpe': mpjpe, 'training': self.training}

        # for item in output_dict:
        #     output_dict[item] = output_dict[item].to(device) 

        return output_dict
    
    # def __len__(self, index):
    #     return len(self.images)

    def __len__(self):
        return len(self.images)

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

def optimize_camera_parameters(joints3d, joints2d, bboxes):

    joints3d = joints3d.to(device)
    joints2d = joints2d.to(device)
    bboxes = bboxes.to(device)

    pred_cam = torch.zeros((joints3d.shape[0], 3))

    pred_cam.requires_grad = True

    optimizer = optim.Adam([pred_cam], lr=.01)

    loss_function = nn.MSELoss()

    bboxes = bboxes.unsqueeze(1).expand(-1, joints2d.shape[1], -1)

    for i in range(3000):

        optimizer.zero_grad()


        joints2d_estimated = projection(joints3d, pred_cam)
        

        joints2d_estimated[:, :, 0] *= bboxes[:, :, 2]/2*1.1
        joints2d_estimated[:, :, 0] += bboxes[:, :, 0]
        joints2d_estimated[:, :, 1] *= bboxes[:, :, 3]/2*1.1
        joints2d_estimated[:, :, 1] += bboxes[:, :, 1]

        zeros  = torch.zeros(joints2d_estimated[:, :, 0].shape).to(device)


        joints2d_estimated[:, :, 0] = torch.where(joints2d[:, :, 0]==0, zeros, joints2d_estimated[:, :, 0])
        joints2d_estimated[:, :, 1] = torch.where(joints2d[:, :, 1]==0, zeros, joints2d_estimated[:, :, 1])

        loss = loss_function(joints2d_estimated, joints2d[:, :, :2])

        loss.backward()

        optimizer.step()



    return pred_cam


def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1).to(pred_joints.device)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2).to(pred_joints.device)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def train_mpjpe_model():

    model = MPJPE_Model().to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    data_dict = load_data("train")

    val_data_dict = load_data("validation")
    
    this_data_set = data_set(data_dict)

    val_data_set = data_set(val_data_dict)

    loss_function = nn.MSELoss()

    loader = torch.utils.data.DataLoader(this_data_set, batch_size = args.training_batch_size, num_workers=4, shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size = args.training_batch_size, num_workers=1, shuffle=True)

    for epoch in range(args.train_epochs):

        total_loss = 0

        iterator = iter(loader)

        val_iterator = iter(val_loader)

        for iteration in tqdm(range(len(loader))):

            batch = next(iterator)

            for item in batch:
                batch[item] = batch[item].to(device) 

            batch['estimated_j3d'].requires_grad = True

            optimizer.zero_grad()
            estimated_loss = model.forward(batch)

            estimated_error_loss = loss_function(estimated_loss, batch['mpjpe'])

            pred_grad = torch.autograd.grad(
                                estimated_error_loss, batch['estimated_j3d'], 
                                retain_graph=True, 
                                create_graph=True
                                )[0]

            # batch['estimated_j3d'].requires_grad = False

            gradient_loss = loss_function(pred_grad, batch['gt_gradient'])

            loss = estimated_error_loss+args.grad_loss_weight*gradient_loss

            total_loss += loss.item()

            loss.backward()

            if(args.wandb_log):

                wandb.log({"loss": loss.item(), "gradient_loss": gradient_loss.item(), "estimated_error_loss": estimated_error_loss.item()})

            optimizer.step()


            del batch

            if(iteration%10==0):

                model.eval()
                val_batch = next(val_iterator)

                for item in val_batch:
                    val_batch[item] = val_batch[item].to(device) 

                estimated_loss = model.forward(val_batch)

                val_loss = loss_function(estimated_loss, val_batch['mpjpe'])

                if(args.wandb_log):
                    wandb.log({"validation loss": val_loss.item()}, commit=False)

                model.train()

                del val_batch

        print(f"epoch: {epoch}, loss: {total_loss}")

        if(args.wandb_log):
            draw_gradients(model, "train", "train")
            draw_gradients(model, "validation", "validation")

        torch.save(model.state_dict(), f"models/linearized_model_{args.crop_scalar}_epoch{epoch}.pt")

        # scheduler.step()
    
    
    return model

def test_mpjpe_model(model):
    # model = MPJPE_Model(num_inputs, num_joints).to(device)
    model.eval()

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
                batch[item] = batch[item].to(device) 


            optimized_joints = batch['estimated_j3d']
            initial_joints = initial_j3d[batch['indices']].to(device)

            optimized_joints.requires_grad = True

            optimizer = optim.SGD([optimized_joints], lr=args.optimization_rate)

            for i in range(10000):

                optimizer.zero_grad()
                estimated_loss = model.forward(batch)

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
                    print(f"loss {estimated_loss.item()}, iteration {i}")

            estimated_j3d[batch['indices']] = batch['estimated_j3d'].cpu().detach()

            # return the error

        evaluate(estimated_j3d, data_dict['gt_j3d'])

        print(f"epoch: {epoch}, estimated_loss_total: {estimated_loss_total}, pose_differences_total: {pose_differences_total}")
    
    return model

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
                

def find_gt_joints(pose, shape, smpl, J_regressor):

    th_pose_rotmat = th_posemap_axisang(pose).view(-1, 24, 3, 3)

    pred_vertices = smpl(global_orient=th_pose_rotmat[:, 0].unsqueeze(1), body_pose=th_pose_rotmat[:, 1:], betas=shape, pose2rot=False).vertices
    J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
    pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
    pred_joints = pred_joints[:, H36M_TO_J14, :]
    pred_pelvis = (pred_joints[:,[2],:] + pred_joints[:,[3],:]) / 2.0

    pred_joints -= pred_pelvis

    return pred_joints

# move the ground truth joints to line up with estimated
def move_gt_pelvis(gt_j3ds, j3ds):
    # move the hip location of gt to estimated
    pred_pelvis = (j3ds[:,[2],:] + j3ds[:,[3],:]) / 2.0
    gt_pelvis = (gt_j3ds[:,[2],:] + gt_j3ds[:,[3],:]) / 2.0

    gt_j3ds -= gt_pelvis
    gt_j3ds += pred_pelvis

    return gt_j3ds

def load_data(set):

    files = sorted(glob.glob(f"data/3dpw/predicted_poses/{set}/*/vibe_output.pkl"))

    if(set == "train"):

        J_regressor = torch.from_numpy(np.load('data/vibe_data/J_regressor_h36m.npy')).float().to(device)
            
        smpl = SMPL(
            '{}'.format(SMPL_MODEL_DIR),
            batch_size=64,
            create_transl=False
        ).to(device)


    images = []
    estimated_j3d = []
    pred_cam = []
    bboxes = []
    gt_j3d = []
    gt_j2d = []

    for file in files:
        data = joblib.load(file)

        for person in data:

            gt_indices = data[person]['gt_indeces']

            images.append(np.array(data[person]['images'])[gt_indices])

            j3ds = torch.Tensor(data[person]['joints3d'][gt_indices])

            estimated_j3d.append(j3ds)
            pred_cam.append(torch.Tensor(data[person]['pred_cam'][gt_indices]))
            bboxes.append(torch.Tensor(data[person]['bboxes'][gt_indices]))
            gt_j2d.append(torch.Tensor(data[person]['gt_joints2d']))

            if(set == "train"):
                pose = torch.Tensor(data[person]['gt_pose'])
                shape = torch.Tensor(data[person]['gt_shape'])
                gt_j3ds = find_gt_joints(pose.to(device), shape.to(device), smpl, J_regressor).cpu()

                gt_j3ds = move_gt_pelvis(gt_j3ds, j3ds)

                gt_j3d.append(gt_j3ds)
            else:
                gt_j3ds = torch.Tensor(data[person]['gt_joints3d'])

                gt_j3ds = move_gt_pelvis(gt_j3ds, j3ds)

                gt_j3d.append(gt_j3ds)
            
        # show what the reprojected person looks like here


    images = np.concatenate(images)
    estimated_j3d = torch.cat(estimated_j3d)
    gt_j3d = torch.cat(gt_j3d)
    gt_j2d = torch.cat(gt_j2d)
    pred_cam = torch.cat(pred_cam)
    bboxes = torch.cat(bboxes)

    gt_cam = optimize_camera_parameters(gt_j3d, gt_j2d, bboxes).detach().cpu()
    
    print("images.shape")
    print(images.shape)
    print("estimated_j3d.shape")
    print(estimated_j3d.shape)
    print("gt_j3d.shape")
    print(gt_j3d.shape)
    print("gt_j2d.shape")
    print(gt_j2d.shape)
    print("pred_cam.shape")
    print(pred_cam.shape)
    print("gt_cam.shape")
    print(gt_cam.shape)
    print("bboxes.shape")
    print(bboxes.shape)

    # project with cam and get 2d error
    projected_2d_estimated_joints = projection(estimated_j3d, gt_cam)
    projected_2d_gt_joints = projection(gt_j3d, gt_cam)
    

    mpjpe = torch.sqrt(((projected_2d_estimated_joints - projected_2d_gt_joints) ** 2).sum(dim=-1))

    print("torch.mean(mpjpe)")
    print(torch.mean(mpjpe))

    

    print("mpjpe.shape")
    print(mpjpe.shape)

    evaluate(estimated_j3d, gt_j3d)

    return {'images':images, 'estimated_j3d':estimated_j3d, 'gt_j3d':gt_j3d, 'gt_j2d':gt_j2d, 'gt_cam':gt_cam, 'pred_cam':pred_cam, 'bboxes':bboxes, 'mpjpe':mpjpe}



def draw_gradients(model, set, name):
    model.eval()

    # 'images':images, 'estimated_j3d':estimated_j3d, 'gt_j3d':gt_j3d, 'gt_j2d':gt_j2d, 'gt_cam':gt_cam, 'pred_cam':pred_cam, 'bboxes':bboxes, 'mpjpe':mpjpe
    data_dict = load_data(set)
    data_dict['estimated_j3d'] = data_dict['gt_j3d']
    data_dict['pred_cam'] = data_dict['gt_cam']
    # change the data dict so it only gets the first image many times over
    this_data_set = data_set(data_dict, training=False)
    loader = torch.utils.data.DataLoader(this_data_set, batch_size = 1, num_workers=0, shuffle=True)
    iterator = iter(loader)
    batch = next(iterator)


    images = torch.cat([batch['image']]*121, dim=0)
    estimated_j3d = torch.cat([batch['estimated_j3d']]*121, dim=0)
    gt_j3d = torch.cat([batch['gt_j3d']]*121, dim=0)
    dims_before = torch.cat([batch['dims_before']]*121, dim=0)
    gt_cam = torch.cat([batch['cam']]*121, dim=0)
    bboxes = torch.cat([batch['bboxes']]*121, dim=0)
    mpjpe = torch.cat([batch['mpjpe']]*121, dim=0)
    gt_gradient = torch.cat([batch['gt_gradient']]*121, dim=0)
    training = torch.cat([batch['training']]*121, dim=0)

    for x in range(11):
        for y in range(11):

            if(x ==5 and y== 5):
                estimated_j3d[x*11+y] = batch['gt_j3d']

                mpjpe[x*11+y] = 0

                continue

            this_index = x*11+y

            # vector pointing to ground truth from estimated
            ground_truth_vector = gt_j3d[this_index]-estimated_j3d[this_index]

            # chose a point along the path between estimated and ground truth
            estimated_j3d[this_index] = torch.rand(1)*ground_truth_vector+estimated_j3d[this_index]

            # add noise to the point
            estimated_j3d[this_index] = estimated_j3d[this_index] + (torch.rand(gt_j3d[this_index].shape)*.1)-.05

            projected_2d_estimated_joints = projection(estimated_j3d[this_index][None], gt_cam[this_index][None])
            projected_2d_gt_joints = projection(gt_j3d[this_index][None], gt_cam[this_index][None])

            this_mpjpe = torch.sqrt(((projected_2d_estimated_joints - projected_2d_gt_joints) ** 2).sum(dim=-1))

            mpjpe[this_index] = this_mpjpe[0]

            gt_gradient[this_index] = 2*(estimated_j3d[this_index]-batch['gt_j3d'])

    locations = []
    directions = []
    estimated_error = []

    batch_size = args.optimization_batch_size
    for i in range(0, images.shape[0], batch_size):

        size = min(images.shape[0]-i, batch_size)

        batch = {   'image': images[i:i+size],
                    'dims_before': dims_before[i:i+size],
                    'estimated_j3d': estimated_j3d[i:i+size],
                    'cam': gt_cam[i:i+size],
                    'bboxes': bboxes[i:i+size],
                    'training': training[i:i+size],
                    'gt_gradient': gt_gradient[i:i+size]}

        for item in batch:
            batch[item] = batch[item].to(device) 

        initial_j3d = batch['estimated_j3d'].clone()

        batch['estimated_j3d'].requires_grad = True

        optimizer = optim.Adam([batch['estimated_j3d']], lr=args.optimization_rate)

        optimizer.zero_grad()

        estimated_loss = model.forward(batch)

        estimated_loss.mean().backward()

        joints2d = projection(initial_j3d, batch['cam'])

        direction = batch['estimated_j3d'].grad

        des_bboxes = batch['bboxes'].unsqueeze(1).expand(-1, joints2d.shape[1], -1)

        joints2d[:, :, 0] *= des_bboxes[:, :, 2]/2*1.1
        joints2d[:, :, 0] += des_bboxes[:, :, 0]
        joints2d[:, :, 1] *= des_bboxes[:, :, 3]/2*1.1
        joints2d[:, :, 1] += des_bboxes[:, :, 1]

        locations.append(joints2d.cpu().detach())
        directions.append(direction.cpu().detach())

        estimated_error.append(estimated_loss.cpu().detach())


    crop_scalar = args.crop_scalar
    crop_size = [64, 64]

    locations = torch.cat(locations, dim=0)
    normalized_locations = locations-locations[60]
    normalized_locations*=(64/(1920/crop_scalar))
    normalized_locations += 32
    directions = torch.cat(directions, dim=0)

    this_error = torch.max(mpjpe)

    mpjpe /= this_error

    estimated_error = torch.cat(estimated_error, dim=0)

    estimated_error /= this_error

    # estimated_error -= torch.min(estimated_error, dim=0).values.unsqueeze(0).expand(estimated_error.shape[0], -1)
    # estimated_error /= torch.max(estimated_error, dim=0).values.unsqueeze(0).expand(estimated_error.shape[0], -1)
    # estimated_error /= torch.max(estimated_error)



    blt = utils.torch_img_to_np_img(images)

    if(dims_before[0, 0]==1920):
        offset = [420, 0]
    else:
        offset = [0, 420]

    plt.imshow(blt[0])
    ax = plt.gca()
    for j in range(locations.shape[1]):

        initial_x = locations[60, j, 0]+offset[0]
        initial_y = locations[60, j, 1]+offset[1]
    
        circ = Circle((initial_x,initial_y),10, color = 'r')

        ax.add_patch(circ)

    wandb.log({f"{name}_overall": wandb.Image(plt)}, commit=False)
    plt.close()

    crops = []
    estimated_errors = []
    gt_errors = []
    gradients = []
    gt_gradients = []

    for j in range(locations.shape[1]):

        dx = locations[60, j, 0]/(dims_before[0, 1]/2)-1
        if(dims_before[0, 0]==1920):
            dx *= 1080/1920
        

        dy = locations[60, j, 1]/(dims_before[0, 0]/2)-1
        if(dims_before[0, 0]==1080):
            dy *= 1080/1920

        vec = torch.Tensor([[0, 1/crop_scalar, 1/crop_scalar, crop_scalar*dx, crop_scalar*dy]])

        transformation_mat = perturbation_helper.vec2mat_for_similarity(vec)

        linearized_sampler = sampling_helper.DifferentiableImageSampler('linearized', 'zeros')

        linearized_transformed_image = linearized_sampler.warp_image(images[0], transformation_mat, out_shape=crop_size)


        plt.imshow(utils.torch_img_to_np_img(linearized_transformed_image)[0])
        crops.append(wandb.Image(plt))
        plt.close()

        # draw error curve
        plt.imshow(utils.torch_img_to_np_img(linearized_transformed_image)[0])
        ax = plt.gca()
        for k in range(normalized_locations.shape[0]):


            error = estimated_error[k, j]

            initial_loc = normalized_locations[k, j, :2]

            # ranging from green for small error and red for high error
            color = (error, 1-error, 0)
            color = np.clip(color, 0, 1)

            circ = Circle((initial_loc[0],initial_loc[1]),1, color = color)

            ax.add_patch(circ)

        estimated_errors.append(wandb.Image(plt))
        
        plt.close()

        plt.imshow(utils.torch_img_to_np_img(linearized_transformed_image)[0])
        ax = plt.gca()
        for k in range(normalized_locations.shape[0]):


            error = mpjpe[k, j]

            initial_loc = normalized_locations[k, j, :2]

            # ranging from green for small error and red for high error
            color = (error, 1-error, 0)
            color = np.clip(color, 0, 1)

            circ = Circle((initial_loc[0],initial_loc[1]),1, color = color)

            ax.add_patch(circ)

        gt_errors.append(wandb.Image(plt))
        plt.close()
        
        # draw gradients
        plt.imshow(utils.torch_img_to_np_img(linearized_transformed_image)[0])
        ax = plt.gca()
        for k in range(normalized_locations.shape[0]):

            initial_loc = normalized_locations[k, j, :2].cpu().detach()

            gradient = directions[k, j, :2].cpu().detach()

            gt_dir = normalized_locations[k, j, :2].cpu().detach()-normalized_locations[60, j, :2].cpu().detach()

            if(k == 60):
                color = 'b'
            else:
                dir_dot = torch.dot(gt_dir/torch.norm(gt_dir), gradient/torch.norm(gradient)).numpy()
                dir_scalar = (dir_dot+1)/2
                color = (dir_scalar, 1-dir_scalar, 0)

                color = np.clip(color, 0, 1)

            circ = Arrow(initial_loc[0].numpy(), initial_loc[1].numpy(), 
                                    gradient[0].numpy()*10, gradient[1].numpy()*10,
                                    width=1, color = color)

            ax.add_patch(circ)

        gradients.append(wandb.Image(plt))
        plt.close()

        # draw gradients
        plt.imshow(utils.torch_img_to_np_img(linearized_transformed_image)[0])
        ax = plt.gca()
        for k in range(normalized_locations.shape[0]):

            initial_loc = normalized_locations[k, j, :2].cpu().detach()

            gt_grad = gt_gradient[k, j, :2].cpu().detach()

            gt_dir = normalized_locations[k, j, :2].cpu().detach()-normalized_locations[60, j, :2].cpu().detach()

            if(k == 60):
                color = 'b'
            else:
                dir_dot = torch.dot(gt_dir/torch.norm(gt_dir), gt_grad/torch.norm(gt_grad)).numpy()
                dir_scalar = (dir_dot+1)/2
                color = (dir_scalar, 1-dir_scalar, 0)

                color = np.clip(color, 0, 1)

            circ = Arrow(initial_loc[0].numpy(), initial_loc[1].numpy(), 
                                    gt_grad[0].numpy()*10, gt_grad[1].numpy()*10,
                                    width=1, color = color)

            ax.add_patch(circ)

        gt_gradients.append(wandb.Image(plt))
        plt.close()


    wandb.log({f"{name}_crops": crops}, commit=False)
    wandb.log({f"{name}_estimated_errors": estimated_errors}, commit=False)
    wandb.log({f"{name}_gt_errors": gt_errors}, commit=False)
    wandb.log({f"{name}_gradients": gradients}, commit=False)
    wandb.log({f"{name}_gt_gradients": gt_gradients})
    # wandb.log({f"{name}_gt_gradients": gt_gradients})

    model.train()

    
    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--opt_epochs', type=int, default=5)
    parser.add_argument('--training_batch_size', type=int, default=16)
    parser.add_argument('--optimization_batch_size', type=int, default=16)
    parser.add_argument('--crop_scalar', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--optimization_rate', type=float, default=1e-3)
    parser.add_argument('--grad_loss_weight', type=float, default=1e0)
    parser.add_argument('--wandb_log', action='store_true')
    args = parser.parse_args()

    print("args")
    print(args)

    device = torch.device("cuda:0")

    
    if(args.wandb_log):
        wandb.init(project="human_body_pose_optimization", name="linearized_6")
        wandb.config.update(args) 

    

    model = train_mpjpe_model()
    # torch.save(model.state_dict(), f"models/linearized_model_{args.crop_scalar}.pt")
    # exit()
    # model = MPJPE_Model().to(device)
    # model.load_state_dict(torch.load(f"models/linearized_model_8_epoch49.pt", map_location=device))

    # for i in range(10):
    #     draw_gradients(model, "train", "demo_images_train")

    # test_mpjpe_model(model)





