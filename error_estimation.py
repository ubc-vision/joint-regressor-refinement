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
sys.path.append('../VIBE')
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.utils.eval_utils import batch_compute_similarity_transform_torch

sys.path.append('/home/willow/Documents/smplpytorch')
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from display_utils import display_model

# from spacepy import pycdf
import numpy as np
import joblib
import glob
import pickle
import random
from time import sleep

import imageio
from utils import utils
from warp import perturbation_helper, sampling_helper 

import math

from tqdm import tqdm


class MPJPE_Model(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self):
        super(MPJPE_Model, self).__init__()

        self.num_inputs = 7168
        self.num_joints = 14


        resnet = models.resnet18(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

        for param in self.resnet.parameters():
            param.requires_grad = False

        # self.linearized_sampler = sampling_helper.DifferentiableImageSampler('linearized', 'zeros')
        self.linearized_sampler = sampling_helper.DifferentiableImageSampler('bilinear', 'zeros')
        self.crop_scalar = 8
        self.crop_size = [64, 64]

        self.linear_operations = nn.Sequential(
            nn.Linear(self.num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_joints),
            nn.ReLU(),
        )

    def forward(self, input_dict):

        image = input_dict['image']
        dims_before = input_dict['dims_before']
        joints3d = input_dict['estimated_j3d']
        cam = input_dict['cam']
        # gt_j2d = input_dict['gt_j2d']
        bboxes = input_dict['bboxes']

        self.input_images = []

        # for every joint, get the crop centred at the reprojected 2d location

        joints2d = projection(joints3d, cam)

        bboxes = bboxes.unsqueeze(1).expand(-1, joints2d.shape[1], -1)

        joints2d[:, :, 0] *= bboxes[:, :, 2]/2*1.1
        joints2d[:, :, 0] += bboxes[:, :, 0]
        joints2d[:, :, 1] *= bboxes[:, :, 3]/2*1.1
        joints2d[:, :, 1] += bboxes[:, :, 1]

        zeros = torch.zeros(image.shape[0]).to('cuda')
        ones = torch.ones(image.shape[0]).to('cuda')

        outputs = []

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

            self.input_images.append(linearized_transformed_image)

            output = self.resnet(linearized_transformed_image)

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

            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)    
        outputs = outputs.reshape(outputs.shape[0], -1)

        error_estimates = self.linear_operations(outputs)

        return error_estimates


def load_image(images, index):

    image = imageio.imread(f"../VIBE/{images[index]}")/255.0
    image = utils.np_img_to_torch_img(image).float()

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

        image, dims_before = load_image(self.images, index)


        # sometimes load noise around the ground truth. 
        if(self.training):
            add_noise = random.random()
        else:
            add_noise = 1

        # only add the noide to gt if training
        if(add_noise > .5 and self.training):
            # makes the noise applied proportional to noise found between estimates and gt
            estimated_j3d = (self.gt_j3d[index] + torch.randn(self.gt_j3d[index].shape)*.06)
            
            # estimated_j3d = self.gt_j3d[index]
            mpjpe = torch.sqrt(((estimated_j3d - self.gt_j3d[index]) ** 2).sum(dim=-1))

            cam = self.gt_cam[index]
            
        else:

            estimated_j3d = self.estimated_j3d[index]

            mpjpe = self.mpjpe[index]

            cam = self.pred_cam[index]
            


        output_dict = {'indices': index, 'image': image, 'dims_before': dims_before, 'estimated_j3d': estimated_j3d, 'gt_j3d': self.gt_j3d[index], 'gt_j2d': self.gt_j2d[index], 'cam': cam, 'bboxes': self.bboxes[index], 'mpjpe': mpjpe}

        # for item in output_dict:
        #     output_dict[item] = output_dict[item].to('cuda') 

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

    joints3d = joints3d.to('cuda')
    joints2d = joints2d.to('cuda')
    bboxes = bboxes.to('cuda')

    pred_cam = torch.zeros((joints3d.shape[0], 3))

    pred_cam.requires_grad = True

    optimizer = optim.Adam([pred_cam], lr=.01)

    loss_function = nn.MSELoss()

    bboxes = bboxes.unsqueeze(1).expand(-1, joints2d.shape[1], -1)

    for i in range(1000):

        optimizer.zero_grad()


        joints2d_estimated = projection(joints3d, pred_cam)

        

        joints2d_estimated[:, :, 0] *= bboxes[:, :, 2]/2*1.1
        joints2d_estimated[:, :, 0] += bboxes[:, :, 0]
        joints2d_estimated[:, :, 1] *= bboxes[:, :, 3]/2*1.1
        joints2d_estimated[:, :, 1] += bboxes[:, :, 1]

        zeros  = torch.zeros(joints2d_estimated[:, :, 0].shape).to('cuda')


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



# def train_mpjpe_model(num_inputs, num_joints):
def train_mpjpe_model():

    # model = MPJPE_Model(num_inputs, num_joints).to('cuda')
    model = MPJPE_Model().to('cuda')
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    data_dict = load_data("train")
    
    this_data_set = data_set(data_dict)

    loss_function = nn.MSELoss()

    for epoch in range(args.train_epochs):

        total_loss = 0

        loader = torch.utils.data.DataLoader(this_data_set, batch_size = 64, num_workers=3, shuffle=True)

        iterator = iter(loader)

        for iteration in tqdm(range(len(loader))):

            batch = next(iterator)

            for item in batch:
                batch[item] = batch[item].to('cuda') 

            optimizer.zero_grad()
            estimated_loss = model.forward(batch)

            loss = loss_function(estimated_loss, batch['mpjpe'])

            total_loss += loss.item()

            loss.backward()

            optimizer.step()

        print(f"epoch: {epoch}, loss: {total_loss}")
    
    return model

def test_mpjpe_model(model):
    # model = MPJPE_Model(num_inputs, num_joints).to('cuda')
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    data_dict = load_data("validation")

    initial_j3d = data_dict['estimated_j3d'].clone()
    estimated_j3d = data_dict['estimated_j3d'].clone()

    mse_loss = nn.MSELoss()

    for epoch in range(args.train_epochs):

        estimated_loss_total = 0
        pose_differences_total = 0

        data_dict['estimated_j3d'] = estimated_j3d

        this_data_set = data_set(data_dict, training=False)

        loader = torch.utils.data.DataLoader(this_data_set, batch_size = 64, num_workers=3, shuffle=False)
        iterator = iter(loader)

        for iteration in tqdm(range(len(loader))):

            batch = next(iterator)

            for item in batch:
                batch[item] = batch[item].to('cuda') 

            this_batch_estimated_j3d = batch['estimated_j3d']

            this_batch_estimated_j3d.requires_grad = True

            optimizer = optim.Adam([this_batch_estimated_j3d], lr=args.optimization_rate)

            optimizer.zero_grad()
            estimated_loss = model.forward(batch)
            estimated_loss = torch.mean(estimated_loss)
            

            # pose_differences = torch.MSE(torch.norm(initial_j3d[indeces].to('cuda') - this_batch_estimated_j3d))
            pose_differences = mse_loss(initial_j3d[batch['indices']].to('cuda'), this_batch_estimated_j3d)


            estimated_loss_total += estimated_loss.item()
            pose_differences_total += pose_differences.item()

            loss = estimated_loss
            # loss = torch.mean(torch.zeros_like(this_batch_estimated_j3d) - this_batch_estimated_j3d)
            # loss = torch.nn.MSELoss()(torch.zeros_like(this_batch_estimated_j3d), this_batch_estimated_j3d)
            loss = pose_differences*1e-1 + estimated_loss

            loss.backward()
            
            optimizer.step()

            estimated_j3d[batch['indices']] = this_batch_estimated_j3d.cpu().detach()

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

    files = glob.glob(f"../VIBE/data/3dpw/predicted_poses/{set}/*/vibe_output.pkl")

    if(set == "train"):

        J_regressor = torch.from_numpy(np.load('../VIBE/data/vibe_data/J_regressor_h36m.npy')).float().to('cuda')
            
        smpl = SMPL(
            '../VIBE/{}'.format(SMPL_MODEL_DIR),
            batch_size=64,
            create_transl=False
        ).to('cuda')


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
                gt_j3ds = find_gt_joints(pose.to('cuda'), shape.to('cuda'), smpl, J_regressor).cpu()

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
    

    mpjpe = torch.sqrt(((estimated_j3d - gt_j3d) ** 2).sum(dim=-1))

    print("mpjpe.shape")
    print(mpjpe.shape)

    evaluate(estimated_j3d, gt_j3d)

    return {'images':images, 'estimated_j3d':estimated_j3d, 'gt_j3d':gt_j3d, 'gt_j2d':gt_j2d, 'gt_cam':gt_cam, 'pred_cam':pred_cam, 'bboxes':bboxes, 'mpjpe':mpjpe}



# this is going to need a dataloader
# but dont modify the points
def draw_gradients(model):
    model.eval()

    for param in model.parameters():
        param.requires_grad = False


    # 'images':images, 'estimated_j3d':estimated_j3d, 'gt_j3d':gt_j3d, 'gt_j2d':gt_j2d, 'gt_cam':gt_cam, 'pred_cam':pred_cam, 'bboxes':bboxes, 'mpjpe':mpjpe
    data_dict = load_data("validation")
    data_dict['estimated_j3d'] = data_dict['gt_j3d']
    data_dict['pred_cam'] = data_dict['gt_cam']
    # change the data dict so it only gets the first image many times over
    this_data_set = data_set(data_dict, training=False)
    loader = torch.utils.data.DataLoader(this_data_set, batch_size = 1, num_workers=0, shuffle=True)
    iterator = iter(loader)
    batch = next(iterator)


    images = torch.cat([batch['image']]*121, dim=0)
    estimated_j3d = torch.cat([batch['estimated_j3d']]*121, dim=0)
    dims_before = torch.cat([batch['dims_before']]*121, dim=0)
    gt_cam = torch.cat([batch['cam']]*121, dim=0)
    bboxes = torch.cat([batch['bboxes']]*121, dim=0)

    for x in range(11):
        for y in range(11):
            estimated_j3d[x*11+y, :, 0] += (x/5-1)*.1
            estimated_j3d[x*11+y, :, 1] -= (y/5-1)*.1

    locations = []
    directions = []

    batch_size = 32
    for i in range(0, images.shape[0], batch_size):

        size = min(images.shape[0]-i, batch_size)

        batch = {   'image': images[i:i+size],
                    'dims_before': dims_before[i:i+size],
                    'estimated_j3d': estimated_j3d[i:i+size],
                    'cam': gt_cam[i:i+size],
                    'bboxes': bboxes[i:i+size]}

        for item in batch:
            batch[item] = batch[item].to('cuda') 

        initial_j3d = batch['estimated_j3d'].clone()

        batch['estimated_j3d'].requires_grad = True

        optimizer = optim.Adam([batch['estimated_j3d']], lr=1)

        # joints_3d[:, :, 0] += .1

        estimated_loss = model.forward(batch)

        estimated_loss.mean().backward()
        optimizer.step()

        joints2d_before = projection(initial_j3d, batch['cam'])
        joints2d = projection(batch['estimated_j3d'], batch['cam'])

        direction = joints2d-joints2d_before

        des_bboxes = batch['bboxes'].unsqueeze(1).expand(-1, joints2d_before.shape[1], -1)

        joints2d_before[:, :, 0] *= des_bboxes[:, :, 2]/2*1.1
        joints2d_before[:, :, 0] += des_bboxes[:, :, 0]
        joints2d_before[:, :, 1] *= des_bboxes[:, :, 3]/2*1.1
        joints2d_before[:, :, 1] += des_bboxes[:, :, 1]


        locations.append(joints2d_before)
        directions.append(direction)


    crop_scalar = 8
    crop_size = [64, 64]

    locations = torch.cat(locations, dim=0)
    normalized_locations = locations-locations[60]
    normalized_locations*=(64/120)
    normalized_locations += 32
    directions = torch.cat(directions, dim=0)


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
    plt.savefig(f"out_folder/bilinear_8/overall")
    plt.close()

    for j in range(locations.shape[1]):

        dx = locations[60, j, 0]/(dims_before[0, 1]/2)-1
        if(dims_before[0, 0]==1920):
            dx *= 1080/1920
        

        dy = locations[60, j, 1]/(dims_before[0, 0]/2)-1
        if(dims_before[0, 0]==1080):
            dy *= 1080/1920

        vec = torch.Tensor([[0, 1/crop_scalar, 1/crop_scalar, crop_scalar*dx, crop_scalar*dy]])

        transformation_mat = perturbation_helper.vec2mat_for_similarity(vec)

        linearized_sampler = sampling_helper.DifferentiableImageSampler('bilinear', 'zeros')

        linearized_transformed_image = linearized_sampler.warp_image(images[0], transformation_mat, out_shape=crop_size)

        plt.imshow(utils.torch_img_to_np_img(linearized_transformed_image)[0])
        ax = plt.gca()
        for k in range(normalized_locations.shape[0]):

            initial_x = normalized_locations[k, j, 0]
            initial_y = normalized_locations[k, j, 1]

            dir_x = directions[k, j, 0]
            dir_y = directions[k, j, 1]

            if(k == 60):
                color = 'b'
            else:
                color = 'r'

            circ = Arrow(initial_x.cpu().detach().numpy(), initial_y.cpu().detach().numpy(), 
                                    dir_x.cpu().detach().numpy(), dir_y.cpu().detach().numpy(),
                                    width=1, color = color)

            ax.add_patch(circ)
        plt.savefig(f"out_folder/bilinear_8/joint_{j}")
        plt.close()

    
    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    device = torch.device("cuda")


    parser.add_argument('--train_epochs', type=int, default=5)
    parser.add_argument('--opt_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--optimization_rate', type=float, default=1e-3)
    args = parser.parse_args()

    

    print("args")
    print(args)

    # model = train_mpjpe_model()
    # torch.save(model.state_dict(), "models/bilinear_model_8.pt")
    # exit()
    # model = MPJPE_Model().to('cuda')
    # model.load_state_dict(torch.load("models/linearized_model.pt"))
    model = MPJPE_Model().to('cuda')
    model.load_state_dict(torch.load("models/bilinear_model_8.pt"))

    draw_gradients(model)

    # test_mpjpe_model(model)

