import torch
from torch import nn, optim
from torch.utils.data import Dataset

from smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14

# from spacepy import pycdf
import numpy as np
import joblib
import glob

import imageio
from utils import utils

from args import args


def load_image(images, index, training):

    image = imageio.imread(f"{images[index]}")/255.0
    image = utils.np_img_to_torch_img(image).float()


    dims_before = torch.Tensor([image.shape[1], image.shape[2]])

    if(dims_before[0]==1080):
        top_bottom_pad = nn.ZeroPad2d((0, 0, 420, 420))
        image = top_bottom_pad(image)
    else:
        sides_pad = nn.ZeroPad2d((420, 420, 0, 0))
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
        self.gt_pose = input_dict['gt_pose']
        self.gt_shape = input_dict['gt_shape']
        self.estimated_pose = input_dict['estimated_pose']
        self.estimated_shape = input_dict['estimated_shape']
        self.training = training

    
    def __getitem__(self, index):

        # print(f"self.images[index] {self.images[index]}")

        image, dims_before = load_image(self.images, index, self.training)

        gt_pose = self.gt_pose[index]
        gt_shape = self.gt_shape[index]
        estimated_pose = self.estimated_pose[index]
        estimated_shape = self.estimated_shape[index]

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

            # work backward to find smpl parameters for joints

        
        else:

            estimated_j3d = self.estimated_j3d[index]

            cam = self.pred_cam[index]

            mpjpe = self.mpjpe[index]

            gt_gradient = 2*(estimated_j3d-self.gt_j3d[index])



        output_dict = {
            'indices': index, 
            'image': image, 
            'dims_before': dims_before, 
            'estimated_j3d': estimated_j3d, 
            'gt_j3d':self.gt_j3d[index], 
            'gt_j2d': self.gt_j2d[index], 
            'gt_gradient': gt_gradient, 
            'cam': cam, 
            'bboxes': self.bboxes[index], 
            'mpjpe': mpjpe, 
            'training': self.training,
            'gt_pose': gt_pose,
            'gt_shape': gt_shape,
            'estimated_pose': estimated_pose,
            'estimated_shape': estimated_shape,
        }

        # for item in output_dict:
        #     output_dict[item] = output_dict[item].to(args.device) 

        return output_dict
    
    # def __len__(self, index):
    #     return len(self.images)

    def __len__(self):
        return len(self.images)


def optimize_camera_parameters(joints3d, joints2d, bboxes):

    joints3d = joints3d.to(args.device)
    joints2d = joints2d.to(args.device)
    bboxes = bboxes.to(args.device)

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

        zeros  = torch.zeros(joints2d_estimated[:, :, 0].shape).to(args.device)


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


def find_joints(pose, shape, smpl, J_regressor):

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

        J_regressor = torch.from_numpy(np.load('data/vibe_data/J_regressor_h36m.npy')).float().to(args.device)
            
        smpl = SMPL(
            '{}'.format(SMPL_MODEL_DIR),
            batch_size=64,
            create_transl=False
        ).to(args.device)


    images = []
    estimated_j3d = []
    pred_cam = []
    bboxes = []
    gt_j3d = []
    gt_j2d = []
    gt_pose = []
    gt_shape = []
    estimated_pose = []
    estimated_shape = []

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

            these_gt_pose = torch.Tensor(data[person]['gt_pose'])
            these_gt_shape = torch.Tensor(data[person]['gt_shape'])
            these_estimated_pose = torch.Tensor(data[person]['pose'][gt_indices])
            these_estimated_shape = torch.Tensor(data[person]['betas'][gt_indices])

            gt_pose.append(these_gt_pose)
            gt_shape.append(these_gt_shape)
            estimated_pose.append(these_estimated_pose)
            estimated_shape.append(these_estimated_shape)


            if(set == "train"):
                gt_j3ds = find_joints(these_gt_pose.to(args.device), these_gt_shape.to(args.device), smpl, J_regressor).cpu()

                gt_j3ds = move_gt_pelvis(gt_j3ds, j3ds)

                gt_j3d.append(gt_j3ds)
            else:
                gt_j3ds = torch.Tensor(data[person]['gt_joints3d'])

                gt_j3ds = move_gt_pelvis(gt_j3ds, j3ds)

                gt_j3d.append(gt_j3ds)

    images = np.concatenate(images)
    estimated_j3d = torch.cat(estimated_j3d)
    gt_j3d = torch.cat(gt_j3d)
    gt_j2d = torch.cat(gt_j2d)
    pred_cam = torch.cat(pred_cam)
    bboxes = torch.cat(bboxes)

    gt_pose = torch.cat(gt_pose)
    gt_shape = torch.cat(gt_shape)
    estimated_pose = torch.cat(estimated_pose)
    estimated_shape = torch.cat(estimated_shape)

    gt_cam = optimize_camera_parameters(gt_j3d, gt_j2d, bboxes).detach().cpu()

    # project with cam and get 2d error
    projected_2d_estimated_joints = projection(estimated_j3d, gt_cam)
    projected_2d_gt_joints = projection(gt_j3d, gt_cam)
    

    mpjpe = torch.sqrt(((projected_2d_estimated_joints - projected_2d_gt_joints) ** 2).sum(dim=-1))

    return_dict = {
        'images':images, 
        'estimated_j3d':estimated_j3d, 
        'gt_j3d':gt_j3d, 
        'gt_j2d':gt_j2d, 
        'gt_cam':gt_cam, 
        'pred_cam':pred_cam, 
        'bboxes':bboxes, 
        'mpjpe':mpjpe, 
        'gt_pose': gt_pose, 
        'gt_shape': gt_shape, 
        'estimated_pose': estimated_pose, 
        'estimated_shape': estimated_shape
    }

    for key in return_dict:
        try:
            print(f"{key} shape: {return_dict[key].shape}")
        except:
            continue

    return return_dict


