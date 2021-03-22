from test import find_crop, convert_back_to_original_dimensions
import torch
from torchvision import transforms
from torch import nn, optim
from data_3dpw import load_data, data_set, find_joints, find_vertices

from args import args

import numpy as np

from time import sleep

from utils import utils


import sys
from SPIN.models import hmr, SMPL
import SPIN.config as config
from SPIN.utils.geometry import rot6d_to_rotmat

from tqdm import tqdm

def estimate_translation_np(S, joints_2d, focal_length=5000, img_size=224):
   """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
   Input:
      S: (17, 3) 3D joint locations
      joints: (17, 3) 2D joint locations and confidence
   Returns:
      (3,) camera translation vector
   """

   num_joints = S.shape[0]
   # focal length
   f = np.array([focal_length,focal_length])
   # optical center
   center = np.array([img_size/2., img_size/2.])

   # transformations
   Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
   XY = np.reshape(S[:,0:2],-1)
   O = np.tile(center,num_joints)
   F = np.tile(f,num_joints)

   # least squares
   Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
   c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

   # square matrix
   A = np.dot(Q.T,Q)
   b = np.dot(Q.T,c)

   # solution
   trans = np.linalg.solve(A, b)

   return trans

def estimate_translation(S, joints_2d, focal_length=5000., img_size=224.):
   """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
   Input:
      S: (B, 49, 3) 3D joint locations
      joints: (B, 49, 3) 2D joint locations and confidence
   Returns:
      (B, 3) camera translation vectors
   """

   trans = np.zeros((S.shape[0], 3), dtype=np.float32)
   # Find the translation for each example in the batch
   for i in range(S.shape[0]):
      S_i = S[i]
      joints_i = joints_2d[i]
      trans[i] = estimate_translation_np(S_i, joints_i, focal_length=focal_length, img_size=img_size)
   return torch.from_numpy(trans).to(args.device)

def perspective_projection(points, rotation, translation, intrinsics):
   """
   This function computes the perspective projection of a set of points.
   Input:
      points (bs, N, 3): 3D points
      rotation (bs, 3, 3): Camera rotation
      translation (bs, 3): Camera translation
      intrinsics (bs, 3, 3) Camera intrinsics
   """
   batch_size = points.shape[0]

   # Transform points
   points = torch.einsum('bij,bkj->bki', rotation, points)
   points = points + translation.unsqueeze(1)

   # Apply perspective distortion
   projected_points = points / points[:,:,-1].unsqueeze(-1)

   # Apply camera intrinsics
   projected_points = torch.einsum('bij,bkj->bki', intrinsics, projected_points)

   return projected_points[:, :, :-1]


# opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

# rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
# projected_joints = perspective_projection(model_joints, rotation, camera_t,
#                                           focal_length, camera_center)



def visualize(image, vertices, name, point_size = .1):

   blt = utils.torch_img_to_np_img(image)

   import matplotlib.cbook as cbook
   from matplotlib.patches import Circle
   import matplotlib.pyplot as plt

   for i in tqdm(range(vertices.shape[0])):

      fig, ax = plt.subplots()

      for j in range(vertices.shape[1]):

         circ = Circle((vertices[i, j, 0],vertices[i, j, 1]),point_size, color = 'r', alpha=0.7)

         ax.add_patch(circ)
         
      im = ax.imshow(blt[i])
      plt.savefig(f"image_{i:03d}_{name}.png", dpi=300)
      plt.close()


def find_error_to_gt(pred_j3ds, target_j3ds):

   pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
   target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0


   pred_j3ds -= pred_pelvis
   target_j3ds -= target_pelvis

   # error = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).mean()*1000
   error = nn.MSELoss()(pred_j3ds, target_j3ds)

   return error

def find_joints(smpl, pred_betas, pred_rotmat, J_regressor):
   pred_vertices = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False).vertices

   J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(args.device)
   pred_joints = torch.matmul(J_regressor_batch, pred_vertices)

   return pred_joints



def create_gt():
   J_regressor = torch.from_numpy(np.load('data/vibe_data/J_regressor_h36m.npy')).float().to(args.device)
        
    smpl = SMPL(
        '{}'.format(SMPL_MODEL_DIR),
        batch_size=64,
        create_transl=False
    ).to(args.device)

   data_dict = load_data("validation")

   normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

   this_data_set = data_set(data_dict)

   loader = torch.utils.data.DataLoader(this_data_set, batch_size = 32, num_workers=0, shuffle=True)
   iterator = iter(loader)

   for iteration in tqdm(range(len(iterator))):
      batch = next(iterator)

      for item in batch:
         batch[item] = batch[item].to(args.device).float()

      pose_rotmat = batch['gt_pose'].view(-1, 24, 3, 3)

      pred_vertices = smpl(global_orient=pose_rotmat[:, 0].unsqueeze(1), body_pose=pose_rotmat[:, 1:], betas=batch['gt_shape'], pose2rot=False).vertices
      joints2d_estimated = projection(pred_vertices, batch['gt_cam'])
      visualize(batch['image'], pred_vertices_2d, "gt")


      # pred_vertices = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False).vertices
      # joints2d_estimated = projection(pred_vertices, batch['gt_cam'])
      # visualize(batch['image'], pred_vertices_2d, "estimated")



if __name__ == "__main__":
   create_gt()