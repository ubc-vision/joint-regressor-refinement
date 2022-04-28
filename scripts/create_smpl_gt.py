# from test import convert_back_to_original_dimensions
import torch
from torchvision import transforms
from torch import nn, optim
from scripts.data import load_data, data_set

from scripts.args import args

import numpy as np

from time import sleep

from scripts import utils

import torch.nn.functional as F

from scripts import constants

import sys
from SPIN.models import hmr, SMPL
import SPIN.config as config
from SPIN.utils.geometry import rot6d_to_rotmat

from tqdm import tqdm

import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, AlphaCompositor, PointsRenderer

import matplotlib.pyplot as plt

from warp import perturbation_helper, sampling_helper


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

# find crop and pass in 224x224 image


def crop_intrinsics(intrinsics, height, width, crop_ci, crop_cj):
    """ Convert to new camera intrinsics for crop of image from original camera.
    Parameters
    ----------
    height : int
        height of crop window
    width : int
        width of crop window
    crop_ci : int
        row of crop window center
    crop_cj : int
        col of crop window center
    Returns
    -------
    :obj:`CameraIntrinsics`
        camera intrinsics for cropped window
    """
    x0 = intrinsics[:, 0, 2]
    y0 = intrinsics[:, 1, 2]
    cx = x0 + (width-1)/2 - crop_cj
    cy = y0 + (height-1)/2 - crop_ci

    cropped_intrinsics = intrinsics.clone()
    cropped_intrinsics[:, 0, 2] = cx
    cropped_intrinsics[:, 1, 2] = cy
    return cropped_intrinsics


def resize_intrinsics(intrinsics, height, width, scale):
    """ Convert to new camera intrinsics with parameters for resized image.

    Parameters
    ----------
    scale : float
        the amount to rescale the intrinsics

    Returns
    -------
    :obj:`CameraIntrinsics`
        camera intrinsics for resized image        
    """
    x0 = intrinsics[:, 0, 2]
    y0 = intrinsics[:, 1, 2]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]

    center_x = (width-1) / 2
    center_y = (height-1) / 2
    orig_cx_diff = x0 - center_x
    orig_cy_diff = y0 - center_y
    height = scale*height
    width = scale*width
    scaled_center_x = (width-1) / 2
    scaled_center_y = (height-1) / 2
    fx = scale * fx
    fy = scale * fy
    cx = scaled_center_x + scale * orig_cx_diff
    cy = scaled_center_y + scale * orig_cy_diff

    scaled_intrinsics = intrinsics.clone()
    scaled_intrinsics[:, 0, 2] = cx
    scaled_intrinsics[:, 1, 2] = cy
    scaled_intrinsics[:, 0, 0] = fx
    scaled_intrinsics[:, 1, 1] = fy
    return scaled_intrinsics

    # scaled_intrinsics = CameraIntrinsics(frame=self.frame,
    #                                      fx=fx, fy=fy, skew=skew, cx=cx, cy=cy,
    #                                      height=height, width=width)
    # return scaled_intrinsics


def find_crop(image, joints_2d, intrinsics=None):

    batch_size = joints_2d.shape[0]
    min_x = torch.min(joints_2d[..., 0], dim=1)[0]
    max_x = torch.max(joints_2d[..., 0], dim=1)[0]
    min_y = torch.min(joints_2d[..., 1], dim=1)[0]
    max_y = torch.max(joints_2d[..., 1], dim=1)[0]

    min_x = (min_x-500)/500
    max_x = (max_x-500)/500
    min_y = (min_y-500)/500
    max_y = (max_y-500)/500

    average_x = (min_x+max_x)/2
    average_y = (min_y+max_y)/2

    scale_x = (max_x-min_x)*1.2
    scale_y = (max_y-min_y)*1.2

    scale = torch.where(scale_x > scale_y, scale_x, scale_y)

    # print(scale[:3])
    # print(average_x[:3])

    scale /= 2

    min_x = (average_x-scale)*500+500
    min_y = (average_y-scale)*500+500

    zeros = torch.zeros(batch_size).to(image.device)
    ones = torch.ones(batch_size).to(image.device)

    bilinear_sampler = sampling_helper.DifferentiableImageSampler(
        'bilinear', 'zeros')

    vec = torch.stack([zeros, scale, scale, average_x /
                       scale, average_y/scale], dim=1)

    average_x = (average_x)*500+500
    average_y = (average_y)*500+500

    transformation_mat = perturbation_helper.vec2mat_for_similarity(vec)

    image = bilinear_sampler.warp_image(
        image, transformation_mat, out_shape=[224, 224]).contiguous()

    if(intrinsics is not None):
        intrinsics = crop_intrinsics(
            intrinsics, 1000*scale, 1000*scale, average_y, average_x)
        intrinsics = resize_intrinsics(
            intrinsics, 1000*scale, 1000*scale, 224/(scale*1000))

    return image, min_x, min_y, scale, intrinsics


# def estimate_translation_np(S, joints_2d, focal_length=5000, img_size=224):
#     """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
#     Input:
#        S: (17, 3) 3D joint locations
#        joints: (17, 3) 2D joint locations and confidence
#     Returns:
#        (3,) camera translation vector
#     """

#     num_joints = S.shape[0]
#     # focal length
#     f = np.array([focal_length, focal_length])
#     # optical center
#     center = np.array([img_size/2., img_size/2.])

#     # transformations
#     Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
#     XY = np.reshape(S[:, 0:2], -1)
#     O = np.tile(center, num_joints)
#     F = np.tile(f, num_joints)

#     # least squares
#     Q = np.array([F*np.tile(np.array([1, 0]), num_joints), F *
#                   np.tile(np.array([0, 1]), num_joints), O-np.reshape(joints_2d, -1)]).T
#     c = (np.reshape(joints_2d, -1)-O)*Z - F*XY

#     # square matrix
#     A = np.dot(Q.T, Q)
#     b = np.dot(Q.T, c)

#     # solution
#     trans = np.linalg.solve(A, b)

#     return trans


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
        trans[i] = estimate_translation_np(
            S_i, joints_i, focal_length=focal_length, img_size=img_size)
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
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum(
        'bij,bkj->bki', intrinsics, projected_points)

    return projected_points[:, :, :-1]


# opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

# rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
# projected_joints = perspective_projection(model_joints, rotation, camera_t,
#                                           focal_length, camera_center)

def normalize_quaternion(quaternion: torch.Tensor,
                         eps: float = 1e-12) -> torch.Tensor:
    r"""Normalizes a quaternion.
    The quaternion should be in (x, y, z, w) format.
    Args:
       quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
       eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.
    Return:
       torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.
    Example:
       >>> quaternion = torch.tensor([1., 0., 1., 0.])
       >>> normalize_quaternion(quaternion)
       tensor([0.7071, 0.0000, 0.7071, 0.0000])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    return F.normalize(quaternion, p=2, dim=-1, eps=eps)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    r"""Converts a quaternion to a rotation matrix.
    The quaternion should be in (x, y, z, w) format.
    Args:
       quaternion (torch.Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape :math:`(*, 4)`.
    Return:
       torch.Tensor: the rotation matrix of shape :math:`(*, 3, 3)`.
    Example[]:
       >>> quaternion = torch.tensor([0., 0., 1., 0.])
       >>> quaternion_to_rotation_matrix(quaternion)
       tensor([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))

    # normalize the input quaternion
    quaternion_norm: torch.Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.tensor(1.)

    matrix: torch.Tensor = torch.stack([
        one - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, one - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, one - (txx + tyy)
    ], dim=-1).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(3).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 3, 3).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx3x3


def rotation_matrix_to_quaternion(
        rotation_matrix: torch.Tensor,
        eps: float = 1e-8) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.
    The quaternion vector has components in (x, y, z, w) format.
    Args:
       rotation_matrix (torch.Tensor): the rotation matrix to convert.
       eps (float): small value to avoid zero division. Default: 1e-8.
    Return:
       torch.Tensor: the rotation in quaternion.
    Shape:
       - Input: :math:`(*, 3, 3)`
       - Output: :math:`(*, 4)`
    Example:
       >>> input = torch.rand(4, 3, 3)  # Nx3x3
       >>> output = rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a (*, 3, 3) tensor. Got {}".format(
                rotation_matrix.shape))

    def safe_zero_division(numerator: torch.Tensor,
                           denominator: torch.Tensor) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny  # type: ignore
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: torch.Tensor = rotation_matrix.view(
        *rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(
        rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(trace + 1.0) * 2.  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_1():
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_2():
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_3():
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat([qx, qy, qz, qw], dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where(
        (m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(
        trace > 0., trace_positive_cond(), where_1)
    return quaternion


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.
    The quaternion should be in (x, y, z, w) format.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
       quaternion (torch.Tensor): tensor with quaternions.
    Return:
       torch.Tensor: tensor with angle axis of rotation.
    Shape:
       - Input: :math:`(*, 4)` where `*` means, any number of dimensions
       - Output: :math:`(*, 3)`
    Example:
       >>> quaternion = torch.rand(2, 4)  # Nx4
       >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(
                quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def visualize(image, joints_2d, name, point_size=.1):

    blt = utils.torch_img_to_np_img(image)

    import matplotlib.cbook as cbook
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt

    for i in tqdm(range(joints_2d.shape[0])):

        fig, ax = plt.subplots()

        for j in range(joints_2d.shape[1]):

            circ = Circle(
                (joints_2d[i, j, 0], joints_2d[i, j, 1]), 1, color='b')

            ax.add_patch(circ)

        im = ax.imshow(blt[i])
        plt.savefig(f"output/image_{i:03d}_{name}.png", dpi=300)
        plt.close()


def find_error_to_gt(pred_j3ds, target_j3ds):

    pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
    target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

    pred_j3ds -= pred_pelvis
    target_j3ds -= target_pelvis

    # error = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).mean()*1000
    error = nn.MSELoss()(pred_j3ds, target_j3ds)

    return error


def find_joints(smpl, pred_betas, orient, pose, J_regressor):

    orient_matrix = quaternion_to_rotation_matrix(
        orient.reshape(-1, 4)).reshape(-1, 1, 3, 3)
    pose_matrix = quaternion_to_rotation_matrix(
        pose.reshape(-1, 4)).reshape(-1, 23, 3, 3)

    pred_vertices = smpl(betas=pred_betas, body_pose=pose_matrix,
                         global_orient=orient_matrix, pose2rot=False).vertices

    J_regressor_batch = J_regressor[None, :].expand(
        pred_vertices.shape[0], -1, -1).to(args.device)
    pred_joints = torch.matmul(J_regressor_batch, pred_vertices)

    return pred_joints


def render_point_cloud(image, point_cloud, cameras, name):

    raster_settings = PointsRasterizationSettings(
        image_size=1000,
        radius=0.003,
        points_per_pixel=10
    )

    feat = torch.ones(point_cloud.shape[0],
                      point_cloud.shape[1], 4).to(args.device)

    point_cloud[:, :, 1] *= -1
    point_cloud[:, :, 0] *= -1
    point_cloud *= 2

    this_point_cloud = Pointclouds(points=point_cloud, features=feat)

    rasterizer = PointsRasterizer(
        cameras=cameras, raster_settings=raster_settings)

    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    # images = renderer(this_point_cloud)
    images = renderer(this_point_cloud)

    for i in range(images.shape[0]):

        plt.figure(figsize=(10, 10))
        plt.imshow(images[i, ..., :3].cpu().detach().numpy())
        plt.imshow(utils.torch_img_to_np_img(image[i]), alpha=0.5)
        plt.savefig(f"output/image_{i:02d}_{name}")

    plt.close()


def project_pytorch(points, K):
    transform = Transform3d(device=self.device)
    transform._matrix = K.transpose(1, 2).contiguous()

    unprojection_transform = to_ndc_transform.inverse()
    xy_inv_depth = torch.cat(
        (points[..., :2], 1.0 / points[..., 2:3]), dim=-1  # type: ignore
    )
    return unprojection_transform.transform_points(xy_inv_depth)


def find_translation_and_pose(image, gt_j3d, gt_j2d, intrinsics, spin_model, smpl, J_regressor):

    batch_size = image.shape[0]

    with torch.no_grad():
        pred_rotmat_initial, pred_betas_initial, pred_camera = spin_model(
            image)

        pred_rotmat_initial = rotation_matrix_to_quaternion(
            pred_rotmat_initial.view(-1, 3, 3)).view(-1, 24, 4)

        pose_initial = pred_rotmat_initial[:, 1:]
        orient_initial = pred_rotmat_initial[:, 0].unsqueeze(1)

        pose = pose_initial.clone()
        orient = orient_initial.clone()

        pred_joints = find_joints(
            smpl, pred_betas_initial, orient, pose, J_regressor)

        estimated_translation = estimate_translation(pred_joints.cpu().detach(
        ).numpy(), gt_j2d.cpu().numpy(), focal_length=700, img_size=224)

        # estimated_translation[:, :2] = estimated_translation[:, :2]*-2

    estimated_translation.requires_grad = True
    orient.requires_grad = True

    optimizer = optim.Adam([orient, estimated_translation], lr=1e-1)

    focal_length = torch.stack(
        [intrinsics[:, 0, 0]/224, intrinsics[:, 1, 1]/224], dim=1)
    principal_point = torch.stack(
        [intrinsics[:, 0, 2]/-112+1, intrinsics[:, 1, 2]/-112+1], dim=1)

    image_size = torch.tensor([224, 224]).unsqueeze(
        0).expand(batch_size, 2).to(args.device)

    for i in range(100):

        optimizer.zero_grad()

        pred_joints = find_joints(
            smpl, pred_betas_initial, orient, pose, J_regressor)

        pred_joints[:, :, 1] *= -1
        pred_joints[:, :, 0] *= -1
        pred_joints *= 2

        cameras = PerspectiveCameras(device=args.device, T=estimated_translation,
                                     focal_length=focal_length, principal_point=principal_point)
        pred_joints_2d = cameras.transform_points_screen(
            pred_joints, image_size)

        error_2d = nn.MSELoss()(gt_j2d, pred_joints_2d[..., :2])*1e-5

        loss = error_2d

        # if(i % 10 == 0):
        # print(f"{i} translation loss 2d {error_2d.item()},\ttotal {loss.item()}")

        # print("mean loss")
        # print(torch.mean(torch.abs(pred_joints_2d[..., :2]-gt_j2d)))
        # print(torch.max(torch.abs(pred_joints_2d[..., :2]-gt_j2d)))

        loss.backward()
        optimizer.step()

    pose.requires_grad = True

    optimizer = optim.Adam(
        [pose], lr=1e-2)

    goal_pose = gt_j3d.float()+torch.randn(gt_j3d.shape).to(args.device)*30
    goal_pose = goal_pose/1000
    # goal_pose = gt_j3d.clone().float()/1000

    for i in range(10):

        optimizer.zero_grad()

        pred_joints = find_joints(
            smpl, pred_betas_initial, orient, pose, J_regressor)

        pred_joints_clone = pred_joints.clone()

        error_3d = find_error_to_gt(pred_joints, goal_pose)

        pred_joints_clone[:, :, 1] *= -1
        pred_joints_clone[:, :, 0] *= -1
        pred_joints_clone *= 2

        cameras = PerspectiveCameras(device=args.device, T=estimated_translation,
                                     focal_length=focal_length, principal_point=principal_point)
        pred_joints_2d = cameras.transform_points_screen(
            pred_joints_clone, image_size)

        error_2d = nn.MSELoss()(
            gt_j2d, pred_joints_2d[..., :2])*1e-5

        # loss = error_2d+error_3d
        loss = error_3d

        # if(i % 10 == 0):
        # print(
        #     f"{i} rotmat loss 2d {error_2d.item()},\tloss 3d {error_3d.item()},\ttotal {loss.item()}")

        loss.backward()

        pose.grad.data[:, constants.HAND_FEET_ROT_INDECES] = 0

        optimizer.step()

    orient = quaternion_to_rotation_matrix(
        orient.reshape(-1, 4)).reshape(-1, 1, 3, 3)
    pose = quaternion_to_rotation_matrix(
        pose.reshape(-1, 4)).reshape(-1, 23, 3, 3)

    return orient.detach(), pose.detach(), pred_betas_initial.detach(), pose_initial.detach(), orient_initial.detach(), estimated_translation.detach()


# def create_gt():
#     J_regressor = torch.from_numpy(
#         np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

#     smpl = SMPL(
#         '{}'.format("SPIN/data/smpl"),
#         batch_size=1,
#     ).to(args.device)

#     joint_mapper_pred = constants.H36M_TO_J14
#     joint_mapper_gt = constants.J24_TO_J14

#     spin_model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
#     checkpoint = torch.load(
#         "SPIN/data/model_checkpoint.pt", map_location=args.device)
#     spin_model.load_state_dict(checkpoint['model'], strict=False)
#     spin_model.eval()

#     data_dict = load_data("validation")

#     normalize = transforms.Normalize(
#         (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

#     this_data_set = data_set(data_dict)

#     loader = torch.utils.data.DataLoader(
#         this_data_set, batch_size=10, num_workers=0, shuffle=False)
#     iterator = iter(loader)

#     for iteration in tqdm(range(len(iterator))):
#         batch = next(iterator)

#         batch['gt_j3d'] /= 1000

#         batch_size = batch['image'].shape[0]

#         for item in batch:
#             batch[item] = batch[item].to(args.device).float()

#         # visualize(batch['image'], batch['gt_j2d'], "before", point_size=10)
#         # exit()

#         initial_image = batch['image'].clone()

#         batch['image'] = normalize(batch['image'])

#         image, min_x, min_y, scale = find_crop(batch)

#         with torch.no_grad():
#             pred_rotmat, pred_betas, pred_camera = spin_model(image)

#         pred_rotmat = rotation_matrix_to_quaternion(
#             pred_rotmat.view(-1, 3, 3)).view(-1, 24, 4)

#         pose = pred_rotmat[:, 1:]
#         orient = pred_rotmat[:, 0].unsqueeze(1)

#         pred_rotmat = pred_rotmat.detach()
#         pred_betas = pred_betas.detach()
#         pred_camera = pred_camera.detach()

#         batch['image'] = initial_image

#         pred_joints = find_joints(smpl, pred_betas, orient, pose, J_regressor)

#         print("pred_joints.shape")
#         print(pred_joints.shape)

#         estimated_translation = estimate_translation(pred_joints.cpu().detach(
#         ).numpy(), batch['gt_j2d'].cpu().numpy(), focal_length=1000, img_size=1000)

#         estimated_translation = estimated_translation.detach()

#         estimated_translation[:, :2] = estimated_translation[:, :2]*-2

#         rotation = torch.eye(3, device=args.device).unsqueeze(
#             0).expand(batch_size, -1, -1)
#         pred_joints_2d = perspective_projection(
#             pred_joints, rotation, estimated_translation, batch['intrinsics'])

#         pred_betas.requires_grad = False
#         estimated_translation.requires_grad = True
#         pose.requires_grad = False
#         orient.requires_grad = True

#         optimizer_position = optim.Adam(
#             [orient, estimated_translation], lr=1e-2)

#         focal_length = torch.stack(
#             [batch['intrinsics'][:, 0, 0]/1000, batch['intrinsics'][:, 1, 1]/1000], dim=1)
#         principal_point = torch.stack(
#             [batch['intrinsics'][:, 0, 2]/-500+1, batch['intrinsics'][:, 1, 2]/-500+1], dim=1)

#         image_size = torch.tensor([1000, 1000]).unsqueeze(
#             0).expand(batch_size, 2).to(args.device)

#         for i in range(100):

#             optimizer_position.zero_grad()

#             pred_joints = find_joints(
#                 smpl, pred_betas, orient, pose, J_regressor)

#             pred_joints[:, :, 1] *= -1
#             pred_joints[:, :, 0] *= -1
#             pred_joints *= 2

#             # project joints
#             # pred_joints_2d = perspective_projection(pred_joints, rotation, estimated_translation, batch['intrinsics'])

#             cameras = PerspectiveCameras(device=args.device, T=estimated_translation,
#                                          focal_length=focal_length, principal_point=principal_point)
#             pred_joints_2d = cameras.transform_points_screen(
#                 pred_joints, image_size)

#             # print(pred_joints_2d)
#             # exit()

#             error_2d = nn.MSELoss()(
#                 batch['gt_j2d'], pred_joints_2d[..., :2])*1e-5

#             # error_3d = find_error_to_gt(pred_joints, batch['gt_j3d'])

#             loss = error_2d

#             if(i % 10 == 0):
#                 print(
#                     f"{i} translation loss 2d {error_2d.item()},\ttotal {loss.item()}")

#             loss.backward()
#             optimizer_position.step()

#         # parameters = list(spin_model.parameters())
#         # parameters.extend(estimated_translation)

#         optimizer_orientation = optim.Adam(
#             [pose, orient, estimated_translation], lr=1e-2)

#         # spin_model.requires_grad = True
#         # pred_rotmat.requires_grad = True
#         # pred_betas.requires_grad = True
#         pose.requires_grad = True

#         this_orient = quaternion_to_rotation_matrix(
#             orient.reshape(-1, 4)).reshape(-1, 1, 3, 3)
#         this_pose = quaternion_to_rotation_matrix(
#             pose.reshape(-1, 4)).reshape(-1, 23, 3, 3)
#         pred_vertices = smpl(betas=pred_betas, body_pose=this_pose,
#                              global_orient=this_orient, pose2rot=False).vertices

#         cameras = PerspectiveCameras(device=args.device, T=estimated_translation,
#                                      focal_length=focal_length, principal_point=principal_point)

#         # render_point_cloud(batch['image'], pred_vertices, cameras, "after_alignment")

#         for i in range(100):

#             # pred_rotmat, pred_betas, pred_camera = spin_model(image)

#             optimizer_orientation.zero_grad()

#             pred_joints = find_joints(
#                 smpl, pred_betas, orient, pose, J_regressor)

#             pred_joints_projected = pred_joints.clone()

#             pred_joints_projected[:, :, 1] *= -1
#             pred_joints_projected[:, :, 0] *= -1
#             pred_joints_projected *= 2

#             error_3d = find_error_to_gt(pred_joints, batch['gt_j3d'])

#             cameras = PerspectiveCameras(device=args.device, T=estimated_translation,
#                                          focal_length=focal_length, principal_point=principal_point)
#             pred_joints_2d = cameras.transform_points_screen(
#                 pred_joints_projected, image_size)

#             error_2d = nn.MSELoss()(
#                 batch['gt_j2d'], pred_joints_2d[..., :2])*1e-5

#             loss = error_2d+error_3d

#             # if(i%10==0):
#             print(
#                 f"{i} rotmat loss 2d {error_2d.item()},\tloss 3d {error_3d.item()},\ttotal {loss.item()}")

#             loss.backward()

#             pose.grad.data[:, constants.HAND_FEET_ROT_INDECES] = 0

#             optimizer_orientation.step()

#         orient = quaternion_to_rotation_matrix(
#             orient.reshape(-1, 4)).reshape(-1, 1, 3, 3)
#         pose = quaternion_to_rotation_matrix(
#             pose.reshape(-1, 4)).reshape(-1, 23, 3, 3)
#         pred_vertices = smpl(betas=pred_betas, body_pose=pose,
#                              global_orient=orient, pose2rot=False).vertices

#         cameras = PerspectiveCameras(device=args.device, T=estimated_translation,
#                                      focal_length=focal_length, principal_point=principal_point)
#         render_point_cloud(batch['image'], pred_vertices,
#                            cameras, "after_pose_optimization")

#         # print("batch['intrinsics']")
#         # print(batch['intrinsics'])
#         # visualize(batch['image'], pred_vertices_2d, batch['gt_j2d'], "visualized")

#         # render_point_cloud(batch['image'], pred_vertices, estimated_translation, batch['intrinsics'])

#         exit()


if __name__ == "__main__":

    create_gt()
# %%
