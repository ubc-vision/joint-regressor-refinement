import numpy as np
import torch
import torch.nn as nn

from scripts.eval_utils import batch_compute_similarity_transform_torch

from scripts.args import args

import random

import torch.nn.functional as F


# def embed_breakpoint(terminate=True):
#     embedding = ('import IPython\n'
#                  'import matplotlib.pyplot as plt\n'
#                  'IPython.embed()\n'
#                  )
#     if terminate:
#         embedding += (
#             'assert 0, \'force termination\'\n'
#         )

#     return embedding


# def torch_img_to_np_img(torch_img):
#     '''convert a torch image to matplotlib-able numpy image
#     torch use Channels x Height x Width
#     numpy use Height x Width x Channels
#     Arguments:
#         torch_img {[type]} -- [description]
#     '''
#     assert isinstance(
#         torch_img, torch.Tensor), 'cannot process data type: {0}'.format(type(torch_img))
#     if len(torch_img.shape) == 4 and (torch_img.shape[1] == 3 or torch_img.shape[1] == 1):
#         return np.transpose(torch_img.detach().cpu().numpy(), (0, 2, 3, 1))
#     if len(torch_img.shape) == 3 and (torch_img.shape[0] == 3 or torch_img.shape[0] == 1):
#         return np.transpose(torch_img.detach().cpu().numpy(), (1, 2, 0))
#     elif len(torch_img.shape) == 2:
#         return torch_img.detach().cpu().numpy()
#     else:
#         raise ValueError('cannot process this image')


def np_img_to_torch_img(np_img):
    """convert a numpy image to torch image
    numpy use Height x Width x Channels
    torch use Channels x Height x Width
    Arguments:
        np_img {[type]} -- [description]
    """
    assert isinstance(
        np_img, np.ndarray), 'cannot process data type: {0}'.format(type(np_img))
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return torch.from_numpy(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return torch.from_numpy(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return torch.from_numpy(np_img)
    else:
        raise ValueError(
            'cannot process this image with shape: {0}'.format(np_img.shape))


# def unit_vector(vector):
#     """ Returns the unit vector of the vector.  """
#     return vector / np.linalg.norm(vector)


# def angle_between(v1, v2):
#     """ Returns the angle in radians between vectors 'v1' and 'v2'::
#             >>> angle_between((1, 0, 0), (0, 1, 0))
#             1.5707963267948966
#             >>> angle_between((1, 0, 0), (1, 0, 0))
#             0.0
#             >>> angle_between((1, 0, 0), (-1, 0, 0))
#             3.141592653589793
#     """
#     v1_u = unit_vector(v1)
#     v2_u = unit_vector(v2)
#     return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def find_joints(smpl, shape, orient, pose, J_regressor, mask=None, return_verts=False):

    if(mask is not None):
        J_regressor = J_regressor*mask

    J_regressor_batch = nn.ReLU()(J_regressor)
    J_regressor_batch = J_regressor_batch / torch.sum(J_regressor_batch, dim=1).unsqueeze(
        1).expand(J_regressor_batch.shape)

    pred_vertices = smpl(global_orient=orient, body_pose=pose,
                         betas=shape, pose2rot=False).vertices
    J_regressor_batch = J_regressor_batch[None, :].expand(
        pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
    pred_joints = torch.matmul(J_regressor_batch, pred_vertices)

    if(return_verts):
        return pred_joints, pred_vertices

    return pred_joints


def move_pelvis(j3ds):
    # move the hip location of gt to estimated
    pelvis = j3ds[:, [0], :].clone()

    j3ds_clone = j3ds.clone()

    j3ds_clone -= pelvis

    return j3ds_clone


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

    return mpjpe, pa_mpjpe


# def render_batch(img_renderer, batch, name, j2d=[], vertices=None):
#     rendered_img = img_renderer(batch)

#     drawing = rendered_img[:, 3:].expand(batch['image'].shape)

#     # drawing = (rendered_img[:, 3:].expand(
#     #     batch['image'].shape)*.5+batch['image']*.5)

#     drawing = batch['image']

#     # colors = ["g", "b"]
#     colors = ["r", "g", "b"]

#     import matplotlib.pyplot as plt
#     from matplotlib.patches import Circle
#     for i in range(drawing.shape[0]):

#         plt.imshow(torch_img_to_np_img(drawing)[i])

#         for index, this_j2d in enumerate(j2d):
#             ax = plt.gca()

#             for j in range(this_j2d.shape[1]):

#                 circ = Circle(
#                     (this_j2d[i, j, 0], this_j2d[i, j, 1]), 1, color=colors[index])

#                 ax.add_patch(circ)

#         plt.savefig(
#             f"output/{i:03d}_{name}.png", dpi=300)
#         plt.close()


def find_j_reg_mask(j_reg):
    ones = torch.ones(j_reg.shape).to(args.device)
    zeros = torch.ones(j_reg.shape).to(args.device)

    mask = torch.where(j_reg == 0, zeros, ones)
    return mask


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
