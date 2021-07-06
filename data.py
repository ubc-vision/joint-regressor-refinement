import torch
from torch import nn, optim
from torch.utils.data import Dataset

# from spacepy import pycdf
import numpy as np
import h5py
import joblib
import glob

from tqdm import tqdm

import imageio
from utils import utils

from args import args

import pickle

import constants

# from pytorch3d.renderer import PerspectiveCameras

from warp import perturbation_helper, sampling_helper

import h5py


class data_set(Dataset):
    def __init__(self, set):

        if(set == "train"):
            location = "data/human3.6m/precomputed_train/"
        else:
            location = "data/human3.6m/precomputed_val/"

        self.bboxes = torch.load(
            f"{location}bboxes.pt", map_location="cpu")
        self.betas = torch.load(
            f"{location}betas.pt", map_location="cpu")
        self.estimated_translation = torch.load(
            f"{location}estimated_translation.pt", map_location="cpu")
        self.gt_j2d = torch.load(
            f"{location}gt_j2d.pt", map_location="cpu")
        self.gt_j3d = torch.load(
            f"{location}gt_j3d.pt", map_location="cpu")
        self.images = pickle.load(open(f"{location}images.pkl", 'rb'))
        self.intrinsics = torch.load(
            f"{location}intrinsics.pt", map_location="cpu")
        self.orient = torch.load(
            f"{location}orient.pt", map_location="cpu")
        self.pixel_annotations = pickle.load(
            open(f"{location}pixel_annotations.pkl", 'rb'))
        self.pose = torch.load(f"{location}pose.pt", map_location="cpu")

        # self.h5py = h5py.File('data/human3.6m/data.h5', 'r')

    def __getitem__(self, index):

        if(args.compute_canada):

            split_path = self.images[index].split("/")[-5:]

            with h5py.File('data/human3.6m/data.h5', 'r') as f:
                image = f.get(
                    f"{split_path[0]}/{split_path[1]}/{split_path[2]}/{split_path[3]}/{split_path[4]}")
                mask_rcnn = f.get(
                    f"{split_path[0]}/{split_path[1]}/maskSequence/{split_path[3]}/{split_path[4]}")

                image = torch.tensor(image).unsqueeze(0)

                mask_rcnn = torch.tensor(mask_rcnn).unsqueeze(0)/255.0

                _, min_x, min_y, scale, intrinsics = find_crop_mask(
                    image, self.bboxes[index].unsqueeze(0), self.intrinsics[index].unsqueeze(0))

        else:
            image = imageio.imread(f"{self.images[index]}")
            image = utils.np_img_to_torch_img(image)[:, :constants.IMG_RES, :constants.IMG_RES]
            image = image.float()/255.0

            mask_name = self.images[index].split("imageSequence")
            mask_name = f"{mask_name[0]}maskSequence{mask_name[1]}"
            mask_rcnn = torch.tensor(imageio.imread(
                f"{mask_name}"), dtype=torch.uint8)

            # TODO reimplement
            mask_rcnn = mask_rcnn.float().unsqueeze(0)/255.0

        
            image, min_x, min_y, scale, intrinsics = find_crop_mask(
                image, self.bboxes[index].unsqueeze(0), self.intrinsics[index].unsqueeze(0))


        # TODO reimplement
        valid = mask_rcnn[0, 0, 0] != 0

        mask_rcnn[:, :2, :2] = 0

        repositioned_j2d = self.gt_j2d[index].clone()
        repositioned_j2d[..., 0] -= min_x
        repositioned_j2d[..., 1] -= min_y
        repositioned_j2d /= scale
        repositioned_j2d /= 1000/224

        output_dict = {
            # "image_name": self.images[index],
            # "mask_name": mask_name,
            "bboxes": self.bboxes[index],
            "betas": self.betas[index],
            "cam": self.estimated_translation[index],
            "gt_j2d": repositioned_j2d,
            "gt_j3d": self.gt_j3d[index],
            # "valid": valid,
            "mask_rcnn": mask_rcnn,
            "image": image[0],
            "intrinsics": intrinsics[0],
            "orient": self.orient[index],
            "pixel_annotations": self.pixel_annotations[index],
            "pose": self.pose[index],
        }

        return output_dict

    def __len__(self):
        return len(self.images)


def find_crop(image, joints_2d, intrinsics):

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

    intrinsics = crop_intrinsics(
        intrinsics, 1000*scale, 1000*scale, average_y, average_x)
    intrinsics = resize_intrinsics(
        intrinsics, 1000*scale, 1000*scale, 224/(scale*1000))

    return image, min_x, min_y, scale, intrinsics


def find_crop_mask(image, mask, intrinsics):

    batch_size = mask.shape[0]
    min_x = mask[:, 1]
    max_x = mask[:, 3]
    min_y = mask[:, 0]
    max_y = mask[:, 2]

    min_x = (min_x-500)/500
    max_x = (max_x-500)/500
    min_y = (min_y-500)/500
    max_y = (max_y-500)/500

    average_x = (min_x+max_x)/2
    average_y = (min_y+max_y)/2

    scale_x = (max_x-min_x)
    scale_y = (max_y-min_y)

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

    intrinsics = crop_intrinsics(
        intrinsics, 1000*scale, 1000*scale, average_y, average_x)
    intrinsics = resize_intrinsics(
        intrinsics, 1000*scale, 1000*scale, 224/(scale*1000))

    return image, min_x, min_y, scale, intrinsics


def load_data(set):

    if(set == "train"):
        actors = ["S1", "S5", "S6", "S7", "S8"]
    elif(set == "validation"):
        actors = ["S9", "S11"]

    scenes = []

    for i in range(len(actors)):
        scenes.extend(
            sorted(glob.glob(f"/scratch/iamerich/human36m/processed/{actors[i]}/*")))

    images = []
    gt_j3d = []
    gt_j2d = []
    intrinsics = []

    print("loading data")

    for scene in tqdm(scenes):
        f = h5py.File(f"{scene}/annot.h5", 'r')

        camera = torch.tensor(f['camera'])
        frame = torch.tensor(f['frame'])

        this_scene_images = np.array(
            [f"{scene}/imageSequence/{camera[i]}/img_{frame[i]:06d}.jpg" for i in range(camera.shape[0])])

        images.extend(this_scene_images)

        # this_scene_gt_j2d = torch.tensor(f['pose/2d'])
        this_scene_gt_j2d = torch.tensor(f['pose/2d'])[:, constants.GT_2_J17]
        # this_scene_gt_j3d = torch.tensor(f['pose/3d'])
        this_scene_gt_j3d = torch.tensor(f['pose/3d'])[:, constants.GT_2_J17]

        batch_size = camera.shape[0]

        intrinsic = torch.zeros(batch_size, 3, 3)
        for i in range(batch_size):
            cam_num = camera[i]
            this_cam_intrinsics = f['intrinsics'][f'{cam_num}']
            intrinsic[i, 0, 0] = this_cam_intrinsics[0]
            intrinsic[i, 0, 2] = this_cam_intrinsics[1]
            intrinsic[i, 1, 1] = this_cam_intrinsics[2]
            intrinsic[i, 1, 2] = this_cam_intrinsics[3]
            intrinsic[i, 2, 2] = 1

        gt_j3d.append(this_scene_gt_j3d)
        gt_j2d.append(this_scene_gt_j2d)
        intrinsics.append(intrinsic)

        # break

        # save the name of the file
        # this will be the location that the file is saved

    # if(set == "train"):
        # load folder for saved results.

    gt_j3d = torch.cat(gt_j3d)
    gt_j2d = torch.cat(gt_j2d)
    intrinsics = torch.cat(intrinsics)

    return_dict = {
        'images':       images,
        'gt_j3d':       gt_j3d,
        'gt_j2d':       gt_j2d,
        'intrinsics':   intrinsics,
    }

    for key in return_dict:
        try:
            print(f"{key} shape: {return_dict[key].shape}")
        except:
            continue

    return return_dict


def load_precompted(epoch=None):
    if(epoch is not None):
        file_location = f"/scratch/iamerich/human36m/processed/saved_output_train/{epoch}"
    else:
        file_location = f"/scratch/iamerich/human36m/processed/saved_output_val"

    estimated_translation = torch.load(
        f"{file_location}/estimated_translation.pt", map_location='cpu')
    j2d_with_noise = torch.load(
        f"{file_location}/j2d_with_noise.pt", map_location='cpu')
    j3d_with_noise = torch.load(
        f"{file_location}/j3d_with_noise.pt", map_location='cpu')
    mpjpe_2d = torch.load(f"{file_location}/mpjpe_2d.pt", map_location='cpu')
    mpjpe_3d = torch.load(f"{file_location}/mpjpe_3d.pt", map_location='cpu')
    orient = torch.load(f"{file_location}/orient.pt", map_location='cpu')
    pose = torch.load(f"{file_location}/pose.pt", map_location='cpu')
    pred_betas = torch.load(
        f"{file_location}/pred_betas.pt", map_location='cpu')

    return{
        "estimated_translation": estimated_translation,
        "j2d_with_noise": j2d_with_noise,
        "j3d_with_noise": j3d_with_noise,
        "mpjpe_2d": mpjpe_2d,
        "mpjpe_3d": mpjpe_3d,
        "orient": orient,
        "pose": pose,
        "pred_betas": pred_betas,
    }


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
