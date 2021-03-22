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

import constants

import create_smpl_gt

from pytorch3d.renderer import PerspectiveCameras


class data_set(Dataset):
    def __init__(self, input_dict, precomputed_dict=None):

        self.includes_precomputed = False

        self.images = input_dict['images']
        self.gt_j3d = input_dict['gt_j3d']
        self.gt_j2d = input_dict['gt_j2d']
        self.intrinsics = input_dict['intrinsics']
        if(precomputed_dict is not None):
            self.includes_precomputed = True
            self.joints3d = precomputed_dict["j3d_with_noise"]
            self.joints2d = precomputed_dict["j2d_with_noise"]
            self.mpjpe_2d = precomputed_dict["mpjpe_2d"]
            self.mpjpe_3d = precomputed_dict["mpjpe_3d"]
            self.orient = precomputed_dict["orient"]
            self.pose = precomputed_dict["pose"]
            self.pred_betas = precomputed_dict["pred_betas"]
            self.estimated_translation = precomputed_dict["estimated_translation"]

    def __getitem__(self, index):

        # print(f"self.images[index] {self.images[index]}")

        image = imageio.imread(f"{self.images[index]}")/255.0
        image = utils.np_img_to_torch_img(image).float(
        )[:, :constants.IMG_RES, :constants.IMG_RES]

        if(self.includes_precomputed):

            output_dict = {
                "image": image,
                "gt_j3d": self.gt_j3d[index],
                "gt_j2d": self.gt_j2d[index],
                "intrinsics": self.intrinsics[index],
                "joints3d": self.joints3d[index],
                "joints2d": self.joints2d[index],
                "mpjpe_2d": self.mpjpe_2d[index],
                "mpjpe_3d": self.mpjpe_3d[index],
                "orient": self.orient[index],
                "pose": self.pose[index],
                "pred_betas": self.pred_betas[index],
                "estimated_translation": self.estimated_translation[index],
            }
        else:
            output_dict = {
                "image": image,
                "gt_j3d": self.gt_j3d[index],
                "gt_j2d": self.gt_j2d[index],
                "intrinsics": self.intrinsics[index],
            }

        return output_dict

    def __len__(self):
        return len(self.images)


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
