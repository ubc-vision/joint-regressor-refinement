import torch
from torch import nn
import torchvision.models as models

from warp import perturbation_helper, sampling_helper 

from args import args

from data import project_points

import matplotlib.pyplot as plt

import constants



class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=num_channels,
                                 eps=1e-5, affine=True)
    
    def forward(self, x):
        x = self.norm(x)
        return x


class Crop_Model(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self, testing=False):
        super(Crop_Model, self).__init__()

        self.num_inputs = 2048*4*4
        self.num_joints = len(constants.GT_2_J17)

        self.testing = testing

        self.resnet = getattr(models, "resnet50")(norm_layer=MyGroupNorm)
        # self.resnet = models.resnet34(pretrained=True)

        self.resnet = nn.Sequential(*list(self.resnet.children())[:3], *list(self.resnet.children())[4:-2])

        self.linearized_sampler = sampling_helper.DifferentiableImageSampler('linearized', 'zeros')
        self.crop_scalar = args.crop_scalar
        self.crop_size = [64, 64]
        self.linear_operations = []

        for i in range(self.num_joints):

            linear_operation = nn.Sequential(
                nn.Linear(self.num_inputs, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            ).to(args.device)
            
            self.linear_operations.append(linear_operation)

        self.linear_operations = nn.ModuleList(self.linear_operations)
        

        # for param in self.linear_operations.parameters():
        #     param.requires_grad = True


    def forward(self, input_dict):  

        image = input_dict['image']
        joints3d = input_dict['joints3d']

        intrinsics = input_dict['intrinsics']

        batch_size  = image.shape[0]

        self.input_images = []

        # for every joint, get the crop centred at the reprojected 2d location

        if(self.testing==False):
            joints2d = project_points(joints3d, intrinsics)
        else:
            joints2d = joints3d[..., :2]

        zeros = torch.zeros(image.shape[0]).to(args.device)
        ones = torch.ones(image.shape[0]).to(args.device)

        # print("image.shape")
        # print(image.shape)

        error_estimates = []


        for i in range(self.num_joints):
            
            dx = joints2d[:, i, 0]/500-1

            dy = joints2d[:, i, 1]/500-1

            vec = torch.stack([zeros, ones/self.crop_scalar, ones/self.crop_scalar, self.crop_scalar*dx, self.crop_scalar*dy], dim=1)

            transformation_mat = perturbation_helper.vec2mat_for_similarity(vec)


            linearized_transformed_image = self.linearized_sampler.warp_image(image, transformation_mat, out_shape=self.crop_size).contiguous()

            output = self.resnet(linearized_transformed_image)

            output = output.reshape(-1, self.num_inputs)

            error_estimate = self.linear_operations[i](output)


            error_estimate = torch.norm(error_estimate, dim=-1)

            # print(error_estimate.shape)

            error_estimates.append(error_estimate)

        error_estimates = torch.stack(error_estimates, dim=-1)

        return error_estimates
