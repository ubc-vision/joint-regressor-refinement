import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms

from warp import perturbation_helper, sampling_helper 

from args import args

from data import projection


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
    def __init__(self):
        super(Crop_Model, self).__init__()

        self.num_inputs = 512*4*4
        self.num_joints = 14

        self.resnet = getattr(models, "resnet18")(norm_layer=MyGroupNorm)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:3], *list(self.resnet.children())[4:-2])

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
            ).to(args.device)
            
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

        zeros = torch.zeros(image.shape[0]).to(args.device)
        ones = torch.ones(image.shape[0]).to(args.device)

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
