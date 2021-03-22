import torch
from torch import optim

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Arrow

import wandb

# from spacepy import pycdf
import numpy as np

from torchvision import transforms


from utils import utils
from warp import perturbation_helper, sampling_helper 


from data import load_data, data_set, project_points

from args import args

def draw_gradients(model, set, name):
    model.eval()

    # 'images':images, 'estimated_j3d':estimated_j3d, 'gt_j3d':gt_j3d, 'gt_j2d':gt_j2d, 'gt_cam':gt_cam, 'pred_cam':pred_cam, 'bboxes':bboxes, 'mpjpe':mpjpe
    data_dict = load_data(set)

    # change the data dict so it only gets the first image many times over
    this_data_set = data_set(data_dict)
    loader = torch.utils.data.DataLoader(this_data_set, batch_size = 1, num_workers=0, shuffle=True)
    iterator = iter(loader)
    batch = next(iterator)
    
    images = torch.cat([batch['image']]*121, dim=0)
    gt_j3d = torch.cat([batch['gt_j3d']]*121, dim=0)
    gt_j2d = torch.cat([batch['gt_j2d']]*121, dim=0)
    intrinsics = torch.cat([batch['intrinsics']]*121, dim=0)
    joints3d = torch.cat([batch['joints3d']]*121, dim=0)
    joints2d = torch.cat([batch['joints2d']]*121, dim=0)
    mpjpe_2d = torch.cat([batch['mpjpe_2d']]*121, dim=0)
    mpjpe_3d = torch.cat([batch['mpjpe_3d']]*121, dim=0)

    for x in range(11):
        for y in range(11):

            if(x ==5 and y== 5):
                joints3d[x*11+y] = batch['gt_j3d']

                mpjpe_2d[x*11+y] = 0

                continue

            this_index = x*11+y

            # add noise to the point
            joints3d[this_index] = gt_j3d[this_index] + (torch.rand(gt_j3d[this_index].shape)*300)-150


            joints2d[this_index] = project_points(joints3d[this_index].unsqueeze(0), intrinsics[this_index].unsqueeze(0))[0]


            mpjpe_2d[this_index] = torch.sqrt(((joints2d[this_index] - gt_j2d[this_index]) ** 2).sum(dim=-1))

    locations = []
    grads = []
    estimated_error = []

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    batch_size = args.optimization_batch_size
    for i in range(0, images.shape[0], batch_size):

        size = min(images.shape[0]-i, batch_size)

        batch = {   'image': images[i:i+size],
                    'joints3d': joints3d[i:i+size],
                    'intrinsics': intrinsics[i:i+size]}

        for item in batch:
            batch[item] = batch[item].to(args.device)

        batch['image'] = normalize(batch['image'])

        batch['joints3d'].requires_grad = True

        optimizer = optim.Adam([batch['joints3d']], lr=args.optimization_rate)

        optimizer.zero_grad()

        estimated_loss = model.forward(batch)

        estimated_loss.mean().backward()


        joints2d = project_points(batch['joints3d'], batch['intrinsics'])

        direction = batch['joints3d'].grad[:, :, :2]

        locations.append(joints2d.cpu().detach())
        grads.append(direction.cpu().detach())

        estimated_error.append(estimated_loss.cpu().detach())


    crop_scalar = args.crop_scalar
    crop_size = [64, 64]

    locations = torch.cat(locations, dim=0)
    normalized_locations = locations-locations[60]
    normalized_locations*=(64/(1000/crop_scalar))
    normalized_locations += 32
    grads = torch.cat(grads, dim=0)

    gt_dir = locations-locations[60]

    this_error = torch.max(mpjpe_2d)

    mpjpe_2d /= this_error

    estimated_error = torch.cat(estimated_error, dim=0)

    estimated_error /= this_error

    # estimated_error -= torch.min(estimated_error, dim=0).values.unsqueeze(0).expand(estimated_error.shape[0], -1)
    # estimated_error /= torch.max(estimated_error, dim=0).values.unsqueeze(0).expand(estimated_error.shape[0], -1)
    # estimated_error /= torch.max(estimated_error)


    blt = utils.torch_img_to_np_img(images)

    plt.imshow(blt[0])
    ax = plt.gca()
    for j in range(locations.shape[1]):

        initial_x = locations[60, j, 0]
        initial_y = locations[60, j, 1]
    
        circ = Circle((initial_x,initial_y),10, color = 'r')

        ax.add_patch(circ)

    wandb.log({f"{name}_overall": wandb.Image(plt)}, commit=False)
    plt.close()

    crops = []
    estimated_errors = []
    gt_errors = []
    gradients = []

    # initial_loc = projection(normalized_locations, gt_cam)

    # gradient = projection(grads, gt_cam)

    # gt_dir = initial_loc-projection(normalized_locations[60][None], gt_cam[60][None])

    for j in range(locations.shape[1]):

        dx = locations[60, j, 0]/500-1
        

        dy = locations[60, j, 1]/500-1

        vec = torch.Tensor([[0, 1/crop_scalar, 1/crop_scalar, crop_scalar*dx, crop_scalar*dy]])

        transformation_mat = perturbation_helper.vec2mat_for_similarity(vec)

        bilinear_sampler = sampling_helper.DifferentiableImageSampler('bilinear', 'zeros')

        bilinear_transformed_image = bilinear_sampler.warp_image(images[0], transformation_mat, out_shape=crop_size)


        plt.imshow(utils.torch_img_to_np_img(bilinear_transformed_image)[0])
        crops.append(wandb.Image(plt))
        plt.close()

        # draw error curve
        plt.imshow(utils.torch_img_to_np_img(bilinear_transformed_image)[0])
        ax = plt.gca()
        for k in range(normalized_locations.shape[0]):


            error = estimated_error[k, j]
        
            # ranging from green for small error and red for high error
            color = (error, 1-error, 0)
            color = np.clip(color, 0, 1)

            circ = Circle((normalized_locations[k, j, 0],normalized_locations[k, j, 1]),1, color = color)

            ax.add_patch(circ)

        estimated_errors.append(wandb.Image(plt))
        
        plt.close()

        plt.imshow(utils.torch_img_to_np_img(bilinear_transformed_image)[0])
        ax = plt.gca()
        for k in range(normalized_locations.shape[0]):

            error = mpjpe_2d[k, j]

            # ranging from green for small error and red for high error
            color = (error, 1-error, 0)
            color = np.clip(color, 0, 1)

            circ = Circle((normalized_locations[k, j, 0],normalized_locations[k, j, 1]),1, color = color)

            ax.add_patch(circ)

        gt_errors.append(wandb.Image(plt))
        plt.close()
        
        # draw gradients
        plt.imshow(utils.torch_img_to_np_img(bilinear_transformed_image)[0])
        ax = plt.gca()
        for k in range(normalized_locations.shape[0]):

            # color = 'b'

            if(k == 60):
                color = 'b'
            else:
                dir_dot = torch.dot(gt_dir[k, j]/torch.norm(gt_dir[k, j]), grads[k, j].float()/torch.norm(grads[k, j].float())).numpy()
                dir_scalar = (dir_dot+1)/2
                color = (dir_scalar, 1-dir_scalar, 0)

                color = np.clip(color, 0, 1)

            circ = Arrow(normalized_locations[k, j, 0].numpy(), normalized_locations[k, j, 1].numpy(), 
                                    grads[k, j, 0].numpy()*10, grads[k, j, 1].numpy()*10,
                                    width=1, color = color)

            ax.add_patch(circ)

        gradients.append(wandb.Image(plt))
        plt.close()



    wandb.log({f"{name}_crops": crops}, commit=False)
    wandb.log({f"{name}_estimated_errors": estimated_errors}, commit=False)
    wandb.log({f"{name}_gt_errors": gt_errors}, commit=False)
    wandb.log({f"{name}_gradients": gradients}, commit=False)
    # wandb.log({f"{name}_gt_gradients": gt_gradients})

    model.train()

    
    return 0