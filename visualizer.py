import torch
from torch import optim

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Arrow

import wandb

# from spacepy import pycdf
import numpy as np


from utils import utils
from warp import perturbation_helper, sampling_helper 


from data import load_data, data_set, projection

from args import args

def draw_gradients(model, set, name):
    model.eval()

    # 'images':images, 'estimated_j3d':estimated_j3d, 'gt_j3d':gt_j3d, 'gt_j2d':gt_j2d, 'gt_cam':gt_cam, 'pred_cam':pred_cam, 'bboxes':bboxes, 'mpjpe':mpjpe
    data_dict = load_data(set)
    data_dict['estimated_j3d'] = data_dict['gt_j3d']
    data_dict['pred_cam'] = data_dict['gt_cam']
    # change the data dict so it only gets the first image many times over
    this_data_set = data_set(data_dict, training=False)
    loader = torch.utils.data.DataLoader(this_data_set, batch_size = 1, num_workers=0, shuffle=False)
    iterator = iter(loader)
    batch = next(iterator)


    images = torch.cat([batch['image']]*121, dim=0)
    estimated_j3d = torch.cat([batch['estimated_j3d']]*121, dim=0)
    gt_j3d = torch.cat([batch['gt_j3d']]*121, dim=0)
    dims_before = torch.cat([batch['dims_before']]*121, dim=0)
    gt_cam = torch.cat([batch['cam']]*121, dim=0)
    bboxes = torch.cat([batch['bboxes']]*121, dim=0)
    mpjpe = torch.cat([batch['mpjpe']]*121, dim=0)
    gt_gradient = torch.cat([batch['gt_gradient']]*121, dim=0)
    training = torch.cat([batch['training']]*121, dim=0)

    for x in range(11):
        for y in range(11):

            if(x ==5 and y== 5):
                estimated_j3d[x*11+y] = batch['gt_j3d']

                mpjpe[x*11+y] = 0

                continue

            this_index = x*11+y

            # vector pointing to ground truth from estimated
            ground_truth_vector = gt_j3d[this_index]-estimated_j3d[this_index]

            # chose a point along the path between estimated and ground truth
            estimated_j3d[this_index] = torch.rand(1)*ground_truth_vector+estimated_j3d[this_index]

            # add noise to the point
            estimated_j3d[this_index] = estimated_j3d[this_index] + (torch.rand(gt_j3d[this_index].shape)*.1)-.05

            projected_2d_estimated_joints = projection(estimated_j3d[this_index][None], gt_cam[this_index][None])
            projected_2d_gt_joints = projection(gt_j3d[this_index][None], gt_cam[this_index][None])

            this_mpjpe = torch.sqrt(((projected_2d_estimated_joints - projected_2d_gt_joints) ** 2).sum(dim=-1))

            mpjpe[this_index] = this_mpjpe[0]

            gt_gradient[this_index] = 2*(estimated_j3d[this_index]-batch['gt_j3d'])

    locations = []
    grads = []
    gt_grads = []
    estimated_error = []

    batch_size = args.optimization_batch_size
    for i in range(0, images.shape[0], batch_size):

        size = min(images.shape[0]-i, batch_size)

        batch = {   'image': images[i:i+size],
                    'dims_before': dims_before[i:i+size],
                    'estimated_j3d': estimated_j3d[i:i+size],
                    'cam': gt_cam[i:i+size],
                    'bboxes': bboxes[i:i+size],
                    'training': training[i:i+size],
                    'gt_gradient': gt_gradient[i:i+size]}

        for item in batch:
            batch[item] = batch[item].to(args.device)

        initial_j3d = batch['estimated_j3d'].clone()

        batch['estimated_j3d'].requires_grad = True

        optimizer = optim.Adam([batch['estimated_j3d']], lr=args.optimization_rate)

        optimizer.zero_grad()

        estimated_loss = model.forward(batch)

        estimated_loss.mean().backward()

        joints2d = projection(initial_j3d, batch['cam'])

        direction = batch['estimated_j3d'].grad[:, :, :2]
        gt_grad = batch['gt_gradient'][:, :, :2]

        des_bboxes = batch['bboxes'].unsqueeze(1).expand(-1, joints2d.shape[1], -1)

        joints2d[:, :, 0] *= des_bboxes[:, :, 2]/2*1.1
        joints2d[:, :, 0] += des_bboxes[:, :, 0]
        joints2d[:, :, 1] *= des_bboxes[:, :, 3]/2*1.1
        joints2d[:, :, 1] += des_bboxes[:, :, 1]

        locations.append(joints2d.cpu().detach())
        grads.append(direction.cpu().detach())
        gt_grads.append(gt_grad.cpu().detach())

        estimated_error.append(estimated_loss.cpu().detach())


    crop_scalar = args.crop_scalar
    crop_size = [64, 64]

    locations = torch.cat(locations, dim=0)
    normalized_locations = locations-locations[60]
    normalized_locations*=(64/(1920/crop_scalar))
    normalized_locations += 32
    grads = torch.cat(grads, dim=0)
    gt_grads = torch.cat(gt_grads, dim=0)

    gt_dir = locations-locations[60]

    this_error = torch.max(mpjpe)

    mpjpe /= this_error

    estimated_error = torch.cat(estimated_error, dim=0)

    estimated_error /= this_error

    # estimated_error -= torch.min(estimated_error, dim=0).values.unsqueeze(0).expand(estimated_error.shape[0], -1)
    # estimated_error /= torch.max(estimated_error, dim=0).values.unsqueeze(0).expand(estimated_error.shape[0], -1)
    # estimated_error /= torch.max(estimated_error)



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

    wandb.log({f"{name}_overall": wandb.Image(plt)}, commit=False)
    plt.close()

    crops = []
    estimated_errors = []
    gt_errors = []
    gradients = []
    gt_gradients = []

    # initial_loc = projection(normalized_locations, gt_cam)

    # gradient = projection(grads, gt_cam)

    # gt_dir = initial_loc-projection(normalized_locations[60][None], gt_cam[60][None])

    for j in range(locations.shape[1]):

        dx = locations[60, j, 0]/(dims_before[0, 1]/2)-1
        if(dims_before[0, 0]==1920):
            dx *= 1080/1920
        

        dy = locations[60, j, 1]/(dims_before[0, 0]/2)-1
        if(dims_before[0, 0]==1080):
            dy *= 1080/1920

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

            error = mpjpe[k, j]

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

            if(k == 60):
                color = 'b'
            else:
                dir_dot = torch.dot(gt_dir[k, j]/torch.norm(gt_dir[k, j]), grads[k, j]/torch.norm(grads[k, j])).numpy()
                dir_scalar = (dir_dot+1)/2
                color = (dir_scalar, 1-dir_scalar, 0)

                color = np.clip(color, 0, 1)

            circ = Arrow(normalized_locations[k, j, 0].numpy(), normalized_locations[k, j, 1].numpy(), 
                                    grads[k, j, 0].numpy()*10, grads[k, j, 1].numpy()*10,
                                    width=1, color = color)

            ax.add_patch(circ)

        gradients.append(wandb.Image(plt))
        plt.close()

        # draw gradients
        plt.imshow(utils.torch_img_to_np_img(bilinear_transformed_image)[0])
        ax = plt.gca()
        for k in range(normalized_locations.shape[0]):

            if(k == 60):
                color = 'b'
            else:
                dir_dot = torch.dot(gt_dir[k, j]/torch.norm(gt_dir[k, j]), gt_grads[k, j]/torch.norm(gt_grads[k, j])).numpy()
                dir_scalar = (dir_dot+1)/2
                color = (dir_scalar, 1-dir_scalar, 0)

                color = np.clip(color, 0, 1)

            circ = Arrow(normalized_locations[k, j, 0].numpy(), normalized_locations[k, j, 1].numpy(), 
                                    gt_grads[k, j, 0].numpy()*10, gt_grads[k, j, 1].numpy()*10,
                                    width=1, color = color)

            ax.add_patch(circ)

        gt_gradients.append(wandb.Image(plt))
        plt.close()


    wandb.log({f"{name}_crops": crops}, commit=False)
    wandb.log({f"{name}_estimated_errors": estimated_errors}, commit=False)
    wandb.log({f"{name}_gt_errors": gt_errors}, commit=False)
    wandb.log({f"{name}_gradients": gradients}, commit=False)
    wandb.log({f"{name}_gt_gradients": gt_gradients})
    # wandb.log({f"{name}_gt_gradients": gt_gradients})

    model.train()

    
    return 0