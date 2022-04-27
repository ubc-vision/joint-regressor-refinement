from args import args
import wandb
import torch
from torch import nn, optim

from tqdm import tqdm

# from render_model import Render_Model
# from pose_estimator import Pose_Estimator
# from pose_refiner import Pose_Refiner
from discriminator import Discriminator, Shape_Discriminator
from renderer import Renderer, return_2d_joints
from mesh_renderer import Mesh_Renderer
# from pose_refiner_transformer import Pose_Refiner_Transformer

from pytorch3d.renderer import PerspectiveCameras

from data import data_set


# from visualizer import draw_gradients

import torchvision
from torchvision import transforms

# from pose_refiner import render_batch

from SPIN.models import hmr, SMPL
import SPIN.config as config

import numpy as np

import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, AlphaCompositor, PointsRenderer

from SPIN.utils.geometry import rot6d_to_rotmat

from utils import utils

import os
import imageio

import copy

from mult_layer_jreg import DL_JReg


def eval(render, mask_rcnn):

    render = torch.where(render > 0.5, torch.ones_like(render), torch.zeros_like(render))

    mask_rcnn = torch.where(mask_rcnn > 0.5, torch.ones_like(mask_rcnn), torch.zeros_like(mask_rcnn))

    intersection = torch.where(mask_rcnn+render == 2, torch.ones_like(mask_rcnn), torch.zeros_like(mask_rcnn))
    union = torch.where(mask_rcnn+render > 0, torch.ones_like(mask_rcnn), torch.zeros_like(mask_rcnn))

    return intersection.sum().item()/union.sum().item()





def viz(render, mask_rcnn, image, gt_joints_2d, spin_joints_2d, joints_2d, name):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    batch_size = render.shape[0]

    # render = 1-render

    render = torch.where(render > 0.5, torch.ones_like(render), torch.zeros_like(render))

    render = render.expand(batch_size, 3, 224, 224)

    # mask_rcnn = 1-mask_rcnn
    mask_rcnn = torch.where(mask_rcnn > 0.8, torch.ones_like(mask_rcnn), torch.zeros_like(mask_rcnn))

    mask_rcnn = mask_rcnn.expand(batch_size, 3, 224, 224)

    # print("truth.shape")
    # print(truth.shape)
    # print(truth)
    # exit()

    # intersection = torch.where(mask_rcnn+render == 2, torch.ones_like(mask_rcnn), torch.zeros_like(mask_rcnn))
    intersection = torch.where(mask_rcnn+render == 1, torch.ones_like(mask_rcnn), torch.zeros_like(mask_rcnn))

    # print("render.shape")
    # print(render.shape)
    # exit()

    render = utils.torch_img_to_np_img(render)
    mask_rcnn = utils.torch_img_to_np_img(mask_rcnn)
    image = utils.torch_img_to_np_img(image)
    intersection = utils.torch_img_to_np_img(intersection)

    gt_joints_2d = gt_joints_2d.cpu().detach().numpy()
    spin_joints_2d = spin_joints_2d.cpu().detach().numpy()
    joints_2d = joints_2d.cpu().detach().numpy()

    


    for i in range(render.shape[0]):
        # plt.imshow(render[i]+image[i])
        plt.imshow(intersection[i])

        # # plt.scatter(gt_joints_2d[i, :, 0], gt_joints_2d[i, :, 1], s=10, c='r')
        plt.scatter(spin_joints_2d[i, :, 0], spin_joints_2d[i, :, 1], s=10, c='g')
        # plt.scatter(joints_2d[i, :, 0], joints_2d[i, :, 1], s=10, c='b')

        ax = plt.gca()

        plt.savefig(
            f"output/{i:03d}_render_{name}.png", dpi=300)
        plt.close()

        plt.imshow(mask_rcnn[i])

        ax = plt.gca()

        plt.savefig(
            f"output/{i:03d}_silhouette_{name}.png", dpi=300)
        plt.close()

        # plt.imshow(image[i])

        # ax = plt.gca()

        # plt.savefig(
        #     f"output/{i:03d}_image_{name}.png", dpi=300)
        # plt.close()

def render_mesh(smpl, silhouette_renderer, betas, orient, pose, batch):
    pred_vertices = smpl(global_orient=orient, body_pose=pose,
                    betas=betas, pose2rot=False).vertices
    pred_vertices[:, :, 1] *= -1
    pred_vertices[:, :, 0] *= -1
    pred_vertices *= 2
    img = silhouette_renderer(batch, pred_vertices)[:, 3].unsqueeze(1)

    return img


def optimize_pose_refiner():

    spin_model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
    checkpoint = torch.load(
        "SPIN/data/model_checkpoint.pt", map_location=args.device)
    spin_model.load_state_dict(checkpoint['model'], strict=False)
    spin_model.eval()

    smpl = SMPL(
        '{}'.format("SPIN/data/smpl"),
        batch_size=1,
    ).to(args.device)

    maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True).to(args.device)
    maskrcnn.eval()

    # new_J_regressor = torch.load(
    #     "models/retrained_J_Regressor.pt").to(args.device)

    # J_reg = torch.from_numpy(
    #     np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)
    # J_regressor = DL_JReg(J_reg)
    # checkpoint = torch.load(
    #     "models/retrained_J_Regressor.pt", map_location=args.device)
    # J_regressor.load_state_dict(checkpoint)
    # J_regressor.eval()
    # J_regressor_retrained = DL_JReg(J_reg.clone())

    SPIN_J_regressor = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)
    J_regressor = torch.load(
        'models/reretrained_J_Regressor.pt').to(args.device).detach()
    J_regressor_retrained = torch.load(
        'models/retrained_J_Regressor.pt').to(args.device).detach()
    # J_regressor_retrained = torch.load(
    #     "models/best_pose_refiner/retrained_J_Regressor.pt", map_location=args.device)
    J_regressor_retrained.requires_grad = True
    

    

    # raster_settings = PointsRasterizationSettings(
    #     image_size=224,
    #     radius=0.04,
    #     points_per_pixel=10
    # )
    # raster_settings_img = PointsRasterizationSettings(
    #     image_size=224,
    #     radius=0.005,
    #     points_per_pixel=10
    # )

    # silhouette_renderer = Renderer(subset=True, image_size=224)
    # img_renderer = Renderer(subset=False)
    silhouette_renderer = Mesh_Renderer(image_size=224)
    # viz_renderer = Mesh_Renderer(image_size=256)
    # silhouette_renderer = nn.DataParallel(silhouette_renderer)

    pose_discriminator = Discriminator().to(args.device)
    checkpoint = torch.load(
        "models/pose_discriminator.pt", map_location=args.device)
    pose_discriminator.load_state_dict(checkpoint)
    pose_discriminator.train()

    disc_optimizer = optim.Adam(
        pose_discriminator.parameters(), lr=args.opt_disc_learning_rate)
    checkpoint = torch.load(
        "models/disc_optimizer.pt", map_location=args.device)
    disc_optimizer.load_state_dict(checkpoint)

    shape_discriminator = Shape_Discriminator().to(args.device)
    checkpoint = torch.load(
        "models/shape_discriminator.pt", map_location=args.device)
    shape_discriminator.load_state_dict(checkpoint)
    shape_discriminator.train()
     
    shape_disc_optimizer = optim.Adam(
        shape_discriminator.parameters(), lr=args.opt_disc_learning_rate)
    checkpoint = torch.load(
        "models/shape_disc_optimizer.pt", map_location=args.device)
    shape_disc_optimizer.load_state_dict(checkpoint)

    J_Regressor_optimizer = optim.Adam(
        [J_regressor_retrained], lr=args.j_reg_lr)

    loss_function = nn.MSELoss()

    j_reg_mask = utils.find_j_reg_mask(J_regressor)

    # data = data_set("train")
    data = data_set("validation")
    val_data = data_set("validation")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=True, drop_last=False)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    eval_before_opt = []
    eval_after_opt = []

    for epoch in range(1):

        iterator = iter(loader)

        for iteration in tqdm(range(len(loader))):

            try:
                batch = next(iterator)
            except:
                import time
                time.sleep(1)
                print("problem loading batch")
                continue

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations"):
                    batch[item] = batch[item].to(args.device).float()

            batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

            spin_image = normalize(batch['spin_image'])

            with torch.no_grad():
                spin_pred_pose, spin_pred_betas, pred_camera = spin_model(
                    spin_image)

            pred_cam_t = torch.stack([-2*pred_camera[:, 1],
                                      -2*pred_camera[:, 2],
                                      2*5000/(224 * pred_camera[:, 0] + 1e-9)], dim=-1)
            batch["cam"] = pred_cam_t

            # pred_rotmat = rot6d_to_rotmat(spin_pred_pose).view(-1, 24, 3, 3)

            batch["pose"] = spin_pred_pose[:, 1:].clone()
            batch["orient"] = spin_pred_pose[:, 0].unsqueeze(1).clone()
            batch["betas"] = spin_pred_betas.clone()

            optimize_parameters = [batch["pose"],
                                   batch["orient"], batch["betas"], batch["cam"]]

            for item in optimize_parameters:
                item.requires_grad = True

            this_batch_optimizer = optim.Adam(
                [batch["cam"]], lr=1e-2)

            for i in range(1000):
            # for i in range(0):

                joints_2d = return_2d_joints(
                    batch, smpl, J_regressor=J_regressor, mask=j_reg_mask)
                loss_j2d = loss_function(batch['gt_j2d'], joints_2d[..., :2])

                this_batch_optimizer.zero_grad()
                loss_j2d.backward()
                this_batch_optimizer.step()

            this_batch_optimizer = optim.Adam(
                optimize_parameters, lr=1e-2)


            # render the poses and perform valuation on IoU and Circle



            # # VIsualize
            pred_rotmat_orient = rot6d_to_rotmat(
                batch['orient'].reshape(-1, 6)).view(-1, 1, 3, 3)

            pred_rotmat_pose = rot6d_to_rotmat(
                batch['pose'].reshape(-1, 6)).view(-1, 23, 3, 3)
            spin_joints_2d = return_2d_joints(
                batch, smpl, J_regressor=SPIN_J_regressor)
            new_joints_2d = return_2d_joints(
                batch, smpl, J_regressor=J_regressor_retrained, mask=j_reg_mask)
            joints_2d = return_2d_joints(
                batch, smpl, J_regressor=J_regressor, mask=j_reg_mask)

            img = render_mesh(smpl, silhouette_renderer, batch["betas"], pred_rotmat_orient, pred_rotmat_pose, batch)

            eval_before_opt.append(eval(img, batch['mask_rcnn']))
            print("torch.mean(eval_before_opt)")
            print(torch.mean(torch.tensor(eval_before_opt)))
            viz(img, batch['mask_rcnn'], batch['spin_image'], spin_joints_2d, new_joints_2d, joints_2d, "")
            exit()



            for i in range(100):

                pred_rotmat_orient = rot6d_to_rotmat(
                    batch['orient'].reshape(-1, 6)).view(-1, 1, 3, 3)

                pred_rotmat_pose = rot6d_to_rotmat(
                    batch['pose'].reshape(-1, 6)).view(-1, 23, 3, 3)

                pred_joints = utils.find_joints(
                    smpl, batch["betas"], pred_rotmat_orient, pred_rotmat_pose, J_regressor, mask=j_reg_mask)

                joints_2d = return_2d_joints(
                    batch, smpl, J_regressor=J_regressor)
                loss_j2d = loss_function(batch['gt_j2d'], joints_2d[..., :2])
                img = render_mesh(smpl, silhouette_renderer, batch["betas"], pred_rotmat_orient, pred_rotmat_pose, batch)
                silhouette_loss = loss_function(img, batch['mask_rcnn'])

                # print("img.shape")
                # print(img.shape)
                # print("batch['mask_rcnn'].shape")
                # print(batch['mask_rcnn'].shape)
                # print(silhouette_loss.item())
                # exit()
                
                joint_loss = loss_function(utils.move_pelvis(
                    pred_joints), batch['gt_j3d']/1000)

                pred_disc = pose_discriminator(
                    torch.cat([batch['orient'], batch['pose']], dim=1))

                pred_shape_disc = shape_discriminator(batch["betas"])

                pose_discriminated_loss = loss_function(pred_disc, torch.ones(
                    pred_disc.shape).to(args.device))

                shape_discriminated_loss = loss_function(pred_shape_disc, torch.ones(
                    pred_shape_disc.shape).to(args.device))

                opt_loss = loss_j2d/100+silhouette_loss*100+joint_loss*10000 + \
                    pose_discriminated_loss*10+shape_discriminated_loss*10

                if i%10==0:
                    print(loss_j2d/100)
                    print(silhouette_loss*100)
                    print(joint_loss*10000)
                    print(pose_discriminated_loss*10)
                    print(shape_discriminated_loss*10)
                    print()
                
                this_batch_optimizer.zero_grad()
                opt_loss.backward()
                this_batch_optimizer.step()


            # if((iteration+1)%100==0):
            print("rendering")

            spin_joints_2d = return_2d_joints(
                batch, smpl, J_regressor=SPIN_J_regressor)
            new_joints_2d = return_2d_joints(
                batch, smpl, J_regressor=J_regressor_retrained, mask=j_reg_mask)
            joints_2d = return_2d_joints(
                batch, smpl, J_regressor=J_regressor, mask=j_reg_mask)

            img = render_mesh(smpl, silhouette_renderer, batch["betas"], pred_rotmat_orient, pred_rotmat_pose, batch)

            eval_after_opt.append(eval(img, batch['mask_rcnn']))
            print("torch.mean(eval_after_opt)")
            print(torch.mean(torch.tensor(eval_after_opt)))

            # viz(img, batch['mask_rcnn'], batch['spin_image'], batch['gt_j2d'], spin_joints_2d, joints_2d, "after_joints")
            # exit()
            # exit()

            # new_joints_2d = return_2d_joints(
            #     batch, smpl, J_regressor=new_J_regressor, mask=j_reg_mask)
            # joints_2d = return_2d_joints(
            #     batch, smpl, J_regressor=J_regressor)
            # utils.render_batch(img_renderer, batch, "final", [
            #                    new_joints_2d, joints_2d, batch["gt_j2d"]])
            # exit()

            pred_gt = pose_discriminator(spin_pred_pose)
            pred_disc = pose_discriminator(
                torch.cat([batch['orient'], batch['pose']], dim=1).detach())
            pose_discriminator_loss = loss_function(pred_disc, torch.zeros(
                pred_disc.shape).to(args.device))+loss_function(pred_gt, torch.ones(
                    pred_disc.shape).to(args.device))
            disc_optimizer.zero_grad()
            pose_discriminator_loss.backward()
            disc_optimizer.step()

            pred_gt = shape_discriminator(spin_pred_betas)
            pred_disc = shape_discriminator(batch["betas"])
            shape_discriminator_loss = loss_function(pred_disc, torch.zeros(
                pred_disc.shape).to(args.device))+loss_function(pred_gt, torch.ones(
                    pred_disc.shape).to(args.device))
            shape_disc_optimizer.zero_grad()
            shape_discriminator_loss.backward()
            shape_disc_optimizer.step()

            # get the joints from the joint regressor retrained
            # get error to gt
            # update j_regressor
            # relu and take norm

            pred_rotmat_orient = rot6d_to_rotmat(
                batch['orient'].reshape(-1, 6)).view(-1, 1, 3, 3)

            pred_rotmat_pose = rot6d_to_rotmat(
                batch['pose'].reshape(-1, 6)).view(-1, 23, 3, 3)

            pred_joints = utils.find_joints(
                smpl, batch["betas"].detach(), pred_rotmat_orient.detach(), pred_rotmat_pose.detach(), J_regressor_retrained, mask=j_reg_mask)
            j_regressor_error = loss_function(utils.move_pelvis(
                pred_joints), batch['gt_j3d']/1000)
            J_Regressor_optimizer.zero_grad()
            j_regressor_error.backward()
            J_Regressor_optimizer.step()

            mpjpe_new_opt, pampjpe_new_opt = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            pred_joints = utils.find_joints(
                smpl, batch["betas"].detach(), pred_rotmat_orient.detach(), pred_rotmat_pose.detach(), J_regressor, mask=j_reg_mask)

            mpjpe_old_opt, pampjpe_old_opt = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            if(args.wandb_log):
                wandb.log(
                    {
                        # "loss_2d": loss_2d.item(),
                        "joint_loss": joint_loss.item(),
                        "pose_discriminated_loss": pose_discriminated_loss.item(),
                        "shape_discriminated_loss": shape_discriminated_loss.item(),
                        "pose_discriminator_loss": pose_discriminator_loss.item(),
                        "shape_discriminator_loss": shape_discriminator_loss.item(),
                        "j_regressor_error": j_regressor_error.item(),
                        "mpjpe": mpjpe_old_opt.item(),
                        "pampjpe": pampjpe_old_opt.item(),
                        "mpjpe difference": mpjpe_new_opt.item()-mpjpe_old_opt.item(),
                        "pampjpe difference": pampjpe_new_opt.item()-pampjpe_old_opt.item(),
                    })

            # if(args.wandb_log and (iteration+1) % 100 == 0):

            #     print("saving model and regressor")

            #     torch.save(pose_discriminator.state_dict(),
            #                f"models/pose_discriminator.pt")
            #     torch.save(disc_optimizer.state_dict(),
            #                f"models/disc_optimizer.pt")

            #     torch.save(shape_discriminator.state_dict(),
            #                f"models/shape_discriminator.pt")
            #     torch.save(shape_disc_optimizer.state_dict(),
            #                f"models/shape_disc_optimizer.pt")

            #     torch.save(J_regressor_retrained,
            #                "models/retrained_J_Regressor.pt")



def optimize_network():

    spin_model = hmr(config.SMPL_MEAN_PARAMS).to(args.device)
    checkpoint = torch.load(
        "SPIN/data/model_checkpoint.pt", map_location=args.device)
    spin_model.load_state_dict(checkpoint['model'], strict=False)
    spin_model.eval()

    smpl = SMPL(
        '{}'.format("SPIN/data/smpl"),
        batch_size=1,
    ).to(args.device)

    maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True).to(args.device)
    maskrcnn.eval()

    J_regressor = torch.from_numpy(
        np.load('SPIN/data/J_regressor_h36m.npy')).float().to(args.device)

    J_regressor_retrained = torch.load(
        "models/best_pose_refiner/retrained_J_Regressor.pt", map_location=args.device)
    J_regressor_retrained.requires_grad = True

    silhouette_renderer = Renderer(subset=True)
    img_renderer = Renderer(subset=False)
    # silhouette_renderer = Mesh_Renderer()
    # silhouette_renderer = nn.DataParallel(silhouette_renderer)

    J_Regressor_optimizer = optim.Adam(
        [J_regressor_retrained], lr=args.j_reg_lr)

    loss_function = nn.MSELoss()

    j_reg_mask = utils.find_j_reg_mask(J_regressor)

    data = data_set("train")
    val_data = data_set("validation")

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=True, drop_last=False)

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for epoch in range(args.train_epochs):

        iterator = iter(loader)
        val_iterator = iter(val_loader)

        for iteration in tqdm(range(len(loader))):

            try:
                batch = next(iterator)
            except:
                import time
                time.sleep(1)
                print("problem loading batch")
                continue

            for item in batch:
                if(item != "valid" and item != "path" and item != "pixel_annotations"):
                    batch[item] = batch[item].to(args.device).float()

            batch['gt_j3d'] = utils.move_pelvis(batch['gt_j3d'])

            spin_image = normalize(batch['image'])

            this_spin_model = copy.deepcopy(spin_model)
            this_spin_model.train()

            this_batch_optimizer = optim.Adam(
                this_spin_model.parameters(), lr=1e-5)

            for i in range(100):

                spin_pred_pose, spin_pred_betas, pred_camera = this_spin_model(
                    spin_image)

                pred_rotmat = rot6d_to_rotmat(
                    spin_pred_pose).view(-1, 24, 3, 3)

                batch["pose"] = spin_pred_pose[:, 1:].clone()
                batch["orient"] = spin_pred_pose[:, 0].unsqueeze(1).clone()
                batch["betas"] = spin_pred_betas.clone()

                # if(i == 0):
                #     utils.render_batch(img_renderer, batch, "og")

                pred_rotmat_orient = rot6d_to_rotmat(
                    batch['orient'].reshape(-1, 6)).view(-1, 1, 3, 3)

                pred_rotmat_pose = rot6d_to_rotmat(
                    batch['pose'].reshape(-1, 6)).view(-1, 23, 3, 3)

                pred_joints = utils.find_joints(
                    smpl, batch["betas"], pred_rotmat_orient, pred_rotmat_pose, J_regressor)

                joint_loss = loss_function(utils.move_pelvis(
                    pred_joints), batch['gt_j3d']/1000)

                # rendered_silhouette = silhouette_renderer(batch)

                # rendered_silhouette = rendered_silhouette[:, 3].unsqueeze(1)

                # silhouette_loss = loss_function(
                #     rendered_silhouette[batch["valid"]], batch["mask_rcnn"][batch["valid"]])

                opt_loss = joint_loss*10000

                # print(i)
                # print(
                #     f"silhouette_loss.item()*100: {silhouette_loss.item()*100}")
                # print(f"joint_loss.item()*10000: {joint_loss.item()*10000}")

                this_batch_optimizer.zero_grad()
                opt_loss.backward()
                this_batch_optimizer.step()

            # utils.render_batch(img_renderer, batch, "optimized")
            # exit()

            # get the joints from the joint regressor retrained
            # get error to gt
            # update j_regressor
            # relu and take norm
            pred_joints = utils.find_joints(
                smpl, batch["betas"].detach(), pred_rotmat_orient.detach(), pred_rotmat_pose.detach(), J_regressor_retrained, mask=j_reg_mask)
            j_regressor_error = loss_function(utils.move_pelvis(
                pred_joints), batch['gt_j3d']/1000)
            J_Regressor_optimizer.zero_grad()
            j_regressor_error.backward()
            J_Regressor_optimizer.step()

            mpjpe_new_opt, pampjpe_new_opt = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            pred_joints = utils.find_joints(
                smpl, batch["betas"].detach(), pred_rotmat_orient.detach(), pred_rotmat_pose.detach(), J_regressor)

            mpjpe_old_opt, pampjpe_old_opt = utils.evaluate(
                pred_joints, batch['gt_j3d'])

            if(args.wandb_log):
                wandb.log(
                    {
                        # "silhouette_loss": silhouette_loss.item(),
                        "joint_loss": joint_loss.item(),
                        # "pose_discriminated_loss": pose_discriminated_loss.item(),
                        # "pose_discriminator_loss": pose_discriminator_loss.item(),
                        "j_regressor_error": j_regressor_error.item(),
                        "mpjpe": mpjpe_old_opt.item(),
                        "pampjpe": pampjpe_old_opt.item(),
                        "mpjpe difference": mpjpe_new_opt.item()-mpjpe_old_opt.item(),
                        "pampjpe difference": pampjpe_new_opt.item()-pampjpe_old_opt.item(),
                    })

            # rendered_img = img_renderer(
            #     batch, smpl, raster_settings_img, subset=False, colored=False)

            # drawing = (rendered_img[:, 3].unsqueeze(1).expand(
            #     batch['image'].shape)+batch['image']*.5)

            # blt = utils.torch_img_to_np_img(drawing)

            # import matplotlib.pyplot as plt
            # from matplotlib.patches import Circle
            # for i in range(blt.shape[0]):

            #     plt.imshow(utils.torch_img_to_np_img(drawing)[i])

            #     ax = plt.gca()

            #     plt.savefig(
            #         f"output/refining_image_{i:03d}_2.png", dpi=300)
            #     plt.close()

            #     plt.imshow(utils.torch_img_to_np_img(rendered_silhouette)[i])

            #     ax = plt.gca()

            #     plt.savefig(
            #         f"output/refining_image_{i:03d}_1.png", dpi=300)
            #     plt.close()

            #     plt.imshow(utils.torch_img_to_np_img(batch["mask_rcnn"])[i])

            #     ax = plt.gca()

            #     plt.savefig(
            #         f"output/refining_image_{i:03d}_0.png", dpi=300)
            #     plt.close()

            # exit()

            if(args.wandb_log and (iteration+1) % 1000 == 0):

                print("saving model and regressor")

                # torch.save(pose_discriminator.state_dict(),
                #            f"models/pose_discriminator_epoch_{epoch}.pt")
                # torch.save(disc_optimizer.state_dict(),
                #            f"models/disc_optimizer_epoch_{epoch}.pt")

                torch.save(J_regressor_retrained,
                           "models/retrained_J_Regressor.pt")
