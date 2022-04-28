import torch

from scripts.args import args

from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras

from scripts import utils


def return_2d_joints(batch, smpl, J_regressor=None, mask=None):

    # start_time = time.time()

    # focal_length = torch.stack(
    #     [batch['intrinsics'][:, 0, 0]/224, batch['intrinsics'][:, 1, 1]/224], dim=1).to(args.device)
    # principal_point = torch.stack(
    #     [batch['intrinsics'][:, 0, 2]/-112+1, batch['intrinsics'][:, 1, 2]/-112+1], dim=1)
    focal_length = torch.ones(
        batch["image"].shape[0], 2).to(args.device)*5000/224
    principal_point = torch.zeros(batch["image"].shape[0], 2).to(args.device)

    pose = utils.rot6d_to_rotmat(
        batch['pose'].reshape(-1, 6)).reshape(-1, 23, 3, 3)
    orient = utils.rot6d_to_rotmat(
        batch['orient'].reshape(-1, 6)).reshape(-1, 1, 3, 3)

    if(J_regressor is not None):
        point_cloud = utils.find_joints(
            smpl, batch['betas'], orient, pose, J_regressor, mask=mask)
    else:

        point_cloud = smpl(betas=batch['betas'], body_pose=pose,
                           global_orient=orient, pose2rot=False).vertices

    point_cloud[:, :, 1] *= -1
    point_cloud[:, :, 0] *= -1
    point_cloud *= 2

    cameras = PerspectiveCameras(device=args.device, T=batch['cam'],
                                 focal_length=focal_length, principal_point=principal_point)

    image_size = torch.tensor([224, 224]).unsqueeze(
        0).expand(batch['intrinsics'].shape[0], 2).to(args.device)

    feat = torch.ones(
        point_cloud.shape[0], point_cloud.shape[1], 4).to(args.device)

    pred_verts_2d = cameras.transform_points_screen(
        point_cloud, image_size)

    return pred_verts_2d
