import wandb
# from train import train_render_model, create_buffer
# from test import test_render_model
from args import args

import torch
# from crop_model import Crop_Model

# from render_model import Render_Model
from optimize import optimize_pose_refiner
from train import train_pose_refiner_model, train_joint_regressor
from test import test_pose_refiner_model, test_pose_refiner_model_VIBE

# from utils import h5py_creator

# from visualizer import draw_gradients

if __name__ == "__main__":

    if(args.wandb_log):
        wandb.init(project="human_body_pose_optimization",
                   name="training discriminator")
        wandb.config.update(args)

    # optimize_pose_refiner()
    # train_pose_refiner_model()
    # test_pose_refiner_model()
    test_pose_refiner_model_VIBE()
    # train_joint_regressor()
