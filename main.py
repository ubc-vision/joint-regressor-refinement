import wandb
# from train import train_render_model, create_buffer
# from test import test_render_model
from args import args

import torch
# from crop_model import Crop_Model

# from render_model import Render_Model
from optimize import optimize_pose_refiner
from train import train_pose_refiner_model

# from visualizer import draw_gradients

if __name__ == "__main__":

    if(args.wandb_log):
        wandb.init(project="human_body_pose_optimization",
                   name="training gt intrinsics")
        wandb.config.update(args)

    # optimize_pose_refiner()
    train_pose_refiner_model()
