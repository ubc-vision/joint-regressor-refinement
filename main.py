import wandb
# from train import train_render_model, create_buffer
# from test import test_render_model
from args import args

import torch
# from crop_model import Crop_Model

# from render_model import Render_Model
from train import train_error_estimator, train_pose_refiner_model, train_error_estimator_parametric
from test import test_pose_refiner_model, test_pose_refiner_model_VIBE, test_pose_refiner_translation_model

from utils import utils

# from utils import h5py_creator

# from visualizer import draw_gradients

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch.nn.functional")

if __name__ == "__main__":

    if(args.wandb_log):
        wandb.init(project="human_body_pose_optimization",
                   name=args.name,
                   settings=wandb.Settings(start_method='fork'))
        wandb.config.update(args)

    utils.set_seed(0)

    # train_error_estimator_parametric()
    train_pose_refiner_model()
