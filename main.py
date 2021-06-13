import wandb
# from train import train_render_model, create_buffer
# from test import test_render_model
from args import args

import torch
# from crop_model import Crop_Model

# from render_model import Render_Model
from optimize import optimize_pose_refiner, optimize_network
from train import train_pose_refiner_model, train_joint_regressor, train_pose_refiner_translation_model
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
                   name="testing black input")
        wandb.config.update(args)

    utils.set_seed(0)

    test_pose_refiner_model(0)

    # for i in range(9):
    #     test_pose_refiner_model(i)

    # optimize_pose_refiner()
    # train_pose_refiner_model()
    # test_pose_refiner_model()
    # test_pose_refiner_model_VIBE()
    # optimize_network()
    # train_joint_regressor()
    # train_pose_refiner_translation_model()
    # test_pose_refiner_translation_model()
