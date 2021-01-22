import wandb
from train import train_crop_model
from test import test_crop_model
from args import args

import torch
from crop_model import Crop_Model

from visualizer import draw_gradients

if __name__ == "__main__":

    
    if(args.wandb_log):
        wandb.init(project="human_body_pose_optimization", name="linearized_6_no_bias")
        wandb.config.update(args) 

    

    # model = train_crop_model()
    # torch.save(model.state_dict(), f"models/linearized_model_{args.crop_scalar}.pt")
    # exit()

    model = Crop_Model().to(args.device)

    model.load_state_dict(torch.load(f"models/linearized_model_8_epoch49.pt", map_location=args.device))

    # for i in range(10):
    #     draw_gradients(model, "train", "demo_images_train")

    test_crop_model(model)





