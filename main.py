import wandb
from train import train_render_model, create_buffer
from test import test_render_model
from args import args

import torch
# from crop_model import Crop_Model

from render_model import Render_Model

# from visualizer import draw_gradients

if __name__ == "__main__":

    if(args.wandb_log):
        wandb.init(project="human_body_pose_optimization",
                   name="render_model")
        wandb.config.update(args)

    # model = train_render_model()
    # create_buffer()
    # torch.save(model.state_dict(), f"models/linearized_model_{args.crop_scalar}.pt")
    # exit()

    model = Render_Model().to(args.device)

    model.load_state_dict(torch.load(
        f"models/render_model_epoch23.pt", map_location=args.device))

    test_render_model(model)
