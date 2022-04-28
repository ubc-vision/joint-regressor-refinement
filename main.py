import wandb
from scripts.args import args

from scripts.test import test_pose_refiner_model, test_pose_refiner_model_VIBE_MEVA
from scripts.optimize import optimize_pose_refiner

from scripts import utils

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

    optimize_pose_refiner()

    test_pose_refiner_model()
    test_pose_refiner_model_VIBE_MEVA(vibe=True)
    test_pose_refiner_model_VIBE_MEVA(vibe=False)
    # optimize_pose_refiner()
    # exit()
