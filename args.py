import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_epochs', type=int, default=50000)
parser.add_argument('--opt_steps', type=int, default=101)
parser.add_argument('--training_batch_size', type=int, default=16)
parser.add_argument('--optimization_batch_size', type=int, default=16)
parser.add_argument('--crop_scalar', type=int, default=6)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--optimization_rate', type=float, default=1e0)
parser.add_argument('--grad_loss_weight', type=float, default=1e-1)
parser.add_argument('--wandb_log', action='store_true')

parser.add_argument('--device', type=str, default = "cuda:0")
args = parser.parse_args()

print("args")
print(args)