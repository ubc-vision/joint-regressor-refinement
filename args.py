import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_epochs', type=int, default=50000)
parser.add_argument('--opt_steps', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--optimization_batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-6)
parser.add_argument('--disc_learning_rate', type=float, default=1e-3)
parser.add_argument('--optimization_rate', type=float, default=1e4)
parser.add_argument('--wandb_log', action='store_true')

parser.add_argument('--device', type=str, default="cuda:0")
args = parser.parse_args()

print("args")
print(args)
