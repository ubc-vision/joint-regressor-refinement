import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str)
parser.add_argument('--train_epochs', type=int, default=1)
parser.add_argument('--opt_steps', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--optimization_batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--opt_lr', type=float, default=1e-2)
parser.add_argument('--disc_learning_rate', type=float, default=1e-4)
parser.add_argument('--opt_disc_learning_rate', type=float, default=1e-3)
parser.add_argument('--translation_lr', type=float, default=1e-6)
parser.add_argument('--j_reg_lr', type=float, default=1e-2)
parser.add_argument('--optimization_rate', type=float, default=1e4)
parser.add_argument('--wandb_log', action='store_true')
parser.add_argument('--compute_canada', action='store_true')


parser.add_argument('--device', type=str, default="cuda:0")


#  Transformer args
#########################################################
# Data related arguments
#########################################################
parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                    help="Directory with all datasets, each in one subfolder")
parser.add_argument("--train_yaml", default='imagenet2012/train.yaml', type=str, required=False,
                    help="Yaml file with all data for training.")
parser.add_argument("--val_yaml", default='human3.6m/valid.protocol2.yaml', type=str, required=False,
                    help="Yaml file with all data for validation.")
parser.add_argument("--num_workers", default=4, type=int,
                    help="Workers in dataloader.")
parser.add_argument("--img_scale_factor", default=1, type=int,
                    help="adjust image resolution.")
#########################################################
# Loading/saving checkpoints
#########################################################
parser.add_argument("--model_name_or_path", default='metro/modeling/bert/bert-base-uncased/', type=str, required=False,
                    help="Path to pre-trained transformer model or model type.")
parser.add_argument("--resume_checkpoint", default="models/metro_release/metro_h36m_state_dict.bin", type=str, required=False,
                    help="Path to specific checkpoint for resume training.")
parser.add_argument("--output_dir", default='output/', type=str, required=False,
                    help="The output directory to save checkpoint and test results.")
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name.")
#########################################################
# Training parameters
#########################################################
parser.add_argument("--per_gpu_train_batch_size", default=30, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
# parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float,
#                     help="The initial lr.")
parser.add_argument("--num_train_epochs", default=200, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--vertices_loss_weight", default=100.0, type=float)
parser.add_argument("--joints_loss_weight", default=1000.0, type=float)
parser.add_argument("--vloss_w_full", default=0.33, type=float)
parser.add_argument("--vloss_w_sub", default=0.33, type=float)
parser.add_argument("--vloss_w_sub2", default=0.33, type=float)
parser.add_argument("--drop_out", default=0.1, type=float,
                    help="Drop out ratio in BERT.")
#########################################################
# Model architectures
#########################################################
parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
parser.add_argument("--num_hidden_layers", default=4, type=int, required=False,
                    help="Update model config if given")
parser.add_argument("--hidden_size", default=-1, type=int, required=False,
                    help="Update model config if given")
parser.add_argument("--num_attention_heads", default=4, type=int, required=False,
                    help="Update model config if given. Note that the division of "
                    "hidden_size / num_attention_heads should be in integer.")
parser.add_argument("--intermediate_size", default=-1, type=int, required=False,
                    help="Update model config if given.")
parser.add_argument("--input_feat_dim", default='2051,512,128', type=str,
                    help="The Image Feature Dimension.")
parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str,
                    help="The Image Feature Dimension.")
parser.add_argument("--legacy_setting", default=True, action='store_true',)
#########################################################
# Others
#########################################################
parser.add_argument("--run_eval_only", default=False, action='store_true',)
parser.add_argument('--logging_steps', type=int, default=1000,
                    help="Log every X steps.")
# parser.add_argument("--device", type=str, default='cuda',
#                     help="cuda or cpu")
parser.add_argument('--seed', type=int, default=88,
                    help="random seed for initialization.")
parser.add_argument("--local_rank", type=int, default=0,
                    help="For distributed training.")


args = parser.parse_args()

print("args")
print(args)
