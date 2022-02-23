from args import args

import torch


import constants

from utils import utils

import sys
sys.path.append('/scratch/iamerich/MeshTransformer')  # noqa
from metro.modeling.bert import METRO_Body_Network as METRO_Network
from metro.modeling._smpl import SMPL, Mesh
from metro.modeling.hrnet.config import config as hrnet_config
from metro.modeling.bert import BertConfig, METRO

from metro.utils.renderer import Renderer

from metro.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from metro.modeling.hrnet.config import update_config as hrnet_update_config



def load_transformer():

    # Mesh and SMPL utils
    mesh_smpl = SMPL().to(args.device)
    mesh_sampler = Mesh(device=args.device)
    # Renderer for visualization
    renderer = Renderer(faces=mesh_smpl.faces.cpu().numpy())
    # Load pretrained model

    # Build model from scratch, and load weights from state_dict.bin
    trans_encoder = []
    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]

    # init three transformer encoders in a loop
    for i in range(len(output_feat_dim)):
        config_class, model_class = BertConfig, METRO
        config = config_class.from_pretrained(args.config_name if args.config_name
                                              else args.model_name_or_path)

        config.output_attentions = True
        config.hidden_dropout_prob = args.drop_out
        config.img_feature_dim = input_feat_dim[i]
        config.output_feature_dim = output_feat_dim[i]
        config.device = args.device
        args.hidden_size = hidden_feat_dim[i]

        if args.legacy_setting == True:
            # During our paper submission, we were using the original intermediate size, which is 3072 fixed
            # We keep our legacy setting here
            args.intermediate_size = -1
        else:
            # We have recently tried to use an updated intermediate size, which is 4*hidden-size.
            # But we didn't find significant performance changes on Human3.6M (~36.7 PA-MPJPE)
            args.intermediate_size = int(args.hidden_size*4)

        # update model structure if specified in arguments
        update_params = ['num_hidden_layers', 'hidden_size',
                         'num_attention_heads', 'intermediate_size']

        for idx, param in enumerate(update_params):
            arg_param = getattr(args, param)
            config_param = getattr(config, param)
            if arg_param > 0 and arg_param != config_param:
                # logger.info(
                #     "Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                setattr(config, param, arg_param)

        # init a transformer encoder and append it to a list
        assert config.hidden_size % config.num_attention_heads == 0
        model = model_class(config=config)

        if(i == len(output_feat_dim)-1):
            final_cls_head = model.cls_head
        # logger.info("Init model from scratch.")
        trans_encoder.append(model)

    # init ImageNet pre-trained backbone model
    if args.arch == 'hrnet':
        hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        # logger.info('=> loading hrnet-v2-w40 model')
    elif args.arch == 'hrnet-w64':
        hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        # logger.info('=> loading hrnet-v2-w64 model')
    else:
        print("=> using pre-trained model '{}'".format(args.arch))
        backbone = models.__dict__[args.arch](pretrained=True)
        # remove the last fc layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    trans_encoder = torch.nn.Sequential(*trans_encoder)
    total_params = sum(p.numel() for p in trans_encoder.parameters())
    # logger.info('Transformers total parameters: {}'.format(total_params))
    backbone_total_params = sum(p.numel() for p in backbone.parameters())
    # logger.info('Backbone total parameters: {}'.format(backbone_total_params))

    # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
    _metro_network = METRO_Network(
        args, config, backbone, trans_encoder, mesh_sampler)

    if args.resume_checkpoint != None and args.resume_checkpoint != 'None':
        # for fine-tuning or resume training or inference, load weights from checkpoint
        # logger.info("Loading state dict from checkpoint {}".format(
        #     args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint,
                                map_location=cpu_device)
        _metro_network.load_state_dict(state_dict, strict=False)
        del state_dict

    return _metro_network, mesh_smpl, mesh_sampler, final_cls_head



# TODO throw away the first num_joints. Wasted space
def find_joints_from_updated_head(cls_head, refined_feats, residual, num_joints, upsampling, upsampling2, smpl):

    refined_feats = cls_head(refined_feats)

    refined_feats = refined_feats + residual[:, num_joints:]

    # pred_vertices_sub2 = refined_feats[:, num_joints:, :]


    temp_transpose = refined_feats.transpose(1, 2)
    pred_vertices_sub = upsampling(temp_transpose)
    pred_vertices_full = upsampling2(pred_vertices_sub)
    pred_vertices_sub = pred_vertices_sub.transpose(1, 2)
    pred_vertices = pred_vertices_full.transpose(1, 2)

    pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

    pred_3d_joints_from_smpl = utils.move_pelvis(
        pred_3d_joints_from_smpl)

    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,
                                                        constants.J17_2_METRO]

    return pred_3d_joints_from_smpl, pred_vertices

    
