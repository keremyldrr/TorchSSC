from __future__ import division

import pycuda.driver as drv
from pycuda import compiler
import pycuda
import os.path as osp
import os
import sys
import time
import argparse
# from typing_extensions import runtime
from tqdm import tqdm
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
from config import config, update_parameters_in_config
from dataloader import ValPre, get_train_loader,get_val_loader
from network import Network
from scannet import ScanNet
from nyu import NYUv2
from eval import SegEvaluator
import matplotlib.pyplot as plt
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR, PolyLR
# from torch_lr_finder import LRFinder
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
# from torch.nn import BatchNorm3d,BatchNorm2d
from sc_utils import export_grid
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
# from tensorboard.plugins.mesh import summary as mesh_summary
# import tensorflow as tf
from tensorboard_plugin_geometry import add_geometry
from loss_utils import compute_loss,process_one_batch
from score_utils import compute_metric
from utils.pyt_utils import load_model
import gc

cudnn.benchmark = True
seed = config.seed

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)




def debug_gpu():
    # Debug out of memory bugs.
    tensor_list = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                # if obj.device
                tensor_list.append(obj)
        except:
            pass
    print(f'Count of tensors = {len(tensor_list)}.')

def evaluate_epoch(args,network,loaders,logger):
    for loader in loaders:
        loader_name = loader.loader_name
        loader  =loader.dataloader
        for cp in args.epochs.split():
            cp = int(cp)
            checkpoint_path = os.path.join(args.snapshot_dir,"epoch-{}.pth".format(cp))
            model = load_model(network,checkpoint_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #  model = DataParallelModel(model, device_ids=engine.devices)
            model.to(device)

    
            # model.train()              
            model.eval()

            with torch.no_grad():
                dset = iter(loader)
                sum_loss = 0
                sum_sem = 0
                sum_com = 0
                sum_rest = 0
                sum_sketch = 0
                sum_sketch_gsnn = 0
                sum_kld = 0
                sum_sketch_raw = 0
                results = []
                print("\n************************** Evaluation of of epoch {} for {} progress *********************".format(cp,loader_name))
                for i in range(len(loader)):
                    if model.training:
                        loss,loss_semantic, loss_sketch_raw, loss_sketch, loss_sketch_gsnn, KLD,results_dict  = process_one_batch(dset,model,config,engine=None)
                        sum_loss +=loss
                        sum_sem  +=loss_semantic
                        sum_sketch  +=loss_sketch
                        sum_sketch_gsnn  +=loss_sketch_gsnn
                        sum_kld  +=KLD
                        sum_sketch_raw  +=loss_sketch_raw
                    else:
                        results_dict = process_one_batch(dset,model,config,engine=None)
                    # results_dictrocess_one_batch(val_dset,model,config,engine)
                    torch.cuda.empty_cache()
                    # debug_gpu()
                    gc.collect()
                    print(i,len(loader),end="\r")
                    
                    results.append(results_dict)
                    # print(val_loss,val_loss_semantic,val_loss_sketch_raw,val_loss_sketch,val_loss_sketch_gsnn,val_KLD)
                sum_loss /= len(loader)
                sum_sem /= len(loader)
                sum_sketch /= len(loader)
                sum_sketch_gsnn /= len(loader)
                sum_kld /= len(loader)
                sum_sketch_raw /= len(loader)
                logger.add_scalar('{}/loss_total'.format(loader_name), sum_loss, cp)
                logger.add_scalar('{}/loss_sem'.format(loader_name), sum_sem, cp)
                logger.add_scalar('{}/loss_sketch'.format(loader_name), sum_sketch, cp)
                logger.add_scalar('{}/loss_sketch_gsnn'.format(loader_name), sum_sketch_gsnn, cp)
                logger.add_scalar('{}/sketch_raw'.format(loader_name), sum_sketch_raw, cp)
                # # # # Need to make this work
                # model.eval()

                results_line, metric = compute_metric(config,results)
                # print("Val loss",val_sum_loss)
                sscmIOU, sscPixel, scIOU, scPixel, scRecall = metric
                logger.add_scalar('{}/sscmIOU'.format(loader_name), sscmIOU, cp)
                logger.add_scalar('{}/sscPixel'.format(loader_name), sscPixel, cp)
                logger.add_scalar('{}/scIOU'.format(loader_name), scIOU, cp)
                logger.add_scalar('{}/scPixel'.format(loader_name), scPixel, cp)
                logger.add_scalar('{}/scRecall'.format(loader_name), scRecall, cp)

                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('--snapshot-dir', default='idontexist', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default='results')
    parser.add_argument('--lr', type=float,
                       default=0.1,
                       dest="lr",
                   help='Learning rate for experiments')
    parser.add_argument('--ew', type=float,
                       default=1.0,
                       dest="ew",
                       help='Empty voxel weight for experiments')
    parser.add_argument('--num_epochs', type=int,
                       default=250,
                       dest="num_epochs",
                        help='Number of epochs to train for experiments')
    parser.add_argument('--only_frustum', dest='only_frustum', type=bool,default=False)
    parser.add_argument('--only_boxes', dest='only_boxes', type=bool,default=False)
    parser.add_argument('--dataset',type=str,default="NYUv2",dest='dataset')
    parser.add_argument('--logdir',type=str,default="logs/evaluation")

    args = parser.parse_args()

    update_parameters_in_config(config,ew=args.ew,lr=args.lr,num_epochs=args.num_epochs,only_frustum=args.only_frustum,only_boxes=args.only_boxes,dataset=args.dataset)
    network = Network(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                norm_layer=nn.BatchNorm3d, eval=True)
  
    config.batch_size = 1

    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'hha_root':config.hha_root_folder,
                    'mapping_root': config.mapping_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'dataset_path': config.dataset_path}
   
    if args.dataset == "scannet":
        val_loader, _ = get_val_loader(
        None, ScanNet, only_frustum=args.only_frustum, only_box=args.only_boxes)
        train_loader, _ = get_train_loader(
        None, ScanNet, only_frustum=args.only_frustum, only_box=args.only_boxes,train_sampler=True)
    else:
     val_loader, val_sampler = get_val_loader(
        None, NYUv2, only_frustum=args.only_frustum, only_box=args.only_boxes)
    #TODO make this work with custom as well
    traindict = edict()
    traindict.dataloader = train_loader
    traindict.loader_name = "train"
    valdict = edict()
    valdict.dataloader = val_loader
    valdict.loader_name = "val"

    print(len(train_loader),"lsqjdklqsjdddj1£1")
    logger = SummaryWriter(args.logdir)
    with torch.no_grad():
        evaluate_epoch(args,network,[traindict,valdict],logger)
        