# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse




def update_logpath_in_config(C,log):
    C.log_dir = osp.abspath(log)
    C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))

    C.log_dir_link = osp.join(C.abs_dir, 'log')
    C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

    exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    C.log_file = C.log_dir + '/log_' + exp_time + '.log'
    C.link_log_file = C.log_file + '/log_last.log'
    C.val_log_file = C.log_dir + '/_' + exp_time + '.lovalg'
    C.link_val_log_file = C.log_dir + '/val_last.log'
    
def update_parameters_in_config(C,lr=0.1,ew=1,num_epochs=200,only_frustum=False,only_boxes=False,dataset="NYUv2"):



    C.log_dir = osp.abspath('log')
    C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))

    C.log_dir_link = osp.join(C.abs_dir, 'log')
    C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

    exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    C.log_file = C.log_dir + '/log_' + exp_time + '.log'
    C.link_log_file = C.log_file + '/log_last.log'
    C.val_log_file = C.log_dir + '/_' + exp_time + '.lovalg'
    C.link_val_log_file = C.log_dir + '/val_last.log'

    """Data Dir and Weight Dir"""

    C.is_test = False

    """Path Config"""


   
    if dataset == "NYUv2":
        C.dataset_path = osp.join(C.volna, 'DATA/NYU/')
        C.train_source = osp.join(C.dataset_path, "train.txt")
        C.eval_source = osp.join(C.dataset_path, "test.txt")
        C.dataset = "NYUv2"
        C.num_classes = 12
        C.image_mean = np.array([0.485, 0.456, 0.406])
        C.image_std = np.array([0.229, 0.224, 0.225])
        C.background = 255
    elif dataset == "scannet":
        C.dataset_path = osp.join(C.volna, 'DATA/ScanNet/')
        C.train_source = osp.join(C.dataset_path, "train_subset.txt")
        C.eval_source = osp.join(C.dataset_path, "val_subset.txt")
        C.dataset = "ScanNet"
        C.num_classes = 19
        C.image_mean = np.array([109.8, 97.2, 83.8])/255

        C.image_std = np.array([1,1,1])
        C.background = 0
    else:
        raise NotImplementedError("Invalid dataset")
    C.img_root_folder = C.dataset_path
    C.gt_root_folder = C.dataset_path
    C.hha_root_folder = osp.join(C.dataset_path, 'HHA')
    C.mapping_root_folder = osp.join(C.dataset_path, 'Mapping')
    """Image Config"""
    
   
    C.image_height = 480
    C.image_width = 640
    C.num_train_imgs = len(open(C.train_source,"r").readlines()) 
    C.num_eval_imgs = len(open(C.eval_source,"r").readlines())

    """ Settings for network, this would be different for each kind of model"""
    C.fix_bias = True
    C.bn_eps = 1e-5
    C.bn_momentum = 0.1
    C.pretrained_model = C.volna + 'DATA/pytorch-weight/resnet50-imagenet.pth'

    """Train Config"""
    C.lr = 0.1#changed by me
    C.lr_power = 0.9
    C.momentum = 0.9
    C.weight_decay = 5e-4
    C.batch_size = 2
    C.nepochs = 250
    C.niters_per_epoch = C.num_train_imgs//C.batch_size 
    C.num_workers = C.batch_size

    C.train_scale_array = [1]
    C.warm_up_epoch = 0

    """Eval Config"""
    C.eval_iter = 30
    C.eval_stride_rate = 2 / 3
    C.eval_scale_array = [1, ]
    C.eval_flip = False
    C.eval_base_size = 503
    C.eval_crop_size = 640

    """Display Config"""
    C.snapshot_iter = 10
    C.record_info_iter = 20
    C.display_iter = 50
    C.sketch_weight = 1
    C.sketch_weight_gsnn = 1.5

    C.kld_weight = 2
    C.samples = 4#C.batch_size
    C.lantent_size = 16
    C.empty_loss_weight = ew# original was 1 
    ## Currently disgusting, delete duplicates 
    C.lr = lr #changed by me
    C.nepochs = num_epochs
    C.niters_per_epoch = C.num_train_imgs//C.batch_size     
    C.only_frustum=only_frustum
    C.only_boxes=only_boxes
    logdir = "logs/{}_ew_{}_lr_{}_of_{}_ob_{}".format(dataset,ew,lr,int(only_frustum),int(only_boxes))
    update_logpath_in_config(C,logdir)
    

C = edict()
config = C
cfg = C
C.seed = 12345

remoteip = os.popen('pwd').read()
C.volna = '/home/yildirir/workspace/kerem/TorchSSC/'      # this is the path to your repo 'TorchSSC'


"""please config ROOT_dir and user when u first using"""
C.repo_name = 'TorchSSC'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]


C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir, 'furnace'))
from utils.pyt_utils import model_urls


def open_tensorboard():
    pass

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()