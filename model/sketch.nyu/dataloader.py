import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.img_utils import  normalize, \
    generate_random_crop_pos, random_crop_pad_to_shape
from torch.utils.data import SubsetRandomSampler
class TrainPre(object):
    def __init__(self, img_mean, img_std,kek=None):
        self.img_mean = img_mean
        self.img_std = img_std
        self.kek = kek
    def __call__(self, img, hha):
        img = normalize(img, self.img_mean, self.img_std)
        # hha = normalize(hha, self.img_mean, self.img_std)

        p_img = img.transpose(2, 0, 1)
        # if self.kek:
        #     print("!!!!!!",p_img.sum(),p_img.shape)

        # print(hha,p_img.shape)
        p_hha = hha#,p.transpose(2, 0, 1)

        extra_dict = {'hha_img': p_hha} #TODO, normlaize depth

        return p_img, extra_dict
class ValPre(object):
    def __call__(self, img, hha):
        extra_dict = {'hha_img': img}#TODO, depth is useless now()
        
        return img, extra_dict


def get_train_loader(engine, dataset, s3client=None,only_frustum=False,only_box=False,train_sampler = None):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'hha_root':config.hha_root_folder,
                    'mapping_root': config.mapping_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    "dataset_path":config.dataset_path}
    train_preprocess = TrainPre(config.image_mean, config.image_std)
    train_dataset = dataset(data_setting, "train", train_preprocess,
                            config.batch_size * config.niters_per_epoch, s3client=s3client,only_frustum=only_frustum,only_box=only_box)
    
    if train_sampler:
        dataset_size = len(train_dataset)
        dataset_indices = list(range(dataset_size))
        np.random.shuffle(dataset_indices)
    train_sampler = SubsetRandomSampler(dataset_indices[:5000])
    is_shuffle = False
    batch_size = config.batch_size
    if engine is not None:
        if engine.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
            batch_size = config.batch_size // engine.world_size
            is_shuffle = False
    train_loader = data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=config.num_workers,
                                    drop_last=True,
                                    shuffle=is_shuffle,
                                    pin_memory=True,
                                    sampler=train_sampler)
 
    return train_loader, train_sampler
def get_val_loader(engine, dataset, s3client=None,only_frustum=False,only_box=False):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'hha_root':config.hha_root_folder,
                    'mapping_root': config.mapping_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    "dataset_path":config.dataset_path}
    val_preprocess = TrainPre(config.image_mean, config.image_std,kek=10)
    val_dataset = dataset(data_setting, "val", val_preprocess,
                            config.batch_size * config.niters_per_epoch, s3client=s3client,only_frustum=only_frustum,only_box=only_box)
    val_sampler = None
 
    is_shuffle = False
    batch_size = 1 # Make this work with config.batch_size
    if engine is not None:
        if engine.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset)
            batch_size = batch_size #config.batch_size // engine.world_size
            is_shuffle = False
    val_loader = data.DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=None)
    
    return val_loader, val_sampler
