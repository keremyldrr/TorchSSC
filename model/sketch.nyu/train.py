from __future__ import division

import pycuda.driver as drv
from pycuda import compiler
import pycuda
import os.path as osp
import os
import sys
import time
import argparse
import pdb

# from typing_extensions import runtime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
from config import config, update_parameters_in_config
from dataloader import ValPre, get_train_loader, get_val_loader
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

# from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d

from torch.nn import BatchNorm3d, BatchNorm2d
from sc_utils import export_grid

from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter
# from tensorboard.plugins.mesh import summary as mesh_summary
# import tensorflow as tf
# from tensorboard_plugin_geometry import add_geometry
from loss_utils import compute_loss, process_one_batch
from score_utils import compute_metric
from evaluate_single import evaluate_epoch

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm

    # pass
except ImportError:
    # raise ImportError("Please install apex from https://www.github.com/nvidia/apex .")
    print("No apex, going on")


from easydict import EasyDict as edict

parser = argparse.ArgumentParser()
port = str(int(float(time.time())) % 20)
os.environ["MASTER_PORT"] = str(10097 + int(port))

drv.init()
print("%d device(s) found." % drv.Device.count())

for ordinal in range(drv.Device.count()):
    dev = drv.Device(ordinal)
    print(ordinal, dev.name())
with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    print(args)
    engine.distributed = False
    update_parameters_in_config(
        config,
        ew=args.ew,
        lr=args.lr,
        num_epochs=args.num_epochs,
        only_frustum=args.only_frustum,
        only_boxes=args.only_boxes,
        dataset=args.dataset,
        prefix=args.prefix,
    )
    print(config)
    cudnn.benchmark = True
    # seed = config.seed
    # if engine.distributed:
    #     seed = engine.local_rank
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    torch.manual_seed(31)
    np.random.seed(31)
    # data loader
    # train_loader, train_sampler = get_train_loader(engine, ScanNet,only_frustum=args.only_frustum,only_box=args.only_boxes)
    if config.dataset == "NYUv2":
        train_loader, train_sampler = get_train_loader(
            engine, NYUv2, only_frustum=args.only_frustum, only_box=args.only_boxes
        )
        val_loader, val_sampler = get_val_loader(
            engine,
            NYUv2,
            only_frustum=args.only_frustum,
            only_box=args.only_boxes,
            val_sampler_size=500,
        )
        train_loader_mini, _ = get_train_loader(
            None,
            NYUv2,
            only_frustum=args.only_frustum,
            only_box=args.only_boxes,
            train_sampler_size=500,
        )
    else:
        train_loader, train_sampler = get_train_loader(
            engine, ScanNet, only_frustum=args.only_frustum, only_box=args.only_boxes
        )
        val_loader, val_sampler = get_val_loader(
            engine,
            ScanNet,
            only_frustum=args.only_frustum,
            only_box=args.only_boxes,
            val_sampler_size=500,
        )

        train_loader_mini, _ = get_train_loader(
            None,
            ScanNet,
            only_frustum=args.only_frustum,
            only_box=args.only_boxes,
            train_sampler_size=500,
        )

    if (engine.distributed and (engine.local_rank == 0)) or len(engine.devices) == 1:
        tb_dir = config.tb_dir + "/{}".format(
            time.strftime("%b%d_%d-%H-%M", time.localtime())
        )
        print(tb_dir)
        res_path = os.path.join(tb_dir, "results")
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        generate_tb_dir = config.tb_dir + "/tb"
        # SummaryWriter.add_geometry = add_geometry

        logger = SummaryWriter(log_dir=tb_dir)
        # writer = tf.summary.create_file_writer(tb_dir)
        # tf.compat.v1.disable_eager_execution()
        engine.link_tb(tb_dir, generate_tb_dir)
        # print("Tensorboard!!!!!!")

    # # config network and criterion
    # criterion = nn.CrossEntropyLoss(reduction='mean',
    #                                 ignore_index=255)

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm
    # BatchNorm2d = BatchNorm3d
    print(config)
    model = Network(
        class_num=config.num_classes,
        feature=128,
        bn_momentum=config.bn_momentum,
        pretrained_model=config.pretrained_model,
        norm_layer=BatchNorm3d,
    )
    init_weight(
        model.business_layer,
        nn.init.kaiming_normal_,
        BatchNorm3d,
        config.bn_eps,
        config.bn_momentum,
        mode="fan_in",
    )  # , nonlinearity='relu')

    state_dict = torch.load(config.pretrained_model)  # ['state_dict']
    transformed_state_dict = {}
    for k, v in state_dict.items():
        transformed_state_dict[k.replace(".bn.", ".")] = v

    model.backbone.load_state_dict(transformed_state_dict, strict=False)

    # group weight and config optimizer
    base_lr = config.lr

    if engine.distributed:
        base_lr = config.lr  # * engine.world_size

    """ fix the weight of resnet"""
    for param in model.backbone.parameters():
        param.requires_grad = False

    params_list = []
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d, base_lr)

    optimizer = torch.optim.SGD(
        params_list,
        lr=base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    # optimizer = torch.optim.AdamW(params_list,
    #                             lr=base_lr,
    #                             # momentum=config.momentum,
    #                             weight_decay=config.weight_decay)

    if engine.distributed:
        print("distributed !!")
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()
    beg_epoch = engine.state.epoch
    # config lr policy
    total_iteration = (beg_epoch + config.nepochs) * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, epochs=beg_epoch + config.nepochs, steps_per_epoch=config.niters_per_epoch)

    # lr_finder = LRFinder(model, optimizer, criterion, device)
    # lr_finder.range_test(train_loader, end_lr=10, num_iter=1000)
    # lr_finder.plot()
    # plt.savefig("LRvsLoss.png")
    # plt.close()
    model.train()
    print("begin train")
    data_setting = {
        "img_root": config.img_root_folder,
        "gt_root": config.gt_root_folder,
        "hha_root": config.hha_root_folder,
        "mapping_root": config.mapping_root_folder,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
        "dataset_path": config.dataset_path,
    }

    beg_epoch = engine.state.epoch
    for epoch in range(beg_epoch, beg_epoch + config.nepochs):
        # if engine.distributed:
        #     train_sampler.set_epoch(epoch)
        bar_format = "{desc}[{elapsed}<{remaining},{rate_fmt}]"
        pbar = tqdm(
            range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format
        )
        dataloader = iter(train_loader)

        sum_loss = 0
        sum_sem = 0
        sum_com = 0
        sum_rest = 0
        sum_sketch = 0
        sum_sketch_gsnn = 0
        sum_kld = 0
        sum_sketch_raw = 0

        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)

            # no results dict here
            # if idx < 6100:
            #     continue
            # print(idx + 1)
            (
                loss,
                loss_semantic,
                loss_sketch_raw,
                loss_sketch,
                loss_sketch_gsnn,
                KLD,
                _,
            ) = process_one_batch(dataloader, model, config, engine)
            loss.backward()
            # print(' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch))
            #     # lr = scheduler.get_last_lr()[0]
            lr = lr_policy.get_lr(epoch * config.niters_per_epoch + idx)
            #     # lr = 0
            optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = lr
            sum_loss += loss.item()
            sum_sem += loss_semantic.item()
            sum_sketch_raw += loss_sketch_raw.item()
            sum_sketch += loss_sketch.item()
            sum_sketch_gsnn += loss_sketch_gsnn.item()
            # print(sum_sketch_gsnn,lr,sum_kld)
            sum_kld += KLD.item()
            optimizer.step()
            # scheduler.step()

            print_str = (
                "Epoch{}/{}".format(epoch, beg_epoch + config.nepochs)
                + " Iter{}/{}:".format(idx + 1, config.niters_per_epoch)
                + " lr=%.2e" % lr
                + " loss=%.5f" % (sum_loss / (idx + 1))
                + " kldloss=%.5f" % (sum_kld / (idx + 1))
                + " sum_sketch_raw=%.5f" % (sum_sketch_raw / (idx + 1))
                + " sum_sketch=%.5f" % (sum_sketch / (idx + 1))
                + " sum_sketch_gsnn=%.5f" % (sum_sketch_gsnn / (idx + 1))
            )

            pbar.set_description(print_str, refresh=False)
            step = epoch * config.niters_per_epoch + idx + 1
            cond_loss = (
                ((idx + 1) % 20 == 0)
                if config.dataset != "NYUv2"
                else ((idx + 1) % 50 == 0)
            )
            if cond_loss:
                logger.add_scalar("train_loss/tot", sum_loss / (idx + 1), step)
                logger.add_scalar("train_loss/semantic", sum_sem / (idx + 1), step)
                logger.add_scalar("train_loss/sketch", sum_sketch / (idx + 1), step)
                logger.add_scalar(
                    "train_loss/sketch_raw", sum_sketch_raw / (idx + 1), step
                )
                logger.add_scalar(
                    "train_loss/sketch_gsnn", sum_sketch_gsnn / (idx + 1), step
                )
                logger.add_scalar("train_loss/KLD", sum_kld / (idx + 1), step)
                logger.add_scalar("lr", optimizer.param_groups[0]["lr"], step)
            # # every 2000 iterations
            # pdb.set_trace()
            cond = ((step) % 2000 == 0) if config.dataset != "NYUv2" else epoch % 5 == 0
            # if cond:
            #     traindict = edict()
            #     traindict.dataloader = train_loader_mini
            #     traindict.loader_name = "train_mini"
            #     valdict = edict()
            #     valdict.dataloader = val_loader
            #     valdict.loader_name = "validation"
            #     # continue

            #     evaluate_epoch(
            #         args=None,
            #         network=None,
            #         model=model,
            #         loaders=[traindict, valdict],
            #         logger=logger,
            #         iter=step,
            #         save_examples=False,
            #     )

            #     engine.save_and_link_checkpoint(
            #         config.snapshot_dir,
            #         config.log_dir,
            #         config.log_dir_link,
            #         iter_num=step,
            #     )

        logger.add_scalar("train_loss_epoch/tot", sum_loss / len(pbar), epoch)
        logger.add_scalar("train_loss_epoch/semantic", sum_sem / len(pbar), epoch)
        logger.add_scalar("train_loss_epoch/sketch", sum_sketch / len(pbar), epoch)
        logger.add_scalar(
            "train_loss_epoch/sketch_raw", sum_sketch_raw / len(pbar), epoch
        )
        logger.add_scalar(
            "train_loss_epoch/sketch_gsnn", sum_sketch_gsnn / len(pbar), epoch
        )
        logger.add_scalar("train_loss_epoch/KLD", sum_kld / len(pbar), epoch)
        # logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        traindict = edict()
        traindict.dataloader = train_loader_mini
        traindict.loader_name = "train_mini"
        valdict = edict()
        valdict.dataloader = val_loader
        valdict.loader_name = "validation"
        evaluate_epoch(
            args=None,
            network=None,
            model=model,
            loaders=[traindict, valdict],
            logger=logger,
            iter=epoch,
            save_examples=False,
        )

        engine.save_and_link_checkpoint(
            config.snapshot_dir,
            config.log_dir,
            config.log_dir_link,
            iter_num=epoch,
        )
