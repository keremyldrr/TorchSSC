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
from config import config, update_parameters_in_config
from network import Network
from scannet import ScanNetSSCDataModule
from nyu import NYUDataModule
import matplotlib.pyplot as plt
from torch.nn import BatchNorm3d, BatchNorm2d
from easydict import EasyDict as edict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
import torch


from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    DeviceStatsMonitor,
)
from pytorch_lightning.strategies import DDPStrategy

parser = argparse.ArgumentParser()
# port = str(int(float(time.time())) % 20)
# os.environ["MASTER_PORT"] = str(10097 + int(port))
parser.add_argument("--only_frustum", action="store_true")
parser.add_argument("--only_boxes", action="store_true")
parser.add_argument("--overfit", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--prefix", type=str, default="dummy")
parser.add_argument("--ckpt", type=str, default="dummy")
parser.add_argument("--dataset", type=str, default="NYUv2")
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--ew", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.1)
args = parser.parse_args()
logger = MLFlowLogger(experiment_name="Eval", run_name=args.prefix)
update_parameters_in_config(
    config,
    ew=args.ew,
    lr=args.lr,
    num_epochs=args.num_epochs,
    only_frustum=args.only_frustum,
    only_boxes=args.only_boxes,
    dataset=args.dataset,
    prefix=args.prefix,
    overfit=args.overfit,
)
config.num_classes = 3
drv.init()
print("%d device(s) found." % drv.Device.count())
print(config)
for ordinal in range(drv.Device.count()):
    dev = drv.Device(ordinal)
    print(ordinal, dev.name())
if config.dataset == "NYUv2":
    data_module = NYUDataModule(config)
else:
    data_module = ScanNetSSCDataModule(config)

config.steps_per_epoch = len(data_module.train_dataloader())
if args.ckpt == "dummy":
    model = Network(
        class_num=config.num_classes,
        feature=128,
        bn_momentum=config.bn_momentum,
        pretrained_model=config.pretrained_model,
        norm_layer=BatchNorm3d,
        config=config,
    )
else:
    model = Network(
        class_num=config.num_classes,
        feature=128,
        bn_momentum=config.bn_momentum,
        pretrained_model=config.pretrained_model,
        norm_layer=BatchNorm3d,
        config=config,
    )
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt["state_dict"])
model.debug = args.debug
lr_monitor = LearningRateMonitor(logging_interval="step")
# TODO: add ckpt and metrics
check_interval = 3
ckpt = ModelCheckpoint(
    every_n_epochs=check_interval,
    save_top_k=2,
    verbose=True,
    mode="max",
    monitor="val/sscmIOU",
)
# TODO Distributed evaluation as well
trainer = pl.Trainer(
    auto_lr_find=False,
    devices=1,
    strategy=DDPStrategy(),
    # logger=mlflow_logger,
    max_epochs=config.nepochs,
    accelerator="gpu",
    check_val_every_n_epoch=3 if config.dataset == "NYUv2" else 1,
    val_check_interval=None if config.dataset == "NYUv2" else 500,
    limit_val_batches=0.25,
    log_every_n_steps=50,
    callbacks=[lr_monitor, ckpt, ModelSummary()],
    logger=logger,
    # overfit_batches=1,  # if not args.overfit else 1,
    # track_grad_norm=2,
    detect_anomaly=True,
)

trainer.validate(model, datamodule=data_module)  # train(FLAGS)
