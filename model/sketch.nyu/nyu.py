#!/usr/bin/env python3
# encoding: utf-8
import numpy as np
import torch

# from datasets.BaseDataset import BaseDataset
import os
import cv2
import io
from io import BytesIO
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from utils.img_utils import normalize


class NYUv2:
    def __init__(
        self,
        setting,
        split_name,
        preprocess=None,
        file_length=None,
        s3client=None,
        only_frustum=False,
        only_box=False,
    ):
        # super(NYUv2, self).__init__(setting, split_name, preprocess, file_length)
        self._split_name = split_name
        self._img_path = setting["img_root"]
        self._gt_path = setting["gt_root"]
        self._hha_path = setting["hha_root"]
        self._mappiing_path = setting["mapping_root"]
        self._train_source = setting["train_source"]
        self._eval_source = setting["eval_source"]
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.s3client = s3client
        self.only_frustum = only_frustum
        self.only_box = only_box

    def read_ceph_img(self, mode, value):
        img_array = np.fromstring(value, dtype=np.uint8)
        img = cv2.imdecode(img_array, mode)
        return img

    def __len__(self):
        return len(self._file_names)

    def read_ceph_npz(self, value):
        f = BytesIO(value)
        data = np.load(f)
        return data

    def read_mc_img(self, mode, filename):
        mclient.Get(filename, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, mode)
        return img

    def read_mc_npz(self, filename):
        mclient.Get(filename, value)
        value_buf = mc.ConvertBuffer(value)
        value_buf = io.BytesIO(value_buf)
        array = np.load(value_buf)
        return array

    def _get_file_names(self, split_name):
        assert split_name in ["train", "val"]
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            item = item.strip()
            item = item.split("\t")
            img_name = item[0]
            file_names.append([img_name, None])

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)  # 原来一轮迭代的长度
        new_file_names = self._file_names * (length // files_len)  # 按照设定获得的一轮迭代的长度

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[: length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def __getitem__(self, index):

        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]

        item_idx = names[0]
        img_path = os.path.join(self._img_path, "RGB", "NYU" + item_idx + "_colors.png")
        hha_path = os.path.join(self._hha_path, item_idx + ".png")
        gt_path = os.path.join(self._gt_path, "Label/" + item_idx + ".npz")
        label_weight_path = os.path.join(self._img_path, "TSDF/" + item_idx + ".npz")
        mapping_path = os.path.join(self._mappiing_path, item_idx + ".npz")
        item_name = item_idx

        (
            img,
            hha,
            tsdf,
            label_weight,
            depth_mapping_3d,
            gt,
            sketch_gt,
        ) = self._fetch_data(
            img_path, hha_path, label_weight_path, mapping_path, gt_path
        )

        img = img[:, :, ::-1]
        if self.preprocess is not None:
            img, extra_dict = self.preprocess(img, hha)  # normalization

        if self._split_name is "train":
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            sketch_gt = torch.from_numpy(np.ascontiguousarray(sketch_gt)).long()
            depth_mapping_3d = torch.from_numpy(
                np.ascontiguousarray(depth_mapping_3d)
            ).long()

            label_weight = torch.from_numpy(np.ascontiguousarray(label_weight)).float()
            # label_weight = torch.ones_like(label_weight).float() #added by kerem
            tsdf = torch.from_numpy(np.ascontiguousarray(tsdf)).float()

            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if "label" in k:
                        extra_dict[k] = extra_dict[k].long()
                    if "img" in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(
            data=img,
            label=gt,
            label_weight=label_weight,
            depth_mapping_3d=depth_mapping_3d,
            tsdf=tsdf,
            sketch_gt=sketch_gt,
            fn=str(item_name),
            n=len(self._file_names),
        )
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(
        self, img_path, hha_path, label_weight_path, mapping_path, gt_path, dtype=None
    ):
        img = np.array(cv2.imread(img_path), dtype=np.float32)
        hha = np.array(cv2.imread(hha_path), dtype=np.float32)
        tsdf = np.load(label_weight_path)["arr_0"].astype(np.float32)
        label_weight = np.load(label_weight_path)["arr_1"].astype(np.float32)
        depth_mapping_3d = np.load(mapping_path)["arr_0"].astype(np.int64)
        gt = np.load(gt_path)["arr_0"].astype(np.int64)
        sketch_gt = np.load(
            gt_path.replace("Label", "sketch3D").replace("npz", "npy")
        ).astype(np.int64)
        if self.only_frustum:
            # print("Only FRUSTUM!!!!")
            f = 8
        else:
            label_weight = np.ones_like(label_weight)
            # print("Full scene!!!!")

        return (
            img,
            hha,
            tsdf.reshape(1, 60, 36, 60),
            label_weight,
            depth_mapping_3d,
            gt,
            sketch_gt.reshape(1, 60, 36, 60),
        )

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 13
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors


class TrainPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, hha):
        img = normalize(img, self.img_mean, self.img_std)
        # hha = normalize(hha, self.img_mean, self.img_std)

        p_img = img.transpose(2, 0, 1)
        p_hha = hha  # ,p.transpose(2, 0, 1)

        extra_dict = {"hha_img": p_hha}  # TODO, normlaize depth

        return p_img, extra_dict


class ValPre(object):
    def __call__(self, img, hha):
        extra_dict = {"hha_img": img}  # TODO, depth is useless now()

        return img, extra_dict


class NYUDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        data_setting = {
            "img_root": config.img_root_folder,
            "gt_root": config.gt_root_folder,
            "hha_root": config.hha_root_folder,
            "mapping_root": config.mapping_root_folder,
            "train_source": config.train_source,
            "eval_source": config.eval_source,
            "dataset_path": config.dataset_path,
        }
        train_preprocess = TrainPre(config.image_mean, config.image_std)

        val_preprocess = TrainPre(config.image_mean, config.image_std)
        self.train_dataset = NYUv2(
            data_setting,
            "train",
            train_preprocess,
            config.batch_size * config.niters_per_epoch,
            s3client=None,
            only_frustum=config.only_frustum,
            only_box=config.only_boxes,
        )

        self.val_dataset = NYUv2(
            data_setting,
            "val",
            val_preprocess,
            1 * config.niters_per_epoch,
            s3client=None,
            only_frustum=config.only_frustum,
            only_box=config.only_boxes,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,  # config.batch_size,
            num_workers=config.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
        )
        self.config = config
        # self.hparams = args

    @staticmethod
    def add_data_loader_arguments(parent_parser):
        # parser = parent_parser.add_argument_group("ScanNetLoader")
        # parser.add_argument("--batch_size", type=int, default=12)
        # parser.add_argument("--sigma", type=float, default=0.0)
        # parser.add_argument("--num_points", default=40000)
        # parser.add_argument("--use_color", action="store_true")
        # parser.add_argument("--use_height", action="store_true")
        # parser.add_argument("--augment", action="store_true")
        # parser.add_argument("--overfit", action="store_true")
        # return parent_parser
        pass

    def train_dataloader(self):

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        return [
            DataLoader(
                self.val_dataset,
                batch_size=1,  # self.config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            ),
            # self.train_subset_dataloader(),
        ]

    # TODO  Implement smaller train_mini and validation, maybe just alter the filename list
    def train_subset_dataloader(self):
        return DataLoader(
            self.train_subset_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
