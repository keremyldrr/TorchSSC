#!/usr/bin/env python3
# encoding: utf-8
from distutils.log import error
from logging import warning
import numpy as np
import torch
from datasets.BaseDataset import BaseDataset
import os
import cv2
import io
import pandas as pd
from io import BytesIO
#     import sysxxx
# sys.path.append(os.path.abspath("."))
# from utils import tsdf_from_depth,get_mapping ,box_filter_label_mapping,frame_to_grid, get_pcd_from_depth,get_grid_to_camera,get_instance_boxes,get_axis_aligned_matrix,get_camera_pose,get_inside_grid,get_pcd_from_depth,get_points_inside_boxes
from sc_utils import get_label_bbox, get_camera_pose,get_pcd_from_depth,get_grid_to_camera,get_instance_boxes,get_points_inside_boxes,frame_to_grid,get_mapping,tsdf_from_depth,get_points_inside_boxes,box_filter_label_mapping,get_axis_aligned_matrix

import time
class ScanNet(BaseDataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None, s3client=None,only_frustum=False, only_box=False):
        super(ScanNet, self).__init__(setting, split_name, preprocess, file_length)
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._hha_path = setting['hha_root']
        self._path = setting['dataset_path']
        self._mappiing_path = setting['mapping_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = len(self._file_names) if file_length is not None else file_length
        self.preprocess = preprocess
        self.s3client = s3client
        self.only_frustum = only_frustum
        self.only_box = only_box

    def read_ceph_img(self, mode, value):
        img_array = np.fromstring(value, dtype=np.uint8)
        img = cv2.imdecode(img_array, mode)
        return img

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
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            item = item.strip()
            item = item.split('\t')
            img_name = item[0]
            file_names.append([img_name, None])

        return file_names


    def __getitem__(self, index):
        st = time.time()
        # if self._file_length is not None:
        #     names = self._construct_new_file_names(self._file_length)[index]
        # else:
        # print(index,len(self._file_names))
        # if self._split_name is "val":
        #     print("idx",index)
        names = self._file_names[index]
        

        item_idx = names[0]
        try:
            scene_name = item_idx[:item_idx.rfind("_")]
            formatted = format(100*int(item_idx[item_idx.rfind("_")+1:]),"06d")
            unformatted = item_idx[item_idx.rfind("_")+1:]
            
            # processed_items = open("processed.txt","a")
            # processed_items.write(item_idx+"\n")
            # processed_items.close()


            img_path = os.path.join(self._path,scene_name, 'RGB', 'rgb'+unformatted+'.png')
            hha_path = os.path.join(self._path,scene_name, "depth/" + formatted+'.png')
            gt_path = os.path.join(self._path, scene_name,'Label/label'+unformatted+'.npz')
            label_weight_path = os.path.join(self._path,scene_name, 'TSDF/tsdf'+unformatted+'.npz')
            mapping_path = os.path.join(self._path, scene_name,'Mapping/mapping'+unformatted+'.npz')
            sketch_path = os.path.join(self._path, scene_name,'sketch3d/sketch'+unformatted+'.npz.npy')
            instancedir = os.path.join("../../DATA/ScanNet/",scene_name,"PCD","instances_{}".format(unformatted))
            path_to_depth_intr = os.path.join("../../DATA/scannet_frames_25k/",scene_name,"intrinsics_depth.txt")
            poses = sorted(os.listdir(os.path.join("../../DATA/scannet_frames_25k/",scene_name, "pose/")))
            depths = sorted(os.listdir(os.path.join("../../DATA/scannet_frames_25k/",scene_name, "depth/")))
            pose_path = os.path.join("../../DATA/scannet_frames_25k/",scene_name, "pose/" + poses[int(unformatted)])
            hha_path = os.path.join("../../DATA/scannet_frames_25k/",scene_name, "depth/" +  depths[int(unformatted)])

            item_name = item_idx

            axis_align_matrix = get_axis_aligned_matrix(scene_name)

            # label_weight_path = "   nsdkfjhdssa"
            img, hha, tsdf, label_weight, depth_mapping_3d, gt, sketch_gt = self._fetch_data(img_path, hha_path, label_weight_path, mapping_path, gt_path,sketch_path,instancedir,path_to_depth_intr,pose_path,axis_align_matrix)
            # print("KMG",img)
            # print(np.unique(gt))
            # gt[gt > 12]=0
            img = img[:, :, ::-1]
            if self.preprocess is not None:
                img, extra_dict = self.preprocess(img, hha)         # normalization

            # if self._split_name is 'train':
            #     img = torch.from_numpy(np.ascontiguousarray(img)).float()
            #     gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            #     sketch_gt = torch.from_numpy(np.ascontiguousarray(sketch_gt)).long()
            #     depth_mapping_3d = torch.from_numpy(np.ascontiguousarray(depth_mapping_3d)).long()

            #     label_weight = torch.from_numpy(np.ascontiguousarray(label_weight)).float()
            #     tsdf = torch.from_numpy(np.ascontiguousarray(tsdf)).float()

            #     if self.preprocess is not None and extra_dict is not None:
            #         for k, v in extra_dict.items():
            #             extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
            #             if 'label' in k:
            #                 extra_dict[k] = extra_dict[k].long()
            #             if 'img' in k:
            #                 extra_dict[k] = extra_dict[k].float()

            output_dict = dict(data=img, label=gt, label_weight=label_weight, depth_mapping_3d=depth_mapping_3d,
                            tsdf=tsdf, sketch_gt=sketch_gt, fn=str(item_name), n=len(self._file_names))
            if self.preprocess is not None and extra_dict is not None:
                output_dict.update(**extra_dict)
            # print(time.time() - st)
        except Exception as error:
            print("Problem with  ",item_idx)
            print(error)
            
        return output_dict

    def _fetch_data(self, img_path, hha_path, label_weight_path, mapping_path, gt_path,sketch_path,instancedir,path_to_depth_intr,pose_path,axis_align_matrix, dtype=None):
        img = np.array(cv2.imread(img_path), dtype=np.float32)
        assert img is not None , img_path
        hha = np.array(cv2.imread(hha_path,-1), dtype=np.float32)

        # assert (hha is not None) & np.isnan(hha) == False , hha_path
        hha = hha / 1000
        # print(img_path)
        grid_shape = np.array([60,36,60])
        VOX_SIZE = 0.08
        TRUNCATION = 0.24
        tsdf = np.load(label_weight_path)['arr_0'].astype(np.float32)#.reshape(1, 60, 36, 60)
        label_weight = np.load(label_weight_path)['arr_1'].astype(np.float32)
        depth_mapping_3d = np.load(mapping_path)['arr_0'].astype(np.int64)
        gt = np.load(gt_path)['arr_0'].flatten().astype(np.int64)
        sketch_gt = np.load(sketch_path).astype(np.int64).flatten()
        if self.only_frustum and not self.only_box:
            # label_weight[tsdf==0 ]= 0
            

            #label# getting only inside of the frustum
            gt[(tsdf == 0) |  (tsdf  == 1) ] = 0
            # print("dsfdsf")
            gt = get_label_bbox(gt,grid_shape)
            label_weight[(tsdf == 0) | (tsdf  == 1) & (gt == 255) ] = 0
            # gt[(tsdf == 0) | (tsdf > 0) | (tsdf  == -1) ]= 255# getting only inside of the frustum
#            print("Only FRUSTUM!!!!")
            sketch_gt[gt == 255] = 0
            # if self.only_box:
            #    raise NotImplementedError(" Only box cant be enabled with frustum")

            
        
        if self.only_box and not self.only_frustum:

            camera_pose = get_camera_pose(pose_path)

            depth_intrinsic = pd.read_csv(path_to_depth_intr,header=None,delimiter=" ").values[:,:-1]
            depth_image = hha
            pts3d, valid_depth_inds = get_pcd_from_depth(depth_image=depth_image,depth_intrinsic=depth_intrinsic)
            transform = np.matmul(axis_align_matrix,camera_pose)
            rot = transform[:3,:3]
            tl = transform[:-1,3]
            points = np.matmul(rot,pts3d[valid_depth_inds].T).T + tl
            worldToCam = get_grid_to_camera(camera_pose=camera_pose,axis_align_matrix=axis_align_matrix)

            boxes = get_instance_boxes(instancedir)
            box_mask = get_points_inside_boxes(points=points,boxes=boxes)
            points,ptrans = frame_to_grid(pts=points,grid_shape=grid_shape,VOX_SIZE=VOX_SIZE)
            points = points[box_mask]
            img_inds = np.indices(depth_image.shape).reshape(2, -1).T
            grid_inds = np.indices(grid_shape).reshape(3, -1).T
            f1 = img_inds[valid_depth_inds]
            imxs = f1[box_mask][:,0]
            imys = f1[box_mask][:,1]

            pts2d = imxs * depth_image.shape[1] + imys 
            im=np.zeros_like(depth_image)
            im[imxs,imys] = 1
            a = np.zeros_like(depth_image.flatten())
            a[imxs * depth_image.shape[1] + imys ] = 1
            a = a.reshape(depth_image.shape)
            pts2d = (imxs * depth_image.shape[1] + imys)#[valid_depth_indsbox_mask] 
            depth_mapping_3d = get_mapping(points,pts2d,grid_shape)
            depth_image_box_filtered = a*depth_image
            tsdf = tsdf_from_depth(grid_shape,VOX_SIZE,TRUNCATION,depth_intrinsic,depth_image_box_filtered,worldToCam,ptrans,upshift=0)

            grid_mask = get_points_inside_boxes(grid_inds,box_filter_label_mapping(boxes,VOX_SIZE,ptrans))
            # t= trimesh.points.PointCloud(grid_inds[grid_mask]).export("ksf.ply")
            # print(depth_mapping_3d.shape)

            gt[grid_mask == 0 ] = 255
            # depth_mapping_3d[grid_mask == 0 ] = 307200

            
            label_weight = grid_mask.astype(int)
        if self.only_frustum and self.only_box:
            camera_pose = get_camera_pose(pose_path)

            depth_intrinsic = pd.read_csv(path_to_depth_intr,header=None,delimiter=" ").values[:,:-1]
            depth_image = hha
            pts3d, valid_depth_inds = get_pcd_from_depth(depth_image=depth_image,depth_intrinsic=depth_intrinsic)
            transform = np.matmul(axis_align_matrix,camera_pose)
            rot = transform[:3,:3]
            tl = transform[:-1,3]
            points = np.matmul(rot,pts3d[valid_depth_inds].T).T + tl
            worldToCam = get_grid_to_camera(camera_pose=camera_pose,axis_align_matrix=axis_align_matrix)

            boxes = get_instance_boxes(instancedir)
            box_mask = get_points_inside_boxes(points=points,boxes=boxes)
            points,ptrans = frame_to_grid(pts=points,grid_shape=grid_shape,VOX_SIZE=VOX_SIZE)
            points = points[box_mask]
            img_inds = np.indices(depth_image.shape).reshape(2, -1).T
            grid_inds = np.indices(grid_shape).reshape(3, -1).T
            f1 = img_inds[valid_depth_inds]
            imxs = f1[box_mask][:,0]
            imys = f1[box_mask][:,1]

            pts2d = imxs * depth_image.shape[1] + imys 
            im=np.zeros_like(depth_image)
            im[imxs,imys] = 1
            a = np.zeros_like(depth_image.flatten())
            a[imxs * depth_image.shape[1] + imys ] = 1
            a = a.reshape(depth_image.shape)
            pts2d = (imxs * depth_image.shape[1] + imys)#[valid_depth_indsbox_mask] 
            depth_mapping_3d = get_mapping(points,pts2d,grid_shape)
            depth_image_box_filtered = a*depth_image
            tsdf = tsdf_from_depth(grid_shape,VOX_SIZE,TRUNCATION,depth_intrinsic,depth_image_box_filtered,worldToCam,ptrans,upshift=0)

            grid_mask = get_points_inside_boxes(grid_inds,box_filter_label_mapping(boxes,VOX_SIZE,ptrans))
            # t= trimesh.points.PointCloud(grid_inds[grid_mask]).export("ksf.ply")
            # print(depth_mapping_3d.shape)

            # depth_mapping_3d[grid_mask == 0 ] = 307200

            gt[(tsdf == 0) |  (tsdf  == 1) ] = 0
            # print("dsfdsf")
            gt = get_label_bbox(gt,grid_shape)
            label_weight[(tsdf == 0) | (tsdf  == 1) & (gt == 255) ] = 0
            # gt[(tsdf == 0) | (tsdf > 0) | (tsdf  == -1) ]= 255# getting only inside of the frustum
#            print("Only FRUSTUM!!!!")
            sketch_gt[gt == 255] = 0
            gt[grid_mask == 0 ] = 255

            label_weight = grid_mask.astype(int)
        
        
        label_weight = (depth_mapping_3d != 307200).astype(int)

        # sketch_gt = np.load(gt_path.replace('Label', 'sketch3D').replace('npz', 'npy')).astype(np.int64)
        # sketch_gt[tsdf==0]=0
        return img, hha, tsdf.reshape(1,60, 36, 60), label_weight, depth_mapping_3d, gt, sketch_gt.reshape(1,60, 36, 60)

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 19
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
