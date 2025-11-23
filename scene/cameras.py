#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
import os
import pickle

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        y, x = torch.meshgrid(torch.arange(0, self.image_height, device='cuda'), torch.arange(0, self.image_width, device='cuda'))
        self.x = x.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
        
    def get_language_feature(self, language_feature_dir, feature_level):
        
        language_feature_name = os.path.join(language_feature_dir, self.image_name.split('.')[0])
        
        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))  # seg_map: torch.Size([4, 730, 988])
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy')) # feature_map: torch.Size([281, 512])
        seg_map = seg_map.cuda()
        feature_map = feature_map.cuda()
        
        seg = seg_map[:, self.y, self.x].squeeze(-1).long()
        mask = seg != -1
        if feature_level == 0: # default
            point_feature1 = feature_map[seg[0:1]].squeeze(0)
            mask = mask[0:1].reshape(1, self.image_height, self.image_width)
        elif feature_level == 1: # s
            point_feature1 = feature_map[seg[1:2]].squeeze(0)
            mask = mask[1:2].reshape(1, self.image_height, self.image_width)
        elif feature_level == 2: # m
            point_feature1 = feature_map[seg[2:3]].squeeze(0)
            mask = mask[2:3].reshape(1, self.image_height, self.image_width)
        elif feature_level == 3: # l
            point_feature1 = feature_map[seg[3:4]].squeeze(0)
            mask = mask[3:4].reshape(1, self.image_height, self.image_width)
        else:
            raise ValueError("feature_level=", feature_level)
        
        point_feature = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
       
        return point_feature, mask
    
    def get_vfm_language_feature(self, language_feature_dir, feature_level):
        
        language_feature_name = os.path.join(language_feature_dir, self.image_name.split('.')[0])
        # language_feature_name = self.language_feature_name
        # print("language_feature_dir: ", language_feature_name)
        # print("language_feature_dir: ", "/home/sen/projects/SuperGSeg/dataset/3dgs/lerf_ovs/teatime/language_features_dim3/frame_00099_s.npy")
        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
        
        # load instance feature
        y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        seg = seg_map[y, x].squeeze(-1).long()
        # seg = seg_map[feature_level:feature_level+1, y, x].squeeze(-1).long()
        
        mask = seg != -1
        
        point_feature1 = feature_map[seg].squeeze(0)
        mask = mask.reshape(1, self.image_height, self.image_width)
        
        point_feature = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
        return point_feature.cuda(), mask.cuda()

    def get_vfm_language_feature_optimized(self, language_feature_dir, feature_level):
        """Optimized version of get_vfm_language_feature
        - Moves seg_map and feature_map to GPU first
        - Reuses precomputed self.x/self.y on GPU
        - Supports optional multi-level seg_map by feature_level
        - Avoids creating large CPU tensors and host->device copies of large outputs
        """

        language_feature_name = os.path.join(language_feature_dir, self.image_name.split('.')[0])

        seg_map_np = np.load(language_feature_name + '_s.npy')
        feature_map_np = np.load(language_feature_name + '_f.npy')

        seg_map = torch.from_numpy(seg_map_np).cuda(non_blocking=True)
        feature_map = torch.from_numpy(feature_map_np).cuda(non_blocking=True)

        if seg_map.ndim == 3:
            gathered_seg_all_levels = seg_map[:, self.y, self.x].squeeze(-1).long()
            channel_index = int(feature_level) if 0 <= int(feature_level) < gathered_seg_all_levels.shape[0] else 0
            selected_seg_indices = gathered_seg_all_levels[channel_index]
            selected_mask = (selected_seg_indices != -1).reshape(1, self.image_height, self.image_width)
        else:
            selected_seg_indices = seg_map[self.y, self.x].squeeze(-1).long()
            selected_mask = (selected_seg_indices != -1).reshape(1, self.image_height, self.image_width)

        selected_point_features = feature_map[selected_seg_indices]
        point_feature = selected_point_features.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)

        return point_feature, selected_mask
    
    
        
class ScanNetCamera(Camera):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid, language_feature_name, instance_mask_name, correlation_name, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(ScanNetCamera, self).__init__(resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                                            image_name, uid, 
                                            trans, scale, data_device, train_test_exp, is_test_dataset, is_test_view)

        self.language_feature_name = language_feature_name
        self.instance_mask_name = instance_mask_name
        self.correlation_name = correlation_name
        
    def get_vfm_language_feature(self, language_feature_dir, feature_level):
        
        language_feature_name = os.path.join(language_feature_dir, self.image_name.split('.')[0])
        # language_feature_name = self.language_feature_name
        # print("language_feature_dir: ", language_feature_name)
        # print("language_feature_dir: ", "/home/sen/projects/SuperGSeg/dataset/3dgs/lerf_ovs/teatime/language_features_dim3/frame_00099_s.npy")
        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
        
        # load instance feature
        y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        seg = seg_map[y, x].squeeze(-1).long()
        # seg = seg_map[feature_level:feature_level+1, y, x].squeeze(-1).long()
        
        mask = seg != -1
        
        point_feature1 = feature_map[seg].squeeze(0)
        mask = mask.reshape(1, self.image_height, self.image_width)
        
        point_feature = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
        return point_feature.cuda(), mask.cuda()
    
    def get_scannet_language_feature(self, language_feature_dir, feature_level):
        
        # language_feature_name = os.path.join(language_feature_dir, self.image_name.split('.')[0])
        language_feature_name = self.language_feature_name
        # print("language_feature_dir: ", language_feature_name)
        # print("language_feature_dir: ", "/home/sen/projects/SuperGSeg/dataset/3dgs/lerf_ovs/teatime/language_features_dim3/frame_00099_s.npy")
        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
        
        # load instance feature
        y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        seg = seg_map[y, x].squeeze(-1).long()
        # seg = seg_map[feature_level:feature_level+1, y, x].squeeze(-1).long()
        
        mask = seg != -1
        
        point_feature1 = feature_map[seg].squeeze(0)
        mask = mask.reshape(1, self.image_height, self.image_width)
        
        point_feature = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
        return point_feature.cuda(), mask.cuda()
        
        
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

