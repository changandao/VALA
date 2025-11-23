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

# from regex import F
import torch
import numpy as np
import torch.nn.functional as F
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture_rgb(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def capture_language_feature(self):
        # Capture standard features if available
        if hasattr(self, "_language_feature"):
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._language_feature,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        # Capture robust features if available
        elif hasattr(self, "_language_feature_robust"):
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._language_feature_robust,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
            raise ValueError("No language features available to capture")
    
    def restore_rgb(self, model_args, training_args, mode='train'):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        
        if mode == 'train':
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
        
        
        
    def restore_language_features(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._language_feature,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        
    def restore_language_features_robust(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._language_feature_robust,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
                
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_language_feature(self):
        # Return standard features if available
        if hasattr(self, "_language_feature") and self._language_feature is not None:
            # Hard normalization
            return torch.nn.functional.normalize(self._language_feature, dim=-1)
        # Return robust features if available
        elif hasattr(self, "_language_feature_robust") and self._language_feature_robust is not None:
            # Robust features are already normalized in finalize_gaussian_features_robust
            return self._language_feature_robust
        else:
            raise ValueError('Language feature has not been set')
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_covariance_inverse_from_gaussian_model(self, scaling_modifier=1.0):
        """
        从GaussianModel获取协方差矩阵的逆
        
        Args:
            gaussian_model: GaussianModel实例
            scaling_modifier: 缩放修正因子
        
        Returns:
            Sigma_inv: (N, 3, 3) 协方差矩阵的逆
        """
        # 1. 获取6维紧凑表示的协方差矩阵
        cov_6d = self.get_covariance(scaling_modifier)  # (N, 6)
        
        # 2. 将6维表示转换为完整的3x3协方差矩阵
        N = cov_6d.shape[0]
        Sigma = torch.zeros((N, 3, 3), device=cov_6d.device, dtype=cov_6d.dtype)
        
        # 根据strip_lowerdiag的格式填充3x3矩阵
        Sigma[:, 0, 0] = cov_6d[:, 0]  # (0,0)
        Sigma[:, 0, 1] = cov_6d[:, 1]  # (0,1)
        Sigma[:, 0, 2] = cov_6d[:, 2]  # (0,2)
        Sigma[:, 1, 0] = cov_6d[:, 1]  # (1,0) = (0,1) 对称
        Sigma[:, 1, 1] = cov_6d[:, 3]  # (1,1)
        Sigma[:, 1, 2] = cov_6d[:, 4]  # (1,2)
        Sigma[:, 2, 0] = cov_6d[:, 2]  # (2,0) = (0,2) 对称
        Sigma[:, 2, 1] = cov_6d[:, 4]  # (2,1) = (1,2) 对称
        Sigma[:, 2, 2] = cov_6d[:, 5]  # (2,2)
        
        # 3. 计算逆矩阵
        Sigma_inv = torch.linalg.inv(Sigma)  # (N, 3, 3)
        
        return Sigma_inv.cpu().detach().numpy()
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)


        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if hasattr(self, "tmp_radii"): 
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, width, height):
        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]
        # Normalize the gradient to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    def accumulate_gaussian_feature_per_view(self, gt_language_feature, gt_mask, mask, significance=None, means2D=None, feature_dim=512):

        if not hasattr(self, "_language_feature"):
            self.create_language_features(feature_dim=feature_dim)
            self._activated_views = torch.zeros((self._xyz.shape[0]), device="cuda")

        batch_y = torch.clamp(means2D[:, 1], max=gt_language_feature.shape[0] - 1)
        batch_x = torch.clamp(means2D[:, 0], max=gt_language_feature.shape[1] - 1)
        gt_batch_features = gt_language_feature[batch_y.long(), batch_x.long()]
        gt_batch_mask = gt_mask[batch_y.long(), batch_x.long()] # (N,)
        
        if not hasattr(self, "_language_feature_weight"):
            self._language_feature_weight = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        self._language_feature_weight[mask] += significance.unsqueeze(-1) * gt_batch_mask.unsqueeze(-1)
        self._language_feature[mask] += significance.unsqueeze(-1) * gt_batch_features * gt_batch_mask.unsqueeze(-1)

        self._activated_views[mask] = torch.where(
            gt_batch_mask,
            1,
            self._activated_views[mask]
        )
        
    def accumulate_gaussian_feature_per_view_gate(self, gt_language_feature, gt_mask, mask, significance=None, means2D=None, 
                                            feature_dim=512, 
                                            ablation_type="ablation_none",
                                            tau_mass=0.95,
                                            tau_abs=0.03,
                                            k_max=10000,
                                            k_target=15):
    
        if not hasattr(self, "_language_feature"):
            self.create_language_features(feature_dim=feature_dim)
            self._activated_views = torch.zeros((self._xyz.shape[0]), device="cuda")

        batch_y = torch.clamp(means2D[:, 1], max=gt_language_feature.shape[0] - 1)
        batch_x = torch.clamp(means2D[:, 0], max=gt_language_feature.shape[1] - 1)
        gt_batch_features = gt_language_feature[batch_y.long(), batch_x.long()]
        valid = gt_mask[batch_y.long(), batch_x.long()] # (N,)
        
        valid = gt_mask[batch_y.long(), batch_x.long()] #& mask
        total_points = significance.numel()
        # total_points = int(total_points)
        # # 使用自适应门控替代硬编码阈值
        # # 参数可以根据需要调整：
        # # tau_mass: 质量覆盖阈值 (建议 0.90-0.95)
        # # tau_abs: 绝对地板阈值 (建议 0.02-0.05) 
        # # k_max: 最大保留高斯数 (建议 4-8)
        # # k_target: 目标平均保留高斯数 (建议 1.5-2.5)
        w_mask_adaptive = self.adaptive_gating(
            significance, 
            lambda_val=None,   # 手动设定较小的lambda值
            tau_mass=tau_mass,    # 质量覆盖85% (更宽松)
            tau_abs=tau_abs,    # 绝对地板0.5% (非常宽松) ##best case 0.13
            k_max=0.15*total_points,         # 最多保留30个高斯
            k_target=15     # 目标平均15个高斯  # no use
        ).view(-1, 1)
        
        # print(f"w_mask_adaptive: {w_mask_adaptive.sum().item()}")
        # 也提供一个基于分位数的简单自适应方法作为对比
        # 动态调整分位数：如果点太少就降低分位数
        
        w_mask_mass = self._mass_coverage_gating(significance, 0.35)
        
        # print(f"w_mask_mass: {w_mask_mass.sum().item()}")
        
        if total_points > 100:
            quantile_threshold = 0.85  # 保留top 15%
        elif total_points > 50:
            quantile_threshold = 0.80  # 保留top 20%
        else:
            quantile_threshold = 0.70  # 保留top 30%
            
        w_mask_quantile_adaptive = significance.view(-1, 1) > significance.quantile(quantile_threshold)
        
        
        # 比较不同方法的效果
        w_mask_hard = significance.view(-1, 1) > 0.10
        w_mask_quantile = significance.view(-1, 1) > significance.quantile(0.85)
        if w_mask_hard.sum() > w_mask_quantile.sum():
            w_final_mask = w_mask_hard
        else:
            w_final_mask = w_mask_quantile
        # w_mask_mass = w_mask_hard.squeeze() & w_mask_mass.squeeze() & w_mask_quantile_adaptive.squeeze()
        
        # print(f"Total points: {significance.numel()}")
        # print(f"Adaptive gating count: {w_mask_adaptive.sum().item()}")
        # print(f"mass coverage gate: {w_mask_mass.sum().item()}")
        # print(f"Quantile adaptive count: {w_mask_quantile_adaptive.sum().item()}")
        # print(f"Hard threshold count: {w_mask_hard.sum().item()}")  
        # print(f"Fixed quantile count: {w_mask_quantile.sum().item()}")
        valid = valid & w_final_mask.squeeze()
        
        if not hasattr(self, "_language_feature_weight"):
            self._language_feature_weight = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        self._language_feature_weight[mask] += significance.unsqueeze(-1) * valid.unsqueeze(-1)
        self._language_feature[mask] += significance.unsqueeze(-1) * gt_batch_features * valid.unsqueeze(-1)

        self._activated_views[mask] = torch.where(
            valid,
            1,
            self._activated_views[mask]
        )
        
        
    def create_language_features(self, feature_dim=512):
        self._language_feature = nn.Parameter(
            torch.zeros((self._xyz.shape[0], feature_dim), dtype=torch.float, device="cuda").requires_grad_(True)
        )

    def finalize_gaussian_features(self):
        self._language_feature = self._language_feature / (self._language_feature_weight + 1e-15)
        
        # Filtering
        prune_mask = self._activated_views == 0
        self.prune_points(prune_mask)
        self._language_feature = self._language_feature[~prune_mask]
        
        print("Pruned {} points".format(prune_mask.sum()))

    def get_language_feature(self):
        return self._language_feature
    
    def _mass_coverage_gating(self, significance, tau_mass=0.95):
        """
        质量覆盖门控
        
        根据权重的累积分布，保留权重总和达到tau_mass比例的高斯
        
        Args:
            significance: 权重张量 (N,)
            tau_mass: 质量覆盖阈值 (0.90-0.95)
            
        Returns:
            s_mass: 质量覆盖门控掩码 (N,) bool tensor
        """
        if significance.numel() == 0:
            return torch.zeros_like(significance, dtype=torch.bool)
            
        # 按权重降序排序
        sorted_weights, sorted_indices = torch.sort(significance, descending=True)
        cumsum_weights = torch.cumsum(sorted_weights, dim=0)
        total_weight = significance.sum()
        
        if total_weight > 0:
            # 找到累积权重达到tau_mass的最小索引
            mass_threshold_idx = torch.where(cumsum_weights >= tau_mass * total_weight)[0]
            if len(mass_threshold_idx) > 0:
                mass_cutoff = mass_threshold_idx[0] + 1  # +1因为索引从0开始
                s_mass = torch.zeros_like(significance, dtype=torch.bool)
                s_mass[sorted_indices[:mass_cutoff]] = True
            else:
                s_mass = torch.ones_like(significance, dtype=torch.bool)
        else:
            s_mass = torch.zeros_like(significance, dtype=torch.bool)
            
        # 输出对应的 significance 值，便于调试/观察
        try:
            selected_significance = significance[s_mass]
            print("[MassCoverage] selected count:", int(s_mass.sum().item()))
            # 仅打印统计信息，避免日志过长
            if selected_significance.numel() > 0:
                ss = selected_significance.detach()
                if ss.is_cuda:
                    ss = ss.cpu()
                print(
                    "[MassCoverage] significance stats -> min: {:.6f}, max: {:.6f}, mean: {:.6f}".format(
                        float(ss.min().item()), float(ss.max().item()), float(ss.mean().item())
                    )
                )
        except Exception:
            pass

        return s_mass

    def adaptive_gating(self, significance, 
                       lambda_val=None, 
                       tau_mass=0.95, 
                       tau_abs=0.03, 
                       k_max=6,
                       k_target=2.0):
        """
        自适应门控机制
        
        Args:
            significance: 权重张量 (N,) 
            lambda_val: 射线相对阈值参数，如果为None则自动确定
            tau_mass: 质量覆盖阈值 (0.90-0.95)
            tau_abs: 绝对地板阈值 (0.02-0.05)
            k_max: 最大保留高斯数量 (4-8)
            k_target: 目标平均保留高斯数 (1.5-2.5)
            
        Returns:
            mask: 门控掩码 (N,) bool tensor
        """
        if not isinstance(k_max, int):
            k_max = int(k_max)
        if not isinstance(k_target, int):
            k_target = int(k_target)
            
        if significance.numel() == 0:
            return torch.zeros_like(significance, dtype=torch.bool)
        
        # 2. 质量覆盖门控
        s_mass = self._mass_coverage_gating(significance, tau_mass)
            
        # 3. 绝对地板过滤
        abs_filter = significance >= tau_abs
        
        # 4. 组合门控：(S_rel ∪ S_mass) ∩ {i: w_i ≥ τ_abs}
        # combined_mask = (s_rel | s_mass) & abs_filter
        combined_mask = s_mass & abs_filter
        
        # 5. 应用上限约束
        if combined_mask.sum() > k_max:
            # 保留权重最大的k_max个
            _, top_indices = torch.topk(significance, k_max)
            final_mask = torch.zeros_like(significance, dtype=torch.bool)
            final_mask[top_indices] = True
            combined_mask = combined_mask & final_mask
            print(f"selected points: k_max: {final_mask.sum().item()}")
        # print(f"Combined mask: {combined_mask.sum().item()}")
            
        # 6. 确保至少保留一个（权重最大的）
        if combined_mask.sum() == 0:
            max_idx = significance.argmax()
            combined_mask[max_idx] = True
            
        return combined_mask
    
    def adaptive_gating_ablation(self, significance, 
                       lambda_val=None, 
                       tau_mass=0.95, 
                       tau_abs=0.03, 
                       k_max=6,
                       k_target=2.0):
        """
        自适应门控机制
        
        Args:
            significance: 权重张量 (N,) 
            lambda_val: 射线相对阈值参数，如果为None则自动确定
            tau_mass: 质量覆盖阈值 (0.90-0.95)
            tau_abs: 绝对地板阈值 (0.02-0.05)
            k_max: 最大保留高斯数量 (4-8)
            k_target: 目标平均保留高斯数 (1.5-2.5)
            
        Returns:
            mask: 门控掩码 (N,) bool tensor
        """
        if not isinstance(k_max, int):
            k_max = int(k_max)
        if not isinstance(k_target, int):
            k_target = int(k_target)
            
        if significance.numel() == 0:
            return torch.zeros_like(significance, dtype=torch.bool)
        
        # 2. 质量覆盖门控
        s_mass = self._mass_coverage_gating(significance, tau_mass)
            
        # 3. 绝对地板过滤
        abs_filter = significance >= tau_abs
        
        # 4. 组合门控：(S_rel ∪ S_mass) ∩ {i: w_i ≥ τ_abs}
        # combined_mask = (s_rel | s_mass) & abs_filter
        combined_mask = s_mass & abs_filter
        
        _, top_indices = torch.topk(significance, k_max, dim=0)
        final_mask = torch.zeros_like(significance, dtype=torch.bool)
        final_mask[top_indices] = True
        combined_mask = combined_mask & final_mask
        # 5. 应用上限约束
        # if combined_mask.sum() < k_max:
        #     # 保留权重最大的k_max个
        #     _, top_indices = torch.topk(significance, k_max)
        #     final_mask = torch.zeros_like(significance, dtype=torch.bool)
        #     final_mask[top_indices] = True
        #     combined_mask = combined_mask & final_mask
        #     print(f"selected points: k_max: {final_mask.sum().item()}")
        # print(f"Combined mask: {combined_mask.sum().item()}")
            
        # 6. 确保至少保留一个（权重最大的）
        if combined_mask.sum() == 0:
            max_idx = significance.argmax()
            combined_mask[max_idx] = True
            
        return combined_mask
    
    def _adaptive_lambda_search(self, significance, k_target, tau_abs, k_max, 
                               lambda_min=0.0, lambda_max=0.8, tolerance=0.5):
        """
        自适应确定lambda参数
        使用二分搜索找到使平均保留高斯数接近k_target的lambda值
        """
        def count_selected_gaussians(lamb):
            w_max = significance.max()
            if w_max <= 0:
                return 1  # 至少保留一个
                
            # 只计算相对阈值部分，不考虑质量覆盖（质量覆盖在主函数中处理）
            s_rel = significance >= (lamb * w_max)
            abs_filter = significance >= tau_abs
            mask = s_rel & abs_filter
            
            # 应用上限
            count = min(mask.sum().item(), k_max)
            return max(count, 1)  # 至少保留一个
            
        # 先检查边界情况
        count_min = count_selected_gaussians(lambda_min)
        count_max = count_selected_gaussians(lambda_max)
        
        # 如果最小lambda的结果仍然小于目标，直接返回最小值
        if count_min <= k_target:
            return lambda_min
            
        # 如果最大lambda的结果仍然大于目标，直接返回最大值  
        if count_max >= k_target:
            return lambda_max
            
        # 二分搜索
        left, right = lambda_min, lambda_max
        best_lambda = left
        best_diff = float('inf')
        
        for _ in range(15):  # 减少迭代次数
            mid = (left + right) / 2
            count = count_selected_gaussians(mid)
            diff = abs(count - k_target)
            
            if diff < best_diff:
                best_diff = diff
                best_lambda = mid
                
            if diff <= tolerance:
                break
                
            if count > k_target:
                left = mid
            else:
                right = mid
                
        return best_lambda

    def accumulate_gaussian_feature_per_view_robust(self,
                                                gt_language_feature,
                                                gt_mask,
                                                mask,
                                                significance,
                                                means2D,
                                                tau_mass=0.5,
                                                tau_abs=0.1,
                                                k_max=10000,
                                                feature_dim=512):

        if not hasattr(self, "_language_feature_robust"):
            self.create_language_features_robust(feature_dim)
            self._activated_views_robust = torch.zeros(self._xyz.shape[0], device="cuda")

        # 1. 取像素 CLIP 向量
        y = torch.clamp(means2D[:, 1], max=gt_language_feature.shape[0]-1).long()
        x = torch.clamp(means2D[:, 0], max=gt_language_feature.shape[1]-1).long()
        f_t = F.normalize(gt_language_feature[y, x], dim=-1)            # ← 归一化
        valid = gt_mask[y, x] #& mask                               # 双重过滤
        if valid.sum() == 0: return
        
        
        total_points = significance.numel()
        # total_points = int(total_points)
       
        # w_mask_adaptive = self.adaptive_gating(
        #     significance, 
        #     lambda_val=None,   # 手动设定较小的lambda值
        #     tau_mass=tau_mass, #tau_mass,    # 质量覆盖85% (更宽松)
        #     tau_abs=tau_abs,    # 绝对地板0.5% (非常宽松) ##best case 0.13
        #     k_max=0.15*total_points,         # 
        #     k_target=15     # 目标平均15个高斯  # no use
        # ).view(-1, 1)
        
        w_mask_adaptive = self.adaptive_gating_ablation(
            significance, 
            lambda_val=None,   # 手动设定较小的lambda值
            tau_mass=tau_mass, #tau_mass,    # 质量覆盖85% (更宽松)
            tau_abs=tau_abs,    # 绝对地板0.5% (非常宽松) ##best case 0.13
            k_max=0.5*total_points,         # 
            k_target=15     # 目标平均15个高斯  # no use
        ).view(-1, 1)
        
        # print(f"w_mask_adaptive: {w_mask_adaptive.sum().item()}")
        # 也提供一个基于分位数的简单自适应方法作为对比
        # 动态调整分位数：如果点太少就降低分位数
        
        w_mask_mass = self._mass_coverage_gating(significance, 0.25)
        
        # print(f"w_mask_mass: {w_mask_mass.sum().item()}")
        
        if total_points > 100:
            quantile_threshold = 0.85  # 保留top 15%
        elif total_points > 50:
            quantile_threshold = 0.80  # 保留top 20%
        else:
            quantile_threshold = 0.70  # 保留top 30%
            
        w_mask_quantile_adaptive = significance.view(-1, 1) > significance.quantile(quantile_threshold)
        
        
        # 比较不同方法的效果
        w_mask_hard = significance.view(-1, 1) > 0.10
        w_mask_quantile = significance.view(-1, 1) > significance.quantile(0.94)
        if w_mask_hard.sum() > w_mask_quantile.sum():
            w_final_mask = w_mask_hard
        else:
            w_final_mask = w_mask_quantile
        # w_mask_mass = w_mask_hard.squeeze() & w_mask_mass.squeeze() & w_mask_quantile_adaptive.squeeze()
        
        # print(f"Total points: {significance.numel()}")
        # print(f"Adaptive gating count: {w_mask_adaptive.sum().item()}")
        # print(f"mass coverage gate: {w_mask_mass.sum().item()}")
        # print(f"Quantile adaptive count: {w_mask_quantile_adaptive.sum().item()}")
        # print(f"Hard threshold count: {w_mask_hard.sum().item()}")  
        # print(f"Fixed quantile count: {w_mask_quantile.sum().item()}")
        # print("the w_mask_mass values is ", significance.quantile(0.94))
        # 使用更宽松的自适应分位数方法
        # w_mask_mass = w_mask_mass & w_mask_hard.squeeze()
        # valid = valid & w_mask_mass
        # valid = valid & w_mask_adaptive.squeeze()
        # valid = valid & w_mask_quantile.squeeze()
        # valid = valid & w_mask_quantile.squeeze()
        # valid = valid & w_final_mask.squeeze()
        ## for no gating
        
        f_t       = f_t[valid]
        w_t       = significance[valid].view(-1, 1)                     # (N_valid,1)
        # w_mask    = (w_t > 0.03).squeeze()
        # w_t       = w_t[w_mask]
        # f_t       = f_t[w_mask]
        # mask = mask[w_mask]
        idx       = torch.nonzero(mask, as_tuple=False).squeeze(1)[valid]

        g_t       = self._language_feature_robust[idx]                  # 现估计
        W_t_prev  = self._cumulative_weights_robust[idx]                # 累计权

        # l1 distance
        # diff      = f_t - g_t
        # dist      = diff.norm(dim=1, keepdim=True).clamp_min(1e-4)      # 稳定
        
        # W_t_new   = W_t_prev + w_t
        # step      = w_t / W_t_new                                       # η_t = w_t / W_t
        # update    = step * diff / dist                                  # ☆ 只乘一次 w_t/W_t

        # # cos similarity
        dot   = (f_t * g_t).sum(dim=-1, keepdim=True)           # (N,1)
        W_t_new   = W_t_prev + w_t
        step      = w_t / W_t_new                                       # η_t = w_t / W_t
        g = f_t - dot * g_t
        update = step * g

        # 写回
        self._language_feature_robust[idx] = F.normalize(g_t + update, dim=-1)
        self._cumulative_weights_robust[idx] = W_t_new
        self._activated_views_robust[idx] += 1

    
    # def accumulate_gaussian_feature_per_view_robust(self, gt_language_feature, gt_mask, mask, significance=None, means2D=None, feature_dim=512):
    #     """
    #     Accumulate Gaussian features using stochastic Weiszfeld algorithm for geometric median
        
    #     This method implements memory-efficient geometric median computation using:
    #     g_{t+1} = g_t + η_t * w_t * (f_t - g_t) / (||f_t - g_t|| + ε)
    #     where η_t = w_t / W_t and only current estimate g_i and cumulative weight W_i are stored
        
    #     Args:
    #         gt_language_feature: Ground truth language features from current view
    #         gt_mask: Mask indicating valid pixels
    #         mask: Mask indicating activated Gaussians
    #         significance: Significance weights for each Gaussian
    #         means2D: 2D projected centers of Gaussians
    #         feature_dim: Dimension of language features
    #     """
        
    #     if not hasattr(self, "_language_feature_robust"):
    #         self.create_language_features_robust(feature_dim=feature_dim)
    #         self._activated_views_robust = torch.zeros((self._xyz.shape[0]), device="cuda")

    #     batch_y = torch.clamp(means2D[:, 1], max=gt_language_feature.shape[0] - 1)
    #     batch_x = torch.clamp(means2D[:, 0], max=gt_language_feature.shape[1] - 1)
    #     gt_batch_features = F.normalize(gt_language_feature[batch_y.long(), batch_x.long()], dim=-1)  # (N, feature_dim)
    #     print(mask.shape)
    #     print(gt_mask[batch_y.long(), batch_x.long()].shape)
    #     print(gt_batch_features.shape)
    #     gt_batch_mask = gt_mask[batch_y.long(), batch_x.long()] & mask # (N,)
        
    #     # Only process valid features with proper mask
    #     valid_feature_mask = gt_batch_mask
    #     if valid_feature_mask.sum() == 0:
    #         return
            
    #     # Extract relevant data for valid features
    #     valid_gt_features = gt_batch_features[valid_feature_mask]  # (N_valid, feature_dim)
    #     valid_significance = significance[valid_feature_mask]  # (N_valid,)
        
    #     # Get the corresponding Gaussian indices (from the mask that are also valid in features)
    #     mask_indices = mask.nonzero(as_tuple=True)[0]  # Get indices where mask is True
    #     valid_gaussian_indices = mask_indices[valid_feature_mask]  # (N_valid,)
        
    #     # Current estimates for these Gaussians  
    #     current_estimates = self._language_feature_robust[valid_gaussian_indices]  # (N_valid, feature_dim)
    #     current_weights = self._cumulative_weights_robust[valid_gaussian_indices]  # (N_valid, 1)
        
    #     # Compute differences and distances
    #     diff = valid_gt_features - current_estimates  # (N_valid, feature_dim)
    #     distances = torch.norm(diff, dim=1, keepdim=True) + 1e-8  # (N_valid, 1) with numerical stability
    #     distances = torch.clamp(distances, min=1e-4)
        
    #     # Update cumulative weights
    #     new_weights = valid_significance.unsqueeze(-1)  # (N_valid, 1)
    #     updated_cumulative_weights = current_weights + new_weights
        
    #     # Compute step size: η_t = w_t / W_t
    #     step_size = new_weights / (updated_cumulative_weights + 1e-15)  # (N_valid, 1)
    #     updates = step_size * diff / distances
        
    #     # # Stochastic Weiszfeld update: g_{t+1} = g_t + η_t * w_t * (f_t - g_t) / ||f_t - g_t||
    #     # # Since step_size already includes w_t/W_t, we just need the normalized direction
    #     # update_direction = diff / distances  # (N_valid, feature_dim)
    #     # # updates = step_size * new_weights * update_direction  # (N_valid, feature_dim)
    #     # updates = step_size * update_direction
        
    #     # Apply updates
    #     self._language_feature_robust[valid_gaussian_indices] = F.normalize(self._language_feature_robust[valid_gaussian_indices] + updates, dim=-1)
    #     self._cumulative_weights_robust[valid_gaussian_indices] = updated_cumulative_weights
        
    #     # Mark as activated
    #     self._activated_views_robust[valid_gaussian_indices] = 1
        
    def create_language_features_robust(self, feature_dim=512):
        """Initialize robust language features and cumulative weights"""
        self._language_feature_robust = nn.Parameter(
            torch.zeros((self._xyz.shape[0], feature_dim), dtype=torch.float, device="cuda").requires_grad_(True)
        )
        # Use a small initial weight to avoid division by zero
        self._cumulative_weights_robust = torch.ones((self._xyz.shape[0], 1), device="cuda") * 1e-10

    # def finalize_gaussian_features_robust(self, weight_threshold=1e-5):
    #     """
    #     Finalize robust Gaussian features by filtering low-weight features
        
    #     Args:
    #         weight_threshold: Minimum cumulative weight threshold for keeping features
    #     """
    #     # Filter out Gaussians with very low cumulative weights
    #     weight_mask = (self._cumulative_weights_robust.squeeze() >= weight_threshold)
    #     activation_mask = (self._activated_views_robust > 0)
        
    #     # Combine both conditions: must be activated and have sufficient weight
    #     keep_mask = weight_mask & activation_mask
    #     prune_mask = ~keep_mask
        
    #     print(f"Robust method: Pruned {prune_mask.sum()} points (weight < {weight_threshold} or not activated)")
    #     print(f"Remaining points: {keep_mask.sum()}")
        
    #     if prune_mask.sum() > 0:
    #         self.prune_points(prune_mask)
    #         # Update robust features to only keep valid ones
    #         self._language_feature_robust = self._language_feature_robust[keep_mask]
    #         self._cumulative_weights_robust = self._cumulative_weights_robust[keep_mask]
            
    #     # Normalize features (geometric median estimate is already normalized by the algorithm)
    #     # Optional: Apply L2 normalization for consistency
    #     self._language_feature_robust = torch.nn.functional.normalize(self._language_feature_robust, dim=-1)
    
    def finalize_gaussian_features_robust(self, w_thr=1e-5):
        # ---- 合并阈值策略 ----
        abs_thr = max(w_thr,
                    0.01 * torch.median(self._cumulative_weights_robust).item())

        keep = (self._cumulative_weights_robust.squeeze() >= abs_thr) & \
            (self._activated_views_robust > 0)
        prune = ~keep

        print(f"[Robust] prune {prune.sum().item()} / {len(keep)} "
            f"(thr={abs_thr:.2e})")

        # ---- 截断所有并行缓冲 ----
        for name in ("_language_feature_robust",
                    "_cumulative_weights_robust",
                    "_activated_views_robust"):
            buf = getattr(self, name, None)
            if buf is not None:
                setattr(self, name, buf[keep].clone())

        # 最后让底层几何/加速结构同步更新
        self.prune_points(prune)

        # 一次性再 normalize（保险）
        self._language_feature_robust = torch.nn.functional.normalize(
            self._language_feature_robust, dim=-1)

    def get_language_feature_robust(self):
        """Get the robust language features (geometric median estimates)"""
        if hasattr(self, "_language_feature_robust"):
            return self._language_feature_robust
        else:
            raise ValueError('Robust language features have not been computed')

    # def calculate_dispersion(self, gt_language_feature,
    #                                             gt_mask,
    #                                             mask,
    #                                             significance,
    #                                             means2D,
    #                                             feature_dim=512):
    #     """Calculate dispersion values for each Gaussian"""
    #     y = torch.clamp(means2D[0, mask, 1], max=gt_language_feature.shape[0]-1).long()
    #     x = torch.clamp(means2D[0, mask, 0], max=gt_language_feature.shape[1]-1).long()
        
    #     x_features = F.normalize(gt_language_feature[y, x], dim=-1)  # (N_valid, feature_dim)
    #     valid_mask = gt_mask.squeeze(0)[y, x]  # (N_valid,)
    #     w_t = significance[0, mask]  # (N_valid,)
        
    #     if valid_mask.sum() == 0:
    #         return
        
    #     x_features = x_features[valid_mask]  # (N_final, feature_dim)
    #     w_t = w_t[valid_mask]  # (N_final,)
    #     gaussian_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)[valid_mask]  # (N_final,)
        
    #     f_hat_i = self._language_feature_robust[gaussian_indices]  # (N_final, feature_dim)
        
    #     cos_sim = (x_features * f_hat_i).sum(dim=-1)  # (N_final,)
    #     res = 1.0 - cos_sim  # (N_final,)
        
    #     sum_res.index_add_(0, gaussian_indices, res)
    #     count.index_add_(0, gaussian_indices, torch.ones_like(res, dtype=torch.int32))
        
    #     weighted_res = w_t * res
    #     sum_wres.index_add_(0, gaussian_indices, weighted_res)
    #     sum_w.index_add_(0, gaussian_indices, w_t)
        
    
    def set_labels(self, labels):
        """
        Store semantic labels for each Gaussian
        
        Args:
            labels: numpy array or torch tensor of shape (N,) containing semantic labels
        """
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(self._xyz.device)
        elif not isinstance(labels, torch.Tensor):
            raise ValueError("Labels must be numpy array or torch tensor")
            
        # Ensure labels are on the same device as other Gaussian parameters
        labels = labels.to(self._xyz.device)
        
        # Store as a regular tensor (not trainable parameter)
        self._labels = labels.long()
        print(f"Stored {len(labels)} semantic labels in Gaussian model")
        
    def get_labels(self):
        """Get the stored semantic labels"""
        if hasattr(self, "_labels") and self._labels is not None:
            return self._labels
        else:
            raise ValueError('Semantic labels have not been set')
            
    def has_labels(self):
        """Check if semantic labels are available"""
        return hasattr(self, "_labels") and self._labels is not None

    def capture_labels(self):
        """
        Capture semantic labels and essential Gaussian parameters for saving to disk
        Similar to capture_language_feature but focused on labels
        
        Returns:
            Tuple containing labels and essential parameters for reconstruction
        """
        if not self.has_labels():
            raise ValueError("No semantic labels available to capture")
            
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._labels,  # The semantic labels
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.spatial_lr_scale,
        )

    
