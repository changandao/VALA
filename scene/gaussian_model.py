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
import numpy as np
import torch.nn.functional as F
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
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
        Get the inverse of covariance matrices from GaussianModel
        
        Args:
            scaling_modifier: Scaling correction factor
        
        Returns:
            Sigma_inv: (N, 3, 3) Inverse of covariance matrices
        """
        # 1. Get 6D compact representation of covariance matrices
        cov_6d = self.get_covariance(scaling_modifier)  # (N, 6)
        
        # 2. Convert 6D representation to full 3x3 covariance matrices
        N = cov_6d.shape[0]
        Sigma = torch.zeros((N, 3, 3), device=cov_6d.device, dtype=cov_6d.dtype)
        
        # Fill 3x3 matrix according to strip_lowerdiag format (symmetric)
        Sigma[:, 0, 0] = cov_6d[:, 0]
        Sigma[:, 0, 1] = cov_6d[:, 1]
        Sigma[:, 0, 2] = cov_6d[:, 2]
        Sigma[:, 1, 0] = cov_6d[:, 1]  # symmetric
        Sigma[:, 1, 1] = cov_6d[:, 3]
        Sigma[:, 1, 2] = cov_6d[:, 4]
        Sigma[:, 2, 0] = cov_6d[:, 2]  # symmetric
        Sigma[:, 2, 1] = cov_6d[:, 4]  # symmetric
        Sigma[:, 2, 2] = cov_6d[:, 5]
        
        # 3. Compute inverse matrices
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
        """
        Accumulate language features per view with adaptive gating.
        
        Args:
            gt_language_feature: Ground truth language features
            gt_mask: Valid mask for ground truth features
            mask: Visibility mask for Gaussians
            significance: Importance weights for each Gaussian
            means2D: 2D projections of Gaussian centers
            feature_dim: Dimension of language features
            ablation_type: Type of ablation study
            tau_mass: Mass coverage threshold (0.90-0.95)
            tau_abs: Absolute floor threshold (0.02-0.05)
            k_max: Maximum number of Gaussians to keep (4-8)
            k_target: Target average number of Gaussians to keep (1.5-2.5)
        """
        if not hasattr(self, "_language_feature"):
            self.create_language_features(feature_dim=feature_dim)
            self._activated_views = torch.zeros((self._xyz.shape[0]), device="cuda")

        batch_y = torch.clamp(means2D[:, 1], max=gt_language_feature.shape[0] - 1)
        batch_x = torch.clamp(means2D[:, 0], max=gt_language_feature.shape[1] - 1)
        gt_batch_features = gt_language_feature[batch_y.long(), batch_x.long()]
        valid = gt_mask[batch_y.long(), batch_x.long()]  # (N,)
        
        total_points = significance.numel()
        
        # Apply adaptive gating
        w_mask_adaptive = self.adaptive_gating(
            significance, 
            lambda_val=None,
            tau_mass=tau_mass,
            tau_abs=tau_abs,
            k_max=0.15 * total_points,
            k_target=15
        ).view(-1, 1)
        
        w_mask_mass = self._mass_coverage_gating(significance, 0.35)
        
        # Dynamically adjust quantile threshold based on point count
        if total_points > 100:
            quantile_threshold = 0.85  # Keep top 15%
        elif total_points > 50:
            quantile_threshold = 0.80  # Keep top 20%
        else:
            quantile_threshold = 0.70  # Keep top 30%
            
        w_mask_quantile_adaptive = significance.view(-1, 1) > significance.quantile(quantile_threshold)
        
        # Compare different gating methods and select the more permissive one
        w_mask_hard = significance.view(-1, 1) > 0.10
        w_mask_quantile = significance.view(-1, 1) > significance.quantile(0.85)
        if w_mask_hard.sum() > w_mask_quantile.sum():
            w_final_mask = w_mask_hard
        else:
            w_final_mask = w_mask_quantile

        valid = valid & w_final_mask.squeeze()

        # l1 distance
        # diff      = f_t - g_t
        # dist      = diff.norm(dim=1, keepdim=True).clamp_min(1e-4)      # stable
        
        # W_t_new   = W_t_prev + w_t
        # step      = w_t / W_t_new                                       # η_t = w_t / W_t
        # update    = step * diff / dist                                  # only multiply once w_t/W_t
        
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
        """Finalize language features by normalizing and pruning inactive Gaussians."""
        self._language_feature = self._language_feature / (self._language_feature_weight + 1e-15)
        
        # Filter out Gaussians that were never activated
        prune_mask = self._activated_views == 0
        self.prune_points(prune_mask)
        self._language_feature = self._language_feature[~prune_mask]
        
        print("Pruned {} points".format(prune_mask.sum()))
    
    
    def get_language_feature(self):
        return self._language_feature
    
    def _mass_coverage_gating(self, significance, tau_mass=0.95):
        """
        Mass coverage gating.
        
        Keep Gaussians whose cumulative weight sum reaches tau_mass proportion.
        
        Args:
            significance: Weight tensor (N,)
            tau_mass: Mass coverage threshold (0.90-0.95)
            
        Returns:
            s_mass: Mass coverage gating mask (N,) bool tensor
        """
        if significance.numel() == 0:
            return torch.zeros_like(significance, dtype=torch.bool)
            
        # Sort weights in descending order
        sorted_weights, sorted_indices = torch.sort(significance, descending=True)
        cumsum_weights = torch.cumsum(sorted_weights, dim=0)
        total_weight = significance.sum()
        
        if total_weight > 0:
            # Find the minimum index where cumulative weight reaches tau_mass
            mass_threshold_idx = torch.where(cumsum_weights >= tau_mass * total_weight)[0]
            if len(mass_threshold_idx) > 0:
                mass_cutoff = mass_threshold_idx[0] + 1  # +1 because index starts from 0
                s_mass = torch.zeros_like(significance, dtype=torch.bool)
                s_mass[sorted_indices[:mass_cutoff]] = True
            else:
                s_mass = torch.ones_like(significance, dtype=torch.bool)
        else:
            s_mass = torch.zeros_like(significance, dtype=torch.bool)

        return s_mass

    def adaptive_gating(self, significance, 
                       lambda_val=None, 
                       tau_mass=0.95, 
                       tau_abs=0.03, 
                       k_max=6,
                       k_target=2.0):
        """
        Adaptive gating mechanism.
        
        Args:
            significance: Weight tensor (N,) 
            lambda_val: Ray relative threshold parameter, auto-determined if None
            tau_mass: Mass coverage threshold (0.90-0.95)
            tau_abs: Absolute floor threshold (0.02-0.05)
            k_max: Maximum number of Gaussians to keep (4-8)
            k_target: Target average number of Gaussians to keep (1.5-2.5)
            
        Returns:
            mask: Gating mask (N,) bool tensor
        """
        if not isinstance(k_max, int):
            k_max = int(k_max)
        if not isinstance(k_target, int):
            k_target = int(k_target)
            
        if significance.numel() == 0:
            return torch.zeros_like(significance, dtype=torch.bool)
        
        # Apply mass coverage gating
        s_mass = self._mass_coverage_gating(significance, tau_mass)
            
        # Apply absolute floor filter
        abs_filter = significance >= tau_abs
        
        # Combined gating: S_mass ∩ {i: w_i ≥ τ_abs}
        combined_mask = s_mass & abs_filter
        
        # Apply upper limit constraint
        if combined_mask.sum() > k_max:
            # Keep only the top k_max by weight
            _, top_indices = torch.topk(significance, k_max)
            final_mask = torch.zeros_like(significance, dtype=torch.bool)
            final_mask[top_indices] = True
            combined_mask = combined_mask & final_mask
            
        # Ensure at least one Gaussian is kept (the one with maximum weight)
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
        Adaptive gating mechanism for ablation study.
        
        Args:
            significance: Weight tensor (N,) 
            lambda_val: Ray relative threshold parameter, auto-determined if None
            tau_mass: Mass coverage threshold (0.90-0.95)
            tau_abs: Absolute floor threshold (0.02-0.05)
            k_max: Maximum number of Gaussians to keep (4-8)
            k_target: Target average number of Gaussians to keep (1.5-2.5)
            
        Returns:
            mask: Gating mask (N,) bool tensor
        """
        if not isinstance(k_max, int):
            k_max = int(k_max)
        if not isinstance(k_target, int):
            k_target = int(k_target)
            
        if significance.numel() == 0:
            return torch.zeros_like(significance, dtype=torch.bool)
        
        # Apply mass coverage gating
        s_mass = self._mass_coverage_gating(significance, tau_mass)
            
        # Apply absolute floor filter
        abs_filter = significance >= tau_abs
        
        # Combined gating: S_mass ∩ {i: w_i ≥ τ_abs}
        combined_mask = s_mass & abs_filter
        
        # Always apply top-k constraint in ablation mode
        _, top_indices = torch.topk(significance, k_max, dim=0)
        final_mask = torch.zeros_like(significance, dtype=torch.bool)
        final_mask[top_indices] = True
        combined_mask = combined_mask & final_mask
            
        # Ensure at least one Gaussian is kept (the one with maximum weight)
        if combined_mask.sum() == 0:
            max_idx = significance.argmax()
            combined_mask[max_idx] = True
            
        return combined_mask
    
    def _adaptive_lambda_search(self, significance, k_target, tau_abs, k_max, 
                               lambda_min=0.0, lambda_max=0.8, tolerance=0.5):
        """
        Adaptively determine lambda parameter.
        Uses binary search to find lambda value that keeps average Gaussian count close to k_target.
        """
        def count_selected_gaussians(lamb):
            w_max = significance.max()
            if w_max <= 0:
                return 1  # Keep at least one
                
            # Only compute relative threshold, mass coverage is handled in main function
            s_rel = significance >= (lamb * w_max)
            abs_filter = significance >= tau_abs
            mask = s_rel & abs_filter
            
            # Apply upper limit
            count = min(mask.sum().item(), k_max)
            return max(count, 1)  # Keep at least one
            
        # Check boundary conditions first
        count_min = count_selected_gaussians(lambda_min)
        count_max = count_selected_gaussians(lambda_max)
        
        # If min lambda result is still below target, return min value
        if count_min <= k_target:
            return lambda_min
            
        # If max lambda result is still above target, return max value
        if count_max >= k_target:
            return lambda_max
            
        # Binary search
        left, right = lambda_min, lambda_max
        best_lambda = left
        best_diff = float('inf')
        
        for _ in range(15):  # Limit iterations
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
        """
        Accumulate language features per view using robust geometric median estimation.
        
        Args:
            gt_language_feature: Ground truth language features from CLIP
            gt_mask: Valid mask for ground truth features
            mask: Visibility mask for Gaussians
            significance: Importance weights for each Gaussian
            means2D: 2D projections of Gaussian centers
            tau_mass: Mass coverage threshold
            tau_abs: Absolute floor threshold
            k_max: Maximum number of Gaussians to keep
            feature_dim: Dimension of language features
        """
        if not hasattr(self, "_language_feature_robust"):
            self.create_language_features_robust(feature_dim)
            self._activated_views_robust = torch.zeros(self._xyz.shape[0], device="cuda")

        # Get pixel CLIP vectors
        y = torch.clamp(means2D[:, 1], max=gt_language_feature.shape[0]-1).long()
        x = torch.clamp(means2D[:, 0], max=gt_language_feature.shape[1]-1).long()
        f_t = F.normalize(gt_language_feature[y, x], dim=-1)  # Normalize
        valid = gt_mask[y, x]
        if valid.sum() == 0:
            return
        
        total_points = significance.numel()
       
        ### gating methods
        w_mask_adaptive = self.adaptive_gating(
            significance, 
            lambda_val=None,   # manually set smaller lambda value
            tau_mass=tau_mass, #tau_mass,    
            tau_abs=tau_abs,    # absolute floor 0.5% (very loose) ##best case 0.13
            k_max=0.5*total_points,         # 
            k_target=15     # target average 15 gaussians  # no use
        ).view(-1, 1)
        
        # w_mask_mass = self._mass_coverage_gating(significance, 0.25)
        
        # Compare different gating methods and select the more permissive one
        w_mask_hard = significance.view(-1, 1) > 0.15
        w_mask_quantile = significance.view(-1, 1) > significance.quantile(0.85)
        if w_mask_hard.sum() > w_mask_quantile.sum():
            w_final_mask = w_mask_hard
        else:
            w_final_mask = w_mask_quantile
        
        w_mask_mass = w_mask_adaptive.squeeze() & w_final_mask.squeeze()

        ### end of gating methods

        ### apply gating mask
        valid = valid & w_mask_mass.squeeze()
        f_t       = f_t[valid]
        w_t       = significance[valid].view(-1, 1)                     # (N_valid,1)

        idx       = torch.nonzero(mask, as_tuple=False).squeeze(1)[valid]

        g_t       = self._language_feature_robust[idx]                  # current estimate
        W_t_prev  = self._cumulative_weights_robust[idx]                # cumulative weight


        # Cosine similarity based update
        dot = (f_t * g_t).sum(dim=-1, keepdim=True)  # (N, 1)
        W_t_new = W_t_prev + w_t
        step = w_t / W_t_new  # η_t = w_t / W_t, step size
        g = f_t - dot * g_t
        update = step * g

        # Write back updated features
        self._language_feature_robust[idx] = F.normalize(g_t + update, dim=-1)
        self._cumulative_weights_robust[idx] = W_t_new
        self._activated_views_robust[idx] += 1

        
    def create_language_features_robust(self, feature_dim=512):
        """Initialize robust language features and cumulative weights"""
        self._language_feature_robust = nn.Parameter(
            torch.zeros((self._xyz.shape[0], feature_dim), dtype=torch.float, device="cuda").requires_grad_(True)
        )
        # Use a small initial weight to avoid division by zero
        self._cumulative_weights_robust = torch.ones((self._xyz.shape[0], 1), device="cuda") * 1e-10

    
    def finalize_gaussian_features_robust(self, w_thr=1e-5):
        """
        Finalize robust language features by pruning low-weight Gaussians.
        
        Args:
            w_thr: Weight threshold for pruning
        """
        # Combined threshold strategy
        abs_thr = max(w_thr,
                    0.01 * torch.median(self._cumulative_weights_robust).item())

        keep = (self._cumulative_weights_robust.squeeze() >= abs_thr) & \
            (self._activated_views_robust > 0)
        prune = ~keep
        print(f"[Robust] prune {prune.sum().item()} / {len(keep)} "
            f"(thr={abs_thr:.2e})")

        # Truncate all parallel buffers
        for name in ("_language_feature_robust",
                    "_cumulative_weights_robust",
                    "_activated_views_robust"):
            buf = getattr(self, name, None)
            if buf is not None:
                setattr(self, name, buf[keep].clone())

        # Sync underlying geometry/acceleration structures
        self.prune_points(prune)

        # Final normalization for safety
        self._language_feature_robust = torch.nn.functional.normalize(
            self._language_feature_robust, dim=-1)
        # # Normalize only valid features, mark the rest as -1 instead of pruning
        # if keep.any():
        #     self._language_feature_robust[keep] = torch.nn.functional.normalize(
        #         self._language_feature_robust[keep], dim=-1
        #     )
        # if prune.any():
        #     self._language_feature_robust[prune] = -1

        # print(f"[Robust] marked {prune.sum().item()} invalid points (-1) instead of pruning (thr={abs_thr:.2e})")

    def get_language_feature_robust(self):
        """Get the robust language features (geometric median estimates)"""
        if hasattr(self, "_language_feature_robust"):
            return self._language_feature_robust
        else:
            raise ValueError('Robust language features have not been computed')
        
    
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
            self._cumulative_weights_robust,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.spatial_lr_scale,
        )

    
