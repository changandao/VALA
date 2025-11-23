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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from sklearn.decomposition import PCA
import torch.utils.dlpack
import matplotlib.pyplot as plt
            
def render_set(model_path, name, iteration, source_path, views, gaussians, pipeline, background, feature_level):
    
    save_path = os.path.join(model_path, name, "ours_{}_langfeat_{}".format(iteration, feature_level))
    render_path = os.path.join(save_path, "renders")
    gts_path = os.path.join(save_path, "gt")
    render_npy_path = os.path.join(save_path, "renders_npy")
    gts_npy_path = os.path.join(save_path,"gt_npy")
    
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(render_npy_path, exist_ok=True)
    os.makedirs(gts_npy_path, exist_ok=True)
    
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, include_feature=True)
        rendering = render_pkg["render"]
        gt, mask = view.get_language_feature(language_feature_dir=f"{source_path}/language_features", feature_level=feature_level) #! modified
        # gt, mask = view.get_scannet_language_feature(language_feature_dir=f"{source_path}/language_features_ins", feature_level=feature_level)
        # np.save(os.path.join(render_npy_path, view.image_name.split('.')[0] + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        # np.save(os.path.join(gts_npy_path, view.image_name.split('.')[0] + ".npy"),gt.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(render_npy_path, view.image_name.split('.')[0] + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(gts_npy_path, view.image_name.split('.')[0] + ".npy"),gt.permute(1,2,0).cpu().numpy())
        
        
        _, H, W = gt.shape
        gt = gt.reshape(512, -1).T.cpu().numpy()
        rendering = rendering.reshape(512, -1).T.cpu().numpy() # (H*W, 512)
        
        # torch.save((rendering.permute(1, 2, 0).detach().cpu()).half(), os.path.join(render_npy_path, '{0:05d}'.format(idx) + "_fmap_HxWxD.pt"))

        # rendering = rendering.reshape(512, -1).T.cpu().numpy() # (H*W, 512)
        pca = PCA(n_components=3)

        combined_np = np.concatenate((gt, rendering), axis=0)
        combined_features = pca.fit_transform(combined_np) # ((n+m)*H*W, 3)
        normalized_features = (combined_features - combined_features.min(axis=0)) / (combined_features.max(axis=0) - combined_features.min(axis=0))
        reshaped_combined_features = normalized_features.reshape(2, H, W, 3)
        
        reduced_rendering = reshaped_combined_features[1]
        reduced_gt = reshaped_combined_features[0]
        
        rendering = torch.tensor(reduced_rendering).permute(2, 0, 1)
        gt = torch.tensor(reduced_gt).permute(2, 0, 1)
        
        # torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name+"_fmap.png" ))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name+"_fmap.png"))

        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name ))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name))
        
def render_sets(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, feature_level : int, ablation_type : str):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)
        
        model_path = os.path.join(args.model_path, ablation_type)
        if ablation_type == "none":
            checkpoint = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}.pth')
        else:
            checkpoint = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}_stochastic_w015_langsplat.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore_language_features(model_params, opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(args.model_path, "train", scene.loaded_iter, dataset.source_path, scene.getTrainCameras(), gaussians, pipeline, background, feature_level)

        if not skip_test:
             render_set(args.model_path, "test", scene.loaded_iter, dataset.source_path, scene.getTestCameras(), gaussians, pipeline, background, feature_level)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ablation_type", type=str, default="none")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.feature_level, args.ablation_type)