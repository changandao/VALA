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
import time
from utils.memory_utils import print_gpu_memory_info, cleanup_memory, check_memory_threshold, MemoryMonitor


def extract_gaussian_features(model_path, iteration, source_path, views, gaussians, pipeline, background, feature_level, omniseg3d=False, ablation_type="none"):

    model_path = os.path.join(model_path, ablation_type)
    if omniseg3d:
        language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_omniseg3d.pth')
    else:
        language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}.pth')
    
    for _, view in enumerate(tqdm(views, desc="Rendering progress")):

        render_pkg= render(view, gaussians, pipeline, background)

        if "waldo_kitchen" in model_path or "figurines" in model_path:
            language_feature_dir = f"{source_path}/language_features"
        else:
            language_feature_dir = f"{source_path}/langsplat/language_features"
        if omniseg3d:
            ##lerf_ovs
            # gt_language_feature, gt_mask = view.get_vfm_language_feature(language_feature_dir=f"{source_path}/omniseg3d/language_features_ins", feature_level=feature_level)
            gt_language_feature, gt_mask = view.get_scannet_language_feature(language_feature_dir=language_feature_dir, feature_level=feature_level)
        else:
            ##scannet
            gt_language_feature, gt_mask = view.get_language_feature(language_feature_dir=f"{source_path}/language_features", feature_level=feature_level)
        
        activated = render_pkg["info"]["activated"]
        significance = render_pkg["info"]["significance"]
        means2D = render_pkg["info"]["means2d"]
        
        mask = activated[0] > 0
        gaussians.accumulate_gaussian_feature_per_view(gt_language_feature.permute(1, 2, 0), gt_mask.squeeze(0), mask, significance[0,mask], means2D[0, mask])
        
        # # print('total gaussians ', mask.shape[0])
        # # print(f'valid gaussians for view {view.image_name} ', means2D[0, mask].shape[0])
        # # Use GaussianModel's robust accumulation method
        # if "teatime" in model_path:
        #     tau_mass = 0.5
        #     tau_abs = 0.1
        # elif "scannet" in model_path:
        #     tau_mass = 0.5
        #     tau_abs = 0.1
        # #     k_max = 0.5*mask.shape[0]
        # else:
        #     tau_mass = 0.75
        #     tau_abs = 0.13
        # gaussians.accumulate_gaussian_feature_per_view_gate(gt_language_feature.permute(1, 2, 0), gt_mask.squeeze(0), mask, significance[0,mask], means2D[0, mask], tau_mass=tau_mass, tau_abs=tau_abs)
    
    gaussians.finalize_gaussian_features()

    # Ensure parent directory exists before saving
    os.makedirs(os.path.dirname(language_feature_save_path), exist_ok=True)
    torch.save((gaussians.capture_language_feature(), 0), language_feature_save_path)
    print("checkpoint saved to: ", language_feature_save_path)
    


def extract_gaussian_features_stochastic(model_path, iteration, source_path, views, gaussians, pipeline, background, feature_level, weight_threshold=1e-5, omniseg3d=False, batch_size=50000, ablation_type="none"):
    """
    Extract Gaussian features using memory-efficient stochastic Weiszfeld algorithm
    Processes Gaussians in batches to handle very large scenes (millions of Gaussians)
    
    Args:
        model_path: path to save the model
        iteration: iteration number
        source_path: source path for language features
        views: list of camera views 
        gaussians: GaussianModel instance
        pipeline: pipeline parameters
        background: background color
        feature_level: feature level to extract
        weight_threshold: minimum weight threshold for filtering low-weight features
        batch_size: batch size for processing Gaussian updates
    """
    # ablation_type = "none" # sam_mask, cos_gate, none
    # language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}_stochastic_o3.pth')
    # language_feature_save_path = os.path.join(model_path,  f'chkpnt{iteration}_omniseg3d_stochastic_cos_gate.pth') \
    # if omniseg3d else os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}_stochastic.pth')
    model_path = os.path.join(model_path, ablation_type)
    if ablation_type == "gate":
        language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_omniseg3d_stochastic_gate.pth') \
        if omniseg3d else os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}_stochastic_gate.pth')
    elif ablation_type == "cos_median":
        language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_omniseg3d_stochastic_cos_median.pth') \
        if omniseg3d else os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}_stochastic_cos_median.pth')
    # elif ablation_type == "none":
    #     language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_omniseg3d_stochastic.pth') \
    #     if omniseg3d else os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}_stochastic.pth')
    else:
        language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_omniseg3d_stochastic_gate.pth') \
        if omniseg3d else os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}_stochastic_gate.pth')    
        
    print(f"Collecting features from all views using stochastic Weiszfeld (batch_size={batch_size})...")
    
    for view_idx, view in enumerate(tqdm(views, desc="Stochastic feature collection")):
        render_pkg = render(view, gaussians, pipeline, background)

        # gt_language_feature, gt_mask = view.get_language_feature(
        #     language_feature_dir=f"{source_path}/language_features", 
        #     feature_level=feature_level
        # )
        if "waldo_kitchen" in model_path or "figurines" in model_path:
            language_feature_dir = f"{source_path}/language_features"
        else:
            language_feature_dir = f"{source_path}/langsplat/language_features"
        if omniseg3d:
            if "lerf_ovs" in model_path:
                gt_language_feature, gt_mask = view.get_vfm_language_feature_optimized(language_feature_dir=f"{source_path}/omniseg3d/language_features_ins", feature_level=feature_level) ##lerf_ovs
            elif "scannet" in model_path:
                gt_language_feature, gt_mask = view.get_scannet_language_feature(language_feature_dir=f"{source_path}/language_features_ins", feature_level=feature_level)
        else:
            gt_language_feature, gt_mask = view.get_language_feature(language_feature_dir=language_feature_dir, feature_level=feature_level)
        
        activated = render_pkg["info"]["activated"]
        significance = render_pkg["info"]["significance"]
        means2D = render_pkg["info"]["means2d"]
        mask = activated[0] > 0
        # print('total gaussians ', mask.shape[0])
        # print(f'valid gaussians for view {view.image_name} ', means2D[0, mask].shape[0])
        # Use GaussianModel's robust accumulation method
        if "teatime" in model_path:
            tau_mass = 0.5
            tau_abs = 0.1
            k_max = 0.3*mask.shape[0]
        elif "scannet" in model_path:
            tau_mass = 0.9
            tau_abs = 0.01
            k_max = 0.5*mask.shape[0]
        else:
            tau_mass = 0.75
            tau_abs = 0.13
        gaussians.accumulate_gaussian_feature_per_view_robust(
            gt_language_feature.permute(1, 2, 0), 
            gt_mask.squeeze(0), 
            mask, 
            significance[0, mask], 
            means2D[0, mask],
            tau_mass=tau_mass,
            tau_abs=tau_abs,
            k_max=k_max
        )
        
        # # Process in batches to save memory
        # mask_indices = mask.nonzero(as_tuple=True)[0]
        
        # if len(mask_indices) == 0:
        #     continue
            
        # for i in range(0, len(mask_indices), batch_size):
        #     end_idx = min(i + batch_size, len(mask_indices))
        #     batch_indices = mask_indices[i:end_idx]
            
        #     # Create batch mask
        #     batch_mask = torch.zeros_like(mask)
        #     batch_mask[batch_indices] = True
            
        #     # Process this batch
        #     gaussians.accumulate_gaussian_feature_per_view_robust(
        #         gt_language_feature.permute(1, 2, 0), 
        #         gt_mask.squeeze(0), 
        #         batch_mask, 
        #         significance[0, batch_mask], 
        #         means2D[0, batch_mask]
        #     )
            
        #     # Clean up GPU memory periodically
        #     if i % (batch_size * 5) == 0:
        #         torch.cuda.empty_cache()
        
        # Periodically clean up memory every 10 views
        if view_idx % 10 == 0:
            torch.cuda.empty_cache()
        
    # Finalize features using robust geometric median
    # gaussians.finalize_gaussian_features_robust(weight_threshold=weight_threshold)
    gaussians.finalize_gaussian_features_robust(w_thr=weight_threshold)
    # gaussians.save_ply(os.path.join(model_path, f'omniseg3d_stochastic_{ablation_type}.ply'))

    # Ensure parent directory exists before saving
    os.makedirs(os.path.dirname(language_feature_save_path), exist_ok=True)
    torch.save((gaussians.capture_language_feature(), 0), language_feature_save_path)
    print(f"Stochastic checkpoint saved to: {language_feature_save_path}")


def process_scene_language_features(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, feature_level : int):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)

        checkpoint = os.path.join(args.model_path, f'chkpnt{iteration}.pth')
        (model_params, _) = torch.load(checkpoint)
        gaussians.restore_rgb(model_params, opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        extract_gaussian_features(args.model_path, iteration, dataset.source_path, scene.getTrainCameras(), gaussians, pipeline, background, feature_level, args.omniseg3d, args.ablation_type)
        # extract_gaussian_features_omniseg3d(args.model_path, iteration, dataset.source_path, scene.getTrainCameras(), gaussians, pipeline, background, feature_level)



def process_scene_language_features_stochastic(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, feature_level : int, weight_threshold : float = 1e-5, omniseg3d : bool = False, batch_size : int = 50000, ablation_type : str = "none"):
    """
    Process scene language features using memory-efficient stochastic Weiszfeld algorithm
    
    Args:
        dataset: model parameters
        opt: optimization parameters  
        iteration: iteration number
        pipeline: pipeline parameters
        feature_level: feature level to extract
        weight_threshold: minimum weight threshold for filtering low-weight features
        batch_size: batch size for processing Gaussian updates
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)

        checkpoint = os.path.join(args.model_path, f'chkpnt{iteration}.pth')
        (model_params, _) = torch.load(checkpoint)
        gaussians.restore_rgb(model_params, opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        extract_gaussian_features_stochastic(args.model_path, iteration, dataset.source_path, scene.getTrainCameras(), gaussians, pipeline, background, feature_level, weight_threshold, omniseg3d, batch_size, ablation_type)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use_robust", action="store_true", help="Use robust Weiszfeld geometric median aggregation")
    parser.add_argument("--use_efficient", action="store_true", help="Use memory-efficient stochastic Weiszfeld algorithm")
    parser.add_argument("--weight_threshold", default=1e-5, type=float, help="Minimum weight threshold for filtering low-weight features")
    parser.add_argument("--omniseg3d", action="store_true", help="Use omniseg3d language features")
    parser.add_argument("--batch_size", default=50000, type=int, help="Batch size for processing updates")
    parser.add_argument("--ablation_type", type=str, default="none", help="ablation type")

    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # if args.use_robust:
    #     print("Using robust Weiszfeld geometric median aggregation...")
    #     process_scene_language_features_robust(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.feature_level, args.weight_threshold)
    if args.use_efficient:
        print("Using memory-efficient stochastic Weiszfeld algorithm...")
        process_scene_language_features_stochastic(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.feature_level, args.weight_threshold, args.omniseg3d, args.batch_size, args.ablation_type)
    else:
        print("Using standard weighted average aggregation...")
        process_scene_language_features(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.feature_level)