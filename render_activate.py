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
import numpy as np
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
from pathlib import Path

from eval.openclip_encoder import OpenCLIPNetwork

import time


from eval import colormaps
import torch.nn.functional as F

import cv2

COLORMAP_OPTIONS = colormaps.ColormapOptions(
    colormap="turbo",
    normalize=True,
    colormap_min=-1.0,
    colormap_max=1.0,
)

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args, label, clip_model, img_label):
    render_path = os.path.join(model_path, name, f"renders_colormap_{img_label}")
    makedirs(render_path, exist_ok=True)
    args.mask = None
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background)

        rendering = output["render"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))


def render_sets(dataset : ModelParams, pipeline : PipelineParams, opt : OptimizationParams, iteration : int, skip_train : bool, skip_test : bool, args, label, clip_model, img_save_label, ablation_type):
    with torch.no_grad():
        start_time = time.time()
        
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)

        if ablation_type == "adaptive_gate":
            checkpoint = os.path.join(args.model_path, args.ablation_type, f'chkpnt{iteration}_langfeat_{args.feature_level}_stochastic_gate.pth')
        elif ablation_type == "cos_median":
            checkpoint = os.path.join(args.model_path, args.ablation_type, f'chkpnt{iteration}_langfeat_{args.feature_level}_stochastic_cos_median.pth')
        elif ablation_type == "supp_occamlgs":
            checkpoint = os.path.join(args.model_path, args.ablation_type, f'chkpnt{iteration}_langfeat_{args.feature_level}.pth')
        else:
            checkpoint = os.path.join(args.model_path, f'chkpnt{iteration}_langfeat_{args.feature_level}.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore_language_features(model_params, opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        features = gaussians._language_feature.clone()
        zero_mask = torch.all(features == -1, dim=-1)

        leaf_lang_feat = features[~zero_mask].to("cuda")
        leaf_lang_feat = F.normalize(leaf_lang_feat, dim=-1)
        activation_features = torch.zeros((features.shape[0], 1), dtype=torch.float32).cuda()
        print("leaf_lang_feat.shape", leaf_lang_feat.shape)
        probs = clip_model.get_relevancy(leaf_lang_feat, label)
        _activation_features = probs[..., 0:1].to(torch.float32)
        activation_features[~zero_mask] = _activation_features

        thr = args.threshold
        
        activation_threshold = torch.where(activation_features.squeeze() > thr)[0]

        features_colormap = colormaps.apply_colormap(activation_features, colormap_options=COLORMAP_OPTIONS)
        features_colormap = (features_colormap.unsqueeze(1) - 0.5) / 0.28209479177387814
        gaussians._features_dc[activation_threshold] = features_colormap[activation_threshold]
        gaussians._features_rest[activation_threshold] = torch.zeros_like(gaussians._features_rest)[activation_threshold].cuda()

        if args.save_ply:
            gaussians.save_ply(os.path.join(dataset.model_path, "point_cloud", "iteration_0", f"{img_save_label}_point_cloud.ply"))

        end_time = time.time()
        print(f'Running time : {end_time - start_time}')
        output_path = os.path.join(dataset.model_path, args.ablation_type)
        makedirs(output_path, exist_ok=True)
        if not skip_train:
             render_set(output_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args, label, clip_model, img_save_label)

        if not skip_test:
             render_set(output_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args, label, clip_model, img_save_label)

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
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--semantic_model", default='dino', type=str)
    # parser.add_argument("--pq_index", type=str, default=None)
    parser.add_argument("--img_save_label", type=str, default=None)
    parser.add_argument("--img_label", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--ablation_type", type=str, default="none")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    
    img_labels = [args.img_label]

    device = "cuda"
    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives(img_labels)
    
    print("include_feature", args.include_feature)

    # index = faiss.read_index(args.pq_index)

    # negative_text_features = torch.from_numpy(np.load('autolabel/text_negative.npy')).to(torch.float32)  # [num_text, 512]

    for label in range(len(img_labels)):
        text_feat = clip_model.encode_text(img_labels[label], device=device).float()
        render_sets(model.extract(args), pipeline.extract(args), opt.extract(args), args.iteration, args.skip_train, args.skip_test, args, label, clip_model, args.img_save_label, args.ablation_type)