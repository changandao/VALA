import time
import numpy as np
import os
import random
import torch
import torchvision

import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union, Any
import json
import numpy as np
from PIL import Image
from eval.utils import polygon_to_mask, stack_mask, vis_mask_save

from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GaussianModel and render directly from their files, not from the package
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
# from autoencoder.model import Autoencoder
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from eval.openclip_encoder import OpenCLIPNetwork
from scene import Scene
from types import SimpleNamespace
from utils.general_utils import safe_state


scene_gt_frames = {
    "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
    "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
    "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
    "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
}
    
scene_texts = {
    "waldo_kitchen": ['Stainless steel pots', 'dark cup', 'refrigerator', 'frog cup', 'pot', 'spatula', 'plate', \
            'spoon', 'toaster', 'ottolenghi', 'plastic ladle', 'sink', 'ketchup', 'cabinet', 'red cup', \
            'pour-over vessel', 'knife', 'yellow desk'],
    "ramen": ['nori', 'sake cup', 'kamaboko', 'corn', 'spoon', 'egg', 'onion segments', 'plate', \
            'napkin', 'bowl', 'glass of water', 'hand', 'chopsticks', 'wavy noodles'],
    "figurines": ['jake', 'pirate hat', 'pikachu', 'rubber duck with hat', 'porcelain hand', \
                'red apple', 'tesla door handle', 'waldo', 'bag', 'toy cat statue', 'miffy', \
                'green apple', 'pumpkin', 'rubics cube', 'old camera', 'rubber duck with buoy', \
                'red toy chair', 'pink ice cream', 'spatula', 'green toy chair', 'toy elephant'],
    "teatime": ['sheep', 'yellow pouf', 'stuffed bear', 'coffee mug', 'tea in a glass', 'apple', 
            'coffee', 'hooves', 'bear nose', 'dall-e brand', 'plate', 'paper napkin', 'three cookies', \
            'bag of cookies'] # 'dall-e brand' is not in the dataset
}


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    print("gt_json_paths:", gt_json_paths)
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            print("save_path:", save_path)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths

def activate_stream(gs_lang_feat, clip_model, gs_xyz, k=10, thresh=0.4):
    valid_map_3d = clip_model.get_max_across_3d(gs_lang_feat)
    n_prompt, n = valid_map_3d.shape
    
    # smooth the relevancy map, similar to in 2D
    gs_xyz_np = gs_xyz.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(gs_xyz_np)
    _, indices = nbrs.kneighbors(gs_xyz_np)
    indices = torch.from_numpy(indices).to(valid_map_3d.device)
    relv_map_smoothed = torch.zeros_like(valid_map_3d)
    gs_mask_pred = torch.zeros_like(valid_map_3d)
    for i in range(n_prompt):
        relv_1d = valid_map_3d[i]  
        neighbors_vals = relv_1d[indices]  
        neighbors_avg = neighbors_vals.mean(dim=1)  
        relv_map_smoothed[i] = 0.5 * (relv_1d + neighbors_avg)
    
        output = relv_map_smoothed[i]
        output = output - torch.min(output)
        output = output / (torch.max(output) + 1e-9)
        output = output * (1.0 - (-1.0)) + (-1.0)
        output = torch.clip(output, 0, 1)
        
        gs_mask_pred[i] = output > thresh
    
    return gs_mask_pred > 0.5  # convert float to bool


def render_set(output_dir, views, gaussians, pipeline, background, scene_name, text_indices=[], gs_masks_pred=[], mask_thresh=0.4, feature_level=1):    
    rgb_path = os.path.join(output_dir, f"predictions_mask_{mask_thresh}", "renders")
    alpha_path = os.path.join(output_dir, f"predictions_mask_{mask_thresh}", "renders_silhouette")
    
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(alpha_path, exist_ok=True)
    
    target_text = scene_texts[scene_name]
    
    opt = {}
    # start_time = time.time()
    count = 0
    for i, text_idx in enumerate(text_indices):
        # print(f"gs_masks_pred[i]: {gs_masks_pred[i]}")
        opt = SimpleNamespace(include_feature=False, mask=gs_masks_pred[i])
        
        print(f"rendering the {text_idx+1}-th query of {len(target_text)} texts: {target_text[text_idx]}")
        # gaussians.prune_points(opt.mask)
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            view_name = (view.image_name).split('.')[0]
            view_dir = os.path.join(rgb_path, view_name)
            os.makedirs(view_dir, exist_ok=True)
            view_alpha_dir = os.path.join(alpha_path, view_name)
            os.makedirs(view_alpha_dir, exist_ok=True)
            output = render(view, gaussians, pipeline, background, include_feature=False, mask=opt.mask)

            rendering = output["render"]
            rendering_alpha = output["render_alpha"]
            
            # Reshape rendering_alpha to [C, H, W] format
            if rendering_alpha.dim() == 4:  # If shape is [1, H, W, 1]
                rendering_alpha = rendering_alpha.squeeze().unsqueeze(0)  # Convert to [1, H, W]
            
            torchvision.utils.save_image(rendering, os.path.join(view_dir, f"{target_text[text_idx]}.png"))
            torchvision.utils.save_image(rendering_alpha, os.path.join(view_alpha_dir, f"{target_text[text_idx]}.png"))
            count += 1
    return count
    # end_time = time.time()
    # print(f"Time taken for rendering: {end_time - start_time} seconds")
    # print(f"Number of renders: {count}")
    # print(f"Average time per render: {(end_time - start_time) / count} seconds")

def evaluate(gaussians, model=None, clip_model=None, thresh=0.4, num_knn=10, device="cuda"):
    assert clip_model is not None
    # assert model is not None  # temporarily disabled
    
    # load language feature field on 3DGS and restore to 512
    gs_lang_feat = gaussians.get_language_feature() ##(N, 512)
    
    valid_map_3d = clip_model.get_max_across_3d(gs_lang_feat)
    n_prompt, n = valid_map_3d.shape
    
    # smooth the relevancy map, similar to in 2D
    gs_xyz_np = gaussians._xyz.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=num_knn).fit(gs_xyz_np)
    _, indices = nbrs.kneighbors(gs_xyz_np)
    indices = torch.from_numpy(indices).to(valid_map_3d.device)
    relv_map_smoothed = torch.zeros_like(valid_map_3d)
    gs_masks_pred = torch.zeros_like(valid_map_3d)
    scores = torch.zeros(n_prompt)
    
    for i in range(n_prompt):
        relv_1d = valid_map_3d[i]  
        neighbors_vals = relv_1d[indices]  
        neighbors_avg = neighbors_vals.mean(dim=1)  
        relv_map_smoothed[i] = 0.5 * (relv_1d + neighbors_avg)
    
        output = relv_map_smoothed[i]
        output = output - torch.min(output)
        output = output / (torch.max(output) + 1e-9)
        output = output * (1.0 - (-1.0)) + (-1.0)
        output = torch.clip(output, 0, 1)
        
        gs_masks_pred[i] = output > thresh
        scores[i] = relv_map_smoothed[i].max()
    
    del gs_lang_feat
    return scores, gs_masks_pred > 0.5
    

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, training_args : OptimizationParams, skip_train : bool, skip_test : bool,
                ae_ckpt_path: str, encoder_hidden_dims: list, decoder_hidden_dims: list, scene_name: str, dataset_name: str, output_dir: str, mask_thresh: float, ablation_type="none", json_dir=None, device="cuda"):
    # if json_dir is not None:
    #     gt_ann, (h, w), img_paths = eval_gt_lerfdata(Path(os.path.join(json_dir, scene_name)), Path(gt_base))
    with torch.no_grad():
        # load AutoEncoder
        # checkpoint = torch.load(ae_ckpt_path, map_location=device)
        # model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
        # model.load_state_dict(checkpoint)
        # model.eval()
        model = None  # temporarily set to None since Autoencoder is not imported
        
        bg_color = [1,1,1]  # white background
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # get text features
        clip_model = OpenCLIPNetwork(device)
        target_text = scene_texts[scene_name]
        clip_model.set_positives(target_text)

        scores = []
        gs_mask_preds = []
        model_path = os.path.join(dataset.model_path, ablation_type)
        for i in range(1,4):
            # dataset.model_path = f"{base_dir}_{i}"
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)
            # checkpoint = os.path.join(model_path, f'chkpnt30000_langfeat_{i}.pth') ##occamlgs
            checkpoint = os.path.join(model_path, f'chkpnt30000_langfeat_{i}_stochastic_gate.pth')

            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore_language_features(model_params, training_args)
            score_lvl, gs_mask_pred_lvl = evaluate(gaussians, model, clip_model, mask_thresh)
            scores.append(score_lvl)
            gs_mask_preds.append(gs_mask_pred_lvl)
        
        
        chosen_levels = torch.argmax(torch.stack(scores), dim=0)
        level_wise_texts = {f"{i}": [] for i in range(1, 4)}
        level_wise_gs_mask_preds = {f"{i}": [] for i in range(1, 4)}
        
        for text_idx, best_level in enumerate(chosen_levels):
            # Map argmax indices (0..2) to saved checkpoints levels (1..3)
            lvl_idx = int(best_level.item())          # 0..2
            lvl_key = lvl_idx + 1                     # 1..3
            level_wise_texts[f"{lvl_key}"].append(text_idx)
            level_wise_gs_mask_preds[f"{lvl_key}"].append(gs_mask_preds[lvl_idx][text_idx])
        
        
        for i in range(1, 4):
            if len(level_wise_texts[f"{i}"]) > 0:
                # dataset.model_path = f"{base_dir}_{i}"
                gaussians = GaussianModel(dataset.sh_degree)
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)
                # checkpoint = os.path.join(model_path, f'chkpnt30000_langfeat_{i}.pth') ##occamlgs
                checkpoint = os.path.join(model_path, f'chkpnt30000_langfeat_{i}_stochastic_gate.pth')
                (model_params, first_iter) = torch.load(checkpoint)
                gaussians.restore_language_features(model_params, training_args)

                print("length of the test cameras:", len(scene.getTestCameras()))
        
                if not skip_train:
                    n_queries = render_set(model_path, scene.getTrainCameras(), gaussians, pipeline, background, scene_name, level_wise_texts[f"{i}"], level_wise_gs_mask_preds[f"{i}"], mask_thresh, i)
                if not skip_test:
                    n_queries = render_set(model_path,scene.getTestCameras(), gaussians, pipeline, background, scene_name, level_wise_texts[f"{i}"], level_wise_gs_mask_preds[f"{i}"], mask_thresh, i)
    return n_queries


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="prompt any label in 3DGS space")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    training_args = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scene_name", type=str, choices=["waldo_kitchen", "ramen", "figurines", "teatime"],
                        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
    parser.add_argument("--dataset_name", type=str, default = "lerf_ovs")
    parser.add_argument("--ae_ckpt_dir", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    parser.add_argument("--base_dir", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--ablation_type", type=str, default="none")
    parser.add_argument("--json_dir", type=str, default=None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if not args.scene_name:
        parser.error("The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh

    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, args.scene_name, "best_ckpt.pth")
    # ae_ckpt_path = "output/3dgs/lerf_ovs/teatime/chkpnt30000_lanfeat_1.pth"

    start_time = time.time()
    count = render_sets(model.extract(args), args.iteration, pipeline.extract(args), training_args.extract(args), args.skip_train, args.skip_test, ae_ckpt_path, args.encoder_dims, args.decoder_dims, args.scene_name, args.dataset_name, args.output_dir, args.mask_thresh, args.ablation_type, args.json_dir)
    # count = render_sets_omniseg3d(model.extract(args), args.iteration, pipeline.extract(args), training_args.extract(args), args.skip_train, args.skip_test, ae_ckpt_path, args.encoder_dims, args.decoder_dims, args.scene_name, args.dataset_name, args.output_dir, args.mask_thresh, args.ablation_type)
    end_time = time.time()
    print(f"Time taken for rendering: {end_time - start_time} seconds")
    print(f"Number of renders: {count}")
    if count > 0:
        print(f"Average time per render: {(end_time - start_time) / count} seconds")
    else:
        print("Average time per render: N/A (no renders)")
