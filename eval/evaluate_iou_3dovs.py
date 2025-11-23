#!/usr/bin/env python
from __future__ import annotations

import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append("..")
from eval import colormaps
from eval.openclip_encoder import OpenCLIPNetwork
from eval.utils import smooth, vis_mask_save, stack_mask

from eval_utils import plot_relevancy_and_threshold, compute_dynamic_threshold


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_3dovsdata(dataset_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    Organizes ground truth annotations from the 3DOVS dataset.
    
    Args:
        dataset_folder: Path to the root directory containing the 3DOVS dataset
        ouput_path: Optional path where visualization images will be saved
        
    Returns:
        Tuple containing:
        - gt_ann: Dictionary where:
            - keys: Frame indices as strings
            - values: Nested dictionary where:
                - keys: Class labels
                - values: Dictionary containing 'mask' with the segmentation mask
        - image_dimensions: Tuple of (height, width)
        - img_paths: List of paths to the original images
    """
    gt_folder = os.path.join(dataset_folder, 'segmentations')
    image_folder = os.path.join(dataset_folder, 'images')
    
    gt_paths = [os.path.join(gt_folder, name) for name in os.listdir(gt_folder) if os.path.isdir(os.path.join(gt_folder, name))]
    gt_paths = sorted(gt_paths, key=lambda x: int(x.split('/')[-1]))
    img_paths = {}
    with open(os.path.join(gt_folder, 'classes.txt'), 'r') as f:
        class_names = [line.strip() for line in f]
        
    gt_ann = {}
    for gt_path in gt_paths:
        img_ann = defaultdict(dict)
        
        idx = int(gt_path.split('/')[-1])
        img_path = os.path.join(image_folder, f"{gt_path.split('/')[-1]}.jpg")
        img_paths[idx] = img_path
        with Image.open(img_path) as img:
            w, h = img.size

        for prompt_data in class_names:
            label = prompt_data
            mask_path = os.path.join(gt_path, f"{label}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
            img_ann[label]['mask'] = mask
            
            # save for visulsization
            save_path = ouput_path / 'gt' / str(idx) / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def activate_stream(sem_map, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    eval_params: Dict = None):
    
    valid_map = clip_model.get_max_across(sem_map)
    n_head, n_prompt, h, w = valid_map.shape
    valid_map = valid_map.cpu()
    
    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    
    for k in range(n_prompt):
        
        chosen_lvl, thresh = compute_dynamic_threshold(valid_map[:, k], clip_model.positives[k], eval_params=eval_params)
        
        for i in range(n_head):
            
            # NOTE [mask] truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) -  torch.min(output) + 1e-9)
            
            save_path = image_name / 'comparison_maps' / f'{clip_model.positives[k]}_level{i}_comparison.png'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plot_relevancy_and_threshold(output, clip_model.positives[k], i, save_path, threshold=thresh)
            
            if i == chosen_lvl:
                # Create Binary Mask through thresholding:
                mask_pred = (output.numpy() > thresh).astype(np.uint8)
                mask_pred = smooth(mask_pred)
                mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
                
                intersection = np.logical_and(mask_gt, mask_pred).sum()
                union = np.logical_or(mask_gt, mask_pred).sum()
                iou = intersection / (union + 1e-9)  # Avoid division by zero
            
        chosen_iou_list.append(iou)
        chosen_lvl_list.append(chosen_lvl)
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_pred, save_path)

    return chosen_iou_list, chosen_lvl_list


def evaluate(feat_dir, output_path, gt_path, logger, eval_params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # colormap_options = colormaps.ColormapOptions(
    #     colormap="turbo",
    #     normalize=True,
    #     colormap_min=-1.0,
    #     colormap_max=1.0,
    # )

    gt_ann, image_shape, image_paths = eval_gt_3dovsdata(Path(gt_path), Path(output_path))
     
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    feat_paths_lvl = []   
    for i in range(len(feat_dir)):
        # Create a mapping of index to file path
        index_to_file = {}
        for file_path in glob.glob(os.path.join(feat_dir[i], '*.npy')):
            file_idx = int(os.path.basename(file_path).split(".npy")[0])
            index_to_file[file_idx] = file_path
        
        feat_paths_lvl.append(index_to_file)
    
    assert len(feat_paths_lvl) == len(feat_dir)   
    
    # instantiate openclip
    clip_model = OpenCLIPNetwork(device)

    chosen_iou_all, chosen_lvl_list = [], []
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx:0>2}'
        image_name.mkdir(exist_ok=True, parents=True)
        
        compressed_sem_feats = np.zeros((len(feat_dir), *image_shape, 512), dtype=np.float32) # compressed_sem_feats: (3, 7, 731, 988, 3) -> (granuity, num_frames, h, w, c)
        for i in range(len(feat_dir)):
            if idx not in feat_paths_lvl[i]:
                raise ValueError(f"Missing feature file for index {idx} in directory {feat_dir[i]}")
            compressed_sem_feats[i] = np.load(feat_paths_lvl[i][idx], mmap_mode='r')
        
        sem_feat = torch.from_numpy(compressed_sem_feats).float().to(device)
        # rgb_img = cv2.imread(image_paths[idx])[..., ::-1]
        # rgb_img = (rgb_img / 255.0).astype(np.float32)
        # rgb_img = torch.from_numpy(rgb_img).to(device)
        print(f"j: {j}, idx: {idx}, image_name: {image_name}, image_path: {image_paths[idx]}") 
        
        img_ann = gt_ann[f'{idx}'] # -> a dictionary of labels, with key as path to mask
        clip_model.set_positives(list(img_ann.keys()))
        
        c_iou_list, c_lvl = activate_stream(sem_feat, clip_model, 
                                            image_name, img_ann,
                                            eval_params=eval_params)

        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

    # iou
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--gt_folder", type=str, default=None)
    parser.add_argument("--feat_folder", type=str, default=None)
    parser.add_argument("--stability_thresh", type=float, default=0.3)
    parser.add_argument("--min_mask_size", type=float, default=0.001)
    parser.add_argument("--max_mask_size", type=float, default=0.95)
    args = parser.parse_args()

    eval_params = {
        "stability_thresh": args.stability_thresh,
        "min_mask_size": args.min_mask_size,
        "max_mask_size": args.max_mask_size,
    }
    dataset_name = args.dataset_name
    feat_dir = [f"./output/3DOVS/{args.dataset_name}/test/{args.feat_folder}_1/renders_npy",
                f"./output/3DOVS/{args.dataset_name}/test/{args.feat_folder}_2/renders_npy",
                f"./output/3DOVS/{args.dataset_name}/test/{args.feat_folder}_3/renders_npy"]
    output_path = f"./eval_results/3DOVS/{args.dataset_name}"
    gt_path = args.gt_folder

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{dataset_name}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    evaluate(feat_dir, output_path, gt_path, logger, eval_params)