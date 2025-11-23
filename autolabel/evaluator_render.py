import cv2
import numpy as np
import os
import cv2
# import open3d as o3d
from plyfile import PlyData
import time
import math
# import hdbscan
from PIL import Image
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json
from autoencoder.model import Autoencoder
import torch.nn.functional as F
import torchvision

from autolabel.constants import COLORS
from autolabel.utils.feature_utils import get_feature_extractor
# from autolabel.dataset import CV_TO_OPENGL
from autolabel import utils
from rich.progress import track
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def compute_iou(p_semantic, gt_semantic, class_index):
    p_semantic = p_semantic == class_index
    gt_semantic = gt_semantic == class_index
    intersection = np.bitwise_and(p_semantic, gt_semantic).sum()
    union = np.bitwise_or(p_semantic, gt_semantic).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def make_legend(axis, semantic_frame, label_mapping):
    classes = np.unique(semantic_frame)
    colors = [COLORS[class_index % COLORS.shape[0]] for class_index in classes]
    prompts = [label_mapping.get(class_id, "unknown") for class_id in classes]
    patches = [
        mpatches.Patch(color=color / 255., label=prompt[:10])
        for color, prompt in zip(colors, prompts)
    ]
    # put those patched as legend-handles into the legend
    axis.legend(handles=patches)

nyu40_dict = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
    26: "paper", 27: "towel", 28: "showercurtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
}

# ScanNet 20 classes
scannet19_dict = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 14: "desk", 16: "curtain",
    24: "refrigerator", 28: "shower curtain", 33: "toilet", 34: "sink",
    36: "bathtub", # 39: "otherfurniture"
}

class OpenVocabEvaluator:

    def __init__(self,
                 device='cuda:0',
                 name="model",
                 features=None,
                 checkpoint=None,
                 debug=False,
                 stride=1,
                 save_figures=None,
                 time=False):
        self.device = device
        self.name = name
        self.debug = debug
        self.stride = stride
        self.model = None
        self.label_id_map = None
        self.label_map = None
        self.features = features
        self.extractor = get_feature_extractor(features, checkpoint, device)
        self.save_figures = save_figures
        self.time = time

    def reset(self, label_map, figure_path):
        # self.model = model
        # target_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34]   # 15
        self.target_id = label_map["id"]
        print("target_id", self.target_id)

        self.label_map = label_map
        self.label_id_map = torch.tensor(self.label_map['id'].values).to(
            self.device)
        self.text_features = self._infer_text_features()
        self.label_mapping = {0: 'void'}
        self.label_to_color_id = np.zeros((label_map['id'].max() + 1),
                                          dtype=int)
        self.label_type_mapping = None
        if 'type' in self.label_map:
            self.label_type_mapping = {0: -1}
            for index, (i, prompt, type) in enumerate(
                    zip(label_map['id'], label_map['prompt'], label_map['type'])):
                self.label_mapping[i] = prompt
                self.label_to_color_id[i] = index + 1
                self.label_type_mapping[i] = type
        else:
            for index, (i, prompt) in enumerate(
                    zip(label_map['id'], label_map['prompt'])):
                if i==39:
                    continue
                self.label_mapping[i] = prompt
                self.label_to_color_id[i] = index + 1
        print("label_mapping", self.label_mapping)
        self.save_figures = figure_path
        os.makedirs(self.save_figures, exist_ok=True)
        if 'evaluated' in self.label_map:
            self.evaluated_labels = label_map[label_map['evaluated'] ==
                                              1]['id'].values
        else:
            self.evaluated_labels = label_map['id'].values
            self.evaluated_labels = self.evaluated_labels[:-1]
        print(self.evaluated_labels)

    def _infer_text_features(self):
        # (2) note: 19 & 15 & 10 classes
        # Given the category ID that needs to be queried (relative to the original NYU40), obtain the corresponding category name.
        # target_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]   # 19
        # target_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34]   # 15
        # target_id = [1,2,4,5,6,7,8,9,10,33] # 10
        target_id = self.target_id

        target_dict = {key: nyu40_dict[key] for key in target_id}
        target_names = list(target_dict.values())
        with open('autolabel/text_features.json', 'r') as f:
            data_loaded = json.load(f)
        all_texts = list(data_loaded.keys())
        text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(torch.float32)  # [num_text, 512]
        
        query_text_feats = torch.zeros(len(target_names), 512).cuda()
        for i, text in enumerate(target_names):
            feat = text_features[all_texts.index(text)].unsqueeze(0)
            query_text_feats[i] = feat
        query_text_feats = F.normalize(query_text_feats, dim=1, p=2)
        return query_text_feats
        # return self.extractor.encode_text(self.label_map['prompt'].values, self.device)

    def eval(self, *args, **kwargs):
        raise NotImplementedError()




class OpenVocabEvaluator2D(OpenVocabEvaluator):

    def eval(self, rgb_path, gt_semantics_path, predict_features_path, ae_ckpt_path, encoder_hidden_dims, decoder_hidden_dims):
        ious = []
        accs = []
        model=None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(ae_ckpt_path, map_location=device)
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
            
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                # print(predict_features_path)
                for i, predict_semantic in enumerate(
                        tqdm(predict_features_path, desc="Evaluating")):

                    rgb = self._read_rgb(rgb_path[i])

                    # print((gt_semantic*mask).shape)
                    # print(gt_semantic.shape[0]*gt_semantic.shape[1])
                    # print(np.sum(mask))
                    predict_feature_path = predict_features_path[i]
                    # p_semantic = np.load(predict_feature_path)
                    # print(predict_feature_path)
                    p_semantic = self._predict_semantic(predict_feature_path, model).cpu().numpy()
                    # p_semantic = self._predict_semantic(predict_feature_path, model).cpu().numpy()
                    # p_semantic = self._predict_semantic(predict_feature_path, model)
                    # if p_semantic.shape[0] != gt_semantic.shape[0] or p_semantic.shape[1] != gt_semantic.shape[1]:
                    #     # print(p_semantic.shape)
                    #     p_semantic = p_semantic.view(1, -1, p_semantic.shape[0], p_semantic.shape[1]).float()
                    #     p_semantic = F.interpolate(p_semantic, size=(gt_semantic.shape[0], gt_semantic.shape[1]), mode='nearest')
                    #     p_semantic = p_semantic.long()
                        
                    # p_semantic = p_semantic.squeeze().cpu().numpy()
                    # p_semantic = self._predict_semantic(predict_feature_path)
                    # if p_semantic.shape[0] != gt_semantic.shape[0] or p_semantic.shape[1] != gt_semantic.shape[1]:
                    #     p_semantic = F.interpolate(p_semantic, size=(gt_semantic.shape[0], gt_semantic.shape[1]), mode='nearest')

                    # p_semantic = p_semantic.cpu().numpy()
                    
                    if self.debug:
                        axis = plt.subplot2grid((1, 2), loc=(0, 0))
                        p_sem = self.label_to_color_id[p_semantic]
                        p_sem_vis = COLORS[p_sem % COLORS.shape[0]]
                        axis.imshow(rgb)
                        axis.imshow(p_sem_vis, alpha=0.5)

                        axis.set_title(f"IoU")
                        axis.axis('off')
                        make_legend(axis, p_sem, self.label_mapping)

                        plt.tight_layout()
                        plt.show()
                        save_debug_path = os.path.join(self.save_figures, "debug")
                        os.makedirs(save_debug_path, exist_ok=True)
                        save_fig = os.path.join(save_debug_path, f"{i:06}.png")
                        plt.savefig(save_fig)

                    if self.save_figures is not None:
                        self._save_figure(p_semantic, rgb, i, predict_feature_path, model)
                        
                    for i, prompt in zip(self.label_map['id'].values,
                                         self.label_map['prompt'].values):
                        if i not in self.evaluated_labels:
                            continue
                        gt_mask = gt_semantic[mask] == i
                        if gt_mask.sum() <= 0:
                            continue
                        p_mask = p_semantic[mask] == i

                        true_positive = np.bitwise_and(p_mask, gt_mask).sum()
                        true_negative = np.bitwise_and(p_mask == False,
                                                       gt_mask == False).sum()
                        false_positive = np.bitwise_and(p_mask,
                                                        gt_mask == False).sum()
                        false_negative = np.bitwise_and(p_mask == False,
                                                        gt_mask).sum()

                        iou[prompt] = (true_positive, true_positive +
                                       false_positive + false_negative)
                        acc[prompt] = (true_positive,
                                       true_positive + false_positive)
                    ious.append(iou)
                    accs.append(acc)

        if len(ious) == 0:
            print(f"Scene {self.name} has no labels in the evaluation set")
            return {}
        out_iou = {}
        out_acc = {}
        for key in ious[0].keys():
            iou_values = [
                iou[key] for iou in ious if iou.get(key, None) is not None
            ]
            acc_values = [
                acc[key] for acc in accs if acc.get(key, None) is not None
            ]
            if len(iou_values) == 0:
                out_iou[key] = None
                out_acc[key] = None
            else:
                intersection = sum([value[0] for value in iou_values])
                union = sum([value[1] for value in iou_values])
                out_iou[key] = intersection / union
                numerator = sum([value[0] for value in acc_values])
                denominator = sum([value[1] for value in acc_values])
                if denominator == 0:
                    out_acc[key] = 0.0
                else:
                    out_acc[key] = numerator / denominator
        out_iou['total'] = np.mean(list(out_iou.values()))
        out_acc['total'] = np.mean(list(out_acc.values()))
        return out_iou, out_acc


    def _save_figure(self, p_semantic, rgb, example_index, predict_feature_path, model):
        rgb_path = os.path.join(self.save_figures, 'rgb')
        p_path = os.path.join(self.save_figures, 'p_semantic')
        feature_path = os.path.join(self.save_figures, "feat_map")
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(p_path, exist_ok=True)
        os.makedirs(feature_path, exist_ok=True)

        # Expand mask to match RGB dimensions
        
        # Apply mask to RGB image
        # masked_rgb = rgb * mask_3d
        masked_rgb = rgb
        Image.fromarray(masked_rgb).save(
            os.path.join(rgb_path, f"{example_index:06}.png"))

        # Handle semantic predictions
        p_sem = self.label_to_color_id[p_semantic]
        p_sem_vis = COLORS[p_sem % COLORS.shape[0]]
        p_sem_vis[p_sem_vis == 0] = 0
        Image.fromarray(p_sem_vis).save(
            os.path.join(p_path, f"{example_index:06}.png"))
 
        overlay = cv2.addWeighted(masked_rgb, 
                                  0.5, p_sem_vis, 0.5, 0)
        Image.fromarray(overlay).save(
            os.path.join(p_path, f"{example_index:06}_overlay.png"))
        
        predict_features = self.load_predict_features(predict_feature_path)
        predict_features = predict_features.float().to(self.device)
        # print(predict_features.shape)
        H, W = predict_features.shape[0], predict_features.shape[1]
        if predict_features.shape[-1]==512:
            language_feature_dim3 = model.encode(predict_features.view(-1, 512).float())
        else:
            language_feature_dim3 = predict_features
        language_feature_dim3 = language_feature_dim3.reshape(H, W, 3).permute(2, 0, 1)
        language_feature_norm = (language_feature_dim3 + 1) / 2
        torchvision.utils.save_image(language_feature_norm, os.path.join(feature_path, f"{example_index:06}.png"))
        
    def load_predict_features(self, predict_feature_path):
        """Load prediction features from file path.
        
        Args:
            predict_feature_path (str): Path to the feature file (.pt or .npy)
            
        Returns:
            torch.Tensor: Loaded features tensor on the specified device
        """
        if predict_feature_path.endswith("pt"):
            predict_features = torch.load(predict_feature_path)
            predict_features = predict_features.float().to(self.device)
        else:
            if predict_feature_path.endswith('.npy'):
                predict_features = np.load(predict_feature_path)
            else:
                predict_features = np.load(predict_feature_path)["features"]
            predict_features = torch.from_numpy(predict_features).float().to(self.device)
        return predict_features

    def _predict_semantic(self, predict_feature_path, model=None):
        if self.time:
            start = time.time()
        predict_features = self.load_predict_features(predict_feature_path)
        if predict_features.shape[-1] ==512:
            restored_feat = predict_features
        elif predict_features.shape[-1] == 3:
            with torch.no_grad():
                h, w, _ = predict_features.shape
                restored_feat = model.decode(predict_features.flatten(0, 1))
                restored_feat = restored_feat.view(h, w, -1)
        else:
            restored_feat = predict_features.permute(1, 2, 0)
        restored_feat = (restored_feat / torch.norm(restored_feat, dim=-1, keepdim=True))
        text_features = self.text_features
        # print(text_features.shape)
        H, W, D = restored_feat.shape
        C = text_features.shape[0]
        similarities = torch.zeros((H, W, C),
                                   dtype=restored_feat.dtype,
                                   device=self.device)
        for i in range(H):
            similarities[i, :, :] = (restored_feat[i, :, None] *
                                     text_features).sum(dim=-1)
        similarities = similarities.argmax(dim=-1)
        if self.time:
            torch.cuda.synchronize()
            end = time.time()
            n_pixels = H * W
            pixels_per_second = n_pixels / (end - start)
            print(
                f"Semantic prediction for {n_pixels} took {end - start} seconds. {pixels_per_second} pixels per second."
            )
        return self.label_id_map[similarities]

    def _read_gt_semantic(self, path):
        semantic = np.array(
            Image.open(path)).astype(np.int64)
        return semantic

    def _read_rgb(self, path):
        rgb = np.array(
            Image.open(path)).astype(np.uint8)
        return rgb

class OpenVocabEvaluator3D(OpenVocabEvaluator):

    def eval(self, dataset, visualize=False):
        point_cloud, gt_semantic = self._read_gt_pointcloud(dataset)
        iou = {}
        acc = {}
        iou_d = {}
        acc_d = {}
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                self.model.eval()
                p_semantic = self._predict_semantic(point_cloud).cpu().numpy()
                p_instance = self._predict_instance(point_cloud)
                p_semantic_denoised = self._denoise_semantic(p_semantic, p_instance)
                mask = np.isin(gt_semantic, self.evaluated_labels)
                intersection = np.bitwise_and(p_semantic == gt_semantic,
                                              mask).sum()

                union = mask.sum()
                if union == 0:
                    print(
                        f"Skipping {self.name} because no labels are in the list of valid labels."
                    )
                    return {}, {}

                if self.debug:
                    pc_vis = point_cloud.cpu().numpy()[mask]
                    pc_vis = o3d.utility.Vector3dVector(pc_vis)
                    pc_vis = o3d.geometry.PointCloud(pc_vis)

                    p_sem = self.label_to_color_id[p_semantic][mask]
                    colors = COLORS[p_sem % COLORS.shape[0]] / 255.
                    pc_vis.colors = o3d.utility.Vector3dVector(colors)

                    o3d.visualization.draw_geometries([pc_vis])

                    gt_sem = self.label_to_color_id[gt_semantic[mask]]
                    gt_colors = COLORS[gt_sem % COLORS.shape[0]] / 255.
                    pc_vis.colors = o3d.utility.Vector3dVector(gt_colors)
                    o3d.visualization.draw_geometries([pc_vis])

                for i, prompt in zip(self.label_map['id'].values,
                                     self.label_map['prompt'].values):
                    if i not in self.evaluated_labels:
                        continue

                    object_mask = gt_semantic[mask] == i
                    if object_mask.sum() == 0:
                        continue
                    p_mask = p_semantic[mask]
                    true_positive = np.bitwise_and(p_mask == i,
                                                   object_mask).sum()
                    true_negative = np.bitwise_and(p_mask != i,
                                                   object_mask == False).sum()
                    false_positive = np.bitwise_and(p_mask == i,
                                                    object_mask == False).sum()
                    false_negative = np.bitwise_and(p_mask != i,
                                                    object_mask).sum()

                    class_iou = float(true_positive) / (
                        true_positive + false_positive + false_negative)
                    iou[prompt] = class_iou
                    acc[prompt] = float(true_positive) / (true_positive +
                                                          false_negative)
                    
                    p_mask = p_semantic_denoised[mask]
                    true_positive = np.bitwise_and(p_mask == i,
                                                   object_mask).sum()
                    true_negative = np.bitwise_and(p_mask != i,
                                                   object_mask == False).sum()
                    false_positive = np.bitwise_and(p_mask == i,
                                                    object_mask == False).sum()
                    false_negative = np.bitwise_and(p_mask != i,
                                                    object_mask).sum()
                    class_iou = float(true_positive) / (
                        true_positive + false_positive + false_negative)
                    iou_d[prompt] = class_iou
                    acc_d[prompt] = float(true_positive) / (true_positive +
                                                          false_negative)
        iou['total'] = np.mean(list(iou.values()))
        acc['total'] = np.mean(list(acc.values()))
        iou_d['total'] = np.mean(list(iou_d.values()))
        acc_d['total'] = np.mean(list(acc_d.values()))
        return iou, acc, iou_d, acc_d

    def _predict_semantic(self, points):
        similarities = torch.zeros(
            (points.shape[0], self.text_features.shape[0]),
            dtype=self.text_features.dtype,
            device=self.device)
        batch_size = 50000
        batches = math.ceil(points.shape[0] / batch_size)
        for batch_index in range(batches):
            batch = points[batch_index * batch_size:(batch_index + 1) *
                           batch_size]
            if self.time:
                start = time.time()
            density = self.model.density(batch)
            features = self.model.semantic(density['geo_feat'])

            if self.time:
                torch.cuda.synchronize()
                first_batch = time.time()

            N = 10
            scale = 1.0 / N
            for _ in range(N - 1):
                noise = torch.randn_like(batch) * 0.02
                density = self.model.density(batch + noise)
                f = self.model.semantic(density['geo_feat'])
                features += f * scale
            features = features / torch.norm(features, dim=-1, keepdim=True)
            if self.time:
                torch.cuda.synchronize()
                duration = time.time() - start
                point_count = batch.shape[0] * N
                points_per_sec = point_count / duration
                print(
                    f"Semantic prediction took {duration:.2f} seconds for {point_count} points. {points_per_sec:.2f} points per second."
                )
                duration_ms = (first_batch - start) * 1000
                print(f"Query latency: {duration_ms:.4f} ms")
            for i in range(self.text_features.shape[0]):
                similarities[batch_index * batch_size:(batch_index + 1) *
                             batch_size,
                             i] = (features *
                                   self.text_features[i][None]).sum(dim=-1)
        similarities = similarities.argmax(dim=-1)
        return self.label_id_map[similarities]

    def _predict_instance(self, points):
        pred_instances = np.zeros((points.shape[0], ), dtype=np.int)
        batch_size = 50000
        batches = math.ceil(points.shape[0] / batch_size)
        for batch_index in range(batches):
            batch = points[batch_index * batch_size:(batch_index + 1) *
                           batch_size]
            xyz_feature_encoding = self.model.feature_encoder(batch, bound=self.model.bound)
            instance_feature = self.model.contrastive(xyz_feature_encoding, None)
            # instance_feature = instance_feature.reshape(-1, feature_dim)
            instance_feature = instance_feature.cpu().numpy()
            sim_mat = cosine_similarity(instance_feature, self.model.instance_centers)
            pred_instance = np.argmax(sim_mat, axis=1) + 1 # start from 1, 0 means noise
            pred_instances[batch_index * batch_size:(batch_index + 1) *
                             batch_size] = pred_instance
        return pred_instances

    def _denoise_semantic(self, p_semantic, p_instance):
        p_semantic_denoised = np.copy(p_semantic)
        instance_ids = np.unique(p_instance)
        for ins_id in instance_ids:
            if ins_id == 0:
                continue
            
            semantic_ids = p_semantic[p_instance == ins_id]
            ids, cnts = np.unique(semantic_ids, return_counts=True)
            p_semantic_denoised[p_instance == ins_id] = ids[np.argmax(cnts)]
        return p_semantic_denoised

    def _read_gt_pointcloud(self, dataset):
        scene_path = dataset.scene.path
        mesh_path = os.path.join(scene_path, 'mesh.ply')
        gt_semantic_path = os.path.join(scene_path, 'mesh_labels.npy')
        semantic = np.load(gt_semantic_path)
        
        plydata = PlyData.read(mesh_path)
        points = np.hstack([
            plydata['vertex']['x'].reshape(-1, 1),
            plydata['vertex']['y'].reshape(-1, 1),
            plydata['vertex']['z'].reshape(-1, 1)
        ])
        aabb = dataset.scene.bbox()
        scene_center = (aabb[0] + aabb[1]) / 2
        points = points - scene_center
        fixed = np.zeros_like(points)
        fixed[:, 0] = points[:, 1]
        fixed[:, 1] = points[:, 2]
        fixed[:, 2] = points[:, 0]
        points = torch.tensor(fixed, dtype=torch.float16, device=self.device)
        semantics = semantic.astype(int)

        return points, semantics


# -----
# Adapt from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
from collections import defaultdict
OFFSET = 256 * 256 * 256
VOID = 0 # or -1

class PanopticStatCat():
        def __init__(self):
            # panoptic segmentation evaluation
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

            # semantic segmentation evaluation
            self.semantic = {'iou': 0.0, 'acc': 0.0}
            self.semantic_denoised = {'iou': 0.0, 'acc': 0.0}
            self.semantic_n = 0

        def __iadd__(self, panoptic_stat_cat):
            self.iou += panoptic_stat_cat.iou
            self.tp += panoptic_stat_cat.tp
            self.fp += panoptic_stat_cat.fp
            self.fn += panoptic_stat_cat.fn
            self.semantic['iou'] += panoptic_stat_cat.semantic['iou']
            self.semantic['acc'] += panoptic_stat_cat.semantic['acc']
            self.semantic_denoised['iou'] += panoptic_stat_cat.semantic_denoised['iou']
            self.semantic_denoised['acc'] += panoptic_stat_cat.semantic_denoised['acc']
            self.semantic_n += panoptic_stat_cat.semantic_n
            return self


class PanopticStat():
    def __init__(self):
        self.panoptic_per_cat = defaultdict(PanopticStatCat)
        self.instance_stat = {
            'coverage': [],
            'gt_inst_area': [],
            'num_pred_inst': 0,
            'num_gt_inst': 0,
        }

    def __getitem__(self, i):
        return self.panoptic_per_cat[i]

    def __iadd__(self, panoptic_stat):
        for label, panoptic_stat_cat in panoptic_stat.panoptic_per_cat.items():
            self.panoptic_per_cat[label] += panoptic_stat_cat
        self.instance_stat['coverage'].extend(panoptic_stat.instance_stat['coverage'])
        self.instance_stat['gt_inst_area'].extend(panoptic_stat.instance_stat['gt_inst_area'])
        self.instance_stat['num_pred_inst'] += panoptic_stat.instance_stat['num_pred_inst']
        self.instance_stat['num_gt_inst'] += panoptic_stat.instance_stat['num_gt_inst']
        return self

    def pq_average(self, categories, label_type_mapping, instance_type='all', verbose=False):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        tp_all, fp_all, fn_all = 0, 0, 0
        for label in categories:
            iou = self.panoptic_per_cat[label].iou
            tp = self.panoptic_per_cat[label].tp
            fp = self.panoptic_per_cat[label].fp
            fn = self.panoptic_per_cat[label].fn
            if tp + fp + fn == 0:
                if verbose:
                    per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'valid': False, 'tp': tp, 'fp': fp, 'fn': fn}
                else:
                    per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'valid': False}
                continue
            
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            if verbose:
                per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'valid': True, 'tp': tp, 'fp': fp, 'fn': fn}
            else:
                per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'valid': True}
            
            # only evaluate instances of "thing" type
            if label_type_mapping is not None:
                if instance_type == 'thing' and label_type_mapping[label] != 1:
                    continue
                if instance_type == 'stuff' and label_type_mapping[label] != 0:
                    continue

            pq += pq_class
            sq += sq_class
            rq += rq_class
            tp_all += tp
            fp_all += fp
            fn_all += fn
            n += 1

        if verbose:
            return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n, 
                    'tp': tp_all / n, 'fp': fp_all / n, 'fn': fn_all / n}, per_class_results
        else:
            return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results
    
    def instance_average(self, iou_threshold=0.5):
        stat_coverage = np.array(self.instance_stat['coverage'])
        stat_gt_inst_area = np.array(self.instance_stat['gt_inst_area'])
        coverage = np.mean(stat_coverage)
        weighted_coverage = np.sum((stat_gt_inst_area / stat_gt_inst_area.sum()) * stat_coverage)
        prec = (stat_coverage > iou_threshold).sum() / self.instance_stat['num_pred_inst']
        rec = (stat_coverage > iou_threshold).sum() / self.instance_stat['num_gt_inst']
        return {'mCov': coverage, 'mWCov': weighted_coverage, 'mPrec': prec, 'mRec': rec}
    
    def semantic_average(self, categories):
        iou, acc, iou_d, acc_d, n = 0, 0, 0, 0, 0
        per_class_results = {}
        for label in categories:
            if self.panoptic_per_cat[label].semantic_n == 0:
                per_class_results[label] = {'iou': 0.0, 'acc': 0.0, 'iou_d': 0.0, 'acc_d': 0.0, 'valid': False}
                continue
            n += 1
            iou_class = self.panoptic_per_cat[label].semantic['iou'] / self.panoptic_per_cat[label].semantic_n
            acc_class = self.panoptic_per_cat[label].semantic['acc'] / self.panoptic_per_cat[label].semantic_n
            iou_d_class = self.panoptic_per_cat[label].semantic_denoised['iou'] / self.panoptic_per_cat[label].semantic_n
            acc_d_class = self.panoptic_per_cat[label].semantic_denoised['acc'] / self.panoptic_per_cat[label].semantic_n
            per_class_results[label] = {'iou': iou_class, 'acc': acc_class, 'iou_d': iou_d_class, 'acc_d': acc_d_class, 'valid': True}
            iou += iou_class
            acc += acc_class
            iou_d += iou_d_class
            acc_d += acc_d_class
        return {'iou': iou / n, 'acc': acc / n, 'iou_d': iou_d / n, 'acc_d': acc_d / n, 'n': n}, per_class_results

# -----

class OpenVocabInstancePQEvaluator(OpenVocabEvaluator):

    def __init__(self, 
                 device='cuda:0', 
                 name="model", 
                 features=None, 
                 checkpoint=None, 
                 debug=False, 
                 stride=1, 
                 save_figures=None, 
                 time=False,
                 denoise_method='average_similarity'):
        super().__init__(device, name, features, checkpoint, debug, stride, save_figures, time)
        self.denoise_method = denoise_method

    def eval(self, dataset):
        self.panoptic_stat = PanopticStat()

        # process frames
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                self.model.eval()
                if self.model.instance_centers is None:
                    self._compute_instance_centers(dataset)
                pred_instances, semantic_similarities, gt_images, gt_semantics, gt_instances, indices = self._process_frames(dataset)

        # predict semantic labels
        pred_semantics, pred_semantic_denoiseds = self._predict_semantics(
            semantic_similarities, pred_instances, method=self.denoise_method)
        # evaluate semantic segmentation
        self._evaluate_semantic(gt_semantics, pred_semantics, pred_semantic_denoiseds, indices)

        # label remapping for gt
        gt_instances, gt_thing_ids = self._instance_label_remapping(gt_instances, gt_semantics)

        # evaluate instance segmentation
        self._evaluate_instance(gt_instances, gt_thing_ids, pred_instances, indices)

        # label remapping for prediction
        pred_instances, pred_thing_ids = self._instance_label_remapping(pred_instances, pred_semantic_denoiseds)

        # evaluate panoptic segmentation
        self._evaluate_panoptic(
            pred_instances=pred_instances,
            pred_semantics=pred_semantics,
            pred_semantic_denoiseds=pred_semantic_denoiseds,
            gt_images=gt_images,
            gt_semantics=gt_semantics,
            gt_instances=gt_instances,
            indices=indices
        )

        return self.panoptic_stat

    def _process_frames(self, dataset):
        pred_instances = []
        semantic_similarities = []
        gt_images = []
        gt_semantics = []
        gt_instances = []
        indices = []
        gt_instance_paths = dataset.scene.gt_instance()
        gt_semantic_paths = dataset.scene.gt_semantic()
        for i, (gt_semantic_path, gt_instance_path) in enumerate(
                tqdm(list(zip(gt_semantic_paths, gt_instance_paths)), desc="Processing")):
            if i % self.stride != 0:
                continue
            indices.append(i)
            batch = dataset._get_test(i)
            gt_images.append(batch['pixels'])

            # read gt semantic and gt instance
            gt_semantic = np.array(
                Image.open(gt_semantic_path).resize(dataset.camera.size, Image.NEAREST)).astype(np.int64)
            gt_instance = np.array(
                Image.open(gt_instance_path).resize(dataset.camera.size, Image.NEAREST)).astype(np.int64)
            gt_semantics.append(gt_semantic)
            gt_instances.append(gt_instance)

            # get instance and semantic features
            rays_o = torch.tensor(batch['rays_o']).to(self.device)
            rays_d = torch.tensor(batch['rays_d']).to(self.device)
            direction_norms = torch.tensor(batch['direction_norms']).to(self.device)
            outputs = self.model.render(rays_o,
                                        rays_d,
                                        direction_norms,
                                        staged=True,
                                        perturb=False)
            instance_feature = outputs['contrastive_features'].cpu().numpy()
            image_height, image_width, feature_dim = instance_feature.shape
            instance_feature = instance_feature.reshape(-1, feature_dim)
            sim_mat = cosine_similarity(instance_feature, self.model.instance_centers)
            pred_instance = np.argmax(sim_mat, axis=1)
            pred_instance = pred_instance.reshape(image_height, image_width) + 1 # start from 1, 0 means noise
            pred_instances.append(pred_instance)

            semantic_feature = outputs['semantic_features']
            semantic_feature = (semantic_feature / torch.norm(semantic_feature, dim=-1, keepdim=True))
            similarity = semantic_feature @ self.text_features.T
            similarity = similarity.cpu().numpy()
            semantic_similarities.append(similarity)
        
        pred_instances = np.stack(pred_instances, axis=0)
        semantic_similarities = np.stack(semantic_similarities, axis=0)
        gt_images = np.stack(gt_images, axis=0)
        gt_semantics = np.stack(gt_semantics, axis=0)
        gt_instances = np.stack(gt_instances, axis=0)
        indices = np.array(indices)
        return pred_instances, semantic_similarities, gt_images, gt_semantics, gt_instances, indices

    def _compute_instance_centers(self, dataset):
        instance_features = []
        for i in tqdm(dataset.indices[::self.stride], desc="Computing instance features"):
            batch = dataset._get_test(i)
            # get instance and semantic features
            rays_o = torch.tensor(batch['rays_o']).to(self.device)
            rays_d = torch.tensor(batch['rays_d']).to(self.device)
            direction_norms = torch.tensor(batch['direction_norms']).to(self.device)
            outputs = self.model.render(rays_o,
                                        rays_d,
                                        direction_norms,
                                        staged=True,
                                        perturb=False)
            instance_feature = outputs['contrastive_features'].cpu().numpy()
            instance_features.append(instance_feature)
        instance_features = np.stack(instance_features, axis=0)
        instance_centers, clusterer = self._feature_clustering(instance_features)
        self.model.set_instance_centers(instance_centers)
        self.model.set_instance_clusterer(clusterer)
        return instance_centers

    def _feature_clustering(self, features):
        # currently cpu implementation
        # TODO: gpu implementation (cuml)
        num_image, image_height, image_width, feature_dim = features.shape
        features = features.reshape(-1, feature_dim)
        clust = hdbscan.HDBSCAN(min_cluster_size=100, gen_min_span_tree=True) # cluster size depends on the image size
        sample_indices = np.random.permutation(features.shape[0])[:200000]
        clust.fit(features[sample_indices, :])

        exemplar = [np.mean(exemplars, axis=0) for exemplars in clust.exemplars_]
        exemplar = np.vstack(exemplar)
        return exemplar, clust
    
    def _predict_semantics(self, semantic_similarities, pred_instances, method='average_similarity'):
        pred_semantics = np.argmax(semantic_similarities, axis=-1)

        # denoised semantic
        pred_semantic_denoiseds = np.copy(pred_semantics)
        instance_ids = np.unique(pred_instances)
        for ins_id in instance_ids:
            if ins_id == VOID:
                continue
            
            if method == 'majority_voting':
                semantic_ids = pred_semantics[pred_instances == ins_id]
                ids, cnts = np.unique(semantic_ids, return_counts=True)
                pred_semantic_denoiseds[pred_instances == ins_id] = ids[np.argmax(cnts)]
            elif method == 'average_similarity':
                sim = semantic_similarities[pred_instances == ins_id]
                sim = np.mean(sim, axis=0)
                s_id = np.argmax(sim, axis=-1)
                pred_semantic_denoiseds[pred_instances == ins_id] = s_id
            elif method == 'average_feature':
                # currently unavailable due to the memory size.
                # need better implementation strategy
                # TODO
                raise NotImplementedError()
                feats = semantic_features[pred_instances == ins_id]
                feats = np.mean(feats, dim=0)
                feats = feats / np.norm(feats, order=2, axis=-1)
                sim = feats @ text_features.T
                s_id = np.argmax(sim, axis=-1)
                pred_semantic_denoiseds[pred_instances == ins_id] = s_id
            else:
                raise NotImplementedError()

        label_id_map = self.label_id_map.cpu().numpy()
        pred_semantics = label_id_map[pred_semantics]
        pred_semantic_denoiseds = label_id_map[pred_semantic_denoiseds]
        return pred_semantics, pred_semantic_denoiseds
    
    def _instance_label_remapping(self, instances, semantics):
        if 'type' not in self.label_map:
            return instances
        
        stuff_id_mapping = {}
        thing_id_list = []
        instance_ids = np.unique(instances)
        new_instance_id = np.max(instance_ids) + 1

        void_mask = np.isin(instances, [VOID])
        if void_mask.sum() != 0:
            s_labels = np.unique(semantics[void_mask])
            for s_id in s_labels:
                if s_id not in self.evaluated_labels:
                    continue
                else:
                    instances[np.logical_and(
                        void_mask, semantics == s_id
                    )] = new_instance_id
                    new_instance_id += 1

        for ins_id in instance_ids:
            if ins_id == VOID:
                continue
            s_labels = semantics[instances == ins_id]
            s_ids, cnts = np.unique(s_labels, return_counts=True)
            s_id = s_ids[np.argmax(cnts)]
            
            if s_id not in self.evaluated_labels:
                instances[instances == ins_id] = VOID

            elif s_id in self.evaluated_labels and self.label_type_mapping[s_id] == 0:
                if s_id not in stuff_id_mapping.keys():
                    stuff_id_mapping[s_id] = ins_id
                else:
                    instances[instances == ins_id] = stuff_id_mapping[s_id]
            
            elif s_id in self.evaluated_labels and self.label_type_mapping[s_id] == 1:
                thing_id_list.append(ins_id)
        return instances, thing_id_list
    
    def _read_gt_panoptic_segmentation(self, semantic, instance):
        gt_segms = {}
        labels, labels_cnt = np.unique(instance, return_counts=True)

        for label, label_cnt in zip(labels, labels_cnt):
            if label == VOID:
                continue
            semantic_ids = semantic[instance == label]
            ids, cnts = np.unique(semantic_ids, return_counts=True)
            gt_segms[label] = {
                'area': label_cnt,
                'category_id': ids[np.argmax(cnts)]
            }
        return gt_segms
    
    def _predict_panoptic_segmentation(self, pred_instance, pred_semantic):
        # construct panoptic segmentation
        pred_segms = {}
        labels, labels_cnt = np.unique(pred_instance, return_counts=True)

        for label, label_cnt in zip(labels, labels_cnt):
            if label == VOID:
                continue
            semantic_ids = pred_semantic[pred_instance == label]
            ids, cnts = np.unique(semantic_ids, return_counts=True)
            pred_segms[label] = {
                'area': label_cnt,
                'category_id': ids[np.argmax(cnts)]
            }

        return pred_segms

    def _evaluate_semantic(self, gt_semantics, pred_semantics, pred_semantic_denoiseds, indices):

        if self.debug:
            semantic_label_color_mapping = {}
            labels = np.unique(
                np.append(
                    np.unique(gt_semantics), np.unique(pred_semantics)
                )
            )
            for label in labels:
                color = np.random.rand(3, )
                semantic_label_color_mapping[label] = color
        
        for gt_semantic, pred_semantic, pred_semantic_denoised, index in tqdm(
            list(zip(gt_semantics, pred_semantics, pred_semantic_denoiseds, indices)), desc="Evaluating semantic segmentation"):

            mask = np.isin(gt_semantic, self.evaluated_labels)
            labels = np.unique(gt_semantic)
            for label in labels:
                if label not in self.evaluated_labels:
                    continue
                object_mask = gt_semantic[mask] == label

                # semantic
                pred_mask = pred_semantic[mask] == label
                true_positive = np.bitwise_and(pred_mask, object_mask).sum()
                false_positive = np.bitwise_and(pred_mask,
                                                object_mask == False).sum()
                false_negative = np.bitwise_and(pred_mask == False,
                                                object_mask).sum()

                class_iou = float(true_positive) / (
                    true_positive + false_positive + false_negative)
                self.panoptic_stat[label].semantic['iou'] += class_iou
                self.panoptic_stat[label].semantic['acc'] += float(true_positive) / (true_positive + false_negative)

                # denoised semantic
                pred_mask_denoised = pred_semantic_denoised[mask] == label
                true_positive = np.bitwise_and(pred_mask_denoised, object_mask).sum()
                false_positive = np.bitwise_and(pred_mask_denoised,
                                                object_mask == False).sum()
                false_negative = np.bitwise_and(pred_mask_denoised == False,
                                                object_mask).sum()

                class_iou = float(true_positive) / (
                    true_positive + false_positive + false_negative)
                self.panoptic_stat[label].semantic_denoised['iou'] += class_iou
                self.panoptic_stat[label].semantic_denoised['acc'] += float(true_positive) / (true_positive + false_negative)

                self.panoptic_stat[label].semantic_n += 1
            
            if self.debug:
                plt.figure(figsize=(30, 10))
                axis = plt.subplot2grid((1, 3), loc=(0, 0))
                p_s = np.zeros((pred_semantic.shape[0], pred_semantic.shape[1], 3))
                labels = np.unique(pred_semantic)
                s_patches = []
                for label in labels:
                    color = semantic_label_color_mapping[label]
                    p_s[pred_semantic == label] = color
                    s_patches.append(mpatches.Patch(color=color, label=self.label_mapping[label][:10]))
                axis.imshow(p_s)
                axis.set_title("Predicted Semantic")
                axis.axis('off')
                axis.legend(handles=s_patches[:20])

                axis = plt.subplot2grid((1, 3), loc=(0, 1))
                p_sd = np.zeros((pred_semantic_denoised.shape[0], pred_semantic_denoised.shape[1], 3))
                labels = np.unique(pred_semantic_denoised)
                s_patches = []
                for label in labels:
                    color = semantic_label_color_mapping[label]
                    p_sd[pred_semantic_denoised == label] = color
                    s_patches.append(mpatches.Patch(color=color, label=self.label_mapping[label][:10]))
                axis.imshow(p_sd)
                axis.set_title("Predicted Denoised Semantic")
                axis.axis('off')
                axis.legend(handles=s_patches[:20])

                axis = plt.subplot2grid((1, 3), loc=(0, 2))
                gt_s = np.zeros((gt_semantic.shape[0], gt_semantic.shape[1], 3))
                labels = np.unique(gt_semantic)
                s_patches = []
                for label in labels:
                    color = semantic_label_color_mapping[label]
                    gt_s[gt_semantic == label] = color
                    s_patches.append(
                        mpatches.Patch(
                            color=color, 
                            label=self.label_mapping[label][:10] if label in self.label_mapping.keys() else "otherprop"
                        )
                    )
                axis.imshow(gt_s)
                axis.set_title("GT Semantic")
                axis.axis('off')
                axis.legend(handles=s_patches[:20])
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_figures, '{:06}_semantic.png'.format(index)))
                plt.close()
    
    def _evaluate_instance(self, gt_instances, gt_thing_ids, pred_instances, indices):
        if self.debug:
            pred_instance_label_color_mapping = {}
            gt_instance_label_color_mapping = {}

        print("Evaluating instance segmentation ...")
        gt_inst_ids, gt_inst_areas = np.unique(gt_instances, return_counts=True)
        for gt_inst_id, gt_inst_area in zip(gt_inst_ids, gt_inst_areas):
            if gt_inst_id not in gt_thing_ids:
                continue
            gt_inst_mask = gt_instances == gt_inst_id
            pred_inst_ids, pred_gt_intersections = np.unique(
                pred_instances[gt_inst_mask], return_counts=True)
            if len(pred_gt_intersections) == 0:
                self.panoptic_stat.instance_stat['coverage'].append(0)
            else:
                index = np.argmax(pred_gt_intersections)
                matched_pred_inst_id = pred_inst_ids[index]
                matched_pred_gt_intersection = pred_gt_intersections[index]
                matched_pred_inst_mask = pred_instances == matched_pred_inst_id
                iou = matched_pred_gt_intersection / (np.sum(matched_pred_inst_mask) + np.sum(gt_inst_mask) - matched_pred_gt_intersection)
                self.panoptic_stat.instance_stat['coverage'].append(iou)
            self.panoptic_stat.instance_stat['gt_inst_area'].append(gt_inst_area)
            
            if self.debug:
                    color = np.random.rand(3, )
                    pred_instance_label_color_mapping[matched_pred_inst_id] = color
                    gt_instance_label_color_mapping[gt_inst_id] = color
        
        gt_inst_mask = np.isin(gt_instances, gt_thing_ids)
        pred_inst_ids = np.unique(pred_instances[gt_inst_mask])
        self.panoptic_stat.instance_stat['num_pred_inst'] += len(pred_inst_ids)
        self.panoptic_stat.instance_stat['num_gt_inst'] += len(gt_thing_ids)

        if self.debug:
            for gt_instance, pred_instance, index in tqdm(
                list(zip(gt_instances, pred_instances, indices)), desc="[DEBUG] visualizing"):
                
                plt.figure(figsize=(20, 10))
                axis = plt.subplot2grid((1, 2), loc=(0, 0))
                p_ins = np.zeros((pred_instance.shape[0], pred_instance.shape[1], 3))
                labels = np.unique(pred_instance)
                for label in labels:
                    p_ins[pred_instance == label] = pred_instance_label_color_mapping.get(label, np.zeros((3, )))
                axis.imshow(p_ins)
                axis.set_title("Predicted Instance")
                axis.axis('off')

                axis = plt.subplot2grid((1, 2), loc=(0, 1))
                gt_ins = np.zeros((gt_instance.shape[0], gt_instance.shape[1], 3))
                labels = np.unique(gt_instance)
                for label in labels:
                    gt_ins[gt_instance == label] = gt_instance_label_color_mapping.get(label, np.zeros((3, )))
                axis.imshow(gt_ins)
                axis.set_title("GT Instance")
                axis.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(self.save_figures, '{:06}_instance.png'.format(index)))
                plt.close()
    
    def _evaluate_panoptic(self, pred_instances, pred_semantics, pred_semantic_denoiseds, gt_images, gt_semantics, gt_instances, indices):

        print("Evaluating panoptic quality ...")
        gt_segms = self._read_gt_panoptic_segmentation(gt_semantics, gt_instances)
        pred_segms = self._predict_panoptic_segmentation(pred_instances, pred_semantic_denoiseds)

        if self.debug:
            pred_panoptic_label_color_mapping = {}
            gt_panoptic_label_color_mapping = {}
        
        ### evaluate panoptic segmentation
        # confusion matrix calculation
        gt_pred_instance = gt_instances.astype(np.uint64) * OFFSET + pred_instances.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(gt_pred_instance, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue

            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                self.panoptic_stat[gt_segms[gt_label]['category_id']].tp += 1
                self.panoptic_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

                if self.debug:
                    color = np.random.rand(3, )
                    pred_panoptic_label_color_mapping[pred_label] = color
                    gt_panoptic_label_color_mapping[gt_label] = color

        # count false negatives
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            self.panoptic_stat[gt_info['category_id']].fn += 1

            if self.debug:
                color = np.random.rand(3, )
                gt_panoptic_label_color_mapping[gt_label] = color

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID region
            if intersection / pred_info['area'] > 0.5:
                continue
            self.panoptic_stat[pred_info['category_id']].fp += 1

            if self.debug:
                color = np.random.rand(3, )
                pred_panoptic_label_color_mapping[pred_label] = color
        
        if self.debug:
            for i, index in enumerate(tqdm(indices, desc="[DEBUG] visualizing")):
                plt.figure(figsize=(30, 10))
                axis = plt.subplot2grid((1, 3), loc=(0, 0))
                gt_image = gt_images[i]
                rgb = (gt_image * 255).astype(np.uint8)
                axis.imshow(rgb)
                axis.set_title("GT Image")
                axis.axis('off')
                
                axis = plt.subplot2grid((1, 3), loc=(0, 1))
                pred_instance = pred_instances[i]
                p_panop = np.zeros((pred_instance.shape[0], pred_instance.shape[1], 3))
                labels = np.unique(pred_instance)
                pred_panop_patches = []
                for label in labels:
                    if label == VOID:
                        continue
                    color = pred_panoptic_label_color_mapping.get(label, np.zeros((3, )))
                    p_panop[pred_instance == label] = color
                    pred_panop_patches.append(
                        mpatches.Patch(color=color, label=self.label_mapping[pred_segms[label]['category_id']][:10])
                    )
                axis.imshow(p_panop)
                axis.set_title("Predicted Panoptic")
                axis.axis('off')
                axis.legend(handles=pred_panop_patches[:30])

                axis = plt.subplot2grid((1, 3), loc=(0, 2))
                gt_instance = gt_instances[i]
                gt_panop = np.zeros((gt_instance.shape[0], gt_instance.shape[1], 3))
                labels = np.unique(gt_instance)
                gt_panop_patches = []
                for label in labels:
                    if label == VOID:
                        continue
                    color = gt_panoptic_label_color_mapping.get(label, np.zeros((3, )))
                    gt_panop[gt_instance == label] = color
                    gt_panop_patches.append(
                        mpatches.Patch(color=color, label=self.label_mapping[gt_segms[label]['category_id']][:10])
                    )
                axis.imshow(gt_panop)
                axis.set_title("GT Panoptic")
                axis.axis('off')
                axis.legend(handles=gt_panop_patches[:30])

                plt.tight_layout()
                plt.savefig(os.path.join(self.save_figures, '{:06}_panoptic.png'.format(index)))
                plt.close()
