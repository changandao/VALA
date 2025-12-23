import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torchvision

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torch import nn
from tqdm import tqdm
from typing import Tuple, Type

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)
    

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


def show_masks(masks, borders=True):
    if len(masks) == 0:
        return
    
    sums = masks.sum(dim=(1, 2))
    sorted_sums, sorted_indices = torch.sort(sums, descending=True)  # Sort in descending order
    sorted_masks = masks[sorted_indices]

    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_masks[0].shape[0], sorted_masks[0].shape[1], 4))
    img[:,:,3] = 0
    for m in sorted_masks:
        m = m.detach().cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1)

    ax.imshow(img)


def apply_erosion(mask, kernel_size=3, iterations=1):
    eroded_mask = torch.zeros_like(mask)
    
    # Define the erosion kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_np = mask.cpu().numpy().astype(np.uint8)  # Convert to NumPy array
    eroded_mask_np = cv2.erode(mask_np, kernel, iterations=iterations)
    eroded_mask = torch.from_numpy(eroded_mask_np).bool()  # Convert back to torch.Tensor

    return eroded_mask


def build_sam_auto_mask_generator(model_type, sam_checkpoint, device, use_langsplat=False): 
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.requires_grad_(False).to(device=device)
    
    if use_langsplat:
        # langsplat
        sam_mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )
    else:
        # omniseg
        sam_mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            points_per_batch=64,      # 256
            pred_iou_thresh=.88,
            stability_score_thresh=.9,  # default: 0.95, LLFF: 0.9
            stability_score_offset=1,
            box_nms_thresh=.7,
            crop_n_layers=1,  # default: 0, LLFF: 1
            crop_nms_thresh=.7,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=128
        )
    return sam_mask_generator

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def _embed_clip_sam_tiles(image, sam_encoder, norm = True):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder.generate(aug_imgs.numpy().astype(np.uint8))

    clip_embeds = {}
    for mode in ['default', 's', 'm', 'l']:
        tiles = seg_images[mode]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
        if norm:
            clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
    
    return clip_embeds, seg_map

def get_seg_img(mask, image):
    image = image.copy()
    # print(mask['segmentation'].shape)
    # print(image.shape)
    # image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    image[mask['segmentation']==0] = np.array([255, 255, 255], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img


def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad


def mask2segmap(masks, image, save_path=None):
    seg_img_list = []
    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    for i in range(len(masks)):
        mask = masks[i]
        seg_img = get_seg_img(mask, image)
        pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
        seg_img_list.append(pad_seg_img)

        seg_map[masks[i]['segmentation']] = i
    seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
    seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')
    
    if save_path is not None:
        for i, seg_img in enumerate(seg_imgs):
            img_pil = torchvision.transforms.ToPILImage()(seg_img)
            img_pil.save(f"{save_path}/seg_{i}.png")

    return seg_imgs, seg_map


def compute_iou(mask1, mask2):
    intersection = (mask1 & mask2).float().sum() 
    union = (mask1 | mask2).float().sum()
    if union == 0:
        return 0.0
    return intersection / union


def match_instance_with_sam_masks(instances, masks_sam):
    instances_matched_sam_masks_idx = []
    instances_sam_masks = []
    for instance_mask in instances:
        ious = []
        for sam_mask in masks_sam:
            iou = compute_iou(instance_mask, torch.tensor(sam_mask['segmentation']))
            ious.append(iou)
        ious = torch.tensor(np.asarray(ious))
        max_iou, max_idx = ious.max(0)
        instances_matched_sam_masks_idx.append(max_idx)
        
        instance_sam_mask = {
            "segmentation": instance_mask,
            "bbox": masks_sam[max_idx]["bbox"]
        }
        instances_sam_masks.append(instance_sam_mask)

    # idx should be distinct as the instance mask is not overlap anymore
    assert len(instances_matched_sam_masks_idx) == len(set(instances_matched_sam_masks_idx))

    return instances_sam_masks


def process_one_scene(sam_mask_generator: SamAutomaticMaskGenerator, scene: str, args: Namespace):
    # load dir
    dataset_name = args.dataset_name
    root_dir = args.root_dir
    rep = args.rep

    print(f"processing {dataset_name}")
    dataset_dir = os.path.join(root_dir, "dataset", rep, dataset_name)
    
    if str(dataset_name).lower() == "lerf_ovs":
        if args.use_langsplat:
            scene_instance_mask_dir = os.path.join(dataset_dir, scene, "langsplat/instance_masks")
            scene_hierarchial_corr_dir = os.path.join(dataset_dir, scene, "langsplat/correlation")
        else:
            scene_instance_mask_dir = os.path.join(dataset_dir, scene, "omniseg3d/instance_masks")
            scene_hierarchial_corr_dir = os.path.join(dataset_dir, scene, "omniseg3d/correlation")
        os.makedirs(scene_instance_mask_dir, exist_ok=True)
        os.makedirs(scene_hierarchial_corr_dir, exist_ok=True)
    elif str(dataset_name).lower() == "mipnerf360":
        if args.use_langsplat:
            scene_instance_mask_dir = os.path.join(dataset_dir, scene, "langsplat/instance_masks")
            scene_hierarchial_corr_dir = os.path.join(dataset_dir, scene, "langsplat/correlation")
        else:
            scene_instance_mask_dir = os.path.join(dataset_dir, scene, "instance_masks")
            scene_hierarchial_corr_dir = os.path.join(dataset_dir, scene, "correlation")
        os.makedirs(scene_instance_mask_dir, exist_ok=True)
        os.makedirs(scene_hierarchial_corr_dir, exist_ok=True)
    elif str(dataset_name).lower() == "waymo":
        if args.use_langsplat:
            scene_instance_mask_dir = os.path.join(dataset_dir, scene, "langsplat/instance_masks")
            scene_hierarchial_corr_dir = os.path.join(dataset_dir, scene, "langsplat/correlation")
        else:
            scene_instance_mask_dir = os.path.join(dataset_dir, scene, "instance_masks")
            scene_hierarchial_corr_dir = os.path.join(dataset_dir, scene, "correlation")
        os.makedirs(scene_instance_mask_dir, exist_ok=True)
        os.makedirs(scene_hierarchial_corr_dir, exist_ok=True)
    else:
        if args.use_langsplat:
            scene_instance_mask_dir = os.path.join(dataset_dir, scene, "train/langsplat/instance_masks")
            test_scene_instance_mask_dir = os.path.join(dataset_dir, scene, "test/langsplat/instance_masks")
            scene_hierarchial_corr_dir = os.path.join(dataset_dir, scene, "train/langsplat/correlation")
            test_scene_hierarchial_corr_dir = os.path.join(dataset_dir, scene, "test/langsplat/correlation")
        else:
            scene_instance_mask_dir = os.path.join(dataset_dir, scene, "train/instance_masks")
            test_scene_instance_mask_dir = os.path.join(dataset_dir, scene, "test/instance_masks")
            scene_hierarchial_corr_dir = os.path.join(dataset_dir, scene, "train/correlation")
            test_scene_hierarchial_corr_dir = os.path.join(dataset_dir, scene, "test/correlation")
    
        os.makedirs(scene_instance_mask_dir, exist_ok=True)
        os.makedirs(scene_hierarchial_corr_dir, exist_ok=True)
        os.makedirs(test_scene_instance_mask_dir, exist_ok=True)
        os.makedirs(test_scene_hierarchial_corr_dir, exist_ok=True)
    
    get_semantic = args.get_semantic
    semantic_dir = None
    if get_semantic:
        print("get semantic")
    
        if str(dataset_name).lower() == "lerf_ovs":
            if args.use_langsplat:
                semantic_folder = os.path.join(dataset_dir, scene, 'langsplat/language_features')
                print(semantic_folder)
            else:
                semantic_folder = os.path.join(dataset_dir, scene, 'language_features_ins')
            # semantic_folder = os.path.join(dataset_dir, scene, 'omniseg3d/language_features_ins')
            semantic_debug_folder = os.path.join(dataset_dir, scene, 'seg_images_frame0')
            os.makedirs(semantic_folder, exist_ok=True)
            os.makedirs(semantic_debug_folder, exist_ok=True)
            semantic_dir = {
                "lang_feat": semantic_folder,
                "seg_img": semantic_debug_folder
            }
        elif str(dataset_name).lower() == "mipnerf360":
            if args.use_langsplat:
                semantic_folder = os.path.join(dataset_dir, scene, 'langsplat/language_features')
            else:
                semantic_folder = os.path.join(dataset_dir, scene, 'language_features_ins')
            semantic_debug_folder = os.path.join(dataset_dir, scene, 'seg_images_frame0')
            os.makedirs(semantic_folder, exist_ok=True)
            os.makedirs(semantic_debug_folder, exist_ok=True)
            semantic_dir = {
                "lang_feat": semantic_folder,
                "seg_img": semantic_debug_folder
            }
        elif str(dataset_name).lower() == "waymo":
            if args.use_langsplat:
                semantic_folder = os.path.join(dataset_dir, scene, 'langsplat/language_features')
            else:
                semantic_folder = os.path.join(dataset_dir, scene, 'language_features_ins')
            semantic_debug_folder = os.path.join(dataset_dir, scene, 'seg_images_frame0')
            os.makedirs(semantic_folder, exist_ok=True)
            os.makedirs(semantic_debug_folder, exist_ok=True)
            semantic_dir = {
                "lang_feat": semantic_folder,
                "seg_img": semantic_debug_folder
            }
        else:
            if args.use_langsplat: 
                semantic_folder = os.path.join(dataset_dir, scene, 'train/langsplat/language_features')
                test_semantic_folder = os.path.join(dataset_dir, scene, 'test/langsplat/language_features')
            else:
                semantic_folder = os.path.join(dataset_dir, scene, 'train/language_features_ins')
                test_semantic_folder = os.path.join(dataset_dir, scene, 'test/language_features_ins')
            semantic_debug_folder = os.path.join(dataset_dir, scene, 'seg_images_frame0')
            os.makedirs(semantic_folder, exist_ok=True)
            os.makedirs(test_semantic_folder, exist_ok=True)
            os.makedirs(semantic_debug_folder, exist_ok=True)
            semantic_dir = {
                "lang_feat": semantic_folder,
                "seg_img": semantic_debug_folder
            }
            test_semantic_dir = {
                "lang_feat": test_semantic_folder,
                "seg_img": semantic_debug_folder
            }
    
    if str(dataset_name).lower() == "lerf_ovs":
        if str(rep).lower() == "3dgs":
            scene_in_dir = os.path.join(dataset_dir, scene)
            img_dir = os.path.join(scene_in_dir, "images")
            image_path_list = glob.glob(os.path.join(img_dir, "*.jpg"))
        else:
            raise NotImplementedError(f"Dataset handling is not implemented yet for 3D representation {rep}")
    elif str(dataset_name).lower() == "replica":
        if str(rep).lower() == "3dgs":
            scene_in_dir = os.path.join(dataset_dir, scene)
            img_dir = os.path.join(scene_in_dir, "color")
            image_path_list = glob.glob(os.path.join(img_dir, "*.jpg"))
        else:
            raise NotImplementedError(f"Dataset handling is not implemented yet for 3D representation {rep}")
    elif str(dataset_name).lower() == "scannet" or str(dataset_name).lower() == "scannet_langsplat" or str(dataset_name).lower() =="scannet_panopli":
        if str(rep).lower() == "3dgs":
            scene_in_dir = os.path.join(dataset_dir, scene)
            img_dir = os.path.join(scene_in_dir, "train/images")
            test_img_dir = os.path.join(scene_in_dir, "test/images")
            image_path_list = glob.glob(os.path.join(img_dir, "*.jpg"))
            test_image_path_list = glob.glob(os.path.join(test_img_dir, "*.jpg"))
        else:
            raise NotImplementedError(f"Dataset handling is not implemented yet for 3D representation {rep}")
    elif str(dataset_name).lower() == "mipnerf360":
        if str(rep).lower() == "3dgs":
            scene_in_dir = os.path.join(dataset_dir, scene)
            img_dir = os.path.join(scene_in_dir, "images_4")
            image_path_list = glob.glob(os.path.join(img_dir, "*.JPG"))
        else:
            raise NotImplementedError(f"Dataset handling is not implemented yet for 3D representation {rep}")
    elif str(dataset_name).lower() == "waymo":
        if str(rep).lower() == "3dgs":
            scene_in_dir = os.path.join(dataset_dir, scene)
            img_dir = os.path.join(scene_in_dir, "images")
            image_path_list = glob.glob(os.path.join(img_dir, "*.png"))
        else:
            raise NotImplementedError(f"Dataset handling is not implemented yet for 3D representation {rep}")
    else:
        raise NotImplementedError(f"Dataset handling is not implemented yet for dataset {dataset_name}")
    # print(img_dir, image_path_list)
    # print(os.path.join("/home/sen/projects/master-thesis/dataset/3dgs/mipnerf360/garden/images_4"))
    print(len(image_path_list))
    image_path_list = sorted(image_path_list)
    if str(dataset_name).lower() == "lerf_ovs":
        test_image_path_list = []
    elif str(dataset_name).lower() == "mipnerf360":
        test_image_path_list = []
    elif str(dataset_name).lower() == "waymo":
        test_image_path_list = []
    else:
        test_image_path_list = sorted(test_image_path_list)
    shape = cv2.imread(image_path_list[0]).shape
    w = shape[1]
    h = shape[0]
    
    images = []
    image_names = []
    for image_path in tqdm(image_path_list, desc="Processing images"):
        name = image_path.split("/")[-1].split(".")[0]
        image_bgr_np = cv2.imread(image_path)
        image_rgb_np = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image_rgb_np)
        images.append(image)
        image_names.append(name)
    imgs = torch.stack(images)
    if args.use_langsplat:
        create_langsplat(sam_mask_generator, imgs, image_names, scene_instance_mask_dir, scene_hierarchial_corr_dir, get_semantic, semantic_dir, args.use_langsplat)
    else:
        create(sam_mask_generator, imgs, image_names, scene_instance_mask_dir, scene_hierarchial_corr_dir, get_semantic, semantic_dir, args.use_langsplat)

    # Process test images
    if str(dataset_name).lower() == "lerf_ovs":
        test_images = []
        test_image_names = []
    elif str(dataset_name).lower() == "mipnerf360":
        test_images = []
        test_image_names = []
    elif str(dataset_name).lower() == "waymo":
        test_images = []
        test_image_names = []
    else:
        test_images = []
        test_image_names = []
        for image_path in tqdm(test_image_path_list, desc="Processing test images"):
            name = image_path.split("/")[-1].split(".")[0]
            image_bgr_np = cv2.imread(image_path)
            image_rgb_np = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image_rgb_np)
            test_images.append(image)
            test_image_names.append(name)
        test_imgs = torch.stack(test_images)

        if args.use_langsplat:
            create_langsplat(sam_mask_generator, test_imgs, test_image_names, test_scene_instance_mask_dir, test_scene_hierarchial_corr_dir, get_semantic, test_semantic_dir, args.use_langsplat)
        else:
            create(sam_mask_generator, test_imgs, test_image_names, test_scene_instance_mask_dir, test_scene_hierarchial_corr_dir, get_semantic, test_semantic_dir, args.use_langsplat)



def create_langsplat(sam_mask_generator: SamAutomaticMaskGenerator,
           image_list: list,
           name_list: list,
           instance_save_folder: str,
           corr_save_folder: str,
           get_semantic: bool,
           semantic_folder: str,
           use_langsplat: bool = False):
    
    assert image_list is not None, "image_list must be provided to generate masks and correlation"
    embed_size=512
    
    if get_semantic:
        model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    else:
        model = None
    
    for idx, image in tqdm(enumerate(image_list), desc="Processing images"):
        name = name_list[idx]
        
        # Check if files already exist
        instance_file = os.path.join(instance_save_folder, f'{name}.npy')
        # corr_file = os.path.join(corr_save_folder, f'{name}.npz')
        semantic_files_exist = True
        if get_semantic:
            feat_file = f"{semantic_folder['lang_feat']}/{name}_f.npy"
            seg_file = f"{semantic_folder['lang_feat']}/{name}_s.npy"
            semantic_files_exist = os.path.exists(feat_file) and os.path.exists(seg_file)
            print(feat_file, seg_file)
        print(instance_file, semantic_files_exist)
        
        if os.path.exists(instance_file) or semantic_files_exist:
            continue
            
        if use_langsplat:
            image = image.numpy().astype(np.uint8)
            masks_default, masks_s, masks_m, masks_l = sam_mask_generator.generate(image)
            masks_default, masks_s, masks_m, masks_l = masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
            
            
            seg_images, seg_maps = {}, {}
            seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image, save_path=semantic_folder['seg_img'])
            if len(masks_s) != 0:
                seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image, save_path=semantic_folder['seg_img']   )
            if len(masks_m) != 0:
                seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image, save_path=semantic_folder['seg_img'])
            if len(masks_l) != 0:
                seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image, save_path=semantic_folder['seg_img'])
                

            clip_embeds = {}
            for mode in ['default', 's', 'm', 'l']:
                tiles = seg_images[mode]
                tiles = tiles.to("cuda")
                with torch.no_grad():
                    clip_embed = model.encode_image(tiles)
                clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
                clip_embeds[mode] = clip_embed.detach().cpu().half()
    

        img_embed = clip_embeds
        seg_map = seg_maps
            
        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        # total_lengths.append(total_length)
        
        # if total_length > img_embeds.shape[1]:
        #     pad = total_length - img_embeds.shape[1]
        #     img_embeds = torch.cat([
        #         img_embeds,
        #         torch.zeros((len(image_list), pad, embed_size))
        #     ], dim=1)

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        # img_embeds[i, :total_length] = img_embed
        
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        save_path_f = f"{semantic_folder['lang_feat']}/{name}_f.npy"
        save_path_s = f"{semantic_folder['lang_feat']}/{name}_s.npy"
        np.save(save_path_f, img_embed)
        np.save(save_path_s, seg_map)
        
        
        
        
def create(sam_mask_generator: SamAutomaticMaskGenerator,
           image_list: list,
           name_list: list,
           instance_save_folder: str,
           corr_save_folder: str,
           get_semantic: bool,
           semantic_folder: str,
           use_langsplat: bool = False):
    
    assert image_list is not None, "image_list must be provided to generate masks and correlation"
    
    if get_semantic:
        model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    else:
        model = None
    
    for idx, image in tqdm(enumerate(image_list), desc="Processing images"):
        name = name_list[idx]
        
        # # Check if files already exist
        # instance_file = os.path.join(instance_save_folder, f'{name}.npy')
        # # corr_file = os.path.join(corr_save_folder, f'{name}.npz')
        # semantic_files_exist = True
        # if get_semantic:
        #     feat_file = f"{semantic_folder['lang_feat']}/{name}_f.npy"
        #     seg_file = f"{semantic_folder['lang_feat']}/{name}_s.npy"
        #     semantic_files_exist = os.path.exists(feat_file) and os.path.exists(seg_file)
        # print(instance_file, semantic_files_exist)
        
        # if os.path.exists(instance_file) and semantic_files_exist:
        #     continue
            
        if use_langsplat:
            masks_default, masks_s, masks_m, masks_l = sam_mask_generator.generate(image.numpy().astype(np.uint8))
            masks_default, masks_s, masks_m, masks_l = masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
            
            
            seg_images, seg_maps = {}, {}
            seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
            if len(masks_s) != 0:
                seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
            if len(masks_m) != 0:
                seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
            if len(masks_l) != 0:
                seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
    
            # 0:default 1:s 2:m 3:l
            # return seg_images, seg_maps
            # masks_sam = masks_default.copy()
            # masks_sam.extend(masks_s)
            # masks_sam.extend(masks_m)
            # masks_sam.extend(masks_l)
        else:
            masks_sam = sam_mask_generator.generate(image.numpy())
            masks_sam = masks_sam[0]
        
        # print(masks_sam[0][0].keys())
        # print(len(masks_sam))
        masks = torch.tensor(np.asarray([mask['segmentation'] for mask in masks_sam]))
        masks = masks[masks.sum((1, 2)).argsort()]
        
        unique, indices = masks.flatten(1).unique(return_inverse=True, dim=1)
        indices = indices.view_as(masks[0])
        patches = []
        
        for i in range(unique.size(1)):
            patch = indices == i
            eroded_patch = apply_erosion(patch, kernel_size=9)
            if eroded_patch.sum() > 0:
                patches.append(patch)
        patches = torch.stack(patches)
        
        mask_patch = torch.zeros((len(masks), len(patches)), dtype=torch.bool)
        patch_index = torch.zeros_like(indices)
        for i, patch in enumerate(patches):
            overlap = (masks & patch.unsqueeze(0)).any(dim=2).any(dim=1)
            mask_patch[:, i] = overlap
            patch_index[patch] = i
        
        mask_patch = mask_patch.float()
        corr = (mask_patch.T @ mask_patch).byte().cpu().numpy()
        
        np.savez_compressed(os.path.join(corr_save_folder, name), indices=np.int16(patch_index), correlation=corr)
        
        binary_patterns = (corr > 0).astype(int)
        unique_patterns, inverse_indices = np.unique(binary_patterns, axis=0, return_inverse=True)
        num_unique_patterns = unique_patterns.shape[0]
        
        instances = []
        instance_seg = -1 * torch.ones_like(patches[0])
        for i in range(num_unique_patterns):
            pattern_indices = np.where(inverse_indices == i)[0]
            merged_patch = torch.zeros_like(patches[0])
            if unique_patterns[i].sum() > 0 and len(pattern_indices) > 0:
                for j in pattern_indices:
                    merged_patch |= patches[j]
                instance_seg[merged_patch] = len(instances)
                instances.append(merged_patch)
                
        instances = torch.stack(instances)
        assert instance_seg.max() == len(instances) - 1
        np.save(os.path.join(instance_save_folder, f'{name}.npy'), instance_seg.detach().cpu().numpy())
        
        if get_semantic:
            assert semantic_folder is not None
            instances_mask_sam = match_instance_with_sam_masks(instances, masks_sam)
            seg_images, seg_map = mask2segmap(instances_mask_sam, image.numpy(), save_path=semantic_folder["seg_img"] if idx == 0 else None)
            
            tiles = seg_images.to("cuda")
            with torch.no_grad():
                clip_embed = model.encode_image(tiles)
            clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
            clip_embed = clip_embed.detach().cpu().half()
            
            # print(clip_embed.shape)
            # print(seg_map.shape)
            
            save_path_f = f"{semantic_folder['lang_feat']}/{name}_f.npy"
            save_path_s = f"{semantic_folder['lang_feat']}/{name}_s.npy"
            np.save(save_path_f, clip_embed)
            np.save(save_path_s, seg_map)
        
        plt.figure(figsize=(20,20))
        plt.imshow(image.numpy())
        show_masks(instances)
        plt.axis('off')
        plt.savefig(os.path.join(instance_save_folder, f'{name}.png'))
        plt.close()
        

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


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)
    
    # arguments
    parser = ArgumentParser("Script to extract hierarchy masks and instance masks from the dataset")
    parser.add_argument("--root_dir", type=str, default="/home/sen.wang/projects/VALA")
    parser.add_argument("--dataset_name", type=str, choices=["replica", "lerf_ovs", "scannet", "scannet_langsplat", "scannet_panopli", "mipnerf360", "waymo"], default="lerf_ovs")
    parser.add_argument("--rep", type=str, choices=["3dgs"], default="3dgs")
    parser.add_argument("--scene", help="If none, preprocess the whole dataset", default=None)
    parser.add_argument("--sam_model", help="model of sam", default="vit_h")
    parser.add_argument("--sam_checkpoint", help="checkpoint of sam", default="/home/sen.wang/projects/VALA/ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument("--get_semantic", help="whether to get CLIP semantic feature from mask", action='store_true', default=False)
    parser.add_argument("--use_langsplat", help="if true use langsplat sam that provides 4 levels of mask", action='store_true')
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    sam_mask_generator = build_sam_auto_mask_generator(args.sam_model, args.sam_checkpoint, args.device, args.use_langsplat)
    
    dataset_name = args.dataset_name
    scene = args.scene
    
    if str(dataset_name).lower() == "lerf_ovs":
        from dataset.lerf_ovs import lerf_ovs
        valid_scenes = lerf_ovs.scenes
    elif str(dataset_name).lower() == "replica":
        from dataset.replica import replica
        valid_scenes = replica.scenes
    elif str(dataset_name).lower() == "scannet" or str(dataset_name).lower() == "scannet_langsplat":
        from dataset.scannet import scannet
        valid_scenes = scannet.scenes
    elif str(dataset_name).lower() == "scannet_panopli":
        from dataset.scannet import scannet_panopli
        valid_scenes = scannet_panopli.scenes
    elif str(dataset_name).lower() == "mipnerf360":
        from dataset.mipnerf360 import mipnerf360
        valid_scenes = mipnerf360.scenes
    elif str(dataset_name).lower() == "waymo":
        from dataset.waymo import waymo
        valid_scenes = waymo.scenes
    else:
        raise NotImplementedError(f"Dataset handling is not implemented yet for dataset {dataset_name}")
    
    scene = args.scene
    if scene is None:
        scene_list = valid_scenes
    elif scene in valid_scenes:
        scene_list = [scene]
    else:
        raise Exception(f"{scene} is not a valid scene of {dataset_name} dataset")
    
    print(scene_list)
    for scene in tqdm(scene_list):
        # if scene == "scene0062_00" or scene == "scene0000_00":
        #     continue
        # if not scene == "scene0050_02":
        #     continue
        # if not scene == "garden":
        #     continue
        # if scene == "garden":
        #     continue
        process_one_scene(sam_mask_generator, scene, args)