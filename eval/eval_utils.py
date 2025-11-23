import time
import matplotlib.pyplot as plt

from scipy.signal import medfilt
import numpy as np
import torch

def calculate_stability_metrics(scores, mask_sizes, thresh_range, eval_params=None):
    """
    Calculate stability metrics for both score and mask size across different thresholds.
    
    This function evaluates how stable segmentation masks are to threshold variations by
    measuring the rate of change (gradient) in both relevancy scores and mask sizes.
    Stable segmentations show minimal changes in mask configuration when thresholds are
    slightly adjusted.
    
    Args:
        scores: Array of segmentation confidence scores at different thresholds
        mask_sizes: Array of corresponding mask sizes (as proportion of image) at different thresholds
        thresh_range: Array of threshold values used for evaluation
        eval_params: Dictionary containing parameters for evaluation:
                     - "min_mask_size": Minimum valid mask size as proportion (e.g., 0.00001)
                     - "max_mask_size": Maximum valid mask size as proportion (e.g., 0.95)
    
    Returns:
        Dictionary containing stability metrics:
        - 'smooth_score_grad': Smoothed gradient of scores (lower values indicate more stability)
        - 'smooth_mask_grad': Smoothed gradient of mask sizes (lower values indicate more stability)
        - 'valid_regions': Boolean mask indicating regions where mask size falls within valid range
        
    """
    # Calculate gradients
    score_gradient = np.abs(np.gradient(scores, thresh_range))
    mask_gradient = np.abs(np.gradient(mask_sizes, thresh_range))
    
    # Smooth gradients
    smooth_score_grad = medfilt(score_gradient, kernel_size=5)
    smooth_mask_grad = medfilt(mask_gradient, kernel_size=5)

    # Filter out regions where mask_size > 0.95 and < 0.00001
    valid_regions = (np.array(mask_sizes) > eval_params["min_mask_size"]) & (np.array(mask_sizes) < eval_params["max_mask_size"])
    
    assert len(smooth_score_grad[valid_regions]) != 0, "No valid regions found"

    return {
        'smooth_score_grad': smooth_score_grad,
        'smooth_mask_grad': smooth_mask_grad,
        'valid_regions': valid_regions
    }

def find_stable_regions(stability_metrics, eval_params=None):
    """
    Find continuous regions where both score and mask size gradients are stable.
    
    This function identifies threshold ranges where segmentation results remain
    consistent (stable), which indicates reliable segmentation performance.
    
    Args:
        stability_metrics: Dictionary containing stability metrics:
            - 'smooth_score_grad': Smoothed gradient of scores
            - 'smooth_mask_grad': Smoothed gradient of mask sizes
            - 'valid_regions': Boolean mask of valid regions
        eval_params: Dictionary with evaluation parameters:
            - "stability_thresh": Maximum gradient value considered stable
        min_region_length: Minimum length of a region to be considered stable
    
    Returns:
        List of tuples containing (start_index, end_index) of stable regions
    """

    score_stable = stability_metrics['smooth_score_grad'] < eval_params["stability_thresh"]
    mask_stable = stability_metrics['smooth_mask_grad'] < eval_params["stability_thresh"]
    valid_regions = stability_metrics['valid_regions']
    
    # Both metrics must be stable
    combined_stable = score_stable & mask_stable & valid_regions
    
    # Find continuous stable regions
    stable_regions = []
    start_idx = None
    
    for i in range(len(combined_stable)):
        if combined_stable[i]:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None and i - start_idx >= 5:
            # Region ends, must be at least 5 points long
                stable_regions.append((start_idx, i))
            start_idx = None
    
    # Handle the case where the last region extends to the end
    if start_idx is not None and len(combined_stable) - start_idx >= 5:
        stable_regions.append((start_idx, len(combined_stable)-1))
    
    return stable_regions
    
def compute_dynamic_threshold(valid_map, object_name, eval_params=None, thresh_range=np.arange(0.01, 1, 0.01)):
    
    """
    Computes the optimal threshold for segmentation by analyzing stability across three levels.
    
    This function normalizes outputs from each feature level, evaluates segmentation performance
    across a range of thresholds, identifies stable regions, and selects the feature level and threshold
    that demonstrate the most stable segmentation behavior.
    
    Process:
        1. For each feature level, normalizes the relevancy scores to [0,1]
        2. Evaluates scores and mask sizes at each threshold value
        3. Calculates stability metrics based on how scores and mask sizes change with threshold
        4. Identifies continuous regions where both metrics are stable
        5. For each level, calculates a score sensitivity metric from the stable region
        6. Selects the level with the lowest score sensitivity (most stable)
        7. Returns the chosen level and its optimal threshold value
    """
    n_head = valid_map.shape[0]
    total_pixels = valid_map.shape[1] * valid_map.shape[2]
    score_gradients = []
    thresholds = []
            
    for head_idx in range(n_head):
        output = valid_map[head_idx]

        output = output - torch.min(output)
        output = output / (torch.max(output) -  torch.min(output) + 1e-9)
        output = output.numpy()
        
        # Calculate metrics
        scores = []
        pixel_counts = []
        
        for thresh in thresh_range:
            mask = output > thresh
            score = np.mean(output[mask]) if np.any(mask) else 0
            scores.append(score)
            
            normalized_count = np.sum(mask) / total_pixels
            pixel_counts.append(normalized_count)

        # Calculate stability metrics
        stability = calculate_stability_metrics(scores, pixel_counts, thresh_range, eval_params=eval_params)
        stable_regions = find_stable_regions(stability, eval_params=eval_params)
        
        if len(stable_regions) == 0:
            print(f"Warning: Found {len(stable_regions)} stable regions for {object_name} head {head_idx}")
            score_gradients.append(999)
            thresholds.append(0.5)
        else:
            valid_mask = stability['valid_regions']
            # Find the last stable region
            (start_idx, end_idx) = stable_regions[-1]
            # Find the longest stable region
            # longest_region = max(stable_regions, key=lambda region: region[1] - region[0])
            # (start_idx, end_idx) = longest_region
            if np.any(valid_mask[start_idx:end_idx+1]):
                score_sensitivity = (scores[end_idx]- scores[start_idx]) / (thresh_range[end_idx] - thresh_range[start_idx] + 1e-9)
                score_gradients.append(score_sensitivity)
                thresholds.append((thresh_range[start_idx] + thresh_range[end_idx]) / 2) # take the median threshold
            else:
                score_gradients.append(999)
                thresholds.append(0.5)
                
    chosen_lvl = np.argmin(score_gradients)
    threshold = thresholds[chosen_lvl]
    
    return chosen_lvl, threshold
    

def plot_relevancy_and_threshold(relevancy_map, prompt_name, head_idx, save_path, threshold=0.5):
    """
    Plot relevancy map and thresholded areas side by side
    """
    if torch.is_tensor(relevancy_map):
        relevancy_map = relevancy_map.numpy()
    
    # Create threshold mask
    threshold_mask = relevancy_map > threshold
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot relevancy map
    im1 = ax1.imshow(relevancy_map, cmap='viridis')
    ax1.set_title(f'Relevancy Map\n{prompt_name}, Level {head_idx}')
    fig.colorbar(im1, ax=ax1, label='Relevancy Score')
    ax1.axis('off')
    
    # Plot thresholded map
    im2 = ax2.imshow(threshold_mask, cmap='binary')
    ax2.set_title(f'Thresholded Map (>{threshold})\n{prompt_name}, Level {head_idx}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    

import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal
from scipy.spatial import cKDTree


# -------------------------------------------------
# 1. 基本工具
# -------------------------------------------------
def mahalanobis_sq(P, mus, Sigma_inv):
    """
    P          : (Q,3) 所有点
    mus        : (N,3) 所有高斯中心
    Sigma_inv  : (N,3,3) 每个高斯协方差的逆矩阵
    返回       : (N,Q)  每颗高斯到所有点的平方马氏距离
    """
    diff = P[None, :, :] - mus[:, None, :]          # (N,Q,3)
    left = np.einsum("...ij,...jk->...ik", diff, Sigma_inv)  # (N,Q,3)
    d2 = np.einsum("...i,...i->... ", left, diff)            # (N,Q)
    return d2


def gaussian_density(P, mus, Sigma):
    """(N,Q) multivariate normal pdf（不含常数项，足够做 argmax)."""
    N, Q = mus.shape[0], P.shape[0]
    d2 = mahalanobis_sq(P, mus, np.linalg.inv(Sigma))  # (N,Q)
    det = np.linalg.det(Sigma)[:, None]               # (N,1)
    coef = 1.0 / np.sqrt((2 * np.pi)**3 * det)        # (N,1)
    return coef * np.exp(-0.5 * d2)                   # (N,Q)


# -------------------------------------------------
# 2. Hard assignment
# -------------------------------------------------
def assign_labels_hard(P, spk, mus, Sigma, *,
                       method="min_dist",   # "min_dist" 或 "max_density"
                       agg="sum"):          # "sum" 或 "mean"（类平衡）
    """
    返回 N 维 int 数组，表示每个 Gaussian 的硬标签。
    """
    Sigma_inv = np.linalg.inv(Sigma)        # (N,3,3)
    n_classes = int(spk.max()) + 1

    if method == "min_dist":
        score = -mahalanobis_sq(P, mus, Sigma_inv)    # 距离越小分越高
    elif method == "max_density":
        score = gaussian_density(P, mus, Sigma)       # 密度越大分越高
    else:
        raise ValueError("method should be 'min_dist' or 'max_density'.")

    # 把点按类别拆分成稀疏 mask，再对 score 做加权
    scores_per_class = np.zeros((mus.shape[0], n_classes))  # (N,L)
    for c in range(n_classes):
        mask = spk == c                    # (Q,)
        if not mask.any():
            continue
        if agg == "mean":
            denom = mask.sum()
        else:  # "sum"
            denom = 1
        scores_per_class[:, c] = score[:, mask].sum(axis=1) / denom

    return scores_per_class.argmax(axis=1).astype(np.int32)  # (N,)


# -------------------------------------------------
# 3. Soft assignment
# -------------------------------------------------
def assign_labels_soft(P, spk, mus, Sigma, *, tau=0.1):
    """
    返回 (N,L) 概率矩阵，行向量对齐所有高斯。
    用 softmax(-d^2 / τ) 做 soft 分配，然后把各类点概率相加。
    """
    Sigma_inv = np.linalg.inv(Sigma)
    d2 = mahalanobis_sq(P, mus, Sigma_inv)       # (N,Q)
    w = softmax(-d2 / tau, axis=0)               # (N,Q) 对每个点在 N 颗高斯归一化
    n_classes = int(spk.max()) + 1
    probs = np.zeros((mus.shape[0], n_classes))  # (N,L)
    for c in range(n_classes):
        mask = spk == c
        if not mask.any():
            continue
        probs[:, c] = w[:, mask].sum(axis=1)
    probs /= probs.sum(axis=1, keepdims=True)    # 再次归一化
    return probs


# -------------------------------------------------
# 4. 显著性权重 & IoU
# -------------------------------------------------
def gaussian_significance(scales, alphas):
    """
    scales : (N,3)  每颗高斯长度 (six, siy, siz)
    alphas : (N,)   α_i
    返回   : (N,)   d_i = six*siy*siz*α_i
    """
    return scales.prod(axis=1) * alphas


def iou_score(pred, gt, scales, alphas, *, soft=False, eps=1e-8):
    """
    pred  : (N,) int 硬标签  或 (N,L) soft 概率
    gt    : (N,) int 伪 GT 标签
    soft  : True 则 pred 视为 soft 概率矩阵
    返回  : float macro-IoU
    """
    d = gaussian_significance(scales, alphas)          # (N,)
    n_classes = int(max(pred.max() if not soft else gt.max(), gt.max())) + 1

    ious = []
    for c in range(n_classes):
        if soft:
            lp = pred[:, c]                            # (N,)
        else:
            lp = (pred == c).astype(np.float32)
        lg = (gt == c).astype(np.float32)
        inter = (d * np.minimum(lp, lg)).sum()
        union = (d * (np.maximum(lp, lg))).sum() + eps
        if lg.sum() > 0:                               # 忽略场景中不存在的类别
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0




def assign_labels_hard_sparse(
        P, spk, mus, Sigma_inv, *,
        radius_factor=2.0,      # 减少搜索半径
        batch_G=16384,          # 每批最多多少高斯
        batch_P=50000,          # 每批最多多少点
        method="min_dist",      # "min_dist" | "max_density"
        agg="mean",             # "sum" | "mean"
        device="cuda:0"):
    """
    P          (Q,3)  numpy  点云坐标
    spk        (Q,)   numpy  点云语义
    mus        (N,3)  numpy  高斯中心
    Sigma_inv  (N,3,3) numpy  协方差逆
    """
    Q, N = len(P), len(mus)
    n_classes = int(spk.max()) + 1
    
    print(f"Processing {Q} points and {N} Gaussians")

    # ---------- 1) KD-tree 找每点的候选高斯 ----------
    # 用各向同性简化：半径 = radius_factor * max(axis sigma)
    # 修复：处理无效值和负值
    diagonal_vals = np.diagonal(Sigma_inv, axis1=1, axis2=2)  # (N,3)
    
    # 处理可能的负值或无效值
    diagonal_vals = np.abs(diagonal_vals)  # 取绝对值避免负数
    diagonal_vals = np.maximum(diagonal_vals, 1e-6)  # 避免除零
    
    sigma_axis = np.sqrt(1. / diagonal_vals)  # (N,3)
    radii = radius_factor * sigma_axis.max(axis=1)                       # (N,)
    
    # 确保半径有合理的范围
    radii = np.clip(radii, 0.01, 0.5)  # 限制最大半径避免内存爆炸
    
    kdtree = cKDTree(mus)                                                # 建树在 CPU
    
    print(f"Building KD-tree with {N} Gaussians, max radius: {radii.max():.4f}, min radius: {radii.min():.4f}")
    
    # ---------- 2) 分批处理点云以避免内存不足 ----------
    sparse_pairs = []
    
    for p_start in range(0, Q, batch_P):
        p_end = min(p_start + batch_P, Q)
        P_batch = P[p_start:p_end]
        
        print(f"Processing points batch {p_start} to {p_end}")
        start_time = time.time()
        
        # 只对当前批次的点进行查询
        idx_list = kdtree.query_ball_point(P_batch, r=radii.max())
        
        end_time = time.time()
        print(f"Batch query time: {end_time - start_time:.2f} seconds")
        
        # 对每个点再根据各自半径过滤 - 向量化版本
        all_point_indices = []
        all_gaussian_indices = []
        
        for local_pk, idxs in enumerate(idx_list):
            if not idxs: continue
            pk = p_start + local_pk  # 全局点索引
            all_point_indices.extend([pk] * len(idxs))
            all_gaussian_indices.extend(idxs)
        
        if all_point_indices:  # 如果有候选对
            # 转换为numpy数组进行向量化计算
            point_indices = np.array(all_point_indices)
            gaussian_indices = np.array(all_gaussian_indices)
            
            # 向量化计算距离
            point_coords = P[point_indices]  # (M, 3)
            gaussian_coords = mus[gaussian_indices]  # (M, 3)
            distances = np.linalg.norm(point_coords - gaussian_coords, axis=1)  # (M,)
            
            # 向量化过滤：距离小于等于对应高斯半径的对
            gaussian_radii = radii[gaussian_indices]  # (M,)
            valid_mask = distances <= gaussian_radii
            
            # 添加有效的对到sparse_pairs
            valid_point_indices = point_indices[valid_mask]
            valid_gaussian_indices = gaussian_indices[valid_mask]
            
            sparse_pairs.extend(zip(valid_point_indices, valid_gaussian_indices))

    print(f"Found {len(sparse_pairs)} sparse pairs")
    
    # 如果仍然没有找到足够的对，尝试增加半径
    if len(sparse_pairs) < 1000:  # 设置一个最小阈值
        print(f"Warning: Only found {len(sparse_pairs)} pairs, trying with larger radius...")
        return assign_labels_hard_sparse(
            P, spk, mus, Sigma_inv,
            radius_factor=radius_factor * 1.5,  # 适度增加
            batch_G=batch_G, batch_P=batch_P, method=method, agg=agg, device=device
        )
    
    # 转成 torch index
    pair_t = torch.tensor(sparse_pairs, dtype=torch.long, device=device)  # (M,2)
    if pair_t.numel() == 0:
        raise RuntimeError("No candidate pairs found even after retries.")

    # ---------- 3) 批遍历高斯 ----------
    scores = torch.zeros((N, n_classes), device=device)

    for g_start in range(0, N, batch_G):
        print(f"Processing Gaussian batch {g_start} to {min(g_start + batch_G, N)}")
        g_end = min(g_start + batch_G, N)
        # 找到这一批 g 的相关 pair
        mask = (pair_t[:,1] >= g_start) & (pair_t[:,1] < g_end)
        if not mask.any(): continue
        pairs = pair_t[mask]
        # 索引点 & 高斯
        p_idx = pairs[:,0]
        g_idx = pairs[:,1] - g_start
        P_chunk = torch.from_numpy(P[p_idx.cpu()]).to(device)  # (M',3)
        mus_chunk = torch.from_numpy(mus[g_start:g_end]).to(device)  # (G,3)
        Sigma_inv_chunk = torch.from_numpy(
            Sigma_inv[g_start:g_end]).to(device)       # (G,3,3)

        # 重新映射 g_idx 连续化
        # 计算 d^2 (G,Q') 需先把 diff (M',3) -> 按 g_idx 分组
        diff = P_chunk - mus_chunk[g_idx]              # (M',3)
        left = torch.bmm(diff.unsqueeze(1),
                         Sigma_inv_chunk[g_idx])       # (M',1,3)
        d2 = (left.squeeze(1) * diff).sum(1)           # (M',)
        if method == "min_dist":
            val = -d2
        else:  # "max_density"
            val = torch.exp(-0.5*d2)  # 常数项可忽略

        # 按类别累积
        labels_p = torch.from_numpy(spk[p_idx.cpu()]).to(device)  # (M',)
        for c in range(n_classes):
            mask_c = labels_p == c
            if not mask_c.any(): continue
            # 修复散布操作
            g_indices = g_idx[mask_c]
            values = val[mask_c]
            
            # 确保索引在正确范围内
            valid_mask = (g_indices >= 0) & (g_indices < (g_end - g_start))
            if not valid_mask.any(): continue
            
            g_indices = g_indices[valid_mask]
            values = values[valid_mask]
            
            # 使用 scatter_add 累积分数
            scores[g_start:g_end, c].scatter_add_(0, g_indices, values)
            
            if agg == "mean":
                # 计算每个高斯的点数用于平均
                counts = torch.zeros(g_end - g_start, device=device)
                counts.scatter_add_(0, g_indices, torch.ones_like(values))
                # 避免除零
                counts = torch.clamp(counts, min=1)
                scores[g_start:g_end, c] /= counts

    labels = scores.argmax(1).cpu().numpy().astype(np.int32)
    return labels