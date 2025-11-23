import numpy as np
from typing import List, Dict, Tuple, Union
import torch
from scipy.spatial.distance import mahalanobis


def _quat_to_rotmat(q):  # q: (N,4) w,x,y,z  -> (N,3,3)
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = torch.stack([
        torch.stack([ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy)], dim=-1),
        torch.stack([2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx)], dim=-1),
        torch.stack([2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz], dim=-1),
    ], dim=-2)
    return R

def _sigma_invsqrt_from_scale_rot(scale, rotation):
    """
    Σ = R diag(s^2) R^T,  Σ^{-1/2} = R diag(1/s) R^T
    scale:   (N,3)
    rotation:(N,4) or (N,3,3)
    return:  (N,3,3)
    """
    if rotation.ndim == 2 and rotation.shape[1] == 4:
        R = _quat_to_rotmat(rotation)
    elif rotation.ndim == 3 and rotation.shape[-2:] == (3,3):
        R = rotation
    else:
        raise ValueError("rotation must be Nx4 (quat) or Nx3x3 (matrix)")
    inv_s = 1.0 / torch.clamp(scale, min=1e-6)
    D = torch.diag_embed(inv_s)  # diag(1/sx,1/sy,1/sz)
    return R @ D @ R.transpose(-1, -2)  # Σ^{-1/2}

class GaussianEvaluationProtocol:
    """
    Implementation of the Gaussian-friendly evaluation protocol for 3D Gaussian Splatting
    semantic segmentation evaluation.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def compute_mahalanobis_distance(self, point: np.ndarray, mu: np.ndarray, 
                                   covariance: np.ndarray) -> float:
        """
        Compute Mahalanobis distance between a point and a 3D Gaussian.
        
        Args:
            point: 3D point coordinates [x, y, z]
            mu: Gaussian mean [μx, μy, μz]
            covariance: 3x3 covariance matrix Σ
            
        Returns:
            Mahalanobis distance
        """
        diff = point - mu
        try:
            # Compute inverse of covariance matrix
            inv_cov = np.linalg.inv(covariance)
            distance = np.sqrt(diff.T @ inv_cov @ diff)
        except np.linalg.LinAlgError:
            # Handle singular matrix by adding small regularization
            inv_cov = np.linalg.inv(covariance + 1e-6 * np.eye(3))
            distance = np.sqrt(diff.T @ inv_cov @ diff)
        
        return distance
    
    def construct_covariance_matrix(self, scale: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """
        Construct covariance matrix from scale and rotation parameters.
        
        Args:
            scale: Scale parameters [sx, sy, sz]
            rotation: Rotation quaternion [w, x, y, z] or rotation matrix
            
        Returns:
            3x3 covariance matrix
        """
        # Create scale matrix
        S = np.diag(scale)
        
        # Handle rotation (assuming quaternion input)
        if rotation.shape == (4,):
            # Convert quaternion to rotation matrix
            w, x, y, z = rotation
            R = np.array([
                [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
            ])
        else:
            # Assume it's already a rotation matrix
            R = rotation
        
        # Covariance matrix: Σ = R * S * S^T * R^T
        covariance = R @ S @ S.T @ R.T
        return covariance
    
    def assign_semantic_labels_optimized(self, gaussians_params: Dict, point_cloud: np.ndarray, 
                                        point_labels: np.ndarray, unique_labels: List,
                                        batch_size: int = 1000, spatial_threshold: float = None) -> np.ndarray:
        """
        Memory-efficient assignment of semantic labels to 3D Gaussians using spatial filtering and batching.
        
        Args:
            gaussians_params: Dictionary containing Gaussian parameters
            point_cloud: Ground truth point cloud [Q, 3]
            point_labels: Semantic labels for each point [Q]
            unique_labels: List of unique semantic labels
            batch_size: Number of Gaussians to process at once
            spatial_threshold: Distance threshold for spatial filtering (if None, auto-compute)
            
        Returns:
            Assigned labels for each Gaussian [N]
        """
        mu = gaussians_params['mu']  # [N, 3]
        scale = gaussians_params['scale']  # [N, 3]
        rotation = gaussians_params['rotation']  # [N, 4] or [N, 3, 3]
        
        N = mu.shape[0]  # Number of Gaussians
        Q = point_cloud.shape[0]  # Number of points
        
        print(f"Processing {N} Gaussians and {Q} points with batch size {batch_size}")
        
        # Auto-compute spatial threshold if not provided
        if spatial_threshold is None:
            # Use 3 times the maximum scale as threshold
            max_scales = np.max(scale, axis=1)  # Max scale for each Gaussian
            spatial_threshold = np.percentile(max_scales, 95) * 3
            print(f"Auto-computed spatial threshold: {spatial_threshold:.4f}")
        
        # Build spatial index for points using simple grid
        from collections import defaultdict
        grid_size = spatial_threshold
        point_grid = defaultdict(list)
        
        for k in range(Q):
            grid_key = tuple((point_cloud[k] // grid_size).astype(int))
            point_grid[grid_key].append(k)
        
        gaussian_labels = np.zeros(N, dtype=int)
        
        # Process Gaussians in batches
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_indices = range(batch_start, batch_end)
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(N-1)//batch_size + 1}")
            
            for i in batch_indices:
                # Get nearby points using spatial filtering
                gaussian_pos = mu[i]
                gaussian_grid_key = tuple((gaussian_pos // grid_size).astype(int))
                
                # Check neighboring grid cells
                nearby_points = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            neighbor_key = (gaussian_grid_key[0] + dx,
                                          gaussian_grid_key[1] + dy,
                                          gaussian_grid_key[2] + dz)
                            nearby_points.extend(point_grid.get(neighbor_key, []))
                
                # If no nearby points found, use all points (fallback)
                if not nearby_points:
                    nearby_points = list(range(Q))
                
                # Further filter by Euclidean distance
                relevant_points = []
                for k in nearby_points:
                    if np.linalg.norm(point_cloud[k] - gaussian_pos) <= spatial_threshold:
                        relevant_points.append(k)
                
                if not relevant_points:
                    # If no relevant points, assign most common label
                    gaussian_labels[i] = np.bincount(point_labels).argmax()
                    continue
                
                # Construct covariance matrix for i-th Gaussian
                cov_matrix = self.construct_covariance_matrix(scale[i], rotation[i])
                
                # Dictionary to store sum of Mahalanobis distances for each label
                label_distance_sums = {label: 0.0 for label in unique_labels}
                
                # Compute distances only to relevant points
                for k in relevant_points:
                    point = point_cloud[k]
                    point_label = point_labels[k]
                    
                    # Compute Mahalanobis distance
                    distance = self.compute_mahalanobis_distance(point, gaussian_pos, cov_matrix)
                    
                    # Add to the sum for this label
                    label_distance_sums[point_label] += distance
                
                # Assign label with highest sum of distances (Eq. 8)
                assigned_label = max(label_distance_sums.keys(), 
                                   key=lambda s: label_distance_sums[s])
                gaussian_labels[i] = assigned_label
        
        return gaussian_labels
    
    def assign_semantic_labels_vectorized(self, gaussians_params: Dict, point_cloud: np.ndarray,
                                        point_labels: np.ndarray, unique_labels: List,
                                        chunk_size: int = 1000) -> np.ndarray:
        """
        Vectorized version for even better performance with GPU support.
        """
        mu = gaussians_params['mu']  # [N, 3]
        scale = gaussians_params['scale']  # [N, 3]
        rotation = gaussians_params['rotation']  # [N, 4] or [N, 3, 3]
        
        N = mu.shape[0]
        Q = point_cloud.shape[0]
        
        # Convert to torch tensors if available
        try:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            mu_torch = torch.from_numpy(mu).float().to(device)
            point_cloud_torch = torch.from_numpy(point_cloud).float().to(device)
            point_labels_torch = torch.from_numpy(point_labels).long().to(device)
            
            gaussian_labels = torch.zeros(N, dtype=torch.long, device=device)
            
            print(f"Using {device} for vectorized computation")
            
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                
                # Compute pairwise distances for chunk
                chunk_mu = mu_torch[i:end_i]  # [chunk_size, 3]
                
                # Simple Euclidean distance first for filtering
                distances = torch.cdist(chunk_mu, point_cloud_torch)  # [chunk_size, Q]
                
                # For each Gaussian in chunk, find closest points and assign labels
                for j, global_idx in enumerate(range(i, end_i)):
                    dists = distances[j]  # [Q]
                    
                    # Get top-k closest points (or use threshold)
                    k_closest = min(1000, Q)  # Limit to 1000 closest points
                    _, closest_indices = torch.topk(dists, k_closest, largest=False)
                    
                    # Count labels among closest points
                    closest_labels = point_labels_torch[closest_indices]
                    label_counts = torch.bincount(closest_labels, minlength=len(unique_labels))
                    
                    # Assign most frequent label
                    gaussian_labels[global_idx] = torch.argmax(label_counts)
                
                if (i // chunk_size + 1) % 10 == 0:
                    print(f"Processed {i + chunk_size}/{N} Gaussians")
            
            return gaussian_labels.cpu().numpy()
            
        except ImportError:
            print("PyTorch not available, falling back to optimized CPU version")
            return self.assign_semantic_labels_optimized(
                gaussians_params, point_cloud, point_labels, unique_labels
            )
    
    @torch.no_grad()
    def assign_semantic_labels_mahalanobis(
        self,
        gaussians_params: dict,
        point_cloud: np.ndarray,           # (Q,3)
        point_labels: np.ndarray,          # (Q,)  假定是 0..L-1 的整型
        unique_labels: list,               # e.g. list(range(L))
        chunk_size: int = 1000,
        k_max: int = 1000,                  # 每颗高斯最多参与投票的候选点数
        radius_factor: float = 5.0,        # 欧氏候选半径 = factor * max(scale)
        method: str = "density",           # "density" 或 "min_dist"
        class_balance: bool = True,        # 是否做类别均衡（按类样本数归一）
        use_alpha_volume: bool = False,    # 是否用 alpha*vol 给“点票”加权
        verbose: bool = False,
    ):
        """
        返回: (N,) numpy int, 每颗 Gaussian 的伪标签
        """
        import os
        # 设置CUDA调试环境变量以获得更好的错误信息
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mu = torch.as_tensor(gaussians_params['mu'], dtype=torch.float32, device=device)        # (N,3)
        scale = torch.as_tensor(gaussians_params['scale'], dtype=torch.float32, device=device)  # (N,3)
        rotation = torch.as_tensor(gaussians_params['rotation'], dtype=torch.float32, device=device)  # (N,4 or N,3,3)
        alpha = torch.as_tensor(gaussians_params.get('opacity', np.ones(len(mu))), dtype=torch.float32, device=device)  # (N,)
        N = mu.shape[0]

        P = torch.as_tensor(point_cloud, dtype=torch.float32, device=device)   # (Q,3)
        y = torch.as_tensor(point_labels, dtype=torch.long, device=device)     # (Q,)
        
        # 安全处理标签：确保所有标签都在有效范围内
        y_min, y_max = y.min().item(), y.max().item()
        if y_min < 0:
            print(f"Warning: Found negative labels (min={y_min}), clamping to 0")
            y = torch.clamp(y, min=0)
        
        L = int(max(int(y.max().item())+1, len(unique_labels)))
        print(f"Debug: Label range: {y.min().item()} to {y.max().item()}, L={L}")

        # Σ^{-1/2} for Mahalanobis via whitened euclidean
        Sigma_invsqrt = _sigma_invsqrt_from_scale_rot(scale, rotation)         # (N,3,3)

        # 预计算每颗高斯的候选半径（欧氏）
        radii = radius_factor * torch.max(scale, dim=1).values                 # (N,)

        labels_out = torch.empty(N, dtype=torch.long, device=device)
        # 若使用 alpha*volume 作为“点票”权重的一部分，需要 per-Gaussian 标量
        if use_alpha_volume:
            g_weight = alpha * scale.prod(dim=1)  # (N,)

        # 分块处理 Gaussians
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            G  = e - s
            
            mu_c = mu[s:e]                               # (G,3)
            rad_c = radii[s:e]                           # (G,)
            S_c = Sigma_invsqrt[s:e]                     # (G,3,3)
            if use_alpha_volume:
                gw_c = g_weight[s:e]    # (G,)

            # 先用欧氏距离做候选裁剪（G×Q）
            # 1) 欧氏距离矩阵 (G, Q)
            d_euc = torch.cdist(mu_c, P)  # 注意内存：G×Q

            # 2) 半径掩码 + 逐行 top-K
            masked = d_euc.masked_fill(d_euc > rad_c[:, None], float('inf'))  # (G,Q)
            vals, idx = torch.topk(masked, k=min(k_max, P.shape[0]), largest=False)  # (G,K)
            valid = torch.isfinite(vals)                                        # (G,K)

            # 3) Gather 候选点 & 标签 (G,K,3)/(G,K)
            idx_safe = idx.masked_fill(~valid, 0)   # 占位
            P_sel = P[idx_safe]                     # (G,K,3)
            y_sel = y[idx_safe]                     # (G,K)
            
            # 确保y_sel中的所有索引都在有效范围内
            y_sel = torch.clamp(y_sel, 0, L-1)

            # 4) Mahalanobis（白化欧氏）：whi = Σ^{-1/2} (p - μ)
            diff = P_sel - mu_c[:, None, :]         # (G,K,3)
            whi  = torch.einsum('gij,gkj->gki', S_c, diff)  # (G,K,3)
            d2   = (whi * whi).sum(-1)              # (G,K)

            # 5) 计算投票权重（掩掉无效候选）
            if method == "density":
                w = torch.exp(-0.5 * d2)
            elif method == "min_dist":
                w = d2
            else:
                raise ValueError("method must be 'density' or 'min_dist'.")
            w = w * valid.float()                   # (G,K)
            if use_alpha_volume:
                w = w * gw_c[:, None]

            # 6) 向量化加权投票: (G,L) = scatter_add_ over (G,K)
            scores = torch.zeros(G, L, device=device)
            
            # 添加调试信息和安全检查
            try:
                if y_sel.min() < 0 or y_sel.max() >= L:
                    print(f"Warning: Invalid label indices detected: min={y_sel.min()}, max={y_sel.max()}, L={L}")
                    y_sel = torch.clamp(y_sel, 0, L-1)
                    
                scores.scatter_add_(dim=1, index=y_sel, src=w)

                if class_balance:
                    counts = torch.zeros(G, L, device=device)
                    counts.scatter_add_(dim=1, index=y_sel, src=valid.float())
                    scores = scores / counts.clamp_min(1.0)
            except RuntimeError as e:
                print(f"Error in scatter_add_: {e}")
                print(f"scores shape: {scores.shape}, y_sel shape: {y_sel.shape}, w shape: {w.shape}")
                print(f"y_sel range: [{y_sel.min()}, {y_sel.max()}], L: {L}")
                # 使用安全的回退方法
                for g in range(G):
                    for k in range(y_sel.shape[1]):
                        if valid[g, k]:
                            label = int(y_sel[g, k].item())
                            if 0 <= label < L:
                                scores[g, label] += w[g, k]

            # 7) 批内一次性取 argmax -> 伪标签
            labels_out[s:e] = torch.argmax(scores, dim=1)

            if verbose and ((s // chunk_size) % 10 == 0 or e == N):
                print(f"[assign] processed {e}/{N}")

        return labels_out.cpu().numpy()
    
    # Keep the original method for backward compatibility
    def assign_semantic_labels(self, gaussians_params: Dict, point_cloud: np.ndarray, 
                             point_labels: np.ndarray, unique_labels: List) -> np.ndarray:
        """
        Original method - use optimized version for large datasets.
        """
        # For large datasets, automatically use optimized version
        if gaussians_params['mu'].shape[0] > 10000 or point_cloud.shape[0] > 10000:
            print("Large dataset detected, using optimized version...")
            # return self.assign_semantic_labels_vectorized(
            #     gaussians_params, point_cloud, point_labels, unique_labels
            # )
            return self.assign_semantic_labels_mahalanobis(
                gaussians_params, point_cloud, point_labels, unique_labels
            )
        
        # Original implementation for small datasets
        mu = gaussians_params['mu']  # [N, 3]
        scale = gaussians_params['scale']  # [N, 3]
        rotation = gaussians_params['rotation']  # [N, 4] or [N, 3, 3]
        
        N = mu.shape[0]  # Number of Gaussians
        Q = point_cloud.shape[0]  # Number of points
        
        gaussian_labels = []
        
        for i in range(N):
            # Construct covariance matrix for i-th Gaussian
            cov_matrix = self.construct_covariance_matrix(scale[i], rotation[i])
            
            # Dictionary to store sum of Mahalanobis distances for each label
            label_distance_sums = {label: 0.0 for label in unique_labels}
            
            # Compute distances to all points
            for k in range(Q):
                point = point_cloud[k]
                point_label = point_labels[k]
                
                # Compute Mahalanobis distance
                distance = self.compute_mahalanobis_distance(point, mu[i], cov_matrix)
                
                # Add to the sum for this label
                label_distance_sums[point_label] += distance
            
            # Assign label with highest sum of distances (Eq. 8)
            assigned_label = max(label_distance_sums.keys(), 
                               key=lambda s: label_distance_sums[s])
            gaussian_labels.append(assigned_label)
        
        return np.array(gaussian_labels)
    
    def compute_significance_scores(self, scale: np.ndarray, opacity: np.ndarray) -> np.ndarray:
        """
        Compute significance scores for each Gaussian based on volume and opacity.
        
        Args:
            scale: Scale parameters [N, 3]
            opacity: Opacity values [N] or [N, 1]
            
        Returns:
            Significance scores [N]
        """
        # Volume of ellipsoid: (4/3) * π * sx * sy * sz
        # We omit the constant factor as it cancels out in IoU computation
        volumes = scale[:, 0] * scale[:, 1] * scale[:, 2]
        
        # Ensure opacity is 1D array to avoid broadcasting issues
        opacity_1d = opacity.flatten() if opacity.ndim > 1 else opacity
        
        significance_scores = volumes * opacity_1d
        return significance_scores
    
    def compute_volume_aware_iou(self, pred_labels: np.ndarray, gt_labels: np.ndarray,
                                significance_scores: np.ndarray, unique_labels: List, 
                                only_present_classes: bool = True) -> Dict:
        """
        Compute volume-aware IoU for each semantic class.
        
        Args:
            pred_labels: Predicted labels for each Gaussian [N]
            gt_labels: Ground truth labels for each Gaussian [N]
            significance_scores: Significance scores for each Gaussian [N]
            unique_labels: List of unique semantic labels
            only_present_classes: If True, only compute IoU for classes present in GT
            
        Returns:
            Dictionary containing IoU for each class and mean IoU
        """
        ious = {}
        pred_labels[gt_labels==0]=0
        # Get classes actually present in GT
        gt_present_labels = np.unique(gt_labels)
        gt_present_labels = gt_present_labels[gt_present_labels > 0]  # Remove unlabeled (0)
        
        if only_present_classes:
            # Only compute IoU for classes present in GT
            eval_labels = [label for label in unique_labels if label in gt_present_labels]
            print(f"Computing IoU for {len(eval_labels)} classes present in GT: {eval_labels}")
        else:
            # Compute for all unique_labels (original behavior)
            eval_labels = unique_labels
            print(f"Computing IoU for all {len(eval_labels)} predefined classes")
        
        for label in eval_labels:
            # Create binary vectors for current label
            pred_binary = (pred_labels == label).astype(float)
            gt_binary = (gt_labels == label).astype(float)
            
            # Compute weighted intersection and union (Eq. 9)
            intersection = np.sum(significance_scores * (pred_binary * gt_binary))
            union = np.sum(significance_scores * (pred_binary + gt_binary - pred_binary * gt_binary))
            
            # Compute IoU
            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0  # No ground truth or prediction for this class
            
            ious[label] = iou
        
        # Compute mean IoU - only over evaluated classes
        iou_values = list(ious.values())
        mean_iou = np.mean(iou_values) if iou_values else 0.0
        ious['mean_iou'] = mean_iou
        ious['evaluated_classes'] = eval_labels
        ious['num_evaluated_classes'] = len(eval_labels)
        ious['gt_present_classes'] = gt_present_labels.tolist()
        
        return ious
    
    def compute_volume_aware_accuracy(self, pred_labels: np.ndarray, gt_labels: np.ndarray,
                                    significance_scores: np.ndarray, unique_labels: List, 
                                    only_present_classes: bool = True) -> Dict:
        """
        Compute volume-aware accuracy metrics including per-class accuracy and mAcc.
        
        Args:
            pred_labels: Predicted labels for each Gaussian [N]
            gt_labels: Ground truth labels for each Gaussian [N]
            significance_scores: Significance scores for each Gaussian [N]
            unique_labels: List of unique semantic labels
            only_present_classes: If True, only compute accuracy for classes present in GT
            
        Returns:
            Dictionary containing accuracy metrics
        """
        accuracies = {}
        
        # Overall accuracy (volume-weighted)
        correct_predictions = (pred_labels == gt_labels).astype(float)
        overall_accuracy = np.sum(significance_scores * correct_predictions) / np.sum(significance_scores)
        
        # Get classes actually present in GT
        gt_present_labels = np.unique(gt_labels)
        gt_present_labels = gt_present_labels[gt_present_labels > 0]  # Remove unlabeled (0)
        
        if only_present_classes:
            # Only compute accuracy for classes present in GT
            eval_labels = [label for label in unique_labels if label in gt_present_labels]
            print(f"Computing accuracy for {len(eval_labels)} classes present in GT: {eval_labels}")
        else:
            # Compute for all unique_labels (original behavior)
            eval_labels = unique_labels
            print(f"Computing accuracy for all {len(eval_labels)} predefined classes")
        
        # Per-class accuracy
        class_accuracies = []
        
        for label in eval_labels:
            # Find all Gaussians with this ground truth label
            gt_mask = (gt_labels == label)
            
            if np.sum(gt_mask) == 0:
                # No ground truth instances for this class
                class_acc = 0.0
            else:
                # Compute weighted accuracy for this class
                class_significance = significance_scores[gt_mask]
                class_correct = correct_predictions[gt_mask]
                
                if np.sum(class_significance) > 0:
                    class_acc = np.sum(class_significance * class_correct) / np.sum(class_significance)
                else:
                    class_acc = 0.0
            
            accuracies[f'class_{label}_acc'] = class_acc
            class_accuracies.append(class_acc)
        
        # Mean class accuracy (mAcc) - now only over classes actually evaluated
        mean_class_accuracy = np.mean(class_accuracies) if class_accuracies else 0.0
        
        accuracies.update({
            'overall_accuracy': overall_accuracy,
            'mean_class_accuracy': mean_class_accuracy,
            'per_class_accuracies': class_accuracies,
            'evaluated_classes': eval_labels,
            'num_evaluated_classes': len(eval_labels),
            'gt_present_classes': gt_present_labels.tolist()
        })
        
        return accuracies
    
    def compute_standard_accuracy(self, pred_labels: np.ndarray, gt_labels: np.ndarray, 
                                unique_labels: List) -> Dict:
        """
        Compute standard (unweighted) accuracy metrics for comparison.
        
        Args:
            pred_labels: Predicted labels for each Gaussian [N]
            gt_labels: Ground truth labels for each Gaussian [N]
            unique_labels: List of unique semantic labels
            
        Returns:
            Dictionary containing standard accuracy metrics
        """
        accuracies = {}
        
        # Overall accuracy
        overall_accuracy = np.mean(pred_labels == gt_labels)
        
        # Per-class accuracy
        class_accuracies = []
        
        for label in unique_labels:
            # Find all Gaussians with this ground truth label
            gt_mask = (gt_labels == label)
            
            if np.sum(gt_mask) == 0:
                # No ground truth instances for this class
                class_acc = 0.0
            else:
                # Standard accuracy for this class
                class_correct = np.sum((pred_labels[gt_mask] == gt_labels[gt_mask]))
                class_total = np.sum(gt_mask)
                class_acc = class_correct / class_total
            
            accuracies[f'std_class_{label}_acc'] = class_acc
            class_accuracies.append(class_acc)
        
        # Mean class accuracy (standard mAcc)
        mean_class_accuracy = np.mean(class_accuracies)
        
        accuracies.update({
            'std_overall_accuracy': overall_accuracy,
            'std_mean_class_accuracy': mean_class_accuracy,
            'std_per_class_accuracies': class_accuracies
        })
        
        return accuracies
    
    def compute_volume_aware_iou_and_accuracy(self, pred_labels: np.ndarray, gt_labels: np.ndarray,
                                significance_scores: np.ndarray, num_classes: int) -> Dict:
        
        pred_labels[gt_labels == 0] = 0
        
        total_classes = num_classes + 1

        ious = np.zeros(total_classes)

        intersection = np.zeros(total_classes)
        union = np.zeros(total_classes)
        correct = np.zeros(total_classes)
        total = np.zeros(total_classes)

        for cls in range(1, total_classes):
            pred_binary = (pred_labels == cls).astype(float)
            gt_binary = (gt_labels == cls).astype(float)
            
            intersection[cls] = np.sum(significance_scores * (gt_binary * pred_binary))
            union[cls] = np.sum(significance_scores * (gt_binary + pred_binary - gt_binary * pred_binary))
            
            correct[cls] = np.sum(significance_scores * (gt_binary * pred_binary))
            total[cls] = np.sum(significance_scores * gt_binary)
            
            # intersection[cls] = torch.sum(significance_scores * (gt_labels == cls) & (pred_labels == cls)).item()
            # union[cls] = torch.sum(significance_scores * ((gt_labels == cls) | (pred_labels == cls))).item()
            # correct[cls] = torch.sum(significance_scores * ((gt_labels == cls) & (pred_labels == cls))).item()
            # total[cls] = torch.sum(significance_scores * (gt_labels == cls)).item()

        valid_union = union != 0
        ious[valid_union] = intersection[valid_union] / union[valid_union]

        # Only consider the categories that exist in the current scene
        gt_classes = np.unique(gt_labels)
        valid_gt_classes = gt_classes[gt_classes != 0]  # ignore 0

        # miou
        mean_iou = ious[valid_gt_classes].mean().item()

        # acc
        valid_mask = (gt_labels != 0).astype(float)
        correct_binary = (gt_labels == pred_labels).astype(float)
        correct_predictions = np.sum(significance_scores * correct_binary * valid_mask)
        total_valid_points = np.sum(significance_scores * valid_mask)
        accuracy = correct_predictions / total_valid_points if total_valid_points > 0 else float('nan')

        class_accuracy = correct / total
        # mAcc.
        mean_class_accuracy = class_accuracy[valid_gt_classes].mean()
        
        iou_and_acc = {}
        iou_and_acc['ious'] = ious
        iou_and_acc['mean_iou'] = mean_iou
        iou_and_acc['accuracy'] = accuracy
        iou_and_acc['mean_class_accuracy'] = mean_class_accuracy

        return iou_and_acc
    
    def evaluate(self, gaussians_params: Dict, predicted_labels: np.ndarray,
                point_cloud: np.ndarray, point_labels: np.ndarray, 
                only_present_classes: bool = True, num_classes: int = 19) -> Dict:
        """
        Complete evaluation pipeline for Gaussian-friendly evaluation.
        
        Args:
            gaussians_params: Dictionary containing Gaussian parameters
            predicted_labels: Predicted semantic labels for Gaussians [N]
            point_cloud: Ground truth point cloud [Q, 3]
            point_labels: Semantic labels for each point [Q]
            only_present_classes: If True, only evaluate classes present in GT
            
        Returns:
            Dictionary containing evaluation results
        """
        # Convert PyTorch tensors to numpy arrays if necessary
        
        if hasattr(predicted_labels, 'cpu'):
            predicted_labels = predicted_labels.cpu().detach().numpy()
        if hasattr(point_cloud, 'cpu'):
            point_cloud = point_cloud.cpu().detach().numpy()
        if hasattr(point_labels, 'cpu'):
            point_labels = point_labels.cpu().detach().numpy()
        
        # Get unique labels
        unique_labels = np.unique(point_labels).tolist()
        
        # Assign ground truth labels to Gaussians
        pseudo_gt_gaussian_labels = self.assign_semantic_labels(
            gaussians_params, point_cloud, point_labels, unique_labels
        )
        print(pseudo_gt_gaussian_labels.shape)
        print(predicted_labels.shape)
        print(point_cloud.shape)
        print(point_labels.shape)
        print(unique_labels)
        print(gaussians_params['scale'].shape)
        print(gaussians_params['opacity'].shape)
        
        # Compute significance scores
        significance_scores = self.compute_significance_scores(
            gaussians_params['scale'], gaussians_params['opacity']
        )
        
        # Compute volume-aware IoU
        
        ious = self.compute_volume_aware_iou(
            predicted_labels, pseudo_gt_gaussian_labels, significance_scores, unique_labels,
            only_present_classes=only_present_classes
        )
        
        # Compute volume-aware accuracy metrics
        volume_aware_acc = self.compute_volume_aware_accuracy(
            predicted_labels, pseudo_gt_gaussian_labels, significance_scores, unique_labels,
            only_present_classes=only_present_classes
        )
        
        # Compute standard accuracy metrics for comparison
        standard_acc = self.compute_standard_accuracy(
            predicted_labels, pseudo_gt_gaussian_labels, unique_labels
        )
        
        ious_and_acc = self.compute_volume_aware_iou_and_accuracy(
            predicted_labels, pseudo_gt_gaussian_labels, significance_scores, num_classes=num_classes
        )
        
        return {
            'ious': ious,
            'ious_and_acc': ious_and_acc,
            'volume_aware_accuracy': volume_aware_acc,
            'standard_accuracy': standard_acc,
            'pseudo_gt_gaussian_labels': pseudo_gt_gaussian_labels,
            'significance_scores': significance_scores
        }


# Example usage for large datasets
def example_usage_large_dataset():
    """
    Example optimized for large datasets (2M Gaussians, 60K points)
    """
    # Initialize evaluator
    evaluator = GaussianEvaluationProtocol()
    
    # Large dataset parameters
    N = 2_000_000  # 2M Gaussians
    Q = 60_000     # 60K points
    num_classes = 20
    
    print(f"Creating large dataset: {N} Gaussians, {Q} points")
    
    # Gaussian parameters (use float32 to save memory)
    gaussians_params = {
        'mu': np.random.randn(N, 3).astype(np.float32),
        'scale': (np.random.rand(N, 3) * 0.1 + 0.01).astype(np.float32),
        'rotation': np.random.randn(N, 4).astype(np.float32),
        'opacity': np.random.rand(N).astype(np.float32)
    }
    
    # Normalize quaternions
    gaussians_params['rotation'] = gaussians_params['rotation'] / np.linalg.norm(
        gaussians_params['rotation'], axis=1, keepdims=True
    )
    
    # Ground truth data
    point_cloud = np.random.randn(Q, 3).astype(np.float32)
    point_labels = np.random.randint(0, num_classes, Q)
    
    # Predicted labels for Gaussians
    predicted_labels = np.random.randint(0, num_classes, N)
    
    print("Starting evaluation with memory optimization...")
    
    # Run evaluation with optimized methods
    import time
    start_time = time.time()
    
    results = evaluator.evaluate(
        gaussians_params, predicted_labels, point_cloud, point_labels
    )
    
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    
    print("\nVolume-aware IoU results:")
    for label, iou in results['ious'].items():
        if isinstance(label, int):  # Skip 'mean_iou' key
            print(f"Class {label}: {iou:.4f}")
    print(f"Mean IoU: {results['ious']['mean_iou']:.4f}")
    
    print("\nVolume-aware Accuracy results:")
    print(f"Overall Accuracy: {results['volume_aware_accuracy']['overall_accuracy']:.4f}")
    print(f"Mean Class Accuracy (mAcc): {results['volume_aware_accuracy']['mean_class_accuracy']:.4f}")
    
    print("\nPer-class Volume-aware Accuracies:")
    for i, acc in enumerate(results['volume_aware_accuracy']['per_class_accuracies']):
        print(f"Class {i}: {acc:.4f}")
    
    print("\nStandard Accuracy results (for comparison):")
    print(f"Standard Overall Accuracy: {results['standard_accuracy']['std_overall_accuracy']:.4f}")
    print(f"Standard mAcc: {results['standard_accuracy']['std_mean_class_accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    # Run example with large dataset
    results = example_usage_large_dataset()