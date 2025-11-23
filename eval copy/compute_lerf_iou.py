import json
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser

def load_image_as_binary(image_path, is_png=False, threshold=10):
    image = Image.open(image_path)
    if is_png:
        image = image.convert('L')
    image_array = np.array(image)
    binary_image = (image_array > threshold).astype(int)
    return binary_image

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def evalute(gt_base, pred_base, output_dir=None, eval_args=None):
    scene_name = eval_args["scene_name"]
    iteration = eval_args["iteration"]
    mask_thresh = eval_args["mask_thresh"]
    
    # gt_base = os.path.join(gt_base, scene_name, 'gt')
    # pred_base = os.path.join(pred_base, scene_name, 'predictions_mask_' + str(mask_thresh))
    
    
    scene_gt_frames = {
        "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
        "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
        "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
        "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
    }
    frame_names = scene_gt_frames[scene_name]

    ious = []
    for frame in frame_names:
        print("frame:", frame)
        gt_floder = os.path.join(gt_base, frame)
        file_names = [f for f in os.listdir(gt_floder) if f.endswith('.jpg')]
        pred_floder = os.path.join(pred_base, "renders_silhouette", frame)
        print("file_names:", file_names)
        for file_name in file_names:
            base_name = os.path.splitext(file_name)[0]
            # print("base_name:", base_name)
            gt_obj_path = os.path.join(gt_floder, file_name)
            pred_obj_path = os.path.join(pred_floder, base_name + '.png')
            # print("pred_obj_path:", pred_obj_path)
            if not os.path.exists(pred_obj_path):
                print(f"Missing pred file for {file_name}, skipping...")
                print(f"IoU for {file_name}: 0")
                ious.append(0.0)
                continue
            mask_gt = load_image_as_binary(gt_obj_path)
            mask_pred = load_image_as_binary(pred_obj_path, is_png=True)
            iou = calculate_iou(mask_gt, mask_pred)
            ious.append(iou)
            # print(f"IoU for {file_name} and {base_name + '.png'}: {iou:.4f}")
    
    # Acc.
    total_count = len(ious)
    count_iou_025 = (np.array(ious) > 0.25).sum()
    count_iou_05 = (np.array(ious) > 0.5).sum()

    # mIoU
    average_iou = np.mean(ious)
    print(f"Average IoU: {average_iou:.4f}")
    print(f"Acc@0.25: {count_iou_025/total_count:.4f}")
    print(f"Acc@0.5: {count_iou_05/total_count:.4f}")
    
    metrics = {
        "mIoU": average_iou,
        "Acc@0.25": count_iou_025/total_count,
        "Acc@0.5": count_iou_05/total_count
    }
    
    if output_dir is not None:
        # output_base = os.path.join(output_dir, scene_name, 'predictions_mask_' + str(mask_thresh))
        os.makedirs(output_dir, exist_ok=True)
        log_file_path = os.path.join(output_dir, "result.txt")
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"mean iou: {average_iou:.4f}\n")
            log_file.write(f"Acc@0.25: {count_iou_025/total_count:.4f}\n")
            log_file.write(f"Acc@0.5: {count_iou_05/total_count:.4f}")
            
    return metrics

if __name__ == "__main__":
    parser = ArgumentParser("Compute LeRF IoU")
    parser.add_argument("--scene_name", type=str, choices=["waldo_kitchen", "ramen", "figurines", "teatime"],
                        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    args = parser.parse_args()
    # if not args.scene_name:
    #     parser.error("The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime")

    # evalute(args.gt_dir, args.pred_dir, args.scene_name, args.output_dir)
    path_gt = args.gt_dir
    path_pred = args.pred_dir
    path_output = args.output_dir
    
    if args.scene_name:
        path_gt = os.path.join(args.gt_dir, args.scene_name, 'gt')
        # renders_cluster_silhouette is the predicted mask
        path_pred = os.path.join(args.pred_dir, args.scene_name, 'predictions')
        # path_pred = args.eval_dir
        eval_args = {
            "scene_name": args.scene_name,
            "iteration": args.iteration,
            "mask_thresh": args.mask_thresh,
            
        }
        single_scene_metrics = evalute(path_gt, path_pred, path_output, eval_args)
        all_scene_metrics = {
            args.scene_name: single_scene_metrics
        }
        
    else:
        # parser.error("The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime")
        all_scene_metrics = {}
        lerf_ovs_scenes = ["waldo_kitchen", "ramen", "figurines", "teatime"]
        # lerf_ovs_scenes = ["ramen", "figurines"]
        # lerf_ovs_scenes = ["teatime"]
        # lerf_ovs_scenes = ["teatime", "waldo_kitchen"]
        
        for scene_name in lerf_ovs_scenes:
            print(f"Processing scene: {scene_name}")
            path_gt = os.path.join(args.gt_dir, scene_name, 'gt')
            path_pred = os.path.join(args.pred_dir, scene_name, 'predictions_mask_' + str(args.mask_thresh))
            path_output = os.path.join(args.output_dir, scene_name, 'predictions_mask_' + str(args.mask_thresh))
            eval_args = {
                "scene_name": scene_name,
                "iteration": args.iteration,
                "mask_thresh": args.mask_thresh,
                
            }
            single_scene_metrics = evalute(path_gt, path_pred, path_output, eval_args)
            all_scene_metrics[scene_name] = single_scene_metrics
            
    # Calculate mean metrics across all evaluated scenes
    mean_scene_metrics = {
        "mIoU": np.mean([metrics["mIoU"] for metrics in all_scene_metrics.values()]),
        "Acc@0.25": np.mean([metrics["Acc@0.25"] for metrics in all_scene_metrics.values()]),
        "Acc@0.5": np.mean([metrics["Acc@0.5"] for metrics in all_scene_metrics.values()])
    }
      # Print mean results
    print("\nMean metrics across all scenes:")
    print(f"Mean mIoU: {mean_scene_metrics['mIoU']:.4f}")
    print(f"Mean Acc@0.25: {mean_scene_metrics['Acc@0.25']:.4f}")
    print(f"Mean Acc@0.5: {mean_scene_metrics['Acc@0.5']:.4f}")
    
    # Save all results including per-scene and mean metrics
    results = {
        "per_scene": all_scene_metrics,
        "mean": mean_scene_metrics
    }
    
    results_path = os.path.join(args.output_dir, f'all_metrics_{args.iteration}_{args.mask_thresh}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)