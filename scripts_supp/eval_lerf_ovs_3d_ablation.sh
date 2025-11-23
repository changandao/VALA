#!/bin/bash
# 统一切换到项目根目录，并设置 PYTHONPATH
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

# python render_lerf_by_feat_langsplat.py -s 
export CUDA_VISIBLE_DEVICES=0
cases="teatime"
# cases="teatime figurines waldo_kitchen ramen"
# cases="waldo_kitchen"
OUTPUT_DIR="lerf_ovs_occamlgs_ablation"
mask_thresh=0.6

for case in $cases
do
    echo "Processing $case"

    BASE_DIR="$ROOT_DIR/output/3dgs/$OUTPUT_DIR/$case"

    # 使用模块方式运行，确保通过 PYTHONPATH 加载到项目根目录
    python -m eval_supp.render_lerf_by_text_ablation -s "$ROOT_DIR/dataset/3dgs/lerf_ovs/$case" -m "$ROOT_DIR/output/3dgs/$OUTPUT_DIR/$case" \
    --iteration 30000 --mask_thresh $mask_thresh --scene_name $case --dataset_name lerf_ovs --ablation_type rpx30_vala_wogate  \
    --ae_ckpt_dir "$ROOT_DIR/dataset/3dgs/lerf_ovs/ckpt/ins" --base_dir "$BASE_DIR" --output_dir "$BASE_DIR" --eval --skip_train

    ablation_output_dir="lerf_ovs_ablation"
python -m eval_supp.compute_lerf_iou_ablation  --gt_dir "$ROOT_DIR/output/3dgs/lerf_ovs" --scene_name $case \
--pred_dir "$ROOT_DIR/output/3dgs/$OUTPUT_DIR" --output_dir "$ROOT_DIR/output/3dgs/$ablation_output_dir" --iteration 30001 --mask_thresh $mask_thresh \
--ablation_type rpx30_vala_wogate    


# # --scene_name $case --dataset_name lerf_ovs \
# # --ae_ckpt_dir output/3dgs --base_dir $BASE_DIR --output_dir $BASE_DIR

# python -m eval.compute_lerf_iou --scene_name $case --gt_dir output/3dgs/lerf_ovs/$case/gt \
# --pred_dir $BASE_DIR/predictions_mask_$mask_thresh --output_dir $BASE_DIR/predictions_mask_$mask_thresh
done

# ablation_output_dir="lerf_ovs_ablation"
# python -m eval_supp.compute_lerf_iou_ablation  --gt_dir "$ROOT_DIR/output/3dgs/lerf_ovs" \
# --pred_dir "$ROOT_DIR/output/3dgs/$OUTPUT_DIR" --output_dir "$ROOT_DIR/output/3dgs/$ablation_output_dir" --iteration 30001 --mask_thresh $mask_thresh \
# --ablation_type rpx10   