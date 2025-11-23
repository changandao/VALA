#!/bin/bash
# 设置 PYTHONPATH 环境变量，添加项目根目录
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd $(dirname "$0")

# python render_lerf_by_feat_langsplat.py -s 
export CUDA_VISIBLE_DEVICES=0
# cases="teatime"
cases="teatime figurines waldo_kitchen ramen"
# cases="waldo_kitchen"
OUTPUT_DIR="lerf_ovs_occamlgs_ablation"
mask_thresh=0.6

for case in $cases
do
echo "Processing $case"

BASE_DIR="output/3dgs/$OUTPUT_DIR/$case"

# 使用模块方式运行，确保当前目录在 Python 路径中
python -m eval_ablation.render_lerf_by_text_langsplat -s dataset/3dgs/lerf_ovs/$case -m output/3dgs/$OUTPUT_DIR/$case \
--iteration 30000 --mask_thresh $mask_thresh --scene_name $case --dataset_name lerf_ovs \
--ae_ckpt_dir dataset/3dgs/lerf_ovs/ckpt/ins --base_dir $BASE_DIR --output_dir $BASE_DIR --eval --skip_train


# # --scene_name $case --dataset_name lerf_ovs \
# # --ae_ckpt_dir output/3dgs --base_dir $BASE_DIR --output_dir $BASE_DIR

# python -m eval.compute_lerf_iou --scene_name $case --gt_dir output/3dgs/lerf_ovs/$case/gt \
# --pred_dir $BASE_DIR/predictions_mask_$mask_thresh --output_dir $BASE_DIR/predictions_mask_$mask_thresh
done

python -m eval_ablation.compute_lerf_iou  --gt_dir output/3dgs/lerf_ovs \
--pred_dir output/3dgs/$OUTPUT_DIR --output_dir output/3dgs/$OUTPUT_DIR --iteration 30015 --mask_thresh $mask_thresh
