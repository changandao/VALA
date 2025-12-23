#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

# python render_lerf_by_feat_langsplat.py -s 
export CUDA_VISIBLE_DEVICES=0
cases="teatime figurines waldo_kitchen ramen"
OUTPUT_DIR="lerf_ovs"
mask_thresh=0.6

for case in $cases
do
    echo "Processing $case"

    BASE_DIR="$ROOT_DIR/output/3dgs/$OUTPUT_DIR/$case"

    # use module to run the script
    python -m eval.render_lerf_by_text_langsplat -s "$ROOT_DIR/dataset/3dgs/lerf_ovs/$case" -m "$ROOT_DIR/output/3dgs/$OUTPUT_DIR/$case" \
    --iteration 30000 --mask_thresh $mask_thresh --scene_name $case --dataset_name lerf_ovs \
    --ae_ckpt_dir "$ROOT_DIR/dataset/3dgs/lerf_ovs/ckpt/ins" --base_dir "$BASE_DIR" --output_dir "$BASE_DIR" --eval --skip_train

    # python -m eval.compute_lerf_iou --scene_name $case --gt_dir output/3dgs/lerf_ovs \
    # --json_dir $ROOT_DIR/dataset/3dgs/lerf_ovs/label --mask_thresh $mask_thresh --iteration 30000 \
    # --pred_dir $ROOT_DIR/output/3dgs/$OUTPUT_DIR --output_dir $BASE_DIR/predictions_mask_$mask_thresh
done

python -m eval.compute_lerf_iou  --gt_dir "$ROOT_DIR/output/3dgs/lerf_ovs" \
--pred_dir "$ROOT_DIR/output/3dgs/$OUTPUT_DIR" --output_dir "$ROOT_DIR/output/3dgs/$OUTPUT_DIR" --iteration 30000 --mask_thresh $mask_thresh \
--ablation_type none   --json_dir $ROOT_DIR/dataset/3dgs/lerf_ovs/label