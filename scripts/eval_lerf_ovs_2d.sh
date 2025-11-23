export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"
# export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

PROJ_PATH="$ROOT_DIR"
# CASE_NAME="teatime"
# CASE_NAME="figurines"
# CASE_NAME="waldo_kitchen"
# CASE_NAME="ramen"

CASES="teatime figurines waldo_kitchen ramen"
for CASE_NAME in $CASES
do
        DATA_NAME=$(echo $CASE_NAME | cut -d'_' -f1)
        if [ $DATA_NAME = "waldo" ]; then
                DATA_NAME="waldo_kitchen"
        fi
        GT_FOLDER="dataset/3dgs/lerf_ovs/label" # path to json GT label file

        echo "Running Python script with case: $CASE_NAME"
        python -m eval_ablation.evaluate_iou_loc \
                --dataset_name $DATA_NAME \
                --feat_dir $PROJ_PATH/output/3dgs/lerf_ovs_occamlgs_ablation/ \
                --output_dir $PROJ_PATH/output/3dgs/lerf_ovs_occamlgs_ablation/ \
                --mask_thresh 0.4 \
                --json_folder $GT_FOLDER \
                --ablation_type gate_weights \
                --direct_512 \
                # --use_dynamic_thresh \
                # --stability_thresh 0.3 \
                # --min_mask_size 0.001 \
                # --max_mask_size 0.95
done