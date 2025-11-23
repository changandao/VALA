# CUDA_VISIBLE_DEVICES=0
# dataset=lerf_ovs

# if [ "$dataset" = "lerf_ovs" ]; then
#     # cases="figurines ramen waldo_kitchen"
#     # cases="teatime figurines ramen waldo_kitchen"
#     cases="teatime"
#     # cases=ramen
# elif [ "$dataset" = "replica" ]; then
#     cases="office0 office1 office2 office3 office4 room0 room1 room2"
# else
#     echo "Error: Unknown dataset '$dataset'" >&2
#     exit 1
# fi

# output_dataset=lerf_ovs_occamlgs_ablation

# for case in $cases; do
#     echo "Processing case: $case"
#     export CUDA_LAUNCH_BLOCKING=1
#     CAMERA_PATH=dataset/3dgs/${dataset}/${case}
#     OUTPUT_PATH=output/3dgs/${output_dataset}/${case}
#     TRAINED_3DGS_PATH=${OUTPUT_PATH}
#     python render_activate.py -s ${CAMERA_PATH} -m ${OUTPUT_PATH} --eval --skip_train\
#     --feature_level 1 --iteration 30000 --img_label red_apple \
#                         --img_save_label red_apple_test \
#                         --threshold 0.5 --ablation_type supp_wopruning \
#                         -l language_features_dim3 --save_ply
# done

CUDA_VISIBLE_DEVICES=0
dataset=waymo

if [ "$dataset" = "lerf_ovs" ]; then
    # cases="figurines ramen waldo_kitchen"
    # cases="teatime figurines ramen waldo_kitchen"
    cases="teatime"
    # cases=ramen
elif [ "$dataset" = "replica" ]; then
    cases="office0 office1 office2 office3 office4 room0 room1 room2"
elif [ "$dataset" = "waymo" ]; then
    cases="1534950_1cams"
else
    echo "Error: Unknown dataset '$dataset'" >&2
    exit 1
fi

output_dataset=waymo

for case in $cases; do
    for feature_level in 1 2 3; do
    echo "Processing case: $case"
    export CUDA_LAUNCH_BLOCKING=1
    CAMERA_PATH=dataset/3dgs/${dataset}/${case}
    OUTPUT_PATH=output/3dgs/${output_dataset}/${case}
    TRAINED_3DGS_PATH=${OUTPUT_PATH}
    python render_activate.py -s ${CAMERA_PATH} -m ${OUTPUT_PATH} --skip_test \
    --feature_level $feature_level --iteration 30000 --img_label house \
                            --img_save_label house_test_${feature_level} \
                            --threshold 0.6 --ablation_type cos_median \
                            -l language_features_dim3 --save_ply
    done
done